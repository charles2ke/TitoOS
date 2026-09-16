"""The platform layer: how agents written for another framework run here.

An :class:`~titoos.integrations.base.Integration` is a driver an agent *calls
out* through. A platform adapter is the opposite direction: it takes an agent
that already exists somewhere else — a LangGraph graph, an OpenAI Agents SDK
agent, a CrewAI crew, a remote A2A peer — and makes it a scheduled
:class:`~titoos.agent.Agent` inside this kernel, so it gets a mailbox,
supervision, restarts and snapshots like anything else.

Keeping those two directions apart is the whole point of this module. An
adapter never decides *when* it runs or *what* it sees: the kernel fixes both
at the tick barrier, hands the foreign agent one turn's worth of messages, and
puts whatever comes back on the bus. That is what lets a run stay reproducible
even though the thing doing the reasoning lives in someone else's library.

An adapter therefore only has to answer three questions:

* ``to_platform`` — what does this turn's mail look like to the platform?
* ``invoke`` / ``ainvoke`` — how is the platform actually run?
* ``from_platform`` — which messages should the result put on the bus?

Nothing here imports a third-party package. Adapters for real frameworks live
next to this module, import their SDK lazily through :func:`require`, and are
registered by name so selecting one never imports the ones you did not
install.
"""

from __future__ import annotations

import inspect
import threading
from dataclasses import dataclass, field
from importlib import import_module
from types import MappingProxyType, ModuleType
from typing import (
    Any,
    Callable,
    Iterator,
    Mapping,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

from ..agent import Agent, Context
from ..integrations.base import IntegrationError
from ..message import Message

#: Shared read-only mapping reused by outbound messages without metadata.
_EMPTY_METADATA: Mapping[str, Any] = MappingProxyType({})


class PlatformError(IntegrationError):
    """Raised when a platform invocation cannot be completed.

    It inherits from :class:`~titoos.integrations.base.IntegrationError` on
    purpose: to an agent, "the foreign framework said no" is the same kind of
    event as "the outside world said no", and code that already handles one
    handles the other. It is an ordinary exception, so it propagates out of
    ``step()`` and the usual supervision rules apply unchanged.
    """

    def __init__(self, message: str, *, platform: str = "", operation: str = "") -> None:
        super().__init__(message, integration=platform, operation=operation)
        self.platform = platform


class MissingDependency(PlatformError):
    """Raised when an adapter's third-party package is not installed.

    Adapters import their SDK lazily so that ``import titoos`` never drags in
    a framework you did not ask for. The price of that is a failure at
    construction time rather than at import time, so the message has to say
    exactly what to install.
    """


def require(module: str, *, extra: str, platform: str = "") -> ModuleType:
    """Import ``module`` or raise :class:`MissingDependency` naming the extra.

    Call it from an adapter's ``__init__``, never at module import time::

        sdk = require("langgraph", extra="langgraph", platform="langgraph")
    """
    try:
        return import_module(module)
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise MissingDependency(
            f"platform {platform or extra!r} needs the {module!r} package, "
            "which is not installed; install it with: pip install "
            f'"titoos[{extra}]"',
            platform=platform or extra,
        ) from exc


@dataclass(frozen=True)
class PlatformTurn:
    """Everything a platform sees for one tick: its mail, and nothing else.

    A turn is immutable and fully determined at the tick barrier, so replaying
    the same turns against the same platform is the closest thing to a
    reproducible run that a non-deterministic framework allows.
    """

    #: Name of the TitoOS agent wrapping the platform.
    agent: str
    tick: int
    messages: tuple[Message, ...] = ()
    #: Input for the very first turn of a platform nobody has written to yet.
    seed: Any = None

    @property
    def payloads(self) -> tuple[Any, ...]:
        """The payload of each message, in delivery order."""
        return tuple(message.payload for message in self.messages)

    @property
    def sender(self) -> str | None:
        """Who to answer by default: the sender of the first message."""
        return self.messages[0].sender if self.messages else None

    @property
    def text(self) -> str:
        """The turn rendered as one string, which is what most SDKs want."""
        if not self.messages:
            return "" if self.seed is None else str(self.seed)
        return "\n".join(str(payload) for payload in self.payloads)


@dataclass(frozen=True)
class Outbound:
    """One message a platform run wants to put on the bus.

    Leaving ``to`` as ``None`` means "answer whoever wrote to us", which is
    what a request/response agent almost always wants and what keeps an
    adapter from having to know the names of the agents around it.
    """

    payload: Any = None
    to: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    broadcast: bool = False

    def __post_init__(self) -> None:
        if self.broadcast and self.to is not None:
            raise ValueError("an outbound message is either broadcast or addressed")
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)) if self.metadata else _EMPTY_METADATA,
        )


@runtime_checkable
class AgentPlatform(Protocol):
    """The surface the kernel needs from any agent framework.

    Deliberately tiny: a translation in, a way to run the thing, a translation
    out. Whatever else an SDK offers is the adapter's business, not the
    kernel's.
    """

    name: str
    is_async: bool

    def to_platform(self, turn: PlatformTurn) -> Any:
        ...

    def from_platform(self, result: Any, turn: PlatformTurn) -> Sequence[Outbound]:
        ...

    def invoke(self, request: Any, turn: PlatformTurn) -> Any:
        ...


class PlatformAdapter:
    """Base class for adapters, with the translation most platforms want.

    Subclasses normally only set :attr:`name` and implement :meth:`invoke` (or
    :meth:`ainvoke`). Override the translation hooks when the platform speaks
    something richer than "one payload in, one payload out".
    """

    #: Name the adapter is registered and addressed under, e.g. ``"langgraph"``.
    name: str = ""
    #: True when :meth:`ainvoke` must be awaited, which requires ``AsyncBackend``.
    is_async: bool = False

    def __init__(self, name: str | None = None) -> None:
        if name is not None:
            self.name = name
        if not self.name:
            raise ValueError("platform adapter name must be a non-empty string")

    # --- translation -------------------------------------------------------

    def to_platform(self, turn: PlatformTurn) -> Any:
        """Turn one tick's mail into whatever the platform takes as input.

        The default passes a lone payload through untouched, a multi-message
        turn as a list, and the seed when the mailbox was empty.
        """
        payloads = turn.payloads
        if not payloads:
            return turn.seed
        if len(payloads) == 1:
            return payloads[0]
        return list(payloads)

    def from_platform(self, result: Any, turn: PlatformTurn) -> Sequence[Outbound]:
        """Turn a platform result into the messages to put on the bus.

        The default treats ``None`` as "nothing to say", passes
        :class:`Outbound` values (or a sequence of them) through, and wraps
        anything else as a single reply to the sender of the turn.
        """
        if result is None:
            return ()
        if isinstance(result, Outbound):
            return (result,)
        if isinstance(result, (list, tuple)) and result and all(
            isinstance(item, Outbound) for item in result
        ):
            return tuple(result)
        return (Outbound(payload=result),)

    # --- execution ---------------------------------------------------------

    def invoke(self, request: Any, turn: PlatformTurn) -> Any:
        """Run the platform for one turn. Synchronous adapters override this."""
        raise NotImplementedError(
            f"platform adapter {self.name!r} implements neither invoke() nor "
            "ainvoke()"
        )

    async def ainvoke(self, request: Any, turn: PlatformTurn) -> Any:
        """Await one turn of the platform.

        The default defers to :meth:`invoke` and awaits its result if it is
        awaitable, so an adapter written against a sync SDK still works on the
        asyncio backend. A genuinely blocking ``invoke`` stalls the tick
        there; run those on ``ThreadBackend`` instead.
        """
        result = self.invoke(request, turn)
        if inspect.isawaitable(result):
            return await result
        return result

    # --- lifecycle ---------------------------------------------------------

    def agent(self, name: str, **options: Any) -> "PlatformAgent":
        """Build the kernel agent that runs this platform under ``name``."""
        cls = AsyncPlatformAgent if self.is_async else PlatformAgent
        return cls(name, self, **options)

    def save_state(self) -> dict[str, Any]:
        """Durable conversation state, if the platform has any.

        The default saves nothing: many SDKs keep their state on a server or
        in an object that cannot be serialized, and pretending otherwise would
        produce snapshots that restore into a subtly different conversation.
        """
        return {}

    def load_state(self, data: dict[str, Any]) -> None:
        """Restore the state previously returned by :meth:`save_state`."""

    def close(self) -> None:
        """Release any resources held, e.g. a session or a subprocess."""

    def _fail(self, message: str, operation: str = "") -> PlatformError:
        return PlatformError(message, platform=self.name, operation=operation)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<{type(self).__name__} {self.name}>"


class CallableAdapter(PlatformAdapter):
    """Adapts a plain callable ``fn(request, turn)`` into an adapter.

    The platform equivalent of :class:`~titoos.agent.FunctionAgent`: useful
    for small bridges and, above all, for exercising agent wiring without
    installing an SDK or reaching the network. The callable may be
    ``async def``, in which case pass ``is_async=True``.
    """

    def __init__(
        self,
        name: str,
        fn: Callable[[Any, PlatformTurn], Any],
        *,
        is_async: bool = False,
    ) -> None:
        super().__init__(name)
        if not callable(fn):
            raise TypeError("CallableAdapter expects a callable")
        self._fn = fn
        self.is_async = is_async

    @property
    def wrapped(self) -> Callable[[Any, PlatformTurn], Any]:
        """The callable this adapter wraps."""
        return self._fn

    def invoke(self, request: Any, turn: PlatformTurn) -> Any:
        return self._fn(request, turn)


class PlatformAgent(Agent):
    """A kernel agent whose behaviour comes from a platform adapter.

    One tick is one platform turn. With an empty mailbox the agent calls
    :meth:`~titoos.agent.Context.wait` instead of invoking anything, so an
    idle foreign agent costs neither a scheduler slot nor a worker thread —
    which matters when the alternative is a model call per idle tick.

    ``seed`` gives the platform something to act on before anyone has written
    to it, for agents that start a conversation rather than answer one.
    ``reply_to`` (or ``broadcast=True``) says where results go when the
    platform does not address them itself; without either, replies go back to
    the sender of the turn.
    """

    def __init__(
        self,
        name: str,
        adapter: PlatformAdapter,
        *,
        seed: Any = None,
        reply_to: str | None = None,
        broadcast: bool = False,
        max_turns: int | None = None,
    ) -> None:
        super().__init__(name)
        if not isinstance(adapter, PlatformAdapter):
            raise TypeError("PlatformAgent expects a PlatformAdapter")
        if reply_to and broadcast:
            raise ValueError("reply_to and broadcast are mutually exclusive")
        if max_turns is not None and max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self.adapter = adapter
        self.seed = seed
        self.reply_to = reply_to
        self.broadcast = broadcast
        self.max_turns = max_turns
        #: Completed platform turns, which is what ``max_turns`` counts.
        self.turns = 0
        self._seeded = seed is None

    # --- the tick ----------------------------------------------------------

    def _begin(self, ctx: Context) -> PlatformTurn | None:
        """The turn to run, or ``None`` when there is nothing to do."""
        if not ctx.inbox and self._seeded:
            ctx.wait()
            return None
        seed = None if self._seeded else self.seed
        return PlatformTurn(
            agent=self.name,
            tick=ctx.tick,
            messages=tuple(ctx.inbox),
            seed=seed,
        )

    def _finish(self, ctx: Context, result: Any, turn: PlatformTurn) -> None:
        try:
            replies = self.adapter.from_platform(result, turn)
        except PlatformError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize the SDK's errors
            raise self._error("from_platform", exc) from exc
        for reply in replies:
            self._dispatch(ctx, reply, turn)
        self._seeded = True
        self.turns += 1
        if self.max_turns is not None and self.turns >= self.max_turns:
            ctx.exit()
            return
        # Nothing more to do until someone writes again; waiting keeps the
        # kernel from spending a platform call on an empty mailbox next tick.
        ctx.wait()

    def _dispatch(self, ctx: Context, reply: Outbound, turn: PlatformTurn) -> None:
        if reply.broadcast or (reply.to is None and self.broadcast):
            ctx.broadcast(reply.payload, **reply.metadata)
            return
        destination = reply.to or self.reply_to or turn.sender
        if not destination:
            raise PlatformError(
                f"platform {self.adapter.name!r} produced a reply for agent "
                f"{self.name!r} with no destination: the turn had no sender, "
                "so set reply_to=, broadcast=True, or address the Outbound",
                platform=self.adapter.name,
                operation="dispatch",
            )
        ctx.send(destination, reply.payload, **reply.metadata)

    def _request(self, turn: PlatformTurn) -> Any:
        try:
            return self.adapter.to_platform(turn)
        except PlatformError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize the SDK's errors
            raise self._error("to_platform", exc) from exc

    def _error(self, operation: str, exc: BaseException) -> PlatformError:
        return PlatformError(
            f"platform {self.adapter.name!r} failed during {operation} for "
            f"agent {self.name!r}: {exc}",
            platform=self.adapter.name,
            operation=operation,
        )

    def step(self, ctx: Context) -> None:
        turn = self._begin(ctx)
        if turn is None:
            return
        request = self._request(turn)
        try:
            result = self.adapter.invoke(request, turn)
        except PlatformError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize the SDK's errors
            raise self._error("invoke", exc) from exc
        if inspect.isawaitable(result):
            # This agent cannot await, and dropping the coroutine would lose
            # the platform's answer silently. Close it and say what to switch
            # to instead.
            close = getattr(result, "close", None)
            if callable(close):
                close()
            raise PlatformError(
                f"platform {self.adapter.name!r} returned an awaitable; set "
                "is_async = True on the adapter and run the kernel on "
                "AsyncBackend",
                platform=self.adapter.name,
                operation="invoke",
            )
        self._finish(ctx, result, turn)

    # --- persistence -------------------------------------------------------

    def save_state(self) -> dict[str, Any]:
        return {
            "turns": self.turns,
            "seeded": self._seeded,
            "platform": self.adapter.save_state(),
        }

    def load_state(self, data: dict[str, Any]) -> None:
        self.turns = int(data.get("turns", 0))
        self._seeded = bool(data.get("seeded", self.seed is None))
        self.adapter.load_state(dict(data.get("platform") or {}))


class AsyncPlatformAgent(PlatformAgent):
    """A :class:`PlatformAgent` for async-first SDKs.

    Requires the asyncio execution backend; the other backends reject it
    through the existing ``supports_async_agents`` check rather than any
    mechanism of its own.
    """

    async def step(self, ctx: Context) -> None:  # type: ignore[override]
        turn = self._begin(ctx)
        if turn is None:
            return
        request = self._request(turn)
        try:
            result = await self.adapter.ainvoke(request, turn)
        except PlatformError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize the SDK's errors
            raise self._error("invoke", exc) from exc
        self._finish(ctx, result, turn)


#: An adapter factory: either a callable, or ``"module:attribute"`` imported on
#: first use so an optional SDK is never imported by registration alone.
AdapterFactory = Union[Callable[..., PlatformAdapter], str]


class PlatformRegistry:
    """The adapters known by name, so a platform can be selected as data.

    Registration must not import anything: an adapter may be registered as
    ``"titoos.platforms.langgraph:LangGraphAdapter"`` and is only imported
    when something actually builds it. That is what keeps ``import titoos``
    free of every framework in the ecosystem.

    Thread-safe like the integration registry.
    """

    def __init__(self) -> None:
        self._factories: dict[str, AdapterFactory] = {}
        self._lock = threading.RLock()

    def register(
        self, name: str, factory: AdapterFactory, *, replace: bool = False
    ) -> None:
        """Register ``factory`` under ``name``."""
        if not name:
            raise ValueError("platform name must be a non-empty string")
        if not callable(factory) and not isinstance(factory, str):
            raise TypeError("factory must be callable or a 'module:attribute' string")
        with self._lock:
            if name in self._factories and not replace:
                raise ValueError(f"platform already registered: {name!r}")
            self._factories[name] = factory

    def unregister(self, name: str) -> None:
        with self._lock:
            self._factories.pop(name, None)

    def get(self, name: str) -> Callable[..., PlatformAdapter]:
        """The factory registered under ``name``, importing it if needed."""
        with self._lock:
            factory = self._factories.get(name)
            if factory is None:
                available = ", ".join(sorted(self._factories)) or "none"
                raise PlatformError(
                    f"no platform registered named {name!r}; available: {available}",
                    platform=name,
                )
            if isinstance(factory, str):
                factory = self._import(name, factory)
                self._factories[name] = factory
        return factory

    def create(self, name: str, /, *args: Any, **kwargs: Any) -> PlatformAdapter:
        """Build the adapter registered under ``name``."""
        adapter = self.get(name)(*args, **kwargs)
        if not isinstance(adapter, PlatformAdapter):
            raise PlatformError(
                f"factory for platform {name!r} returned "
                f"{type(adapter).__name__}, not a PlatformAdapter",
                platform=name,
            )
        return adapter

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._factories))

    @staticmethod
    def _import(name: str, target: str) -> Callable[..., PlatformAdapter]:
        module_name, _, attribute = target.partition(":")
        if not module_name or not attribute:
            raise PlatformError(
                f"platform {name!r} is registered as {target!r}, which is not "
                "of the form 'module:attribute'",
                platform=name,
            )
        try:
            module = import_module(module_name)
        except ImportError as exc:
            raise MissingDependency(
                f"platform {name!r} could not be imported from "
                f"{module_name!r}: {exc}",
                platform=name,
            ) from exc
        factory = getattr(module, attribute, None)
        if factory is None:
            raise PlatformError(
                f"module {module_name!r} has no attribute {attribute!r} for "
                f"platform {name!r}",
                platform=name,
            )
        return factory

    def __contains__(self, name: object) -> bool:
        with self._lock:
            return name in self._factories

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        with self._lock:
            return len(self._factories)


#: The registry adapters are published in, including out-of-tree ones.
registry = PlatformRegistry()


def register_platform(
    name: str, factory: AdapterFactory, *, replace: bool = False
) -> None:
    """Publish an adapter in the default :data:`registry`."""
    registry.register(name, factory, replace=replace)


def create_platform(name: str, /, *args: Any, **kwargs: Any) -> PlatformAdapter:
    """Build the adapter published under ``name`` in the default registry."""
    return registry.create(name, *args, **kwargs)


def available_platforms() -> tuple[str, ...]:
    """Every platform name published in the default registry."""
    return registry.names()
