"""Agent primitives: the base :class:`Agent` and its execution context."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from .message import Message

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .kernel import Kernel


class RestartPolicy(str, Enum):
    """What the kernel does when an agent raises out of :meth:`Agent.step`."""

    #: Leave the agent FAILED. The default, and the historical behaviour.
    NEVER = "never"
    #: Reset the agent to READY and run it again, up to ``max_restarts``.
    ON_FAILURE = "on_failure"


class AgentState(str, Enum):
    """Lifecycle state of an agent inside the kernel."""

    READY = "ready"
    RUNNING = "running"
    WAITING = "waiting"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Context:
    """Everything an agent needs during a single scheduling tick."""

    kernel: "Kernel"
    agent: "Agent"
    tick: int
    inbox: list[Message] = field(default_factory=list)

    def send(self, to: str, payload: Any = None, **metadata: Any) -> Message:
        return self.kernel.bus.post(self.agent.name, to, payload, **metadata)

    def broadcast(self, payload: Any = None, **metadata: Any) -> Message:
        return self.kernel.bus.broadcast(self.agent.name, payload, **metadata)

    def spawn(self, agent: "Agent") -> "Agent":
        """Register ``agent`` as a child of the running agent.

        The child is supervised by its parent: if it fails and exhausts its
        restarts, the parent is notified (see :meth:`Kernel.register`).
        """
        return self.kernel.register(agent, parent=self.agent.name)

    def children(self) -> tuple["Agent", ...]:
        """The agents spawned by the running agent that still exist."""
        return self.kernel.children_of(self.agent.name)

    def wait(self) -> None:
        """Block the agent until a message arrives.

        The agent stays alive but is skipped on subsequent ticks until its
        mailbox is non-empty, so an idle agent costs neither a scheduler slot
        nor a worker thread. Calling :meth:`exit` afterwards still ends it.
        """
        self.agent.state = AgentState.WAITING

    def exit(self) -> None:
        """Mark the running agent as finished; it will not be scheduled again."""
        self.agent.state = AgentState.DONE


class Agent:
    """Base class for every unit of work scheduled by the kernel."""

    #: How the kernel reacts when this agent raises. Override per subclass or
    #: per instance.
    restart_policy: "RestartPolicy" = RestartPolicy.NEVER
    #: Maximum number of restarts before the failure becomes permanent.
    max_restarts: int = 3

    def __init__(self, name: str) -> None:
        if not name:
            raise ValueError("agent name must be a non-empty string")
        self.name = name
        self.state = AgentState.READY
        #: Name of the agent that spawned this one, if any.
        self.parent: str | None = None
        #: How often the kernel has restarted this agent so far.
        self.restarts = 0

    def on_restart(self) -> None:
        """Hook called just before a restarted agent becomes runnable again.

        Override to reset any internal state the failed step may have left
        inconsistent. The default does nothing.
        """

    @property
    def is_alive(self) -> bool:
        """True while the agent may still run; waiting agents are alive."""
        return self.state in (
            AgentState.READY,
            AgentState.RUNNING,
            AgentState.WAITING,
        )

    def step(self, ctx: Context) -> None:
        """Perform one unit of work. Subclasses must override this.

        May also be declared ``async def``; such agents require the asyncio
        execution backend.
        """
        raise NotImplementedError

    def save_state(self) -> dict[str, Any]:
        """Return this agent's durable state as a plain, serializable dict.

        The default saves nothing: an agent is restored with its lifecycle
        state and supervision counters, but no internal attributes. Override
        together with :meth:`load_state` to persist real work.
        """
        return {}

    def load_state(self, data: dict[str, Any]) -> None:
        """Restore the state previously returned by :meth:`save_state`."""

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<{type(self).__name__} {self.name} {self.state.value}>"


class FunctionAgent(Agent):
    """Adapts a plain callable ``fn(ctx)`` into an :class:`Agent`."""

    def __init__(self, name: str, fn: Callable[[Context], None]) -> None:
        super().__init__(name)
        self._fn = fn

    def step(self, ctx: Context) -> None:
        self._fn(ctx)
