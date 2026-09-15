"""The integration layer: how agents reach the world outside the kernel.

An :class:`Integration` is the agent equivalent of a device driver. The kernel
routes messages between agents; an integration is the only sanctioned way for
an agent to touch anything else — an HTTP API, the filesystem, a subprocess,
the clock.

Integrations are installed on the kernel, not on agents, so the same agent code
can run against a real HTTP endpoint or a fake one in a test without changing a
line. Every call goes through :meth:`~titoos.agent.Context.call`, which keeps
the side effects of a tick attributable to a named agent and a named
integration.
"""

from __future__ import annotations

import threading
from typing import Any, Iterable, Iterator, Mapping


class IntegrationError(RuntimeError):
    """Raised when an integration call cannot be completed.

    Integrations wrap the errors of whatever library or protocol they speak in
    this type, so an agent can handle "the outside world said no" without
    knowing how the outside world was reached. It is an ordinary exception: it
    propagates out of ``step()`` and the usual supervision rules apply.
    """

    def __init__(self, message: str, *, integration: str = "", operation: str = "") -> None:
        super().__init__(message)
        self.integration = integration
        self.operation = operation


class Integration:
    """Base class for everything an agent can reach outside the kernel.

    Subclasses declare a :attr:`name` and implement operations as methods
    listed in :attr:`operations`. Only listed operations are callable, so an
    integration's surface is exactly what it advertises and never the rest of
    its Python attributes.
    """

    #: Name agents use to address this integration, e.g. ``"http"``.
    name: str = ""
    #: Operation names callable through :meth:`call`.
    operations: tuple[str, ...] = ()

    def __init__(self, name: str | None = None) -> None:
        if name is not None:
            self.name = name
        if not self.name:
            raise ValueError("integration name must be a non-empty string")

    def call(self, operation: str, /, *args: Any, **kwargs: Any) -> Any:
        """Invoke ``operation`` on this integration."""
        if operation not in self.operations:
            raise IntegrationError(
                f"unknown operation {operation!r} on integration {self.name!r}; "
                f"available: {', '.join(self.operations) or 'none'}",
                integration=self.name,
                operation=operation,
            )
        return getattr(self, operation)(*args, **kwargs)

    def close(self) -> None:
        """Release any resources held. Called by :meth:`Kernel.shutdown`."""

    def _fail(self, message: str, operation: str = "") -> IntegrationError:
        return IntegrationError(message, integration=self.name, operation=operation)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<{type(self).__name__} {self.name}>"


class IntegrationProxy:
    """A restricted view of an integration, limited to its operations.

    :meth:`Integration.call` already refuses anything outside
    :attr:`Integration.operations`; handing an agent the driver object itself
    would hand it every other attribute too, including ``close()``. Agents get
    this proxy instead, so the advertised surface is the whole surface.
    """

    __slots__ = ("_integration",)

    def __init__(self, integration: Integration) -> None:
        object.__setattr__(self, "_integration", integration)

    @property
    def name(self) -> str:
        return self._integration.name

    @property
    def operations(self) -> tuple[str, ...]:
        return self._integration.operations

    def call(self, operation: str, /, *args: Any, **kwargs: Any) -> Any:
        """Invoke ``operation``, exactly as :meth:`Context.call` would."""
        return self._integration.call(operation, *args, **kwargs)

    def __getattr__(self, attribute: str) -> Any:
        integration: Integration = object.__getattribute__(self, "_integration")
        if attribute not in integration.operations:
            raise AttributeError(
                f"integration {integration.name!r} does not advertise "
                f"{attribute!r}; available: "
                f"{', '.join(integration.operations) or 'none'}"
            )

        def operation(*args: Any, **kwargs: Any) -> Any:
            return integration.call(attribute, *args, **kwargs)

        operation.__name__ = attribute
        return operation

    def __setattr__(self, attribute: str, value: Any) -> None:
        raise AttributeError(
            f"integration {self.name!r} is not writable through a proxy"
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<IntegrationProxy {self.name}>"


class IntegrationRegistry:
    """The integrations installed on a kernel, addressed by name.

    Thread-safe like the message bus: agents running concurrently in a worker
    pool resolve integrations through the same registry.
    """

    def __init__(self) -> None:
        self._integrations: dict[str, Integration] = {}
        self._lock = threading.RLock()

    def install(self, integration: Integration) -> Integration:
        """Add ``integration``. Names must be unique."""
        if not isinstance(integration, Integration):
            raise TypeError("install() expects an Integration instance")
        with self._lock:
            if integration.name in self._integrations:
                raise ValueError(
                    f"integration already installed: {integration.name!r}"
                )
            self._integrations[integration.name] = integration
        return integration

    def uninstall(self, name: str) -> None:
        """Remove and close the integration named ``name``, if installed."""
        with self._lock:
            integration = self._integrations.pop(name, None)
        if integration is not None:
            integration.close()

    def get(self, name: str) -> Integration:
        with self._lock:
            integration = self._integrations.get(name)
        if integration is None:
            raise IntegrationError(
                f"no integration installed named {name!r}", integration=name
            )
        return integration

    def call(self, name: str, operation: str, /, *args: Any, **kwargs: Any) -> Any:
        return self.get(name).call(operation, *args, **kwargs)

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._integrations)

    def close(self) -> None:
        """Close every installed integration, even if some raise."""
        with self._lock:
            integrations = tuple(self._integrations.values())
        errors: list[BaseException] = []
        for integration in integrations:
            try:
                integration.close()
            except Exception as exc:  # noqa: BLE001 - one bad driver must not block the rest
                errors.append(exc)
        if errors:
            raise errors[0]

    def __contains__(self, name: object) -> bool:
        with self._lock:
            return name in self._integrations

    def __iter__(self) -> Iterator[Integration]:
        with self._lock:
            return iter(tuple(self._integrations.values()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._integrations)


def normalize_allowlist(values: Iterable[str] | None, what: str) -> frozenset[str]:
    """Return a non-empty, lower-cased allowlist or raise.

    Integrations that reach dangerous resources are default-deny: the caller
    must say what is permitted, rather than opting out of what is not.
    """
    allowed = frozenset(v.strip().lower() for v in (values or ()) if v and v.strip())
    if not allowed:
        raise ValueError(f"{what} must list at least one entry")
    return allowed


def redact(mapping: Mapping[str, Any] | None, keys: Iterable[str]) -> dict[str, Any]:
    """Copy ``mapping`` with ``keys`` replaced by a placeholder.

    Used when integrations describe themselves or a call in logs, so tokens in
    headers never end up in a report or a snapshot.
    """
    secret = {k.lower() for k in keys}
    return {
        k: ("***" if k.lower() in secret else v) for k, v in (mapping or {}).items()
    }
