"""TitoOS - a minimal operating system for agents."""

from typing import TYPE_CHECKING, Any

from .agent import Agent, AgentState, Context, FunctionAgent, RestartPolicy
from .backends import AsyncBackend, ExecutionBackend, SerialBackend, ThreadBackend
from .bus import MessageBus
from .integrations import (
    ClockIntegration,
    CommandResult,
    FileSystemIntegration,
    HttpIntegration,
    HttpResponse,
    Integration,
    IntegrationError,
    IntegrationRegistry,
    ShellIntegration,
)
from .kernel import CHILD_FAILED, Kernel, StopReason, TickReport
from .message import BROADCAST, Message
from .persistence import AgentFactory, AgentRecord, Snapshot

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .platforms import (
        AsyncPlatformAgent,
        CallableAdapter,
        Outbound,
        PlatformAdapter,
        PlatformAgent,
        PlatformError,
        PlatformTurn,
    )

__all__ = [
    "Agent",
    "AgentFactory",
    "AgentRecord",
    "AgentState",
    "AsyncBackend",
    "AsyncPlatformAgent",
    "BROADCAST",
    "CHILD_FAILED",
    "CallableAdapter",
    "ClockIntegration",
    "CommandResult",
    "Context",
    "ExecutionBackend",
    "FileSystemIntegration",
    "FunctionAgent",
    "HttpIntegration",
    "HttpResponse",
    "Integration",
    "IntegrationError",
    "IntegrationRegistry",
    "Kernel",
    "Message",
    "MessageBus",
    "Outbound",
    "PlatformAdapter",
    "PlatformAgent",
    "PlatformError",
    "PlatformTurn",
    "RestartPolicy",
    "SerialBackend",
    "ShellIntegration",
    "Snapshot",
    "StopReason",
    "ThreadBackend",
    "TickReport",
]

#: Names served lazily from :mod:`titoos.platforms`. Importing that package is
#: cheap, but adapters for real frameworks are reached through it, and pulling
#: the platform layer in on every ``import titoos`` would invite a future
#: adapter to make the base import expensive. Resolving on first use keeps the
#: core import free of anything the caller did not ask for.
_PLATFORM_EXPORTS = frozenset(
    {
        "AsyncPlatformAgent",
        "CallableAdapter",
        "Outbound",
        "PlatformAdapter",
        "PlatformAgent",
        "PlatformError",
        "PlatformTurn",
    }
)


def __getattr__(name: str) -> Any:
    if name in _PLATFORM_EXPORTS:
        from . import platforms

        value = getattr(platforms, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


__version__ = "0.2.0"
