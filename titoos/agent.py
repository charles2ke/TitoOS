"""Agent primitives: the base :class:`Agent` and its execution context."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from .message import Message

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .kernel import Kernel


class AgentState(str, Enum):
    """Lifecycle state of an agent inside the kernel."""

    READY = "ready"
    RUNNING = "running"
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
        return self.kernel.register(agent)

    def exit(self) -> None:
        """Mark the running agent as finished; it will not be scheduled again."""
        self.agent.state = AgentState.DONE


class Agent:
    """Base class for every unit of work scheduled by the kernel."""

    def __init__(self, name: str) -> None:
        if not name:
            raise ValueError("agent name must be a non-empty string")
        self.name = name
        self.state = AgentState.READY

    @property
    def is_alive(self) -> bool:
        return self.state in (AgentState.READY, AgentState.RUNNING)

    def step(self, ctx: Context) -> None:
        """Perform one unit of work. Subclasses must override this."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<{type(self).__name__} {self.name} {self.state.value}>"


class FunctionAgent(Agent):
    """Adapts a plain callable ``fn(ctx)`` into an :class:`Agent`."""

    def __init__(self, name: str, fn: Callable[[Context], None]) -> None:
        super().__init__(name)
        self._fn = fn

    def step(self, ctx: Context) -> None:
        self._fn(ctx)
