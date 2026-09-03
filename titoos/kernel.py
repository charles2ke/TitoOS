"""The TitoOS kernel: registers agents and runs them cooperatively."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

from .agent import Agent, AgentState, Context, FunctionAgent
from .bus import MessageBus


@dataclass(frozen=True)
class TickReport:
    """Summary of a single scheduling tick."""

    tick: int
    ran: tuple[str, ...]
    failed: tuple[str, ...]


class Kernel:
    """A cooperative, single-threaded scheduler for :class:`Agent` objects.

    Agents are scheduled round-robin in registration order. Each tick every
    live agent gets exactly one :meth:`Agent.step` call with its pending
    messages delivered in the context inbox.
    """

    def __init__(self) -> None:
        self.bus = MessageBus()
        self._agents: dict[str, Agent] = {}
        self._tick = 0
        self.errors: list[tuple[str, BaseException]] = []

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def agents(self) -> tuple[Agent, ...]:
        return tuple(self._agents.values())

    def get(self, name: str) -> Agent:
        return self._agents[name]

    def register(self, agent: Agent) -> Agent:
        """Add ``agent`` to the kernel. Names must be unique."""
        if agent.name in self._agents:
            raise ValueError(f"agent already registered: {agent.name!r}")
        self._agents[agent.name] = agent
        self.bus.register(agent.name)
        return agent

    def spawn(self, name: str, fn: Callable[[Context], None]) -> Agent:
        """Register a callable ``fn(ctx)`` as an agent named ``name``."""
        return self.register(FunctionAgent(name, fn))

    def unregister(self, name: str) -> None:
        self._agents.pop(name, None)
        self.bus.unregister(name)

    def live_agents(self) -> Iterator[Agent]:
        return (agent for agent in self._agents.values() if agent.is_alive)

    def step(self) -> TickReport:
        """Run one scheduling tick and return what happened."""
        self._tick += 1
        ran: list[str] = []
        failed: list[str] = []
        for agent in list(self.live_agents()):
            inbox = self.bus.receive(agent.name)
            ctx = Context(kernel=self, agent=agent, tick=self._tick, inbox=inbox)
            agent.state = AgentState.RUNNING
            try:
                agent.step(ctx)
            except Exception as exc:  # noqa: BLE001 - a failing agent must not kill the OS
                agent.state = AgentState.FAILED
                self.errors.append((agent.name, exc))
                failed.append(agent.name)
                continue
            ran.append(agent.name)
            if agent.state is AgentState.RUNNING:
                agent.state = AgentState.READY
        return TickReport(tick=self._tick, ran=tuple(ran), failed=tuple(failed))

    def run(self, max_ticks: int = 100) -> list[TickReport]:
        """Run until every agent is finished or ``max_ticks`` is reached."""
        if max_ticks < 0:
            raise ValueError("max_ticks must not be negative")
        reports: list[TickReport] = []
        for _ in range(max_ticks):
            if not any(self.live_agents()):
                break
            reports.append(self.step())
        return reports
