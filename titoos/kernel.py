"""The TitoOS kernel: registers agents and runs them cooperatively."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from types import TracebackType
from typing import Callable, Iterator

from .agent import Agent, AgentState, Context, FunctionAgent
from .bus import MessageBus
from .message import Message


class StopReason(str, Enum):
    """Why :meth:`Kernel.run` stopped."""

    #: Every agent reached DONE or FAILED.
    FINISHED = "finished"
    #: Live agents remain but all are waiting with empty mailboxes, so no
    #: further tick could change anything. Either the workflow is complete or
    #: the agents are deadlocked.
    QUIESCENT = "quiescent"
    #: The ``max_ticks`` budget ran out while agents were still runnable.
    MAX_TICKS = "max_ticks"


@dataclass(frozen=True)
class TickReport:
    """Summary of a single scheduling tick."""

    tick: int
    ran: tuple[str, ...]
    failed: tuple[str, ...]
    #: Live agents skipped this tick because they were waiting for a message.
    waiting: tuple[str, ...] = ()


class Kernel:
    """A cooperative scheduler for :class:`Agent` objects.

    Agents are scheduled round-robin in registration order. Each tick every
    live agent gets exactly one :meth:`Agent.step` call with its pending
    messages delivered in the context inbox.

    With ``max_workers=1`` (the default) agents run one after another on the
    calling thread. With ``max_workers > 1`` the agents of a tick run
    concurrently in a thread pool, which is useful when agents spend their
    time on I/O such as model or tool calls. Ticks stay as barriers: every
    agent of a tick finishes before the next tick begins, so messages sent
    during a tick are always delivered in the following one regardless of the
    worker count.

    The kernel can be used as a context manager to shut the pool down::

        with Kernel(max_workers=8) as kernel:
            kernel.run()
    """

    def __init__(self, max_workers: int = 1) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        self.bus = MessageBus()
        self.max_workers = max_workers
        self._agents: dict[str, Agent] = {}
        self._tick = 0
        self.errors: list[tuple[str, BaseException]] = []
        self.stop_reason: StopReason | None = None
        self._lock = threading.RLock()
        self._pool: ThreadPoolExecutor | None = None

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def agents(self) -> tuple[Agent, ...]:
        with self._lock:
            return tuple(self._agents.values())

    def get(self, name: str) -> Agent:
        with self._lock:
            return self._agents[name]

    def register(self, agent: Agent) -> Agent:
        """Add ``agent`` to the kernel. Names must be unique.

        Safe to call from a worker thread, e.g. via :meth:`Context.spawn`.
        Agents registered during a tick start running on the next one.
        """
        with self._lock:
            if agent.name in self._agents:
                raise ValueError(f"agent already registered: {agent.name!r}")
            self._agents[agent.name] = agent
            self.bus.register(agent.name)
        return agent

    def spawn(self, name: str, fn: Callable[[Context], None]) -> Agent:
        """Register a callable ``fn(ctx)`` as an agent named ``name``."""
        return self.register(FunctionAgent(name, fn))

    def unregister(self, name: str) -> None:
        with self._lock:
            self._agents.pop(name, None)
        self.bus.unregister(name)

    def _unregister_finished_mailbox(self, agent: Agent) -> None:
        if agent.state in (AgentState.DONE, AgentState.FAILED):
            self.bus.unregister(agent.name)

    def live_agents(self) -> Iterator[Agent]:
        """Every agent that may still run, including waiting ones."""
        with self._lock:
            snapshot = tuple(self._agents.values())
        return (agent for agent in snapshot if agent.is_alive)

    def is_quiescent(self) -> bool:
        """True when live agents remain but none of them can make progress.

        That is the case when every live agent is waiting and no message is
        pending for any of them, so running further ticks is pointless.
        """
        live = tuple(self.live_agents())
        if not live:
            return False
        return all(
            agent.state is AgentState.WAITING and not self.bus.pending(agent.name)
            for agent in live
        )

    def _run_agent(self, agent: Agent, tick: int, inbox: list[Message]) -> bool:
        """Run a single agent for one tick. Returns ``True`` if it succeeded."""
        ctx = Context(kernel=self, agent=agent, tick=tick, inbox=inbox)
        agent.state = AgentState.RUNNING
        try:
            agent.step(ctx)
        except Exception as exc:  # noqa: BLE001 - a failing agent must not kill the OS
            agent.state = AgentState.FAILED
            with self._lock:
                self.errors.append((agent.name, exc))
            return False
        if agent.state is AgentState.RUNNING:
            agent.state = AgentState.READY
        return True

    def _executor(self) -> ThreadPoolExecutor:
        with self._lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(
                    max_workers=self.max_workers, thread_name_prefix="titoos"
                )
            return self._pool

    def step(self) -> TickReport:
        """Run one scheduling tick and return what happened.

        Mailboxes are drained before any agent runs, so a message sent during
        a tick is always delivered on the following one. The report lists
        agents in registration order, not completion order, so results stay
        deterministic regardless of the worker count.

        Waiting agents are skipped unless a message was pending for them at
        the tick boundary, in which case they wake and run with it.
        """
        self._tick += 1
        tick = self._tick
        # Drain every mailbox up front so the set of messages an agent sees is
        # fixed before any agent runs. Without this both the delivery of a
        # message and the wake-up of a waiting agent would depend on the order
        # in which workers happen to be scheduled.
        scheduled: list[Agent] = []
        inboxes: list[list[Message]] = []
        waiting: list[str] = []
        for agent in self.live_agents():
            inbox = self.bus.receive(agent.name)
            if agent.state is AgentState.WAITING and not inbox:
                waiting.append(agent.name)
                continue
            scheduled.append(agent)
            inboxes.append(inbox)

        if self.max_workers == 1 or len(scheduled) < 2:
            outcomes = [
                self._run_agent(agent, tick, inbox)
                for agent, inbox in zip(scheduled, inboxes)
            ]
        else:
            futures = [
                self._executor().submit(self._run_agent, agent, tick, inbox)
                for agent, inbox in zip(scheduled, inboxes)
            ]
            outcomes = [future.result() for future in futures]

        # Reclaim mailboxes only once the whole tick is over: dropping them
        # mid-tick would make sending to an agent that finished concurrently
        # fail or not depending on thread timing.
        for agent in scheduled:
            self._unregister_finished_mailbox(agent)

        ran = tuple(a.name for a, ok in zip(scheduled, outcomes) if ok)
        failed = tuple(a.name for a, ok in zip(scheduled, outcomes) if not ok)
        return TickReport(tick=tick, ran=ran, failed=failed, waiting=tuple(waiting))

    def run(self, max_ticks: int = 100) -> list[TickReport]:
        """Run until the agents finish, go quiescent, or ``max_ticks`` is hit.

        Check :attr:`stop_reason` afterwards to tell those cases apart.
        """
        if max_ticks < 0:
            raise ValueError("max_ticks must not be negative")
        reports: list[TickReport] = []
        self.stop_reason = StopReason.MAX_TICKS
        for _ in range(max_ticks):
            if not any(self.live_agents()):
                self.stop_reason = StopReason.FINISHED
                break
            if self.is_quiescent():
                self.stop_reason = StopReason.QUIESCENT
                break
            reports.append(self.step())
        else:
            if not any(self.live_agents()):
                self.stop_reason = StopReason.FINISHED
            elif self.is_quiescent():
                self.stop_reason = StopReason.QUIESCENT
        return reports

    def shutdown(self, wait: bool = True) -> None:
        """Release the worker pool, if one was created."""
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=wait)

    def __enter__(self) -> "Kernel":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.shutdown()
