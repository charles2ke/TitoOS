"""The TitoOS kernel: registers agents and runs them cooperatively."""

from __future__ import annotations

import inspect
import threading
from dataclasses import dataclass
from enum import Enum
from types import TracebackType
from typing import Callable, Iterator, Sequence

from .agent import Agent, AgentState, Context, FunctionAgent, RestartPolicy
from .backends import AsyncBackend, ExecutionBackend, SerialBackend, ThreadBackend
from .bus import MessageBus
from .message import Message
from .persistence import AgentFactory, AgentRecord, FinishedAgent, Snapshot


#: Payload of the message a supervisor receives when a child fails for good.
CHILD_FAILED = "titoos.child_failed"

_FINISHED = (AgentState.DONE, AgentState.FAILED)


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
    #: Agents restarted by their supervisor after failing this tick.
    restarted: tuple[str, ...] = ()


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

    def __init__(
        self,
        max_workers: int = 1,
        backend: ExecutionBackend | None = None,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if backend is None:
            backend = SerialBackend() if max_workers == 1 else ThreadBackend(max_workers)
        self.bus = MessageBus()
        self.max_workers = max_workers
        self.backend = backend
        self._agents: dict[str, Agent] = {}
        self._tick = 0
        self.errors: list[tuple[str, BaseException]] = []
        self.stop_reason: StopReason | None = None
        self._lock = threading.RLock()

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

    def register(self, agent: Agent, parent: str | None = None) -> Agent:
        """Add ``agent`` to the kernel. Names must be unique.

        Safe to call from a worker thread, e.g. via :meth:`Context.spawn`.
        Agents registered during a tick start running on the next one.
        ``parent`` links the agent to its supervisor.
        """
        with self._lock:
            if agent.name in self._agents:
                raise ValueError(f"agent already registered: {agent.name!r}")
            if parent is not None:
                agent.parent = parent
            self._agents[agent.name] = agent
            self.bus.register(agent.name)
        return agent

    def children_of(self, name: str) -> tuple[Agent, ...]:
        """Every registered agent spawned by ``name``."""
        with self._lock:
            return tuple(a for a in self._agents.values() if a.parent == name)

    def spawn(self, name: str, fn: Callable[[Context], None]) -> Agent:
        """Register a callable ``fn(ctx)`` as an agent named ``name``."""
        return self.register(FunctionAgent(name, fn))

    def unregister(self, name: str) -> None:
        with self._lock:
            self._agents.pop(name, None)
        self.bus.unregister(name)

    def _supervise(self, agent: Agent, inbox: list[Message]) -> bool:
        """Apply ``agent``'s restart policy after a failure.

        Returns ``True`` if the agent was restarted. Runs at the tick barrier
        rather than inside a worker so the decision and the restart counter
        are deterministic. The messages handed to the failed step are put back
        so a crash mid-job does not lose work.
        """
        if agent.restart_policy is not RestartPolicy.ON_FAILURE:
            return False
        if agent.restarts >= agent.max_restarts:
            return False
        agent.restarts += 1
        agent.on_restart()
        agent.state = AgentState.READY
        self.bus.requeue(agent.name, inbox)
        return True

    def _escalate(self, agent: Agent) -> None:
        """Tell ``agent``'s parent that it failed for good, if it has one."""
        if agent.parent is None:
            return
        with self._lock:
            parent = self._agents.get(agent.parent)
        if parent is None or not parent.is_alive:
            return
        self.bus.post(
            agent.name,
            agent.parent,
            CHILD_FAILED,
            child=agent.name,
            restarts=agent.restarts,
        )

    def _unregister_finished_mailbox(self, agent: Agent) -> None:
        if agent.state in (AgentState.DONE, AgentState.FAILED):
            self.bus.unregister(agent.name)

    def live_agents(self) -> Iterator[Agent]:
        """Every agent that may still run, including waiting ones."""
        return iter(self._live_snapshot())

    def _live_snapshot(self) -> tuple[Agent, ...]:
        """The live agents as of now, in registration order."""
        with self._lock:
            agents = tuple(self._agents.values())
        return tuple(agent for agent in agents if agent.is_alive)

    def is_quiescent(self) -> bool:
        """True when live agents remain but none of them can make progress.

        That is the case when every live agent is waiting and no message is
        pending for any of them, so running further ticks is pointless.
        """
        return self._is_quiescent(self._live_snapshot())

    def _is_quiescent(self, live: Sequence[Agent]) -> bool:
        if not live:
            return False
        if any(agent.state is not AgentState.WAITING for agent in live):
            return False
        return not self.bus.any_pending([agent.name for agent in live])

    def _run_agent(self, agent: Agent, tick: int, inbox: list[Message]) -> bool:
        """Run a single agent for one tick. Returns ``True`` if it succeeded."""
        ctx = Context(kernel=self, agent=agent, tick=tick, inbox=inbox)
        agent.state = AgentState.RUNNING
        try:
            agent.step(ctx)
        except Exception as exc:  # noqa: BLE001 - a failing agent must not kill the OS
            return self._record_failure(agent, exc)
        if agent.state is AgentState.RUNNING:
            agent.state = AgentState.READY
        return True

    async def _run_agent_async(self, agent: Agent, tick: int, inbox: list[Message]) -> bool:
        """Await one tick of ``agent``, which may define a sync or async step."""
        ctx = Context(kernel=self, agent=agent, tick=tick, inbox=inbox)
        agent.state = AgentState.RUNNING
        try:
            result = agent.step(ctx)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # noqa: BLE001 - a failing agent must not kill the OS
            return self._record_failure(agent, exc)
        if agent.state is AgentState.RUNNING:
            agent.state = AgentState.READY
        return True

    def _record_failure(self, agent: Agent, exc: BaseException) -> bool:
        agent.state = AgentState.FAILED
        with self._lock:
            self.errors.append((agent.name, exc))
        return False

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
        live = self._live_snapshot()
        for agent, inbox in zip(live, self.bus.receive_many([a.name for a in live])):
            if agent.state is AgentState.WAITING and not inbox:
                waiting.append(agent.name)
                continue
            scheduled.append(agent)
            inboxes.append(inbox)

        outcomes = self.backend.run_tick(self, scheduled, inboxes, tick)

        # Supervision runs at the barrier, in registration order, so restarts
        # and escalation messages do not depend on worker timing. Restarts are
        # applied first and escalation second, so a parent that failed and was
        # restarted in this same tick is still notified about its children.
        casualties = [
            (agent, inbox)
            for agent, inbox, ok in zip(scheduled, inboxes, outcomes)
            if not ok
        ]
        restarted = tuple(
            agent.name for agent, inbox in casualties if self._supervise(agent, inbox)
        )
        for agent, _ in casualties:
            if agent.state is AgentState.FAILED:
                self._escalate(agent)

        # Reclaim mailboxes only once the whole tick is over: dropping them
        # mid-tick would make sending to an agent that finished concurrently
        # fail or not depending on thread timing. Restarted agents are alive
        # again by now, so they keep theirs.
        for agent in scheduled:
            self._unregister_finished_mailbox(agent)

        ran = tuple(a.name for a, ok in zip(scheduled, outcomes) if ok)
        failed = tuple(a.name for a, ok in zip(scheduled, outcomes) if not ok)
        return TickReport(
            tick=tick,
            ran=ran,
            failed=failed,
            waiting=tuple(waiting),
            restarted=restarted,
        )

    def run(self, max_ticks: int = 100) -> list[TickReport]:
        """Run until the agents finish, go quiescent, or ``max_ticks`` is hit.

        Check :attr:`stop_reason` afterwards to tell those cases apart.
        """
        if max_ticks < 0:
            raise ValueError("max_ticks must not be negative")
        reports: list[TickReport] = []
        self.stop_reason = StopReason.MAX_TICKS
        for _ in range(max_ticks):
            live = self._live_snapshot()
            if not live:
                self.stop_reason = StopReason.FINISHED
                break
            if self._is_quiescent(live):
                self.stop_reason = StopReason.QUIESCENT
                break
            reports.append(self.step())
        else:
            live = self._live_snapshot()
            if not live:
                self.stop_reason = StopReason.FINISHED
            elif self._is_quiescent(live):
                self.stop_reason = StopReason.QUIESCENT
        return reports

    def snapshot(self) -> Snapshot:
        """Capture the kernel as of the current tick boundary.

        Must be called between ticks — never from inside an agent's
        ``step()``, where agents are mid-execution and the picture would be
        inconsistent.
        """
        with self._lock:
            running = [a.name for a in self._agents.values() if a.state is AgentState.RUNNING]
            if running:
                raise RuntimeError(
                    "cannot snapshot while agents are running: "
                    f"{', '.join(sorted(running))}"
                )
            records = tuple(
                AgentRecord(
                    name=agent.name,
                    # Placeholders keep reporting the kind they stood in for,
                    # so snapshots stay stable across repeated save/restore.
                    kind=(
                        agent.kind
                        if isinstance(agent, FinishedAgent)
                        else type(agent).__name__
                    ),
                    state=agent.state.value,
                    parent=agent.parent,
                    restarts=agent.restarts,
                    data=agent.save_state(),
                )
                for agent in self._agents.values()
            )
            mailboxes = {
                name: tuple(messages)
                for name, messages in self.bus.dump().items()
                if messages
            }
            return Snapshot(tick=self._tick, agents=records, mailboxes=mailboxes)

    @classmethod
    def restore(
        cls,
        snapshot: Snapshot,
        factories: dict[str, AgentFactory],
        max_workers: int = 1,
        backend: "ExecutionBackend | None" = None,
    ) -> "Kernel":
        """Rebuild a kernel from ``snapshot``.

        ``factories`` maps the ``kind`` recorded for each agent (its class
        name) to a callable building a bare agent of that kind for a given
        name. Behaviour cannot be serialized, so supplying it is the caller's
        job. A factory is required for every agent that is still alive; a
        missing one is an error rather than a silently dropped agent. Agents
        that had already finished are restored as
        :class:`~titoos.persistence.FinishedAgent` placeholders when no
        factory is given, since they can never run again.
        """
        live_kinds = {
            r.kind for r in snapshot.agents if AgentState(r.state) not in _FINISHED
        }
        missing = sorted(live_kinds - set(factories))
        if missing:
            raise KeyError(f"no factory for agent kind(s): {', '.join(missing)}")

        kernel = cls(max_workers=max_workers, backend=backend)
        kernel._tick = snapshot.tick
        for record in snapshot.agents:
            factory = factories.get(record.kind)
            if factory is None:
                agent: Agent = FinishedAgent(record.name, kind=record.kind)
            else:
                agent = factory(record.name)
            if agent.name != record.name:
                raise ValueError(
                    f"factory for {record.kind!r} built agent named "
                    f"{agent.name!r}, expected {record.name!r}"
                )
            agent.parent = record.parent
            agent.restarts = record.restarts
            agent.load_state(dict(record.data))
            kernel.register(agent)
            # Set the lifecycle state after registration so a restored WAITING
            # agent is not scheduled until a message actually arrives.
            agent.state = AgentState(record.state)
        kernel.bus.load(
            {name: list(messages) for name, messages in snapshot.mailboxes.items()}
        )
        # Mailbox load replaces the registry wholesale, so re-add empty
        # mailboxes for live agents that had no pending messages. Finished
        # agents stay without one, exactly as when they were reclaimed.
        for agent in kernel.agents:
            if agent.is_alive:
                kernel.bus.register(agent.name)
            elif not kernel.bus.pending(agent.name):
                kernel.bus.unregister(agent.name)
        return kernel

    def shutdown(self, wait: bool = True) -> None:
        """Release any resources held by the execution backend."""
        self.backend.shutdown(wait=wait)

    def __enter__(self) -> "Kernel":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.shutdown()
