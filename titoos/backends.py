"""Execution backends: how the agents of a single tick are run.

A backend only decides *how* the scheduled agents of a tick are executed. It
never decides *which* agents run or what they see in their inbox — the kernel
fixes both at the tick boundary before handing work over. That is what keeps
results identical across backends.
"""

from __future__ import annotations

import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Protocol, Sequence

from .agent import Agent, FunctionAgent
from .message import Message

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .kernel import Kernel


class Job(Protocol):
    """A single agent to run for a tick."""

    agent: Agent
    inbox: list[Message]


class ExecutionBackend:
    """Runs the scheduled agents of one tick and reports success per agent."""

    #: Whether this backend can await ``async def step`` agents.
    supports_async_agents = False

    def run_tick(
        self, kernel: "Kernel", agents: Sequence[Agent], inboxes: Sequence[list[Message]], tick: int
    ) -> list[bool]:
        raise NotImplementedError

    def shutdown(self, wait: bool = True) -> None:
        """Release any resources held by the backend."""

    @staticmethod
    def _is_async_agent(agent: Agent) -> bool:
        """True if running ``agent`` produces a coroutine to await.

        ``FunctionAgent`` keeps its callable behind a uniform sync ``step``,
        so the wrapped function has to be inspected too.
        """
        target = agent.wrapped if isinstance(agent, FunctionAgent) else agent.step
        return inspect.iscoroutinefunction(target)

    def _reject_async(self, agents: Sequence[Agent]) -> None:
        if self.supports_async_agents:
            return
        for agent in agents:
            if self._is_async_agent(agent):
                raise TypeError(
                    f"agent {agent.name!r} defines an async step() but "
                    f"{type(self).__name__} cannot await it; use AsyncBackend"
                )


class SerialBackend(ExecutionBackend):
    """Runs agents one after another on the calling thread."""

    def run_tick(self, kernel, agents, inboxes, tick):
        self._reject_async(agents)
        return [
            kernel._run_agent(agent, tick, inbox) for agent, inbox in zip(agents, inboxes)
        ]


class ThreadBackend(ExecutionBackend):
    """Runs the agents of a tick concurrently in a thread pool.

    Suited to agents that block on I/O such as model or tool calls.
    """

    def __init__(self, max_workers: int) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        self.max_workers = max_workers
        self._pool: ThreadPoolExecutor | None = None

    def _executor(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=self.max_workers, thread_name_prefix="titoos"
            )
        return self._pool

    def run_tick(self, kernel, agents, inboxes, tick):
        self._reject_async(agents)
        if len(agents) < 2:
            return [
                kernel._run_agent(agent, tick, inbox)
                for agent, inbox in zip(agents, inboxes)
            ]
        futures = [
            self._executor().submit(kernel._run_agent, agent, tick, inbox)
            for agent, inbox in zip(agents, inboxes)
        ]
        # Collecting in submission order keeps the outcome list aligned with
        # registration order regardless of which worker finishes first.
        return [future.result() for future in futures]

    def shutdown(self, wait: bool = True) -> None:
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=wait)


class AsyncBackend(ExecutionBackend):
    """Runs the agents of a tick concurrently on an asyncio event loop.

    Agents may define either ``def step(ctx)`` or ``async def step(ctx)``.
    Synchronous agents run inline on the loop, so a blocking one will stall
    the tick; use :class:`ThreadBackend` for those.
    """

    supports_async_agents = True

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def run_tick(self, kernel, agents, inboxes, tick):
        if not agents:
            return []

        async def gather_all() -> list[bool]:
            tasks = [
                kernel._run_agent_async(agent, tick, inbox)
                for agent, inbox in zip(agents, inboxes)
            ]
            # gather preserves input order, so outcomes stay aligned with
            # registration order even though the agents finish out of order.
            return list(await asyncio.gather(*tasks))

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self._get_loop().run_until_complete(gather_all())
        raise RuntimeError(
            "AsyncBackend.run_tick() cannot be called from a running event "
            "loop; await Kernel.run_async() instead"
        )

    def shutdown(self, wait: bool = True) -> None:
        loop, self._loop = self._loop, None
        if loop is not None and not loop.is_closed():
            loop.close()
