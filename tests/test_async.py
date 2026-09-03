import asyncio

import pytest

from titoos import (
    Agent,
    AgentState,
    AsyncBackend,
    Kernel,
    SerialBackend,
    StopReason,
    ThreadBackend,
)


class AsyncCounter(Agent):
    def __init__(self, name, limit, delay=0.0):
        super().__init__(name)
        self.limit = limit
        self.delay = delay
        self.count = 0

    async def step(self, ctx):
        await asyncio.sleep(self.delay)
        self.count += 1
        if self.count >= self.limit:
            ctx.exit()


class SyncCounter(Agent):
    def __init__(self, name, limit):
        super().__init__(name)
        self.limit = limit
        self.count = 0

    def step(self, ctx):
        self.count += 1
        if self.count >= self.limit:
            ctx.exit()


def test_async_agents_run_on_async_backend():
    kernel = Kernel(backend=AsyncBackend())
    agent = kernel.register(AsyncCounter("a", 3))

    with kernel:
        kernel.run(max_ticks=10)

    assert agent.count == 3
    assert kernel.stop_reason is StopReason.FINISHED


def test_async_backend_also_runs_sync_agents():
    kernel = Kernel(backend=AsyncBackend())
    sync = kernel.register(SyncCounter("sync", 2))
    coro = kernel.register(AsyncCounter("async", 2))

    with kernel:
        kernel.run(max_ticks=10)

    assert sync.count == 2 and coro.count == 2


@pytest.mark.parametrize("backend", [SerialBackend, lambda: ThreadBackend(4)])
def test_sync_backends_reject_async_agents(backend):
    kernel = Kernel(backend=backend())
    kernel.register(AsyncCounter("a", 1))
    kernel.register(SyncCounter("b", 1))

    with pytest.raises(TypeError, match="async step"):
        kernel.step()


def test_async_agents_of_a_tick_overlap():
    kernel = Kernel(backend=AsyncBackend())
    order = []

    class Recorder(Agent):
        def __init__(self, name, delay):
            super().__init__(name)
            self.delay = delay

        async def step(self, ctx):
            await asyncio.sleep(self.delay)
            order.append(self.name)
            ctx.exit()

    kernel.register(Recorder("slow", 0.05))
    kernel.register(Recorder("fast", 0.0))

    with kernel:
        (report,) = kernel.run(max_ticks=3)

    # They genuinely overlapped: the later-registered agent finished first...
    assert order == ["fast", "slow"]
    # ...but the report still follows registration order.
    assert report.ran == ("slow", "fast")


def test_async_backend_messaging_matches_serial():
    def build(backend):
        kernel = Kernel(backend=backend)
        seen = []

        class Pinger(Agent):
            async def step(self, ctx):
                if ctx.tick == 1:
                    ctx.send("ponger", "ping")
                    return
                if ctx.inbox:
                    seen.extend(m.payload for m in ctx.inbox)
                    ctx.exit()
                else:
                    ctx.wait()

        class Ponger(Agent):
            async def step(self, ctx):
                for message in ctx.inbox:
                    ctx.send(message.sender, "pong")
                    ctx.exit()
                    return
                ctx.wait()

        kernel.register(Pinger("pinger"))
        kernel.register(Ponger("ponger"))
        with kernel:
            reports = kernel.run(max_ticks=10)
        return seen, kernel.tick, [(r.ran, r.waiting) for r in reports]

    # Two independent AsyncBackend runs must agree with each other.
    assert build(AsyncBackend()) == build(AsyncBackend())
    seen, _, _ = build(AsyncBackend())
    assert seen == ["pong"]


def test_async_failure_is_isolated():
    kernel = Kernel(backend=AsyncBackend())

    class Boom(Agent):
        async def step(self, ctx):
            raise RuntimeError("boom")

    kernel.register(Boom("boom"))
    survivor = kernel.register(AsyncCounter("survivor", 2))

    with kernel:
        reports = kernel.run(max_ticks=5)

    assert kernel.get("boom").state is AgentState.FAILED
    assert survivor.count == 2
    assert [name for name, _ in kernel.errors] == ["boom"]
    assert reports[0].failed == ("boom",)


def test_async_waiting_agents_are_skipped():
    kernel = Kernel(backend=AsyncBackend())

    class Waiter(Agent):
        def __init__(self, name):
            super().__init__(name)
            self.steps = 0

        async def step(self, ctx):
            self.steps += 1
            ctx.wait()

    waiter = kernel.register(Waiter("waiter"))

    with kernel:
        kernel.run(max_ticks=50)

    assert waiter.steps == 1
    assert kernel.stop_reason is StopReason.QUIESCENT


def test_backend_results_are_identical_across_backends():
    def build(backend):
        kernel = Kernel(backend=backend)
        for i in range(5):
            kernel.register(SyncCounter(f"a{i}", i + 1))
        with kernel:
            reports = kernel.run(max_ticks=20)
        return (
            kernel.tick,
            kernel.stop_reason,
            [(r.ran, r.failed, r.waiting) for r in reports],
            [kernel.get(f"a{i}").count for i in range(5)],
        )

    serial = build(SerialBackend())
    assert build(ThreadBackend(4)) == serial
    assert build(AsyncBackend()) == serial


def test_async_backend_refuses_reentrant_use():
    kernel = Kernel(backend=AsyncBackend())
    kernel.register(AsyncCounter("a", 1))

    async def main():
        kernel.step()

    with pytest.raises(RuntimeError, match="running event loop"):
        asyncio.run(main())

    kernel.shutdown()


def test_max_workers_still_selects_a_backend():
    assert isinstance(Kernel().backend, SerialBackend)
    assert isinstance(Kernel(max_workers=4).backend, ThreadBackend)
    with pytest.raises(ValueError):
        Kernel(max_workers=0)
