import threading
import time

import pytest

from titoos import Agent, AgentState, Kernel


class Counter(Agent):
    def __init__(self, name, limit):
        super().__init__(name)
        self.limit = limit
        self.count = 0

    def step(self, ctx):
        self.count += 1
        if self.count >= self.limit:
            ctx.exit()


def test_max_workers_must_be_positive():
    with pytest.raises(ValueError):
        Kernel(max_workers=0)


def test_agents_of_a_tick_run_on_different_threads():
    kernel = Kernel(max_workers=4)
    threads = set()
    lock = threading.Lock()

    def record(ctx):
        with lock:
            threads.add(threading.current_thread().name)
        time.sleep(0.05)
        ctx.exit()

    for i in range(4):
        kernel.spawn(f"a{i}", record)

    with kernel:
        kernel.run(max_ticks=2)

    assert len(threads) > 1
    assert all(name.startswith("titoos") for name in threads)


def test_parallel_tick_is_faster_than_serial():
    def sleeper(ctx):
        time.sleep(0.1)
        ctx.exit()

    def timed(max_workers):
        kernel = Kernel(max_workers=max_workers)
        for i in range(4):
            kernel.spawn(f"a{i}", sleeper)
        start = time.monotonic()
        with kernel:
            kernel.run(max_ticks=2)
        return time.monotonic() - start

    assert timed(4) < timed(1)


def test_tick_is_a_barrier_so_messages_arrive_next_tick():
    kernel = Kernel(max_workers=4)
    seen = []

    def sender(ctx):
        if ctx.tick == 1:
            ctx.send("receiver", "hello")
            ctx.exit()

    def receiver(ctx):
        if ctx.tick == 1:
            assert ctx.inbox == []  # sent this tick, delivered on the next
            return
        seen.extend(m.payload for m in ctx.inbox)
        ctx.exit()

    kernel.spawn("sender", sender)
    kernel.spawn("receiver", receiver)

    with kernel:
        kernel.run(max_ticks=5)

    assert seen == ["hello"]


def test_concurrent_broadcasts_are_all_delivered():
    kernel = Kernel(max_workers=8)
    counts = {}

    def chatter(ctx):
        if ctx.tick == 1:
            ctx.broadcast("hi")
        else:
            counts[ctx.agent.name] = len(ctx.inbox)
            ctx.exit()

    for i in range(8):
        kernel.spawn(f"a{i}", chatter)

    with kernel:
        kernel.run(max_ticks=5)

    assert counts == {f"a{i}": 7 for i in range(8)}


def test_reports_are_deterministic_despite_completion_order():
    kernel = Kernel(max_workers=4)

    def slow(ctx):
        time.sleep(0.05)
        ctx.exit()

    def fast(ctx):
        ctx.exit()

    kernel.spawn("slow", slow)
    kernel.spawn("fast", fast)

    with kernel:
        (report,) = kernel.run(max_ticks=3)

    assert report.ran == ("slow", "fast")  # registration order, not finish order


def test_failing_agent_is_isolated_when_threaded():
    kernel = Kernel(max_workers=4)

    def boom(ctx):
        raise RuntimeError("boom")

    kernel.spawn("boom", boom)
    survivor = kernel.register(Counter("survivor", 2))

    with kernel:
        reports = kernel.run(max_ticks=5)

    assert kernel.get("boom").state is AgentState.FAILED
    assert survivor.count == 2
    assert [name for name, _ in kernel.errors] == ["boom"]
    assert reports[0].failed == ("boom",)


def test_spawning_from_a_worker_thread_is_safe():
    kernel = Kernel(max_workers=4)

    def parent(ctx):
        ctx.spawn(Counter(f"child-{ctx.agent.name}", 1))
        ctx.exit()

    for i in range(4):
        kernel.spawn(f"p{i}", parent)

    with kernel:
        kernel.run(max_ticks=5)

    for i in range(4):
        child = kernel.get(f"child-p{i}")
        assert child.count == 1 and child.state is AgentState.DONE


def test_threaded_run_matches_serial_result():
    def build(max_workers):
        kernel = Kernel(max_workers=max_workers)
        for i in range(5):
            kernel.register(Counter(f"a{i}", i + 1))
        with kernel:
            kernel.run(max_ticks=20)
        return kernel.tick, [kernel.get(f"a{i}").count for i in range(5)]

    assert build(4) == build(1)


def test_shutdown_is_idempotent_and_kernel_still_usable():
    kernel = Kernel(max_workers=2)
    kernel.register(Counter("a", 1))
    kernel.register(Counter("b", 1))

    kernel.run(max_ticks=2)
    kernel.shutdown()
    kernel.shutdown()

    kernel.register(Counter("c", 1))
    kernel.register(Counter("d", 1))
    kernel.run(max_ticks=2)
    kernel.shutdown()

    assert kernel.get("c").state is AgentState.DONE
    assert kernel.get("d").state is AgentState.DONE
