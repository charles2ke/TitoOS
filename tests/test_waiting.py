import pytest

from titoos import Agent, AgentState, FunctionAgent, Kernel, StopReason


class Waiter(Agent):
    """Waits for messages, recording every payload it is woken with."""

    def __init__(self, name):
        super().__init__(name)
        self.steps = 0
        self.seen = []

    def step(self, ctx):
        self.steps += 1
        self.seen.extend(m.payload for m in ctx.inbox)
        ctx.wait()


def test_waiting_agent_is_skipped_until_a_message_arrives():
    kernel = Kernel()
    waiter = kernel.register(Waiter("waiter"))

    def poker(ctx):
        if ctx.tick == 3:
            ctx.send("waiter", "wake")
            ctx.exit()

    kernel.spawn("poker", poker)
    reports = kernel.run(max_ticks=6)

    # Ran on tick 1 (initial) and tick 4 (woken by the tick-3 message).
    assert waiter.steps == 2
    assert waiter.seen == ["wake"]
    assert reports[1].waiting == ("waiter",)
    assert "waiter" not in reports[1].ran


def test_waiting_agent_stays_alive():
    kernel = Kernel()
    waiter = kernel.register(Waiter("waiter"))

    kernel.run(max_ticks=5)

    assert waiter.state is AgentState.WAITING
    assert waiter.is_alive
    assert waiter in list(kernel.live_agents())


def test_idle_waiters_stop_the_run_instead_of_spinning():
    kernel = Kernel()
    for i in range(3):
        kernel.register(Waiter(f"w{i}"))

    reports = kernel.run(max_ticks=100)

    assert kernel.stop_reason is StopReason.QUIESCENT
    assert len(reports) == 1  # one tick to run, then nothing can change
    assert all(kernel.get(f"w{i}").steps == 1 for i in range(3))


def test_quiescent_is_distinct_from_finished():
    kernel = Kernel()
    kernel.spawn("done", lambda ctx: ctx.exit())
    kernel.run(max_ticks=5)
    assert kernel.stop_reason is StopReason.FINISHED
    assert not kernel.is_quiescent()

    stuck = Kernel()
    stuck.register(Waiter("waiter"))
    stuck.run(max_ticks=5)
    assert stuck.stop_reason is StopReason.QUIESCENT
    assert stuck.is_quiescent()


def test_max_ticks_stop_reason_when_agents_still_runnable():
    kernel = Kernel()
    kernel.spawn("forever", lambda ctx: None)

    kernel.run(max_ticks=3)

    assert kernel.stop_reason is StopReason.MAX_TICKS
    assert kernel.tick == 3


def test_mutual_wait_is_reported_as_quiescent_deadlock():
    kernel = Kernel()
    kernel.register(Waiter("a"))
    kernel.register(Waiter("b"))

    kernel.run(max_ticks=50)

    assert kernel.stop_reason is StopReason.QUIESCENT
    assert kernel.tick == 1


def test_waiter_can_exit_after_being_woken():
    kernel = Kernel()

    def worker(ctx):
        if ctx.inbox:
            ctx.exit()
        else:
            ctx.wait()

    def boss(ctx):
        if ctx.tick == 1:
            ctx.send("worker", "go")
            ctx.exit()

    kernel.spawn("worker", worker)
    kernel.spawn("boss", boss)

    kernel.run(max_ticks=10)

    assert kernel.get("worker").state is AgentState.DONE
    assert kernel.stop_reason is StopReason.FINISHED


def test_broadcast_wakes_every_waiter():
    kernel = Kernel()
    for i in range(3):
        kernel.register(Waiter(f"w{i}"))

    def caller(ctx):
        if ctx.tick == 2:
            ctx.broadcast("hello")
            ctx.exit()

    kernel.spawn("caller", caller)
    kernel.run(max_ticks=10)

    for i in range(3):
        assert kernel.get(f"w{i}").seen == ["hello"]


def test_waiting_is_deterministic_across_worker_counts():
    def build(max_workers):
        kernel = Kernel(max_workers=max_workers)
        for i in range(4):
            kernel.register(Waiter(f"w{i}"))

        def poker(ctx):
            if ctx.tick == 2:
                ctx.send("w0", "one")
            elif ctx.tick == 3:
                ctx.broadcast("all")
                ctx.exit()

        kernel.spawn("poker", poker)
        with kernel:
            reports = kernel.run(max_ticks=20)
        return (
            kernel.tick,
            kernel.stop_reason,
            [(r.ran, r.waiting) for r in reports],
            {f"w{i}": kernel.get(f"w{i}").seen for i in range(4)},
        )

    assert build(4) == build(1)


def test_waiting_agents_do_not_occupy_worker_threads():
    kernel = Kernel(max_workers=4)
    for i in range(6):
        kernel.register(Waiter(f"w{i}"))
    kernel.spawn("active", lambda ctx: None)

    with kernel:
        reports = kernel.run(max_ticks=4)

    # After the first tick only the active agent is scheduled.
    assert reports[1].ran == ("active",)
    assert reports[1].waiting == tuple(f"w{i}" for i in range(6))


def test_wait_then_message_in_same_tick_is_delivered_next_tick():
    kernel = Kernel()
    waiter = kernel.register(Waiter("waiter"))

    def sender(ctx):
        if ctx.tick == 1:
            ctx.send("waiter", "same-tick")
            ctx.exit()

    kernel.spawn("sender", sender)
    kernel.run(max_ticks=5)

    # The waiter waits on tick 1 and is woken on tick 2 by that message.
    assert waiter.seen == ["same-tick"]
    assert waiter.steps == 2


def test_empty_kernel_is_not_quiescent():
    kernel = Kernel()

    assert not kernel.is_quiescent()
    assert kernel.run(max_ticks=5) == []
    assert kernel.stop_reason is StopReason.FINISHED


def test_spawned_agent_wakes_a_quiescent_kernel():
    kernel = Kernel()
    waiter = kernel.register(Waiter("waiter"))

    def child(ctx):
        ctx.send("waiter", "from-child")
        ctx.exit()

    def spawner(ctx):
        ctx.spawn(FunctionAgent("child", child))
        ctx.exit()

    kernel.spawn("spawner", spawner)
    kernel.run(max_ticks=10)

    assert waiter.seen == ["from-child"]


@pytest.mark.parametrize("max_workers", [1, 4])
def test_failed_waiter_does_not_keep_kernel_alive(max_workers):
    kernel = Kernel(max_workers=max_workers)

    def boom(ctx):
        if ctx.inbox:
            raise RuntimeError("boom")
        ctx.wait()

    kernel.spawn("boom", boom)
    kernel.spawn("poker", lambda ctx: (ctx.send("boom", "x"), ctx.exit()))

    with kernel:
        kernel.run(max_ticks=10)

    assert kernel.get("boom").state is AgentState.FAILED
    assert kernel.stop_reason is StopReason.FINISHED
