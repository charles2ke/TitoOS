import pytest

from titoos import (
    CHILD_FAILED,
    Agent,
    AgentState,
    Kernel,
    RestartPolicy,
    StopReason,
)


class Flaky(Agent):
    """Fails ``fail_times`` times, then completes."""

    restart_policy = RestartPolicy.ON_FAILURE

    def __init__(self, name, fail_times, max_restarts=3):
        super().__init__(name)
        self.max_restarts = max_restarts
        self.fail_times = fail_times
        self.attempts = 0
        self.restarted_hooks = 0

    def on_restart(self):
        self.restarted_hooks += 1

    def step(self, ctx):
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise RuntimeError(f"attempt {self.attempts} failed")
        ctx.exit()


def test_default_policy_leaves_a_failed_agent_dead():
    kernel = Kernel()

    def boom(ctx):
        raise RuntimeError("boom")

    kernel.spawn("boom", boom)
    reports = kernel.run(max_ticks=5)

    assert kernel.get("boom").state is AgentState.FAILED
    assert kernel.get("boom").restarts == 0
    assert reports[0].restarted == ()


def test_on_failure_policy_restarts_until_success():
    kernel = Kernel()
    flaky = kernel.register(Flaky("flaky", fail_times=2))

    reports = kernel.run(max_ticks=10)

    assert flaky.attempts == 3
    assert flaky.restarts == 2
    assert flaky.restarted_hooks == 2
    assert flaky.state is AgentState.DONE
    assert reports[0].restarted == ("flaky",)
    assert reports[1].restarted == ("flaky",)
    assert reports[2].ran == ("flaky",)
    assert kernel.stop_reason is StopReason.FINISHED


def test_restarts_are_capped_by_max_restarts():
    kernel = Kernel()
    doomed = kernel.register(Flaky("doomed", fail_times=99, max_restarts=2))

    kernel.run(max_ticks=10)

    assert doomed.restarts == 2
    assert doomed.attempts == 3  # initial run plus two restarts
    assert doomed.state is AgentState.FAILED
    assert kernel.stop_reason is StopReason.FINISHED


def test_spawned_agents_record_their_parent():
    kernel = Kernel()

    def parent(ctx):
        if ctx.tick == 1:
            ctx.spawn(Flaky("child", fail_times=0))
            assert [c.name for c in ctx.children()] == ["child"]
        ctx.exit()

    kernel.spawn("parent", parent)
    kernel.run(max_ticks=5)

    assert kernel.get("child").parent == "parent"
    assert [a.name for a in kernel.children_of("parent")] == ["child"]


def test_top_level_agents_have_no_parent():
    kernel = Kernel()
    agent = kernel.register(Flaky("solo", fail_times=0))

    assert agent.parent is None
    assert kernel.children_of("solo") == ()


def test_parent_is_notified_when_a_child_fails_for_good():
    kernel = Kernel()
    notices = []

    def supervisor(ctx):
        if ctx.tick == 1:
            ctx.spawn(Flaky("worker", fail_times=99, max_restarts=1))
            ctx.wait()
            return
        for message in ctx.inbox:
            if message.payload == CHILD_FAILED:
                notices.append((message.metadata["child"], message.metadata["restarts"]))
                ctx.exit()

    kernel.spawn("supervisor", supervisor)
    kernel.run(max_ticks=10)

    assert notices == [("worker", 1)]
    assert kernel.get("worker").state is AgentState.FAILED
    assert kernel.get("supervisor").state is AgentState.DONE


def test_escalation_wakes_a_waiting_supervisor():
    kernel = Kernel()

    def supervisor(ctx):
        if ctx.tick == 1:
            ctx.spawn(Flaky("worker", fail_times=99, max_restarts=0))
        if ctx.inbox:
            ctx.exit()
            return
        ctx.wait()

    kernel.spawn("supervisor", supervisor)
    kernel.run(max_ticks=10)

    # Without the escalation message the supervisor would sit waiting forever.
    assert kernel.get("supervisor").state is AgentState.DONE
    assert kernel.stop_reason is StopReason.FINISHED


def test_no_escalation_while_restarts_remain():
    kernel = Kernel()
    received = []

    def supervisor(ctx):
        if ctx.tick == 1:
            ctx.spawn(Flaky("worker", fail_times=1, max_restarts=3))
        received.extend(m.payload for m in ctx.inbox)

    kernel.spawn("supervisor", supervisor)
    kernel.run(max_ticks=6)

    # The child recovered on its restart, so the parent is never told.
    assert received == []
    assert kernel.get("worker").state is AgentState.DONE


def test_escalation_to_a_dead_parent_is_dropped():
    kernel = Kernel()

    def parent(ctx):
        ctx.spawn(Flaky("child", fail_times=99, max_restarts=0))
        ctx.exit()

    kernel.spawn("parent", parent)
    kernel.run(max_ticks=6)

    assert kernel.get("child").state is AgentState.FAILED
    assert kernel.stop_reason is StopReason.FINISHED


@pytest.mark.parametrize("max_workers", [1, 4])
def test_supervision_is_deterministic_across_worker_counts(max_workers):
    kernel = Kernel(max_workers=max_workers)
    for i in range(4):
        kernel.register(Flaky(f"f{i}", fail_times=i, max_restarts=5))

    with kernel:
        reports = kernel.run(max_ticks=20)

    assert kernel.tick == 4
    assert [kernel.get(f"f{i}").restarts for i in range(4)] == [0, 1, 2, 3]
    assert reports[0].restarted == ("f1", "f2", "f3")
    assert reports[1].restarted == ("f2", "f3")
    assert reports[2].restarted == ("f3",)


def test_restarted_agent_keeps_its_pending_messages():
    kernel = Kernel()

    class Picky(Agent):
        restart_policy = RestartPolicy.ON_FAILURE

        def __init__(self, name):
            super().__init__(name)
            self.seen = []

        def step(self, ctx):
            if self.restarts == 0:
                raise RuntimeError("first attempt always fails")
            if ctx.inbox:
                self.seen.extend(m.payload for m in ctx.inbox)
                ctx.exit()
                return
            ctx.wait()

    picky = kernel.register(Picky("picky"))

    def sender(ctx):
        if ctx.tick == 3:
            ctx.send("picky", "after-restart")
            ctx.exit()

    kernel.spawn("sender", sender)
    kernel.run(max_ticks=10)

    # The mailbox survived the restart rather than being reclaimed.
    assert picky.seen == ["after-restart"]
    assert picky.state is AgentState.DONE
