import pytest

from titoos import Agent, AgentState, Kernel


class Counter(Agent):
    """Steps a fixed number of times, then exits."""

    def __init__(self, name, limit):
        super().__init__(name)
        self.limit = limit
        self.count = 0

    def step(self, ctx):
        self.count += 1
        if self.count >= self.limit:
            ctx.exit()


def test_register_rejects_duplicate_names():
    kernel = Kernel()
    kernel.register(Counter("a", 1))

    with pytest.raises(ValueError):
        kernel.register(Counter("a", 1))


def test_run_stops_when_all_agents_are_done():
    kernel = Kernel()
    short = kernel.register(Counter("short", 2))
    long = kernel.register(Counter("long", 3))

    reports = kernel.run(max_ticks=10)

    assert len(reports) == 3
    assert kernel.tick == 3
    assert short.count == 2 and long.count == 3
    assert short.state is AgentState.DONE and long.state is AgentState.DONE
    assert reports[-1].ran == ("long",)


def test_done_agent_mailbox_is_unregistered():
    kernel = Kernel()
    kernel.register(Counter("done", 1))

    kernel.run(max_ticks=1)

    assert "done" not in kernel.bus.mailboxes()
    with pytest.raises(KeyError):
        kernel.bus.post("sender", "done", "late")


def test_run_respects_max_ticks():
    kernel = Kernel()
    forever = kernel.register(Counter("forever", 1_000))

    kernel.run(max_ticks=5)

    assert kernel.tick == 5
    assert forever.count == 5
    assert forever.state is AgentState.READY


def test_failing_agent_is_isolated():
    kernel = Kernel()

    def boom(ctx):
        raise RuntimeError("boom")

    kernel.spawn("boom", boom)
    survivor = kernel.register(Counter("survivor", 2))

    reports = kernel.run(max_ticks=10)

    assert kernel.get("boom").state is AgentState.FAILED
    assert survivor.count == 2
    assert [name for name, _ in kernel.errors] == ["boom"]
    assert reports[0].failed == ("boom",)
    assert reports[1].ran == ("survivor",)


def test_failed_agent_mailbox_is_unregistered():
    kernel = Kernel()

    def boom(ctx):
        raise RuntimeError("boom")

    kernel.spawn("boom", boom)
    kernel.run(max_ticks=1)

    assert "boom" not in kernel.bus.mailboxes()
    with pytest.raises(KeyError):
        kernel.bus.post("sender", "boom", "late")


def test_agents_exchange_messages_across_ticks():
    kernel = Kernel()
    received = []

    def pinger(ctx):
        if ctx.tick == 1:
            ctx.send("ponger", "ping")
        else:
            received.extend(m.payload for m in ctx.inbox)
            ctx.exit()

    def ponger(ctx):
        for message in ctx.inbox:
            ctx.send(message.sender, "pong")
            ctx.exit()

    kernel.spawn("pinger", pinger)
    kernel.spawn("ponger", ponger)

    kernel.run(max_ticks=10)

    assert received == ["pong"]


def test_agent_can_spawn_another_agent():
    kernel = Kernel()

    def parent(ctx):
        ctx.spawn(Counter("child", 1))
        ctx.exit()

    kernel.spawn("parent", parent)
    kernel.run(max_ticks=5)

    assert kernel.get("child").count == 1
    assert kernel.get("child").state is AgentState.DONE


def test_base_agent_step_is_abstract():
    kernel = Kernel()
    kernel.register(Agent("bare"))

    kernel.run(max_ticks=1)

    assert kernel.get("bare").state is AgentState.FAILED
    assert isinstance(kernel.errors[0][1], NotImplementedError)


def test_negative_max_ticks_rejected():
    with pytest.raises(ValueError):
        Kernel().run(max_ticks=-1)
