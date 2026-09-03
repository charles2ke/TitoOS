import pytest

from titoos import (
    Agent,
    AgentState,
    Kernel,
    RestartPolicy,
    Snapshot,
    StopReason,
)


class Accumulator(Agent):
    """Sums the numbers it is sent, persisting the running total."""

    def __init__(self, name, total=0):
        super().__init__(name)
        self.total = total

    def step(self, ctx):
        for message in ctx.inbox:
            self.total += message.payload
        ctx.wait()

    def save_state(self):
        return {"total": self.total}

    def load_state(self, data):
        self.total = data["total"]


FACTORIES = {"Accumulator": Accumulator}


def test_snapshot_captures_tick_agents_and_state():
    kernel = Kernel()
    acc = kernel.register(Accumulator("acc"))
    kernel.spawn("feeder", lambda ctx: (ctx.send("acc", 5), ctx.exit()))

    kernel.run(max_ticks=5)
    snapshot = kernel.snapshot()

    assert snapshot.tick == kernel.tick
    assert acc.total == 5
    record = next(r for r in snapshot.agents if r.name == "acc")
    assert record.kind == "Accumulator"
    assert record.data == {"total": 5}
    assert record.state == AgentState.WAITING.value


def test_restore_round_trips_agent_state():
    kernel = Kernel()
    kernel.register(Accumulator("acc", total=7))
    kernel.run(max_ticks=2)

    restored = Kernel.restore(kernel.snapshot(), FACTORIES)

    assert restored.tick == kernel.tick
    assert restored.get("acc").total == 7
    assert restored.get("acc").state is AgentState.WAITING


def test_restored_kernel_continues_the_workflow():
    kernel = Kernel()
    kernel.register(Accumulator("acc"))
    kernel.spawn("feeder", lambda ctx: (ctx.send("acc", 10), ctx.exit()))
    kernel.run(max_ticks=5)
    assert kernel.get("acc").total == 10

    restored = Kernel.restore(kernel.snapshot(), FACTORIES)
    restored.spawn("feeder2", lambda ctx: (ctx.send("acc", 32), ctx.exit()))
    restored.run(max_ticks=5)

    assert restored.get("acc").total == 42


def test_pending_messages_survive_a_snapshot():
    kernel = Kernel()
    kernel.register(Accumulator("acc"))
    kernel.spawn("feeder", lambda ctx: (ctx.send("acc", 3), ctx.exit()))

    kernel.step()  # feeder sends; the message is still pending
    snapshot = kernel.snapshot()

    assert [m.payload for m in snapshot.mailboxes["acc"]] == [3]

    restored = Kernel.restore(snapshot, FACTORIES)
    restored.run(max_ticks=5)

    assert restored.get("acc").total == 3


def test_json_round_trip():
    kernel = Kernel()
    kernel.register(Accumulator("acc", total=11))
    kernel.spawn("feeder", lambda ctx: (ctx.send("acc", 4), ctx.exit()))
    kernel.step()

    text = kernel.snapshot().to_json()
    snapshot = Snapshot.from_json(text)
    restored = Kernel.restore(snapshot, FACTORIES)
    restored.run(max_ticks=5)

    assert restored.get("acc").total == 15


def test_snapshot_preserves_supervision_metadata():
    class Flaky(Accumulator):
        restart_policy = RestartPolicy.ON_FAILURE

        def step(self, ctx):
            if self.restarts == 0:
                raise RuntimeError("boom")
            ctx.wait()

    kernel = Kernel()

    def parent(ctx):
        ctx.spawn(Flaky("child"))
        ctx.exit()

    kernel.spawn("parent", parent)
    kernel.run(max_ticks=5)

    record = next(r for r in kernel.snapshot().agents if r.name == "child")
    assert record.parent == "parent"
    assert record.restarts == 1

    restored = Kernel.restore(kernel.snapshot(), {"Flaky": Flaky})
    assert restored.get("child").parent == "parent"
    assert restored.get("child").restarts == 1


def test_restore_rejects_unknown_agent_kind():
    kernel = Kernel()
    kernel.register(Accumulator("acc"))

    with pytest.raises(KeyError, match="Accumulator"):
        Kernel.restore(kernel.snapshot(), {})


def test_restore_rejects_a_factory_that_renames_the_agent():
    kernel = Kernel()
    kernel.register(Accumulator("acc"))

    with pytest.raises(ValueError, match="expected 'acc'"):
        Kernel.restore(kernel.snapshot(), {"Accumulator": lambda name: Accumulator("wrong")})


def test_snapshot_refuses_while_an_agent_is_running():
    kernel = Kernel()
    captured = []

    def sneaky(ctx):
        try:
            ctx.kernel.snapshot()
        except RuntimeError as exc:
            captured.append(str(exc))
        ctx.exit()

    kernel.spawn("sneaky", sneaky)
    kernel.run(max_ticks=2)

    assert captured and "cannot snapshot while agents are running" in captured[0]


def test_snapshot_version_is_checked():
    kernel = Kernel()
    kernel.register(Accumulator("acc"))
    data = kernel.snapshot().to_dict()
    data["version"] = 999

    with pytest.raises(ValueError, match="unsupported snapshot version"):
        Snapshot.from_dict(data)


def test_finished_agents_are_restored_without_a_mailbox():
    kernel = Kernel()
    kernel.spawn("transient", lambda ctx: ctx.exit())
    kernel.register(Accumulator("acc"))
    kernel.run(max_ticks=3)

    restored = Kernel.restore(kernel.snapshot(), FACTORIES)

    assert restored.get("transient").state is AgentState.DONE
    assert "transient" not in restored.bus.mailboxes()
    assert "acc" in restored.bus.mailboxes()


def test_default_agent_saves_no_state():
    class Bare(Agent):
        def step(self, ctx):
            ctx.exit()

    kernel = Kernel()
    kernel.register(Bare("bare"))

    record = next(r for r in kernel.snapshot().agents if r.name == "bare")
    assert record.data == {}


def test_restored_kernel_reports_quiescence():
    kernel = Kernel()
    kernel.register(Accumulator("acc"))
    kernel.run(max_ticks=3)

    restored = Kernel.restore(kernel.snapshot(), FACTORIES)
    restored.run(max_ticks=10)

    assert restored.stop_reason is StopReason.QUIESCENT
    assert restored.tick == kernel.tick  # nothing left to do, no ticks burned
