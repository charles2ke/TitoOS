"""Tests for the platform layer: foreign agents running on this kernel."""

from __future__ import annotations

import pytest

from titoos import (
    AsyncBackend,
    FunctionAgent,
    Kernel,
    SerialBackend,
    StopReason,
    ThreadBackend,
)
from titoos.platforms import (
    AsyncPlatformAgent,
    CallableAdapter,
    MissingDependency,
    Outbound,
    PlatformAdapter,
    PlatformAgent,
    PlatformError,
    PlatformRegistry,
    PlatformTurn,
    available_platforms,
    create_platform,
    register_platform,
    registry,
    require,
)


class FakePlatform(PlatformAdapter):
    """Stands in for an SDK: records what it was asked, answers a string."""

    name = "fake"

    def __init__(self, name: str | None = None, replies: list | None = None) -> None:
        super().__init__(name)
        self.requests: list = []
        self.turns: list[PlatformTurn] = []
        self.replies = replies
        self.closed = False
        self.memory: list[str] = []

    def invoke(self, request, turn):
        self.requests.append(request)
        self.turns.append(turn)
        self.memory.append(str(request))
        if self.replies is not None:
            return self.replies.pop(0)
        return f"fake:{request}"

    def save_state(self):
        return {"memory": list(self.memory)}

    def load_state(self, data):
        self.memory = list(data.get("memory") or [])

    def close(self):
        self.closed = True


class AsyncFakePlatform(PlatformAdapter):
    name = "afake"
    is_async = True

    def __init__(self) -> None:
        super().__init__()
        self.requests: list = []

    async def ainvoke(self, request, turn):
        self.requests.append(request)
        return f"async:{request}"


def collector(sink: list):
    """An agent that records everything it receives and then waits."""

    def agent(ctx):
        for message in ctx.inbox:
            sink.append((message.sender, message.payload))
        ctx.wait()

    return agent


# --- turn and outbound ------------------------------------------------------


def test_turn_exposes_payloads_sender_and_text():
    kernel = Kernel()
    seen: list[PlatformTurn] = []
    platform = FakePlatform()
    kernel.register(platform.agent("assistant"))

    def boss(ctx):
        if ctx.tick == 1:
            ctx.send("assistant", "one")
            ctx.send("assistant", "two")
        for _ in ctx.inbox:
            ctx.exit()

    kernel.spawn("boss", boss)
    kernel.run(max_ticks=5)

    seen = platform.turns
    assert len(seen) == 1
    turn = seen[0]
    assert turn.agent == "assistant"
    assert turn.payloads == ("one", "two")
    assert turn.sender == "boss"
    assert turn.text == "one\ntwo"


def test_turn_text_falls_back_to_the_seed():
    turn = PlatformTurn(agent="a", tick=1, seed="hello")
    assert turn.text == "hello"
    assert PlatformTurn(agent="a", tick=1).text == ""


def test_outbound_cannot_be_broadcast_and_addressed():
    with pytest.raises(ValueError):
        Outbound(payload="x", to="sink", broadcast=True)


def test_outbound_metadata_is_read_only():
    outbound = Outbound(payload="x", metadata={"trace": 1})
    with pytest.raises(TypeError):
        outbound.metadata["trace"] = 2  # type: ignore[index]


# --- the default translation ------------------------------------------------


def test_single_payload_is_passed_through_untouched():
    adapter = FakePlatform()
    turn = PlatformTurn(agent="a", tick=1, messages=())
    assert adapter.to_platform(turn) is None

    kernel = Kernel()
    kernel.register(adapter.agent("assistant"))
    kernel.spawn("boss", lambda ctx: (ctx.send("assistant", {"q": 1}), ctx.wait()))
    kernel.run(max_ticks=3)
    assert adapter.requests == [{"q": 1}]


def test_multiple_messages_arrive_as_a_list_in_delivery_order():
    adapter = FakePlatform()
    kernel = Kernel()
    kernel.register(adapter.agent("assistant"))

    def boss(ctx):
        if ctx.tick == 1:
            ctx.send("assistant", "a")
            ctx.send("assistant", "b")
        ctx.wait()

    kernel.spawn("boss", boss)
    kernel.run(max_ticks=4)
    assert adapter.requests == [["a", "b"]]


def test_none_result_sends_nothing():
    received: list = []
    adapter = FakePlatform(replies=[None])
    kernel = Kernel()
    kernel.register(adapter.agent("assistant"))
    kernel.spawn("sink", collector(received))
    kernel.spawn("boss", lambda ctx: (ctx.send("assistant", "hi"), ctx.wait()))
    kernel.run(max_ticks=5)
    assert received == []


def test_a_sequence_of_outbound_fans_out():
    received: list = []
    adapter = FakePlatform(
        replies=[[Outbound(payload="one", to="sink"), Outbound(payload="two", to="sink")]]
    )
    kernel = Kernel()
    kernel.register(adapter.agent("assistant"))
    kernel.spawn("sink", collector(received))
    kernel.spawn("boss", lambda ctx: (ctx.send("assistant", "go"), ctx.wait()))
    kernel.run(max_ticks=6)
    assert received == [("assistant", "one"), ("assistant", "two")]


# --- routing ----------------------------------------------------------------


def test_reply_goes_back_to_the_sender_by_default():
    received: list = []
    adapter = FakePlatform()
    kernel = Kernel()
    kernel.register(adapter.agent("assistant"))

    def boss(ctx):
        if ctx.tick == 1:
            ctx.send("assistant", "ping")
        for message in ctx.inbox:
            received.append((message.sender, message.payload))
        ctx.wait()

    kernel.spawn("boss", boss)
    kernel.run(max_ticks=5)
    assert received == [("assistant", "fake:ping")]


def test_reply_to_overrides_the_sender():
    received: list = []
    adapter = FakePlatform()
    kernel = Kernel()
    kernel.register(adapter.agent("assistant", reply_to="sink"))
    kernel.spawn("sink", collector(received))
    kernel.spawn("boss", lambda ctx: (ctx.send("assistant", "ping"), ctx.wait()))
    kernel.run(max_ticks=6)
    assert received == [("assistant", "fake:ping")]


def test_broadcast_sends_to_everyone_else():
    received: list = []
    adapter = FakePlatform()
    kernel = Kernel()
    kernel.register(adapter.agent("assistant", seed="start", broadcast=True))
    kernel.spawn("sink", collector(received))
    kernel.run(max_ticks=4)
    assert received == [("assistant", "fake:start")]


def test_metadata_survives_the_round_trip():
    seen: list = []
    adapter = FakePlatform(replies=[Outbound(payload="x", to="sink", metadata={"trace": 7})])
    kernel = Kernel()
    kernel.register(adapter.agent("assistant"))

    def sink(ctx):
        for message in ctx.inbox:
            seen.append(dict(message.metadata))
        ctx.wait()

    kernel.spawn("sink", sink)
    kernel.spawn("boss", lambda ctx: (ctx.send("assistant", "go"), ctx.wait()))
    kernel.run(max_ticks=6)
    assert seen == [{"trace": 7}]


def test_an_undeliverable_reply_fails_the_agent():
    adapter = FakePlatform()
    kernel = Kernel()
    kernel.register(adapter.agent("assistant", seed="start"))
    kernel.run(max_ticks=3)
    assert [name for name, _ in kernel.errors] == ["assistant"]
    error = kernel.errors[0][1]
    assert isinstance(error, PlatformError)
    assert "no destination" in str(error)


# --- scheduling -------------------------------------------------------------


def test_an_idle_platform_agent_waits_instead_of_invoking():
    adapter = FakePlatform()
    kernel = Kernel()
    kernel.register(adapter.agent("assistant"))
    kernel.run(max_ticks=5)
    assert adapter.requests == []
    assert kernel.stop_reason is StopReason.QUIESCENT


def test_the_seed_runs_one_turn_before_any_mail_arrives():
    received: list = []
    adapter = FakePlatform()
    kernel = Kernel()
    kernel.register(adapter.agent("assistant", seed="kickoff", reply_to="sink"))
    kernel.spawn("sink", collector(received))
    kernel.run(max_ticks=6)
    assert adapter.requests == ["kickoff"]
    assert received == [("assistant", "fake:kickoff")]


def test_max_turns_ends_the_agent():
    adapter = FakePlatform()
    kernel = Kernel()
    agent = adapter.agent("assistant", seed="a", reply_to="sink", max_turns=1)
    kernel.register(agent)
    kernel.spawn("sink", collector([]))
    kernel.run(max_ticks=6)
    assert agent.turns == 1
    assert not agent.is_alive


def test_a_platform_failure_is_a_platform_error_and_is_supervised():
    class Broken(PlatformAdapter):
        name = "broken"

        def invoke(self, request, turn):
            raise ValueError("model exploded")

    from titoos import RestartPolicy

    kernel = Kernel()
    agent = Broken().agent("assistant")
    agent.restart_policy = RestartPolicy.ON_FAILURE
    agent.max_restarts = 2
    kernel.register(agent)
    # The crashed turn's mail is requeued, so the same message is redelivered
    # until the restart budget runs out.
    kernel.spawn("boss", lambda ctx: (ctx.send("assistant", "go"), ctx.wait()))
    kernel.run(max_ticks=10)

    assert agent.restarts == 2
    assert len(kernel.errors) == 3
    error = kernel.errors[0][1]
    assert isinstance(error, PlatformError)
    assert error.platform == "broken"
    assert error.operation == "invoke"
    assert "model exploded" in str(error)


def test_a_sync_agent_refuses_an_awaitable_result():
    kernel = Kernel()
    kernel.register(
        PlatformAgent("assistant", CallableAdapter("coro", _coroutine_factory), seed="go")
    )
    kernel.run(max_ticks=3)
    error = kernel.errors[0][1]
    assert isinstance(error, PlatformError)
    assert "AsyncBackend" in str(error)


async def _async_reply(request, turn):
    return f"async:{request}"


def _coroutine_factory(request, turn):
    return _async_reply(request, turn)


# --- determinism ------------------------------------------------------------


def _run_conversation(backend):
    received: list = []
    kernel = Kernel(backend=backend)
    for index in range(4):
        kernel.register(
            FakePlatform(name=f"p{index}").agent(f"assistant{index}", reply_to="sink")
        )
    kernel.spawn("sink", collector(received))

    def boss(ctx):
        if ctx.tick == 1:
            for index in range(4):
                ctx.send(f"assistant{index}", f"q{index}")
        ctx.wait()

    kernel.spawn("boss", boss)
    reports = kernel.run(max_ticks=8)
    kernel.shutdown()
    return received, [(r.tick, r.ran, r.failed) for r in reports]


def test_platform_agents_behave_identically_on_serial_and_thread_backends():
    serial = _run_conversation(SerialBackend())
    threaded = _run_conversation(ThreadBackend(4))
    assert serial == threaded


def test_async_platform_agents_run_on_the_async_backend():
    received: list = []
    adapter = AsyncFakePlatform()
    agent = adapter.agent("assistant", seed="go", reply_to="sink")
    assert isinstance(agent, AsyncPlatformAgent)

    kernel = Kernel(backend=AsyncBackend())
    kernel.register(agent)
    kernel.spawn("sink", collector(received))
    kernel.run(max_ticks=6)
    kernel.shutdown()
    assert received == [("assistant", "async:go")]


def test_async_platform_agents_are_rejected_by_the_serial_backend():
    kernel = Kernel()
    kernel.register(AsyncFakePlatform().agent("assistant", seed="go"))
    with pytest.raises(TypeError, match="AsyncBackend"):
        kernel.step()


def test_a_sync_adapter_still_runs_on_the_async_backend():
    received: list = []
    kernel = Kernel(backend=AsyncBackend())
    # A sync adapter wrapped in an async agent is legitimate: ainvoke defers
    # to invoke, so the same adapter works on either backend.
    agent = AsyncPlatformAgent("assistant", FakePlatform(), seed="go", reply_to="sink")
    kernel.register(agent)
    kernel.spawn("sink", collector(received))
    kernel.run(max_ticks=6)
    kernel.shutdown()
    assert received == [("assistant", "fake:go")]


# --- persistence ------------------------------------------------------------


def test_platform_state_survives_a_snapshot():
    adapter = FakePlatform()
    kernel = Kernel()
    kernel.register(adapter.agent("assistant", reply_to="sink"))
    kernel.spawn("sink", collector([]))
    kernel.spawn("boss", lambda ctx: (ctx.send("assistant", "one"), ctx.exit()))
    kernel.run(max_ticks=4)

    snapshot = kernel.snapshot()
    record = next(r for r in snapshot.agents if r.name == "assistant")
    assert record.data["turns"] == 1
    assert record.data["platform"] == {"memory": ["one"]}

    restored_adapter = FakePlatform()
    restored = Kernel.restore(
        snapshot,
        {
            "PlatformAgent": lambda name: restored_adapter.agent(name, reply_to="sink"),
            "FunctionAgent": lambda name: FunctionAgent(name, collector([])),
        },
    )
    assert restored.get("assistant").turns == 1
    assert restored_adapter.memory == ["one"]


# --- construction -----------------------------------------------------------


def test_an_adapter_needs_a_name():
    with pytest.raises(ValueError):
        PlatformAdapter()


def test_reply_to_and_broadcast_are_mutually_exclusive():
    with pytest.raises(ValueError):
        FakePlatform().agent("assistant", reply_to="sink", broadcast=True)


def test_max_turns_must_be_positive():
    with pytest.raises(ValueError):
        FakePlatform().agent("assistant", max_turns=0)


def test_platform_agent_rejects_a_non_adapter():
    with pytest.raises(TypeError):
        PlatformAgent("assistant", object())  # type: ignore[arg-type]


def test_an_adapter_without_invoke_says_so():
    kernel = Kernel()
    kernel.register(PlatformAdapter("bare").agent("assistant", seed="go"))
    kernel.run(max_ticks=3)
    assert isinstance(kernel.errors[0][1], PlatformError)
    assert "neither invoke() nor ainvoke()" in str(kernel.errors[0][1])


def test_callable_adapter_wraps_a_function():
    adapter = CallableAdapter("echo", lambda request, turn: f"echo:{request}")
    assert adapter.wrapped is not None
    kernel = Kernel()
    received: list = []
    kernel.register(adapter.agent("assistant", seed="hi", reply_to="sink"))
    kernel.spawn("sink", collector(received))
    kernel.run(max_ticks=5)
    assert received == [("assistant", "echo:hi")]


def test_callable_adapter_rejects_a_non_callable():
    with pytest.raises(TypeError):
        CallableAdapter("echo", "not callable")  # type: ignore[arg-type]


# --- the registry -----------------------------------------------------------


def test_registry_creates_a_registered_adapter():
    local = PlatformRegistry()
    local.register("fake", FakePlatform)
    assert local.names() == ("fake",)
    assert "fake" in local
    assert len(local) == 1
    adapter = local.create("fake", "renamed")
    assert isinstance(adapter, FakePlatform)
    assert adapter.name == "renamed"


def test_registry_refuses_a_duplicate_unless_replacing():
    local = PlatformRegistry()
    local.register("fake", FakePlatform)
    with pytest.raises(ValueError):
        local.register("fake", FakePlatform)
    local.register("fake", FakePlatform, replace=True)


def test_registry_reports_what_is_available():
    local = PlatformRegistry()
    local.register("fake", FakePlatform)
    with pytest.raises(PlatformError, match="available: fake"):
        local.get("nope")


def test_registry_rejects_a_bad_factory():
    local = PlatformRegistry()
    with pytest.raises(TypeError):
        local.register("fake", 42)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        local.register("", FakePlatform)


def test_registry_resolves_a_dotted_target_lazily():
    local = PlatformRegistry()
    local.register("fake", f"{FakePlatform.__module__}:FakePlatform")
    assert isinstance(local.create("fake"), FakePlatform)


def test_registry_reports_a_missing_module_as_a_missing_dependency():
    local = PlatformRegistry()
    local.register("ghost", "titoos_does_not_exist:Ghost")
    with pytest.raises(MissingDependency):
        local.create("ghost")


def test_registry_reports_a_missing_attribute():
    local = PlatformRegistry()
    local.register("ghost", f"{FakePlatform.__module__}:NoSuchAdapter")
    with pytest.raises(PlatformError, match="has no attribute"):
        local.create("ghost")


def test_registry_rejects_a_malformed_target():
    local = PlatformRegistry()
    local.register("ghost", "not-a-target")
    with pytest.raises(PlatformError, match="module:attribute"):
        local.create("ghost")


def test_registry_rejects_a_factory_returning_the_wrong_type():
    local = PlatformRegistry()
    local.register("ghost", lambda: object())
    with pytest.raises(PlatformError, match="not a PlatformAdapter"):
        local.create("ghost")


def test_the_default_registry_is_usable_and_starts_empty_of_our_name():
    assert "titoos-test-platform" not in available_platforms()
    register_platform("titoos-test-platform", FakePlatform)
    try:
        assert "titoos-test-platform" in available_platforms()
        assert isinstance(create_platform("titoos-test-platform"), FakePlatform)
        assert list(registry)  # iterating yields names
    finally:
        registry.unregister("titoos-test-platform")
    assert "titoos-test-platform" not in available_platforms()


# --- optional dependencies --------------------------------------------------


def test_require_returns_an_installed_module():
    assert require("json", extra="json").__name__ == "json"


def test_require_names_the_extra_to_install():
    with pytest.raises(MissingDependency) as excinfo:
        require("titoos_not_a_real_sdk", extra="fancy", platform="fancy")
    assert 'pip install "titoos[fancy]"' in str(excinfo.value)


def test_importing_titoos_does_not_import_the_platform_layer():
    import subprocess
    import sys

    code = (
        "import sys, titoos;"
        "assert 'titoos.platforms' not in sys.modules;"
        "assert titoos.PlatformAgent is not None;"
        "assert 'titoos.platforms' in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_platform_errors_are_integration_errors():
    from titoos import IntegrationError

    assert issubclass(PlatformError, IntegrationError)
    assert issubclass(MissingDependency, PlatformError)
