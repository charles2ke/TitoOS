"""Tests for the A2A client, the integration and the platform adapter."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import titoos.integrations.a2a as a2a_module
from titoos import (
    A2AAdapter,
    A2AIntegration,
    A2APeer,
    FunctionAgent,
    IntegrationError,
    JsonRpcError,
    Kernel,
)
from titoos.platforms import PlatformError, create_platform


def _task(task_id: str, context_id: str, state: str, *, says: str | None = None, artifacts=None) -> dict:
    """A task in A2A 1.0 JSON."""
    task: dict = {"id": task_id, "contextId": context_id, "status": {"state": f"TASK_STATE_{state}"}}
    if says is not None:
        task["status"]["message"] = {
            "messageId": f"{task_id}-status",
            "role": "ROLE_AGENT",
            "parts": [{"text": says}],
        }
    if artifacts is not None:
        task["artifacts"] = [{"artifactId": f"{task_id}-out", "parts": artifacts}]
    return task


class A2AServer:
    """What the fake agent does, and what it has seen."""

    def __init__(self) -> None:
        self.base = ""
        self.card: dict = {}
        self.card_redirect: str | None = None
        self.sse = False
        self.polls = 0
        #: (verb, path, lower-cased headers, JSON body or None), in arrival order.
        self.seen: list[tuple[str, str, dict, dict | None]] = []

    def v1_card(self) -> dict:
        return {
            "name": "Helper",
            "description": "Answers questions.",
            "version": "1.0.0",
            "supportedInterfaces": [
                {"url": f"{self.base}/grpc", "protocolBinding": "GRPC", "protocolVersion": "1.0"},
                {"url": f"{self.base}/rpc", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"},
            ],
            "capabilities": {},
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["text/plain"],
            "skills": [],
        }

    def v03_card(self) -> dict:
        return {
            "name": "Old helper",
            "url": f"{self.base}/rpc03",
            "preferredTransport": "JSONRPC",
            "protocolVersion": "0.3.0",
            "capabilities": {},
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["text/plain"],
            "skills": [],
        }

    def gets(self) -> list[str]:
        return [path for verb, path, _, _ in self.seen if verb == "GET"]

    def posts(self) -> list[tuple[dict, dict]]:
        return [(headers, body) for verb, _, headers, body in self.seen if verb == "POST"]

    def sent_messages(self) -> list[dict]:
        return [body["params"]["message"] for _, body in self.posts() if "message" in body["params"]]


class _A2AHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence the test output
        pass

    @property
    def fake(self) -> A2AServer:
        return self.server.fake  # type: ignore[attr-defined]

    def _reply(self, status: int, payload, extra=()):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for key, value in extra:
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - http.server API
        fake = self.fake
        fake.seen.append(("GET", self.path, {k.lower(): v for k, v in self.headers.items()}, None))
        if not self.path.endswith("/.well-known/agent-card.json"):
            self._reply(404, {})
        elif fake.card_redirect:
            self._reply(302, {}, [("Location", fake.card_redirect)])
        else:
            self._reply(200, fake.card)

    def do_POST(self):  # noqa: N802 - http.server API
        fake = self.fake
        headers = {k.lower(): v for k, v in self.headers.items()}
        body = json.loads(self.rfile.read(int(headers["content-length"])))
        fake.seen.append(("POST", self.path, headers, body))
        handler = {"/rpc": self._v1, "/rpc03": self._v03}.get(self.path)
        if handler is None:
            self._reply(404, {})
            return
        response = {"jsonrpc": "2.0", "id": body["id"], **handler(body["method"], body["params"])}
        if not fake.sse:
            self._reply(200, response)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(f": hello\n\ndata: {json.dumps(response)}\n\n".encode())

    def _v1(self, method: str, params: dict) -> dict:
        if method == "SendMessage":
            message = params["message"]
            text = message["parts"][0].get("text")
            context = message.get("contextId") or "ctx-1"
            if message.get("taskId") == "t-ask":
                return {"result": {"task": _task("t-ask", context, "COMPLETED", artifacts=[{"text": f"painted {text}"}])}}
            if text == "hello":
                reply = {"messageId": "m-1", "role": "ROLE_AGENT", "contextId": context, "parts": [{"text": "hi there"}]}
                return {"result": {"message": reply}}
            if text == "slow":
                self.fake.polls = 0
                return {"result": {"task": _task("t-slow", context, "WORKING")}}
            if text == "ask":
                return {"result": {"task": _task("t-ask", context, "INPUT_REQUIRED", says="which colour?")}}
            if text == "break":
                return {"result": {"task": _task("t-break", context, "FAILED", says="kaput")}}
            # Anything else is echoed back as it arrived.
            reply = {"messageId": "m-2", "role": "ROLE_AGENT", "contextId": context, "parts": message["parts"]}
            return {"result": {"message": reply}}
        if method == "GetTask":
            if params["id"] != "t-slow":
                return {"error": {"code": -32001, "message": "Task not found"}}
            self.fake.polls += 1
            if self.fake.polls < 2:
                return {"result": _task("t-slow", "ctx-1", "WORKING")}
            return {"result": _task("t-slow", "ctx-1", "COMPLETED", artifacts=[{"text": "finally"}])}
        if method == "CancelTask":
            return {"result": _task(params["id"], "ctx-1", "CANCELED")}
        return {"error": {"code": -32601, "message": "Method not found"}}

    def _v03(self, method: str, params: dict) -> dict:
        if method != "message/send":
            return {"error": {"code": -32601, "message": "Method not found"}}
        message = params["message"]
        first = message["parts"][0]
        context = message.get("contextId") or "ctx-03"
        if message.get("taskId") == "t-ask":
            artifact = {"artifactId": "a", "parts": [{"kind": "text", "text": f"painted {first['text']}"}]}
            task = {"kind": "task", "id": "t-ask", "contextId": context, "status": {"state": "completed"}, "artifacts": [artifact]}
            return {"result": task}
        if first.get("text") == "ask":
            question = {"kind": "message", "messageId": "s", "role": "agent", "parts": [{"kind": "text", "text": "which colour?"}]}
            task = {"kind": "task", "id": "t-ask", "contextId": context, "status": {"state": "input-required", "message": question}}
            return {"result": task}
        reply = {"kind": "message", "messageId": "m-03", "role": "agent", "contextId": context, "parts": message["parts"]}
        return {"result": reply}


@pytest.fixture
def agent_server():
    fake = A2AServer()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _A2AHandler)
    server.fake = fake  # type: ignore[attr-defined]
    fake.base = f"http://127.0.0.1:{server.server_port}"
    fake.card = fake.v1_card()
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield fake
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def fast_polling(monkeypatch):
    monkeypatch.setattr(a2a_module, "_POLL_START", 0.01)
    monkeypatch.setattr(a2a_module, "_POLL_MAX", 0.01)


# --- the integration --------------------------------------------------------


def test_a2a_1_0_message_round_trip(agent_server):
    peer = A2APeer(agent_server.base, headers={"Authorization": "Bearer s3cret"})
    a2a = A2AIntegration({"helper": peer})
    first = a2a.send_message("helper", "hello")
    second = a2a.send_message("helper", "hello", context_id=first.context_id)

    assert (first.kind, first.text, first.context_id) == ("message", "hi there", "ctx-1")
    assert first.ok and first.done and first.state is None
    assert second.text == "hi there"
    assert a2a.agent_card("helper")["name"] == "Helper"
    assert agent_server.gets() == ["/.well-known/agent-card.json"]  # read once
    headers, body = agent_server.posts()[0]
    assert body["method"] == "SendMessage"
    assert headers["a2a-version"] == "1.0"
    assert headers["authorization"] == "Bearer s3cret"
    message = body["params"]["message"]
    assert (message["role"], message["parts"]) == ("ROLE_USER", [{"text": "hello"}])
    assert body["params"]["configuration"] == {"returnImmediately": False}
    assert agent_server.sent_messages()[1]["contextId"] == "ctx-1"
    assert "s3cret" not in repr(peer) + repr(a2a.describe())


def test_a2a_waits_for_a_working_task_to_finish(agent_server, fast_polling):
    result = A2AIntegration({"helper": agent_server.base}).send_message("helper", "slow")
    assert (result.state, result.text, result.task_id) == ("completed", "finally", "t-slow")
    polls = [body["params"] for _, body in agent_server.posts() if body["method"] == "GetTask"]
    assert polls == [{"id": "t-slow"}, {"id": "t-slow"}]


def test_a2a_follows_up_a_task_it_did_not_wait_for(agent_server):
    a2a = A2AIntegration({"helper": agent_server.base})
    started = a2a.send_message("helper", "slow", wait=False)
    assert (started.state, started.done) == ("working", False)
    assert agent_server.posts()[0][1]["params"]["configuration"] == {"returnImmediately": True}
    assert a2a.get_task("helper", "t-slow").state == "working"
    assert a2a.get_task("helper", "t-slow").text == "finally"
    assert a2a.cancel_task("helper", "t-slow").state == "canceled"


def test_a2a_continues_a_task_that_asks_for_input(agent_server):
    a2a = A2AIntegration({"helper": agent_server.base})
    question = a2a.send_message("helper", "ask")
    assert question.needs_input and not question.done
    assert (question.text, question.task_id) == ("which colour?", "t-ask")

    answer = a2a.send_message("helper", "blue", task_id=question.task_id, context_id=question.context_id)
    assert (answer.state, answer.text) == ("completed", "painted blue")
    assert agent_server.sent_messages()[1]["taskId"] == "t-ask"


def test_a2a_reports_failed_tasks_and_refusals(agent_server):
    a2a = A2AIntegration({"helper": agent_server.base})
    failed = a2a.send_message("helper", "break")
    assert (failed.state, failed.ok, failed.done, failed.text) == ("failed", False, True, "kaput")

    with pytest.raises(JsonRpcError, match="TaskNotFoundError") as caught:
        a2a.get_task("helper", "t-unknown")
    assert caught.value.code == -32001
    with pytest.raises(IntegrationError, match="unknown A2A peer 'ghost'; configured peers: helper"):
        a2a.send_message("ghost", "hello")


def test_a2a_speaks_0_3_to_a_0_3_agent(agent_server):
    agent_server.card = agent_server.v03_card()
    a2a = A2AIntegration({"old": agent_server.base})
    question = a2a.send_message("old", "ask")
    answer = a2a.send_message("old", "blue", task_id=question.task_id)

    assert (question.state, question.text) == ("input_required", "which colour?")
    assert (answer.state, answer.text) == ("completed", "painted blue")
    headers, body = agent_server.posts()[0]
    assert body["method"] == "message/send"
    assert "a2a-version" not in headers
    message = body["params"]["message"]
    assert (message["kind"], message["role"]) == ("message", "user")
    assert message["parts"] == [{"kind": "text", "text": "ask"}]
    assert body["params"]["configuration"] == {"blocking": True}


def test_a2a_writes_parts_in_each_version_shape(agent_server):
    parts = [b"\x00\x01", {"url": "https://example.com/a.png", "mediaType": "image/png", "filename": "a.png"}]
    A2AIntegration({"new": agent_server.base}).send_message("new", data={"k": 1}, parts=parts)
    agent_server.card = agent_server.v03_card()
    old = A2AIntegration({"old": agent_server.base})
    reply = old.send_message("old", data={"k": 1}, parts=parts)

    new_parts, old_parts = (m["parts"] for m in agent_server.sent_messages())
    assert new_parts == [
        {"data": {"k": 1}},
        {"raw": "AAE="},
        {"url": "https://example.com/a.png", "mediaType": "image/png", "filename": "a.png"},
    ]
    assert old_parts == [
        {"kind": "data", "data": {"k": 1}},
        {"kind": "file", "file": {"bytes": "AAE="}},
        {"kind": "file", "file": {"uri": "https://example.com/a.png", "name": "a.png", "mimeType": "image/png"}},
    ]
    # Replies come back in the 1.0 shape whatever the version.
    assert reply.parts[1] == {"raw": "AAE="}
    with pytest.raises(IntegrationError, match="0.3 data parts must be objects"):
        old.send_message("old", data=[1, 2])


def test_a2a_reads_a_reply_from_an_event_stream(agent_server):
    agent_server.sse = True
    assert A2AIntegration({"helper": agent_server.base}).send_message("helper", "hello").text == "hi there"


def test_a2a_will_not_follow_a_card_to_another_origin(agent_server):
    agent_server.card["supportedInterfaces"] = [
        {"url": "http://elsewhere.invalid/rpc", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}
    ]
    with pytest.raises(IntegrationError, match=r"another origin \(http://elsewhere.invalid/rpc\).*endpoint="):
        A2AIntegration({"helper": agent_server.base}).send_message("helper", "hello")

    pinned = A2APeer(agent_server.base, endpoint=f"{agent_server.base}/rpc")
    assert A2AIntegration({"helper": pinned}).send_message("helper", "hello").text == "hi there"
    assert len(agent_server.gets()) == 1  # the pinned peer skipped the card


def test_a2a_endpoint_pin_suggests_0_3_when_1_0_is_refused(agent_server):
    guessed = A2APeer(agent_server.base, endpoint=f"{agent_server.base}/rpc03")
    with pytest.raises(JsonRpcError, match="protocol_version='0.3'"):
        A2AIntegration({"old": guessed}).send_message("old", "hello")

    pinned = A2APeer(agent_server.base, endpoint=f"{agent_server.base}/rpc03", protocol_version="0.3")
    assert A2AIntegration({"old": pinned}).send_message("old", "hello").text == "hello"


def test_a2a_never_follows_a_redirect(agent_server):
    agent_server.card_redirect = "http://127.0.0.1:9/card.json"
    with pytest.raises(IntegrationError, match="redirects are not followed"):
        A2AIntegration({"helper": agent_server.base}).agent_card("helper")


def test_agents_message_a2a_peers_through_the_kernel(agent_server):
    seen: list = []
    kernel = Kernel()
    kernel.install(A2AIntegration({"helper": agent_server.base}))

    def worker(ctx):
        seen.append(ctx.call("a2a", "send_message", "helper", "hello").text)
        ctx.exit()

    kernel.spawn("worker", worker)
    kernel.run(max_ticks=3)
    kernel.shutdown()
    assert seen == ["hi there"]


@pytest.mark.parametrize(
    "build, match",
    [
        (lambda: A2AIntegration({}), "non-empty mapping"),
        (lambda: A2APeer("ftp://example.com/agent"), "http or https"),
        (lambda: A2APeer("https://example.com", protocol_version="2.0"), "unsupported A2A protocol_version"),
    ],
)
def test_a2a_configuration_is_validated(build, match):
    with pytest.raises(ValueError, match=match):
        build()


# --- the adapter ------------------------------------------------------------


def collector(sink: list):
    def agent(ctx):
        sink.extend((message.payload, dict(message.metadata)) for message in ctx.inbox)
        ctx.wait()

    return agent


def test_adapter_holds_one_conversation_with_a_remote_agent(agent_server):
    replies: list = []
    kernel = Kernel()
    kernel.register(create_platform("a2a", agent_server.base, name="remote").agent("remote"))
    script = iter(["hello", "ask", "blue", "hello"])

    def boss(ctx):
        if ctx.tick == 1:
            ctx.send("remote", next(script))
        for message in ctx.inbox:
            replies.append((message.payload, dict(message.metadata)))
            following = next(script, None)
            if following is not None:
                ctx.send("remote", following)
        ctx.wait()

    kernel.spawn("boss", boss)
    kernel.run(max_ticks=20)
    kernel.shutdown()

    assert [payload for payload, _ in replies] == ["hi there", "which colour?", "painted blue", "hi there"]
    assert replies[0][1] == {"a2a_context_id": "ctx-1"}
    assert replies[1][1] == {"a2a_state": "input_required", "a2a_task_id": "t-ask", "a2a_context_id": "ctx-1"}
    sent = agent_server.sent_messages()
    assert [m.get("contextId") for m in sent] == [None, "ctx-1", "ctx-1", "ctx-1"]
    # The task is only continued while it waits for input.
    assert [m.get("taskId") for m in sent] == [None, None, "t-ask", None]


def test_adapter_sends_mappings_as_data_and_returns_rich_replies_whole(agent_server):
    replies: list = []
    kernel = Kernel()
    kernel.register(A2AAdapter(agent_server.base).agent("remote", reply_to="sink"))
    kernel.spawn("sink", collector(replies))
    kernel.spawn("boss", lambda ctx: (ctx.send("remote", {"q": 1}), ctx.exit()))
    kernel.run(max_ticks=6)
    kernel.shutdown()

    [(payload, _)] = replies
    assert (payload["kind"], payload["parts"]) == ("message", [{"data": {"q": 1}}])


def test_adapter_raises_a_platform_error_for_a_failed_task(agent_server):
    kernel = Kernel()
    kernel.register(A2AAdapter(agent_server.base).agent("remote"))
    kernel.spawn("boss", lambda ctx: (ctx.send("remote", "break"), ctx.wait()))
    kernel.run(max_ticks=5)
    kernel.shutdown()

    [(name, error)] = kernel.errors
    assert name == "remote" and isinstance(error, PlatformError)
    assert "ended task t-break as failed: kaput" in str(error)


def test_adapter_conversation_survives_a_snapshot(agent_server):
    kernel = Kernel()
    kernel.register(A2AAdapter(agent_server.base).agent("remote", reply_to="sink"))
    kernel.spawn("sink", collector([]))
    kernel.spawn("boss", lambda ctx: (ctx.send("remote", "ask"), ctx.exit()))
    kernel.run(max_ticks=4)
    snapshot = kernel.snapshot()
    kernel.shutdown()

    record = next(r for r in snapshot.agents if r.name == "remote")
    assert record.data["platform"] == {"context_id": "ctx-1", "task_id": "t-ask"}
    restored = A2AAdapter(agent_server.base)
    Kernel.restore(
        snapshot,
        {
            "PlatformAgent": lambda name: restored.agent(name, reply_to="sink"),
            "FunctionAgent": lambda name: FunctionAgent(name, collector([])),
        },
    )
    assert (restored.context_id, restored.task_id) == ("ctx-1", "t-ask")
