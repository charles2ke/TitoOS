"""Tests for the MCP client: stdio and Streamable HTTP, modern and legacy servers."""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from titoos import IntegrationError, JsonRpcError, Kernel, McpIntegration
from titoos.integrations.mcp import _header_value

FAKE_SERVER = Path(__file__).with_name("fake_mcp_server.py")
MODERN = "2026-07-28"


# --- stdio ------------------------------------------------------------------


@pytest.fixture
def mcp_log(tmp_path):
    return tmp_path / "received.jsonl"


def stdio_server(mode: str, log: Path, **kwargs) -> McpIntegration:
    env = {"FAKE_MCP_LOG": str(log), **kwargs.pop("env", {})}
    return McpIntegration(command=[sys.executable, str(FAKE_SERVER), mode], env=env, **kwargs)


def received(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text().splitlines()]


def methods(log: Path) -> list[str]:
    return [message["method"] for message in received(log) if "method" in message]


def params_of(log: Path, method: str) -> list[dict]:
    return [m.get("params", {}) for m in received(log) if m.get("method") == method]


def test_stdio_modern_server_lists_and_calls_tools(mcp_log):
    mcp = stdio_server("modern", mcp_log)
    try:
        names = [tool["name"] for tool in mcp.list_tools()]
        result = mcp.call_tool("echo", {"text": "hi"})
        info = mcp.server_info()
    finally:
        mcp.close()

    assert names[:4] == ["echo", "fail", "retry", "elicit"] and "getenv" in names
    assert result.ok and result.text == "hi"
    assert (info["era"], info["protocol_version"]) == ("modern", MODERN)
    assert info["server"] == {"name": "fake-modern", "version": "1.0"}
    assert info["instructions"] == "Be brief."
    assert "initialize" not in methods(mcp_log)
    assert [p.get("cursor") for p in params_of(mcp_log, "tools/list")] == [None, "page-2"]
    meta = params_of(mcp_log, "tools/call")[0]["_meta"]
    assert meta["io.modelcontextprotocol/protocolVersion"] == MODERN
    assert meta["io.modelcontextprotocol/clientInfo"]["name"] == "titoos"


def test_stdio_retries_input_required_with_the_request_state(mcp_log):
    mcp = stdio_server("modern", mcp_log)
    try:
        assert mcp.call_tool("retry").text == "retried"
        with pytest.raises(IntegrationError, match=r"needs input titoos cannot provide \(confirm\)"):
            mcp.call_tool("elicit")
    finally:
        mcp.close()
    retries = [p for p in params_of(mcp_log, "tools/call") if p["name"] == "retry"]
    assert [p.get("requestState") for p in retries] == [None, "state-1"]


def test_stdio_falls_back_to_initialize_for_a_legacy_server(mcp_log):
    mcp = stdio_server("legacy", mcp_log)
    try:
        assert mcp.call_tool("echo", {"text": "old"}).text == "old"
        info = mcp.server_info()
    finally:
        mcp.close()

    assert (info["era"], info["protocol_version"]) == ("legacy", "2025-11-25")
    assert info["server"]["name"] == "fake-legacy"
    assert methods(mcp_log)[:3] == ["server/discover", "initialize", "notifications/initialized"]
    assert params_of(mcp_log, "initialize")[0]["capabilities"] == {}
    assert "_meta" not in params_of(mcp_log, "tools/call")[0]


def test_stdio_treats_an_unanswered_probe_as_a_legacy_server(mcp_log):
    mcp = stdio_server("silent", mcp_log, timeout=1.0)
    try:
        assert mcp.call_tool("echo", {"text": "quiet"}).text == "quiet"
    finally:
        mcp.close()
    # The probe is abandoned, not cancelled: initialize must come first.
    assert methods(mcp_log)[:3] == ["server/discover", "initialize", "notifications/initialized"]


def test_stdio_pinned_legacy_version_skips_the_probe(mcp_log):
    mcp = stdio_server("legacy", mcp_log, protocol_version="2025-06-18")
    try:
        mcp.list_tools()
        assert mcp.server_info()["protocol_version"] == "2025-06-18"
    finally:
        mcp.close()
    assert methods(mcp_log)[0] == "initialize"


def test_stdio_answers_server_pings_and_ignores_stray_output(mcp_log):
    mcp = stdio_server("legacy", mcp_log)
    try:
        assert mcp.call_tool("ping_me").text == "pong"
        assert mcp.call_tool("noisy").text == "still fine"
    finally:
        mcp.close()


def test_stdio_reports_a_crash_with_its_stderr_and_then_restarts(mcp_log):
    mcp = stdio_server("modern", mcp_log)
    try:
        with pytest.raises(IntegrationError, match="exited with code 3") as caught:
            mcp.call_tool("die")
        assert "fatal: asked to die" in str(caught.value)
        assert mcp.call_tool("echo", {"text": "back"}).text == "back"
    finally:
        mcp.close()
    assert sum("started" in message for message in received(mcp_log)) == 2


def test_stdio_drops_a_server_whose_message_exceeds_max_bytes(mcp_log):
    mcp = stdio_server("modern", mcp_log, max_bytes=4096)
    try:
        with pytest.raises(IntegrationError, match="larger than max_bytes"):
            mcp.call_tool("huge")
    finally:
        mcp.close()


def test_stdio_server_inherits_only_a_minimal_environment(mcp_log, monkeypatch):
    monkeypatch.setenv("TITOOS_TEST_SECRET", "hunter2")
    mcp = stdio_server("modern", mcp_log, env={"GIVEN": "yes"})
    try:
        assert mcp.call_tool("getenv", {"name": "TITOOS_TEST_SECRET"}).text == "<unset>"
        assert mcp.call_tool("getenv", {"name": "GIVEN"}).text == "yes"
    finally:
        mcp.close()


def test_allowed_tools_narrow_the_grant(mcp_log):
    mcp = stdio_server("modern", mcp_log, allowed_tools=["echo"])
    try:
        assert [tool["name"] for tool in mcp.list_tools()] == ["echo"]
        with pytest.raises(IntegrationError, match="not in allowed_tools"):
            mcp.call_tool("fail")
    finally:
        mcp.close()
    assert params_of(mcp_log, "tools/call") == []


def test_a_tool_error_is_returned_unless_check_is_set(mcp_log):
    mcp = stdio_server("modern", mcp_log)
    try:
        result = mcp.call_tool("fail")
        assert result.is_error and not result.ok and result.text == "it broke"
        with pytest.raises(IntegrationError, match="it broke"):
            mcp.call_tool("fail", check=True)
    finally:
        mcp.close()


def test_resources_and_prompts(mcp_log):
    mcp = stdio_server("modern", mcp_log)
    try:
        assert mcp.list_resources() == [{"uri": "mem://notes", "name": "notes"}]
        assert mcp.read_resource("mem://notes")[0]["text"] == "remember the milk"
        prompt = mcp.get_prompt("greet", {"who": "Ada"})
    finally:
        mcp.close()
    assert prompt["messages"][0]["content"]["text"] == "Hello, Ada"
    assert "resultType" not in prompt


def test_agents_call_mcp_tools_through_the_kernel(mcp_log):
    seen: list = []
    kernel = Kernel()
    kernel.install(stdio_server("modern", mcp_log))

    def worker(ctx):
        seen.append(ctx.call("mcp", "call_tool", "echo", {"text": "via the kernel"}).text)
        ctx.exit()

    kernel.spawn("worker", worker)
    kernel.run(max_ticks=3)
    kernel.shutdown()

    assert seen == ["via the kernel"]
    assert "exited" in received(mcp_log)[-1]  # shutdown closed the server's stdin


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({}, "exactly one"),
        ({"command": "python server.py"}, "not a string"),
        ({"command": ["titoos-no-such-program"]}, "not found on PATH"),
        ({"url": "file:///etc/passwd"}, "http or https"),
        ({"url": "http://example.com/mcp", "protocol_version": "1999-01-01"}, "unsupported"),
    ],
)
def test_construction_is_validated(kwargs, match):
    with pytest.raises(ValueError, match=match):
        McpIntegration(**kwargs)


def test_describe_hides_arguments_and_credentials():
    stdio = McpIntegration(command=[sys.executable, "-c", "pass", "--token=s3cret"])
    http = McpIntegration(
        url="https://user:pw@example.com/mcp?key=s3cret",
        headers={"Authorization": "Bearer s3cret"},
    )
    assert stdio.describe()["command"] == os.path.basename(sys.executable)
    assert http.describe()["url"] == "https://example.com/mcp"
    assert http.describe()["headers"] == {"Authorization": "***"}
    assert "s3cret" not in repr(stdio.describe()) + repr(http.describe())


@pytest.mark.parametrize(
    "value, expected",
    [  # the examples in the Streamable HTTP specification
        ("Hello, 世界", "=?base64?SGVsbG8sIOS4lueVjA==?="),
        (" padded ", "=?base64?IHBhZGRlZCA=?="),
        ("line1\nline2", "=?base64?bGluZTEKbGluZTI=?="),
        ("=?base64?literal?=", "=?base64?PT9iYXNlNjQ/bGl0ZXJhbD89?="),
        ("us-west-2", "us-west-2"),
    ],
)
def test_header_values_are_encoded_as_the_specification_shows(value, expected):
    assert _header_value(value) == expected


# --- Streamable HTTP --------------------------------------------------------


def _text(value: str) -> dict:
    return {"content": [{"type": "text", "text": value}]}


def _error(request_id, code: int, message: str, data=None) -> dict:
    error = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


class McpHttpServer:
    """What the fake HTTP server does, and what it has seen.

    It is modern while ``modern_versions`` is non-empty, and answers the
    legacy ``initialize`` handshake while ``legacy_versions`` is.
    """

    def __init__(self) -> None:
        self.url = ""
        self.modern_versions = [MODERN]
        self.legacy_versions = ["2025-11-25"]
        self.sse = False
        self.token: str | None = None
        self.redirect: str | None = None
        #: Refuse every POST, like a server that only speaks 2024-11-05's HTTP+SSE.
        self.sse_transport_only = False
        self.header_name = "Region"
        self.sessions: set[str] = set()
        self.minted = 0
        #: Legacy streams that can be resumed: Last-Event-ID -> request id.
        self.resumable: dict[str, object] = {}
        #: (verb, lower-cased headers, JSON body or None), in arrival order.
        self.seen: list[tuple[str, dict, dict | None]] = []

    def posts(self, method: str) -> list[tuple[dict, dict]]:
        return [(h, m) for verb, h, m in self.seen if verb == "POST" and m.get("method") == method]

    def verbs(self, verb: str) -> list[dict]:
        return [h for v, h, _ in self.seen if v == verb]

    def tools(self) -> list[dict]:
        return [
            {"name": "echo", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}},
            {
                "name": "lookup",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "x-mcp-header": self.header_name},
                        "count": {"type": "integer", "x-mcp-header": "Count"},
                        "flag": {"type": ["boolean", "null"], "x-mcp-header": "Flag"},
                    },
                },
            },
            # Only strings, integers and booleans may be mirrored into headers.
            {
                "name": "bad",
                "inputSchema": {
                    "type": "object",
                    "properties": {"filter": {"type": "object", "x-mcp-header": "Filter"}},
                },
            },
            {"name": "ping_me", "inputSchema": {"type": "object"}},
        ]


class _McpHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence the test output
        pass

    @property
    def fake(self) -> McpHttpServer:
        return self.server.fake  # type: ignore[attr-defined]

    def _headers(self) -> dict:
        return {key.lower(): value for key, value in self.headers.items()}

    def _send(self, status: int, body: bytes = b"", content_type: str | None = None, extra=()):
        self.send_response(status)
        if content_type:
            self.send_header("Content-Type", content_type)
        if content_type != "text/event-stream":
            self.send_header("Content-Length", str(len(body)))
        for key, value in extra:
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload, extra=()):
        self._send(status, json.dumps(payload).encode(), "application/json", extra)

    def _events(self, *events: str):
        self._send(200, "".join(events).encode(), "text/event-stream")

    def do_DELETE(self):  # noqa: N802 - http.server API
        headers = self._headers()
        self.fake.seen.append(("DELETE", headers, None))
        self.fake.sessions.discard(headers.get("mcp-session-id"))
        self._send(200)

    def do_GET(self):  # noqa: N802 - http.server API
        headers = self._headers()
        self.fake.seen.append(("GET", headers, None))
        request_id = self.fake.resumable.pop(headers.get("last-event-id"), None)
        if request_id is None:
            self._send(405)
            return
        answered = any(
            m is not None and m.get("id") == "server-ping" and m.get("result") == {}
            for _, _, m in self.fake.seen
        )
        response = {"jsonrpc": "2.0", "id": request_id, "result": _text("pong" if answered else "no pong")}
        self._events(f"id: e2\nevent: message\ndata: {json.dumps(response)}\n\n")

    def do_POST(self):  # noqa: N802 - http.server API
        headers = self._headers()
        message = json.loads(self.rfile.read(int(headers["content-length"])))
        fake = self.fake
        fake.seen.append(("POST", headers, message))
        if fake.redirect:
            self._send(307, extra=[("Location", fake.redirect)])
        elif fake.sse_transport_only:
            self._send(405)
        elif fake.token and headers.get("authorization") != fake.token:
            self._send(401, b"unauthorized", "text/plain")
        elif "method" not in message:  # the client answering one of our requests
            self._send(202)
        elif message["method"] == "initialize":
            self._initialize(message)
        elif "mcp-session-id" in headers or message["method"].startswith("notifications/"):
            self._legacy(message, headers)
        else:
            self._modern(message, headers)

    def _initialize(self, message):
        fake = self.fake
        if not fake.legacy_versions:
            self._json(400, _error(message["id"], -32601, f"use {', '.join(fake.modern_versions)}"))
            return
        asked = message["params"]["protocolVersion"]
        fake.minted += 1
        session = f"session-{fake.minted}"
        fake.sessions.add(session)
        result = {
            "protocolVersion": asked if asked in fake.legacy_versions else fake.legacy_versions[0],
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake-http-legacy", "version": "1.5"},
        }
        self._json(200, {"jsonrpc": "2.0", "id": message["id"], "result": result}, [("Mcp-Session-Id", session)])

    def _legacy(self, message, headers):
        fake = self.fake
        if headers.get("mcp-session-id") not in fake.sessions:
            self._json(404, _error(None, -32001, "Session not found"))
            return
        if "id" not in message:
            self._send(202)
            return
        params = message.get("params") or {}
        if message["method"] == "tools/call" and params["name"] == "ping_me":
            # Ask for a ping, then drop the stream: the answer comes on resumption.
            fake.resumable["e1"] = message["id"]
            ping = {"jsonrpc": "2.0", "id": "server-ping", "method": "ping"}
            self._events("id: e0\ndata:\n\n", "retry: 10\n\n", f"id: e1\ndata: {json.dumps(ping)}\n\n")
            return
        self._json(200, {"jsonrpc": "2.0", "id": message["id"], "result": self._result(message)})

    def _modern(self, message, headers):
        fake = self.fake
        request_id, method = message["id"], message["method"]
        params = message.get("params") or {}
        if not fake.modern_versions:  # a legacy-only server, before initialize
            self._json(400, _error(None, -32000, "Bad Request: Server not initialized"))
            return
        version = headers.get("mcp-protocol-version")
        if version not in fake.modern_versions:
            supported = fake.modern_versions + fake.legacy_versions
            self._json(400, _error(request_id, -32022, "Unsupported protocol version", {"supported": supported}))
            return
        expected = {"mcp-method": method}
        if method == "tools/call":
            expected["mcp-name"] = params["name"]
            if params["name"] == "lookup":
                expected[f"mcp-param-{fake.header_name.lower()}"] = params["arguments"]["region"]
        if any(key not in headers for key in expected):
            self._json(400, _error(request_id, -32020, "Header mismatch"))
            return
        if method == "server/discover":
            result = {
                "supportedVersions": fake.modern_versions + fake.legacy_versions,
                "capabilities": {"tools": {}},
                "_meta": {"io.modelcontextprotocol/serverInfo": {"name": "fake-http", "version": "2.0"}},
            }
        else:
            result = self._result(message)
        response = {"jsonrpc": "2.0", "id": request_id, "result": {"resultType": "complete", **result}}
        if fake.sse:
            progress = {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progress": 1}}
            self._events(
                ": keep-alive\n\n",
                f"event: message\ndata: {json.dumps(progress)}\n\n",
                f"event: message\ndata: {json.dumps(response)}\n\n",
            )
        else:
            self._json(200, response)

    def _result(self, message) -> dict:
        params = message.get("params") or {}
        if message["method"] == "tools/list":
            return {"tools": self.fake.tools()}
        arguments = params.get("arguments") or {}
        if params.get("name") == "echo":
            return _text(arguments["text"])
        if params.get("name") == "lookup":
            return _text(f"looked up {arguments['region']}")
        raise AssertionError(f"unexpected request {message}")


@pytest.fixture
def mcp_http():
    fake = McpHttpServer()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _McpHandler)
    server.fake = fake  # type: ignore[attr-defined]
    fake.url = f"http://127.0.0.1:{server.server_port}/mcp"
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield fake
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_http_modern_requests_carry_routing_and_parameter_headers(mcp_http):
    mcp = McpIntegration(url=mcp_http.url, headers={"Authorization": "Bearer t"})
    try:
        names = [tool["name"] for tool in mcp.list_tools()]
        result = mcp.call_tool("lookup", {"region": "Zürich", "count": 3.0, "flag": None})
        with pytest.raises(IntegrationError, match="'bad' was excluded"):
            mcp.call_tool("bad", {"filter": {}})
    finally:
        mcp.close()

    assert names == ["echo", "lookup", "ping_me"]  # "bad" mirrors an object
    assert result.text == "looked up Zürich"
    [(headers, _)] = mcp_http.posts("tools/call")
    assert headers["mcp-protocol-version"] == MODERN
    assert (headers["mcp-method"], headers["mcp-name"]) == ("tools/call", "lookup")
    assert headers["mcp-param-region"] == "=?base64?WsO8cmljaA==?="
    assert headers["mcp-param-count"] == "3"
    assert "mcp-param-flag" not in headers  # a null argument omits its header
    assert headers["authorization"] == "Bearer t"
    assert "mcp-session-id" not in headers
    assert mcp_http.verbs("DELETE") == []  # modern servers have no session to end


def test_http_reads_a_response_from_an_event_stream(mcp_http):
    mcp_http.sse = True
    mcp = McpIntegration(url=mcp_http.url)
    try:
        assert mcp.call_tool("echo", {"text": "streamed"}).text == "streamed"
    finally:
        mcp.close()


def test_http_refreshes_the_tools_after_a_header_mismatch(mcp_http):
    mcp = McpIntegration(url=mcp_http.url)
    try:
        mcp.list_tools()
        mcp_http.header_name = "Zone"  # the tool changed since it was listed
        assert mcp.call_tool("lookup", {"region": "eu"}).text == "looked up eu"
    finally:
        mcp.close()
    calls = [headers for headers, _ in mcp_http.posts("tools/call")]
    assert [("mcp-param-region" in h, "mcp-param-zone" in h) for h in calls] == [(True, False), (False, True)]
    assert len(mcp_http.posts("tools/list")) == 2


def test_http_renegotiates_when_the_server_drops_the_version(mcp_http):
    mcp = McpIntegration(url=mcp_http.url)
    try:
        mcp.list_tools()
        mcp_http.modern_versions = ["2027-01-01"]  # upgraded past us; still speaks 2025-11-25
        assert mcp.call_tool("echo", {"text": "again"}).text == "again"
        info = mcp.server_info()
    finally:
        mcp.close()
    assert (info["era"], info["protocol_version"]) == ("legacy", "2025-11-25")


def test_http_names_the_versions_a_server_offers_when_none_fit(mcp_http):
    mcp_http.modern_versions = ["2027-01-01"]
    mcp_http.legacy_versions = []
    mcp = McpIntegration(url=mcp_http.url)
    with pytest.raises(IntegrationError, match="supports protocol versions 2027-01-01"):
        mcp.list_tools()


def test_http_legacy_session_is_renewed_and_finally_ended(mcp_http):
    mcp_http.modern_versions = []
    mcp = McpIntegration(url=mcp_http.url)
    assert mcp.call_tool("echo", {"text": "one"}).text == "one"
    mcp_http.sessions.clear()  # the server restarted and forgot the session
    assert mcp.call_tool("echo", {"text": "two"}).text == "two"
    mcp.close()

    assert len(mcp_http.posts("initialize")) == 2
    headers, _ = mcp_http.posts("tools/call")[-1]
    assert (headers["mcp-session-id"], headers["mcp-protocol-version"]) == ("session-2", "2025-11-25")
    assert [h["mcp-session-id"] for h in mcp_http.verbs("DELETE")] == ["session-2"]


def test_http_legacy_answers_a_ping_on_the_stream_and_resumes_it(mcp_http):
    mcp_http.modern_versions = []
    mcp = McpIntegration(url=mcp_http.url)
    try:
        assert mcp.call_tool("ping_me").text == "pong"
    finally:
        mcp.close()
    assert [h.get("last-event-id") for h in mcp_http.verbs("GET")] == ["e1"]


def test_http_asks_for_credentials_on_401(mcp_http):
    mcp_http.token = "Bearer good"
    with pytest.raises(IntegrationError, match="HTTP 401.*headers="):
        McpIntegration(url=mcp_http.url).list_tools()
    mcp = McpIntegration(url=mcp_http.url, headers={"Authorization": "Bearer good"})
    try:
        assert mcp.list_tools()
    finally:
        mcp.close()


def test_http_never_follows_a_redirect(mcp_http):
    mcp_http.redirect = "http://127.0.0.1:9/elsewhere"
    with pytest.raises(IntegrationError, match="redirect to http://127.0.0.1:9/elsewhere not followed"):
        McpIntegration(url=mcp_http.url).list_tools()


def test_http_names_the_unsupported_http_sse_transport(mcp_http):
    mcp_http.sse_transport_only = True
    with pytest.raises(IntegrationError, match=r"rejected initialize with HTTP 405.*HTTP\+SSE transport is not supported"):
        McpIntegration(url=mcp_http.url).list_tools()
    assert [message["method"] for _, _, message in mcp_http.seen] == ["server/discover", "initialize"]


def test_a_refusal_keeps_the_json_rpc_error_code(mcp_http):
    mcp_http.modern_versions = []
    mcp_http.legacy_versions = []
    with pytest.raises(JsonRpcError) as caught:
        McpIntegration(url=mcp_http.url).list_tools()
    assert caught.value.code == -32601
