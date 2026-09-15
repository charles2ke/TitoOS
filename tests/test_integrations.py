"""Tests for the integration layer and the drivers shipped with it."""

from __future__ import annotations

import json
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from titoos import (
    Agent,
    ClockIntegration,
    FileSystemIntegration,
    HttpIntegration,
    Integration,
    IntegrationError,
    Kernel,
    ShellIntegration,
    StopReason,
)


def interpreter_name() -> str:
    """Base name of the running interpreter, e.g. ``python3``."""
    return Path(sys.executable).name


class Echo(Integration):
    """A tiny in-memory integration used to exercise the registry."""

    name = "echo"
    operations = ("say", "describe")

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.calls: list[str] = []
        self.closed = False

    def say(self, text: str) -> str:
        self.calls.append(text)
        return f"echo:{text}"

    def describe(self) -> dict[str, str]:
        return {"name": self.name}

    def hidden(self) -> str:  # not in operations
        return "nope"

    def close(self) -> None:
        self.closed = True


# --- registry and context wiring -------------------------------------------


def test_agent_calls_an_installed_integration():
    kernel = Kernel()
    echo = kernel.install(Echo())
    seen = []

    def caller(ctx):
        seen.append(ctx.call("echo", "say", "hello"))
        ctx.exit()

    kernel.spawn("caller", caller)
    kernel.run(max_ticks=3)

    assert seen == ["echo:hello"]
    assert echo.calls == ["hello"]
    assert kernel.stop_reason is StopReason.FINISHED


def test_context_integration_returns_the_object():
    kernel = Kernel()
    echo = kernel.install(Echo())
    found = []

    def caller(ctx):
        found.append(ctx.integration("echo"))
        ctx.exit()

    kernel.spawn("caller", caller)
    kernel.run(max_ticks=3)
    assert found == [echo]


def test_duplicate_names_are_rejected():
    kernel = Kernel()
    kernel.install(Echo())
    with pytest.raises(ValueError, match="already installed"):
        kernel.install(Echo())


def test_installing_a_non_integration_is_a_type_error():
    kernel = Kernel()
    with pytest.raises(TypeError):
        kernel.install(object())  # type: ignore[arg-type]


def test_unknown_integration_and_operation_raise():
    kernel = Kernel()
    kernel.install(Echo())
    with pytest.raises(IntegrationError, match="no integration installed"):
        kernel.integrations.call("nope", "say", "x")
    with pytest.raises(IntegrationError, match="unknown operation"):
        kernel.integrations.call("echo", "hidden")


def test_a_failing_integration_call_fails_only_that_agent():
    kernel = Kernel()
    kernel.install(Echo())

    def broken(ctx):
        ctx.call("echo", "missing")

    def fine(ctx):
        ctx.exit()

    kernel.spawn("broken", broken)
    kernel.spawn("fine", fine)
    kernel.run(max_ticks=3)

    assert [name for name, _ in kernel.errors] == ["broken"]
    assert isinstance(kernel.errors[0][1], IntegrationError)


def test_shutdown_closes_integrations():
    with Kernel() as kernel:
        echo = kernel.install(Echo())
    assert echo.closed


def test_uninstall_closes_and_removes():
    kernel = Kernel()
    echo = kernel.install(Echo())
    kernel.integrations.uninstall("echo")
    assert "echo" not in kernel.integrations
    assert echo.closed
    assert len(kernel.integrations) == 0


def test_registry_reports_names():
    kernel = Kernel()
    kernel.install(Echo())
    kernel.install(Echo("echo-2"))
    assert kernel.integrations.names() == ("echo", "echo-2")


def test_unnamed_integration_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        Integration()


def test_integrations_are_usable_from_worker_threads():
    with Kernel(max_workers=4) as kernel:
        echo = kernel.install(Echo())

        def caller(ctx):
            ctx.call("echo", "say", ctx.agent.name)
            ctx.exit()

        for i in range(4):
            kernel.spawn(f"caller-{i}", caller)
        kernel.run(max_ticks=3)

    assert sorted(echo.calls) == ["caller-0", "caller-1", "caller-2", "caller-3"]


# --- HTTP -------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - http.server API
        if self.path.startswith("/boom"):
            self.send_error(500, "boom")
            return
        if self.path.startswith("/elsewhere"):
            self.send_response(302)
            self.send_header("Location", "http://example.invalid/nope")
            self.end_headers()
            return
        body = json.dumps({"path": self.path}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", 0))
        payload = self.rfile.read(length)
        self.send_response(201)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # silence the test output
        pass


@pytest.fixture
def http_server():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_http_get_and_post(http_server):
    http = HttpIntegration(allowed_hosts=["127.0.0.1"])
    response = http.get(f"{http_server}/hello", params={"a": "1"})
    assert response.ok
    assert response.json() == {"path": "/hello?a=1"}

    created = http.post_json(f"{http_server}/things", {"x": 2})
    assert created.status == 201
    assert created.json() == {"x": 2}


def test_http_error_status_is_returned_not_raised(http_server):
    http = HttpIntegration(allowed_hosts=["127.0.0.1"])
    response = http.get(f"{http_server}/boom")
    assert response.status == 500
    assert not response.ok


def test_http_rejects_hosts_outside_the_allowlist():
    http = HttpIntegration(allowed_hosts=["api.example.com"])
    with pytest.raises(IntegrationError, match="not in the allowlist"):
        http.get("https://evil.example.org/data")


def test_http_subdomain_wildcard():
    http = HttpIntegration(allowed_hosts=[".example.com"])
    assert http._host_allowed("api.example.com")
    assert http._host_allowed("example.com")
    assert not http._host_allowed("example.com.evil.org")


def test_http_rejects_non_http_schemes():
    http = HttpIntegration(allowed_hosts=["127.0.0.1"])
    with pytest.raises(IntegrationError, match="scheme"):
        http.get("file:///etc/passwd")


def test_http_rejects_unknown_methods():
    http = HttpIntegration(allowed_hosts=["127.0.0.1"])
    with pytest.raises(IntegrationError, match="method"):
        http.request("TRACE", "http://127.0.0.1/x")


def test_http_redirect_outside_the_allowlist_is_blocked(http_server):
    http = HttpIntegration(allowed_hosts=["127.0.0.1"])
    with pytest.raises(IntegrationError):
        http.get(f"{http_server}/elsewhere")


def test_http_requires_an_allowlist():
    with pytest.raises(ValueError, match="allowed_hosts"):
        HttpIntegration(allowed_hosts=[])


def test_http_enforces_max_bytes(http_server):
    http = HttpIntegration(allowed_hosts=["127.0.0.1"], max_bytes=4)
    with pytest.raises(IntegrationError, match="max_bytes"):
        http.get(f"{http_server}/hello")


def test_http_describe_redacts_credentials():
    http = HttpIntegration(
        allowed_hosts=["api.example.com"], headers={"Authorization": "******"}
    )
    assert http.describe()["headers"] == {"Authorization": "***"}


def test_http_transport_failure_raises():
    http = HttpIntegration(allowed_hosts=["127.0.0.1"], timeout=0.5)
    with pytest.raises(IntegrationError, match="failed"):
        # Port 1 is reserved and nothing listens on it.
        http.get("http://127.0.0.1:1/nothing")


# --- filesystem -------------------------------------------------------------


def test_files_round_trip(tmp_path):
    files = FileSystemIntegration(tmp_path)
    files.write_text("notes/a.txt", "hello")
    files.append_text("notes/a.txt", " world")
    assert files.read_text("notes/a.txt") == "hello world"
    assert files.list_dir("notes") == ["notes/a.txt"]
    assert files.exists("notes/a.txt")
    assert files.delete("notes/a.txt") is True
    assert files.delete("notes/a.txt") is False
    assert not files.exists("notes/a.txt")


def test_files_reject_traversal(tmp_path):
    files = FileSystemIntegration(tmp_path)
    with pytest.raises(IntegrationError, match="escapes the sandbox"):
        files.read_text("../outside.txt")
    with pytest.raises(IntegrationError, match="relative"):
        files.read_text("/etc/passwd")


def test_files_reject_symlink_escape(tmp_path):
    outside = tmp_path.parent / "outside-secret.txt"
    outside.write_text("secret")
    root = tmp_path / "root"
    root.mkdir()
    (root / "link.txt").symlink_to(outside)
    files = FileSystemIntegration(root)
    with pytest.raises(IntegrationError, match="escapes the sandbox"):
        files.read_text("link.txt")


def test_files_read_only(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    files = FileSystemIntegration(tmp_path, read_only=True)
    assert files.read_text("a.txt") == "x"
    with pytest.raises(IntegrationError, match="read-only"):
        files.write_text("a.txt", "y")


def test_files_enforce_max_bytes(tmp_path):
    files = FileSystemIntegration(tmp_path, max_bytes=4)
    with pytest.raises(IntegrationError, match="max_bytes"):
        files.write_text("a.txt", "too long")
    (tmp_path / "b.txt").write_text("also too long")
    with pytest.raises(IntegrationError, match="max_bytes"):
        files.read_text("b.txt")


def test_files_missing_root(tmp_path):
    with pytest.raises(ValueError, match="not an existing directory"):
        FileSystemIntegration(tmp_path / "nope")


def test_files_read_missing_file(tmp_path):
    files = FileSystemIntegration(tmp_path)
    with pytest.raises(IntegrationError, match="cannot read"):
        files.read_text("missing.txt")


# --- shell ------------------------------------------------------------------


def test_shell_runs_an_allowed_command():
    shell = ShellIntegration(allowed_commands=[interpreter_name()])
    result = shell.run([sys.executable, "-c", "print('hi')"])
    assert result.ok
    assert result.stdout.strip() == "hi"


def test_shell_rejects_commands_outside_the_allowlist():
    shell = ShellIntegration(allowed_commands=["git"])
    with pytest.raises(IntegrationError, match="not in the allowlist"):
        shell.run([sys.executable, "-c", "print(1)"])


def test_shell_rejects_a_string_command():
    shell = ShellIntegration(allowed_commands=[interpreter_name()])
    with pytest.raises(IntegrationError, match="list of arguments"):
        shell.run("echo hi; rm -rf /")  # type: ignore[arg-type]


def test_shell_arguments_are_data_not_shell_syntax(tmp_path):
    shell = ShellIntegration(allowed_commands=[interpreter_name()])
    result = shell.run([sys.executable, "-c", "import sys; print(sys.argv[1])", "; ls"])
    assert result.stdout.strip() == "; ls"


def test_shell_reports_failure_and_can_check():
    shell = ShellIntegration(allowed_commands=[interpreter_name()])
    result = shell.run([sys.executable, "-c", "raise SystemExit(3)"])
    assert result.returncode == 3
    assert not result.ok
    with pytest.raises(IntegrationError, match="exit code 3"):
        shell.run([sys.executable, "-c", "raise SystemExit(3)"], check=True)


def test_shell_times_out():
    shell = ShellIntegration(allowed_commands=[interpreter_name()], timeout=0.2)
    with pytest.raises(IntegrationError, match="timed out"):
        shell.run([sys.executable, "-c", "import time; time.sleep(5)"])


def test_shell_requires_an_allowlist():
    with pytest.raises(ValueError, match="allowed_commands"):
        ShellIntegration(allowed_commands=[])


def test_shell_unknown_executable():
    shell = ShellIntegration(allowed_commands=["definitely-not-a-real-binary"])
    with pytest.raises(IntegrationError, match="not found"):
        shell.run(["definitely-not-a-real-binary"])


def test_shell_stdin_is_forwarded():
    shell = ShellIntegration(allowed_commands=[interpreter_name()])
    result = shell.run(
        [sys.executable, "-c", "import sys; print(sys.stdin.read().upper())"],
        stdin="abc",
    )
    assert result.stdout.strip() == "ABC"


# --- clock ------------------------------------------------------------------


def test_clock_now_is_timezone_aware():
    clock = ClockIntegration()
    assert clock.now().tzinfo is not None


def test_frozen_clock_is_deterministic_and_never_sleeps():
    fixed = datetime(2030, 1, 1, tzinfo=timezone.utc)
    clock = ClockIntegration(fixed=fixed)
    assert clock.now() == fixed == clock.now()
    assert clock.timestamp() == fixed.timestamp()
    assert clock.sleep(3600) == 5.0


def test_clock_sleep_is_capped():
    clock = ClockIntegration(max_sleep=0.01)
    assert clock.sleep(10) == 0.01
    with pytest.raises(IntegrationError, match="negative"):
        clock.sleep(-1)


# --- end-to-end -------------------------------------------------------------


class Reporter(Agent):
    """Fetches a URL and writes the result through the filesystem driver."""

    def __init__(self, name: str, url: str) -> None:
        super().__init__(name)
        self.url = url

    def step(self, ctx):
        response = ctx.call("http", "get", self.url)
        stamp = ctx.call("clock", "now").isoformat()
        ctx.call("files", "write_text", "report.json",
                 json.dumps({"status": response.status, "at": stamp}))
        ctx.send("archivist", "report.json")
        ctx.exit()


def test_pipeline_across_integrations(http_server, tmp_path):
    archived: list[dict] = []

    def archivist(ctx):
        if not ctx.inbox:
            ctx.wait()
            return
        for message in ctx.inbox:
            archived.append(json.loads(ctx.call("files", "read_text", message.payload)))
        ctx.exit()

    with Kernel() as kernel:
        kernel.install(HttpIntegration(allowed_hosts=["127.0.0.1"]))
        kernel.install(FileSystemIntegration(tmp_path))
        kernel.install(ClockIntegration(fixed=datetime(2030, 1, 1, tzinfo=timezone.utc)))
        kernel.register(Reporter("reporter", f"{http_server}/report"))
        kernel.spawn("archivist", archivist)
        kernel.run(max_ticks=10)

    assert archived == [{"status": 200, "at": "2030-01-01T00:00:00+00:00"}]
    assert kernel.stop_reason is StopReason.FINISHED


def test_a_fake_integration_swaps_in_for_a_real_one(tmp_path):
    """The same agent runs unchanged against a stub HTTP driver."""

    class FakeHttp(Integration):
        name = "http"
        operations = ("get",)

        def get(self, url, **kwargs):
            from titoos import HttpResponse

            return HttpResponse(status=200, url=url, body="{}")

    with Kernel() as kernel:
        kernel.install(FakeHttp())
        kernel.install(FileSystemIntegration(tmp_path))
        kernel.install(ClockIntegration(fixed=datetime(2030, 1, 1, tzinfo=timezone.utc)))
        kernel.register(Reporter("reporter", "https://anything.example/x"))
        kernel.spawn("archivist", lambda ctx: ctx.exit())
        kernel.run(max_ticks=5)

    assert json.loads((tmp_path / "report.json").read_text())["status"] == 200
