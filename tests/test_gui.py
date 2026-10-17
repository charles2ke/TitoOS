import json
import urllib.error
import urllib.request

import pytest

from titoos import Agent, Kernel
from titoos.gui import GUIError, KernelGUI
from titoos.gui import server as gui_server
from titoos.gui.__main__ import demo_kernel


class Echo(Agent):
    def __init__(self, name):
        super().__init__(name)
        self.seen = []

    def step(self, ctx):
        self.seen.extend(m.payload for m in ctx.inbox)
        ctx.wait()


@pytest.fixture
def gui():
    kernel = Kernel()
    kernel.register(Echo("echo"))
    with KernelGUI(kernel).start() as gui:
        yield gui


def request(gui, method, path, body=None, headers=None):
    data = None
    all_headers = dict(headers or {})
    if body is not None:
        data = json.dumps(body).encode()
        all_headers.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(gui.url + path.lstrip("/"), data=data, method=method, headers=all_headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as err:
        return err.code, err.headers, err.read()


def test_serves_dashboard_page(gui):
    status, headers, body = request(gui, "GET", "/")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert b"TitoOS" in body
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]


def test_state_lists_agents_and_tick(gui):
    status, _, body = request(gui, "GET", "/api/state")
    state = json.loads(body)
    assert status == 200
    assert state["tick"] == 0
    assert [a["name"] for a in state["agents"]] == ["echo"]
    assert state["agents"][0]["state"] == "ready"


def test_step_send_and_run_drive_the_kernel(gui):
    status, _, body = request(gui, "POST", "/api/step", {})
    assert status == 200
    assert json.loads(body)["ran"] == ["echo"]

    status, _, body = request(gui, "POST", "/api/send", {"to": "echo", "payload": {"job": 1}})
    assert status == 200
    assert json.loads(body)["sender"] == "gui"
    state = gui.state()
    assert state["mailboxes"]["echo"]["messages"][0]["payload"] == {"job": 1}
    assert state["agents"][0]["pending"] == 1

    status, _, body = request(gui, "POST", "/api/run", {"max_ticks": 10})
    result = json.loads(body)
    assert result["stop_reason"] == "quiescent"
    assert gui.kernel.get("echo").seen == [{"job": 1}]
    assert [r["tick"] for r in gui.state()["reports"]] == [2, 1]


def test_send_to_unknown_agent_is_a_bad_request(gui):
    status, _, body = request(gui, "POST", "/api/send", {"to": "nobody", "payload": 1})
    assert status == 400
    assert "unknown recipient" in json.loads(body)["error"]


def test_run_rejects_bad_tick_counts(gui):
    for value in (-1, "5", True, 10**9):
        status, _, _ = request(gui, "POST", "/api/run", {"max_ticks": value})
        assert status == 400


def test_post_requires_json_content_type(gui):
    status, _, _ = request(gui, "POST", "/api/step", headers={"Content-Type": "text/plain"}, body={})
    assert status == 415
    assert gui.kernel.tick == 0


def test_cross_origin_post_is_refused(gui):
    status, _, _ = request(gui, "POST", "/api/step", {}, headers={"Origin": "http://evil.example"})
    assert status == 403
    assert gui.kernel.tick == 0


def test_same_origin_post_is_accepted(gui):
    origin = f"http://127.0.0.1:{gui.port}"
    status, _, _ = request(gui, "POST", "/api/step", {}, headers={"Origin": origin})
    assert status == 200


def test_unknown_host_header_is_refused(gui):
    status, _, _ = request(gui, "GET", "/api/state", headers={"Host": f"attacker.example:{gui.port}"})
    assert status == 403


def test_non_json_payloads_are_shown_by_repr():
    kernel = Kernel()
    kernel.register(Echo("echo"))
    kernel.bus.post("someone", "echo", object())
    with KernelGUI(kernel) as gui:
        payload = gui.state()["mailboxes"]["echo"]["messages"][0]["payload"]
    assert payload.startswith("<object object")


def test_state_message_values_are_detached():
    kernel = Kernel()
    kernel.register(Echo("echo"))
    payload = {"nested": [1]}
    metadata = {"details": {"nested": [2]}}
    kernel.bus.post("someone", "echo", payload, **metadata)

    with KernelGUI(kernel) as gui:
        state = gui.state()
        message = state["mailboxes"]["echo"]["messages"][0]
        payload["nested"].append(3)
        metadata["details"]["nested"].append(4)

    assert message["payload"] == {"nested": [1]}
    assert message["metadata"] == {"details": {"nested": [2]}}


def test_send_builds_response_while_holding_lock(monkeypatch):
    kernel = Kernel()
    kernel.register(Echo("echo"))
    with KernelGUI(kernel) as gui:
        original_message = gui_server._message

        def checked_message(message):
            assert gui._lock.locked()
            return original_message(message)

        monkeypatch.setattr(gui_server, "_message", checked_message)
        response = gui.send("echo", {"nested": [1]})

    assert response["payload"] == {"nested": [1]}


def test_send_validates_arguments():
    with KernelGUI(Kernel()) as gui:
        with pytest.raises(GUIError):
            gui.send("")
        with pytest.raises(GUIError):
            gui.send("*", metadata=[1])


def test_snapshot_endpoint(gui):
    status, _, body = request(gui, "GET", "/api/snapshot")
    assert status == 200
    assert json.loads(body)["tick"] == 0


def test_demo_kernel_runs_to_quiescence_with_a_restart():
    with KernelGUI(demo_kernel()) as gui:
        result = gui.run(50)
        state = gui.state()
    assert result["stop_reason"] == "quiescent"
    restarts = {a["name"]: a["restarts"] for a in state["agents"]}
    assert restarts["worker-2"] == 1
    assert len(gui.kernel.get("collector").results) == 3
