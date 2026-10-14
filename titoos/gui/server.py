"""A browser dashboard for a running :class:`~titoos.kernel.Kernel`.

:class:`KernelGUI` wraps a kernel in a small HTTP server built on the standard
library only. It serves a single-page dashboard plus a JSON API, so the same
operations are available to the page, to ``curl`` and to tests:

========================  =====================================================
``GET  /``                The dashboard page.
``GET  /api/state``       Tick, stop reason, agents, mailboxes, errors, reports.
``GET  /api/snapshot``    :meth:`Kernel.snapshot` as JSON.
``POST /api/step``        Run one tick.
``POST /api/run``         Run up to ``{"max_ticks": N}`` ticks.
``POST /api/send``        Post ``{"to": ..., "payload": ...}`` onto the bus.
========================  =====================================================

The GUI drives the kernel itself: every tick it runs happens under one lock,
so state is only ever read at a tick boundary. Do not also call ``step()`` or
``run()`` from another thread while a GUI is attached.

The server is local by default. It binds to ``127.0.0.1``, answers only to the
host names it was bound under (which defeats DNS rebinding), and accepts
``POST`` only with a JSON body and, when the browser sends one, a same-origin
``Origin`` header — so a web page on another site cannot drive the kernel.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from types import TracebackType
from typing import TYPE_CHECKING, Any, Iterable

from ..message import Message

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..kernel import Kernel, TickReport

_logger = logging.getLogger(__name__)

#: Largest request body the API accepts. Every request is a few fields.
MAX_BODY = 64 * 1024
#: Upper bound on ``POST /api/run``; a run holds the GUI lock until it ends.
MAX_RUN_TICKS = 10_000
#: Messages listed per mailbox in ``/api/state``; the count is always exact.
MAX_MESSAGES_SHOWN = 50

_LOOPBACK_NAMES = ("127.0.0.1", "localhost", "[::1]")


def _jsonable(value: Any) -> Any:
    """``value`` if it survives JSON encoding, otherwise its ``repr``."""
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)
    return value


def _message(message: Message) -> dict[str, Any]:
    return {
        "sender": message.sender,
        "to": message.to,
        "payload": _jsonable(message.payload),
        "metadata": {key: _jsonable(v) for key, v in message.metadata.items()},
    }


def _report(report: "TickReport") -> dict[str, Any]:
    return {
        "tick": report.tick,
        "ran": list(report.ran),
        "failed": list(report.failed),
        "waiting": list(report.waiting),
        "restarted": list(report.restarted),
    }


class GUIError(ValueError):
    """A request the GUI refuses; reported to the client as ``400``."""


class KernelGUI:
    """Serve a dashboard for ``kernel`` over HTTP.

    ``port=0`` picks a free port; read the result from :attr:`url`. Use
    :meth:`start` to serve from a background thread, or :meth:`serve_forever`
    to block. The object is also a context manager that stops the server::

        with KernelGUI(kernel).start() as gui:
            print(gui.url)

    ``sender`` is the name messages posted from the GUI carry. ``history``
    bounds how many tick reports are kept. ``allowed_hosts`` adds host names
    (without port) the server answers to, for a non-loopback ``host``.
    """

    def __init__(
        self,
        kernel: "Kernel",
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        sender: str = "gui",
        history: int = 50,
        allowed_hosts: Iterable[str] = (),
    ) -> None:
        if not sender:
            raise ValueError("sender must be a non-empty string")
        if history < 1:
            raise ValueError("history must be at least 1")
        self.kernel = kernel
        self.sender = sender
        self.reports: deque[TickReport] = deque(maxlen=history)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._server = ThreadingHTTPServer((host, port), _make_handler(self))
        self._server.daemon_threads = True
        self.host = host
        self.port: int = self._server.server_address[1]
        names = {host, *allowed_hosts}
        if host in ("127.0.0.1", "localhost"):
            names.update(_LOOPBACK_NAMES)
        self._allowed_hosts = frozenset(
            f"{_bracket(name)}:{self.port}".lower() for name in names
        )

    # -- operations ---------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """Everything the dashboard shows, as JSON-friendly data."""
        with self._lock:
            kernel = self.kernel
            mailboxes = kernel.bus.dump()
            agents = [
                {
                    "name": agent.name,
                    "kind": type(agent).__name__,
                    "state": agent.state.value,
                    "alive": agent.is_alive,
                    "parent": agent.parent,
                    "restarts": agent.restarts,
                    "pending": len(mailboxes.get(agent.name, ())),
                }
                for agent in kernel.agents
            ]
            return {
                "tick": kernel.tick,
                "stop_reason": kernel.stop_reason.value if kernel.stop_reason else None,
                "quiescent": kernel.is_quiescent(),
                "backend": type(kernel.backend).__name__,
                "sender": self.sender,
                "agents": agents,
                "mailboxes": {
                    name: {
                        "count": len(messages),
                        "messages": [_message(m) for m in messages[:MAX_MESSAGES_SHOWN]],
                    }
                    for name, messages in mailboxes.items()
                },
                "errors": [
                    {"agent": name, "type": type(exc).__name__, "error": str(exc)}
                    for name, exc in kernel.errors
                ],
                "integrations": list(kernel.integrations.names()),
                "reports": [_report(r) for r in reversed(self.reports)],
            }

    def step(self) -> dict[str, Any]:
        """Run one tick and return its report."""
        with self._lock:
            report = self.kernel.step()
            self.reports.append(report)
        return _report(report)

    def run(self, max_ticks: int = 100) -> dict[str, Any]:
        """Run like :meth:`Kernel.run` and return the reports and stop reason."""
        if isinstance(max_ticks, bool) or not isinstance(max_ticks, int):
            raise GUIError("max_ticks must be an integer")
        if not 0 <= max_ticks <= MAX_RUN_TICKS:
            raise GUIError(f"max_ticks must be between 0 and {MAX_RUN_TICKS}")
        with self._lock:
            reports = self.kernel.run(max_ticks=max_ticks)
            self.reports.extend(reports)
            reason = self.kernel.stop_reason
        return {
            "reports": [_report(r) for r in reports],
            "stop_reason": reason.value if reason else None,
        }

    def send(
        self,
        to: str,
        payload: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Post a message from :attr:`sender`; ``to="*"`` broadcasts it."""
        if not isinstance(to, str) or not to:
            raise GUIError("'to' must be a non-empty string")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise GUIError("'metadata' must be an object")
        message = Message(sender=self.sender, to=to, payload=payload, metadata=metadata)
        with self._lock:
            try:
                self.kernel.bus.send(message)
            except KeyError as exc:
                raise GUIError(exc.args[0]) from None
        return _message(message)

    def snapshot(self) -> str:
        """The kernel's snapshot as JSON text."""
        with self._lock:
            return self.kernel.snapshot().to_json()

    # -- server lifecycle ---------------------------------------------------

    @property
    def url(self) -> str:
        return f"http://{_bracket(self.host)}:{self.port}/"

    def start(self) -> "KernelGUI":
        """Serve from a daemon thread and return ``self``."""
        if self._thread is not None:
            raise RuntimeError("GUI already started")
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="titoos-gui", daemon=True
        )
        self._thread.start()
        return self

    def serve_forever(self) -> None:
        """Serve on the calling thread until :meth:`stop` is called elsewhere."""
        self._server.serve_forever()

    def stop(self) -> None:
        """Stop serving and release the port. The kernel is left untouched."""
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join()
            self._thread = None
        self._server.server_close()

    def __enter__(self) -> "KernelGUI":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    def _host_allowed(self, host: str | None) -> bool:
        return host is not None and host.lower() in self._allowed_hosts


def _bracket(host: str) -> str:
    """``host`` as it appears before ``:port`` — IPv6 literals in brackets."""
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def _page() -> bytes:
    return resources.files(__package__).joinpath("static/index.html").read_bytes()


def _make_handler(gui: KernelGUI) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "TitoOS-GUI"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            _logger.debug("%s - " + format, self.address_string(), *args)

        # -- helpers --------------------------------------------------------

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; script-src 'unsafe-inline'; "
                "style-src 'unsafe-inline'; connect-src 'self'; "
                "base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: HTTPStatus, data: Any) -> None:
            body = json.dumps(data, default=repr).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def _error(self, status: HTTPStatus, message: str) -> None:
            self._json(status, {"error": message})

        def _guard(self) -> bool:
            if not gui._host_allowed(self.headers.get("Host")):
                self._error(HTTPStatus.FORBIDDEN, "host not allowed")
                return False
            return True

        def _body(self) -> dict[str, Any] | None:
            content_type = self.headers.get("Content-Type", "")
            if content_type.split(";")[0].strip().lower() != "application/json":
                self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "expected application/json")
                return None
            origin = self.headers.get("Origin")
            if origin is not None:
                scheme, _, rest = origin.partition("://")
                if scheme != "http" or not gui._host_allowed(rest):
                    self._error(HTTPStatus.FORBIDDEN, "cross-origin request refused")
                    return None
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if not 0 <= length <= MAX_BODY:
                self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "invalid body length")
                return None
            raw = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(raw)
            except ValueError:
                self._error(HTTPStatus.BAD_REQUEST, "body is not valid JSON")
                return None
            if not isinstance(data, dict):
                self._error(HTTPStatus.BAD_REQUEST, "body must be a JSON object")
                return None
            return data

        # -- routes ---------------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802
            if not self._guard():
                return
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, _page(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._json(HTTPStatus.OK, gui.state())
            elif path == "/api/snapshot":
                try:
                    text = gui.snapshot()
                except (RuntimeError, TypeError, ValueError) as exc:
                    self._error(HTTPStatus.CONFLICT, f"cannot snapshot: {exc}")
                    return
                self._send(HTTPStatus.OK, text.encode("utf-8"), "application/json; charset=utf-8")
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")

        def do_POST(self) -> None:  # noqa: N802
            if not self._guard():
                return
            path = self.path.split("?", 1)[0]
            if path not in ("/api/step", "/api/run", "/api/send"):
                self._error(HTTPStatus.NOT_FOUND, "not found")
                return
            data = self._body()
            if data is None:
                return
            try:
                if path == "/api/step":
                    result = gui.step()
                elif path == "/api/run":
                    result = gui.run(data.get("max_ticks", 100))
                else:
                    result = gui.send(
                        data.get("to", ""), data.get("payload"), data.get("metadata")
                    )
            except GUIError as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except Exception as exc:  # the kernel itself refused, e.g. TypeError
                _logger.exception("GUI request %s failed", path)
                self._error(
                    HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}"
                )
                return
            self._json(HTTPStatus.OK, result)

    return Handler
