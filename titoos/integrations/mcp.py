"""Model Context Protocol (MCP) client, over stdio or Streamable HTTP.

An MCP server is a program or an HTTP endpoint that offers tools, resources
and prompts. :class:`McpIntegration` connects to the one server the operator
names — a command to start, or a URL to reach — and lets agents list and use
what it offers through :meth:`~titoos.agent.Context.call`.

It speaks the 2026-07-28 revision of the protocol, where every request carries
its own version and capabilities, and falls back to the ``initialize``
handshake of the revisions most servers still run (2025-11-25 back to
2024-11-05). Which one a server speaks is detected on first use, following
the specification's backward-compatibility rules, unless the operator pins a
version with ``protocol_version=``.
"""

from __future__ import annotations

import base64
import copy
import itertools
import json
import logging
import os
import shutil
import signal
import subprocess
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .base import Integration
from .jsonrpc import (
    TRANSPORT_ERRORS,
    JsonRpcError,
    ResponseTooLarge,
    SseEvent,
    SseStream,
    check_headers,
    check_url,
    display_url,
    error_parts,
    find_response,
    http_open,
    is_timeout,
    read_body,
    redact_headers,
    snippet,
)

logger = logging.getLogger(__name__)

#: The revision this client prefers.
LATEST_PROTOCOL_VERSION = "2026-07-28"
#: Revisions that carry version and capabilities on every request.
MODERN_PROTOCOL_VERSIONS: tuple[str, ...] = ("2026-07-28",)
#: Revisions that open a session with an ``initialize`` handshake.
LEGACY_PROTOCOL_VERSIONS: tuple[str, ...] = (
    "2025-11-25",
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
)
#: Every revision this client speaks, in order of preference.
SUPPORTED_PROTOCOL_VERSIONS = MODERN_PROTOCOL_VERSIONS + LEGACY_PROTOCOL_VERSIONS

_HEADER_MISMATCH = -32020
_MISSING_CLIENT_CAPABILITY = -32021
_UNSUPPORTED_PROTOCOL_VERSION = -32022

#: Pages a list operation follows before deciding the server is looping.
_MAX_PAGES = 100
#: Retries of a request the server answered with ``input_required``.
_MAX_INPUT_ROUNDS = 8
#: Reconnections to a legacy event stream while waiting for one response.
_MAX_RESUMES = 5
#: How long a stdio server gets to exit after each shutdown step.
_EXIT_GRACE = 2.0
#: How long to wait for a reader thread once the server is gone.
_DRAIN_GRACE = 1.0
#: How long to give a best-effort stdio write after its request timed out.
_CANCEL_WRITE_TIMEOUT = 0.1
#: Bytes of a server's stderr kept for error messages.
_STDERR_TAIL = 4096
#: Bound on best-effort requests: answers to pings, cancellations, DELETE.
_COURTESY_TIMEOUT = 5.0
_CHUNK = 8192
_SAFE_INTEGER = 2**53 - 1
_ACCEPT = "application/json, text/event-stream"

#: Variables a stdio server inherits from the kernel's environment.
_INHERITED_ENV: tuple[str, ...] = (
    (
        "APPDATA",
        "HOMEDRIVE",
        "HOMEPATH",
        "LOCALAPPDATA",
        "PATH",
        "PATHEXT",
        "PROCESSOR_ARCHITECTURE",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "USERNAME",
        "USERPROFILE",
    )
    if os.name == "nt"
    else ("HOME", "LOGNAME", "PATH", "SHELL", "TERM", "USER")
)


def _default_env() -> dict[str, str]:
    """The part of the kernel's environment a stdio server inherits.

    The same short list the official MCP SDKs pass down: enough for launchers
    such as ``npx`` and ``uvx`` to find their runtimes and caches, and none of
    the rest of the kernel's environment, which is where tokens tend to live.
    Values starting with ``()`` are exported shell functions — code, not
    configuration — and are dropped.
    """
    env: dict[str, str] = {}
    for key in _INHERITED_ENV:
        value = os.environ.get(key)
        if value is not None and not value.startswith("()"):
            env[key] = value
    return env


def _client_info() -> dict[str, str]:
    from .. import __version__  # deferred: titoos/__init__ imports this module

    return {"name": "titoos", "version": __version__}


def _client_meta(version: str) -> dict[str, Any]:
    """The ``_meta`` every modern request carries in place of a handshake."""
    return {
        "io.modelcontextprotocol/protocolVersion": version,
        "io.modelcontextprotocol/clientInfo": _client_info(),
        # titoos offers no sampling, elicitation or roots.
        "io.modelcontextprotocol/clientCapabilities": {},
    }


# --- launching a stdio server ------------------------------------------------


def _resolve_program(program: str, env: Mapping[str, str], cwd: str | None) -> str:
    """Resolve ``argv[0]`` to an absolute path once, at construction.

    Symlinks are deliberately left alone: multi-call binaries decide what to
    be from the name they were started under.
    """
    if not program.strip():
        raise ValueError("command must start with a program name")
    if os.sep in program or (os.altsep and os.altsep in program):
        path = os.path.abspath(os.path.join(cwd or os.getcwd(), program))
        if not (os.path.isfile(path) and os.access(path, os.X_OK)):
            raise ValueError(f"MCP server program is not an executable file: {program!r}")
        return path
    found = shutil.which(program, path=env.get("PATH", os.defpath))
    if found is None:
        raise ValueError(f"MCP server program not found on PATH: {program!r}")
    return os.path.abspath(found)


def _stdio_launch(
    command: Sequence[str],
    env: Mapping[str, str] | None,
    cwd: "str | os.PathLike[str] | None",
) -> tuple[tuple[str, ...], dict[str, str], str | None]:
    if isinstance(command, (str, bytes)):
        raise ValueError(
            "command must be a list of arguments, not a string; there is no "
            "shell to split it"
        )
    argv = [os.fspath(part) if isinstance(part, os.PathLike) else part for part in command]
    if not argv or not all(isinstance(part, str) for part in argv):
        raise ValueError("command must be a non-empty list of strings")
    workdir = None
    if cwd is not None:
        workdir = os.path.abspath(os.path.expanduser(os.fspath(cwd)))
        if not os.path.isdir(workdir):
            raise ValueError(f"cwd is not an existing directory: {workdir}")
    extra = dict(env or {})
    for key, value in extra.items():
        if (
            not isinstance(key, str)
            or not isinstance(value, str)
            or not key
            or "=" in key
            or "\0" in key + value
        ):
            raise ValueError(f"invalid environment variable for the MCP server: {key!r}")
    child_env = {**_default_env(), **extra}
    argv[0] = _resolve_program(argv[0], child_env, workdir)
    return tuple(argv), child_env, workdir


# --- transports --------------------------------------------------------------


class _TransportError(Exception):
    """The exchange with the server broke before an answer arrived."""


class _TransportTimeout(_TransportError):
    """No answer arrived before the deadline."""


class _SessionExpired(Exception):
    """A legacy HTTP server no longer knows the session (it answered 404)."""


@dataclass(frozen=True)
class _Reply:
    """What a transport got back for one request."""

    status: int
    #: The JSON-RPC response, or ``None`` if the server sent none.
    message: dict[str, Any] | None
    detail: str = ""


def _answer_for(request: dict[str, Any]) -> dict[str, Any]:
    """The reply to a request a legacy server sends the client.

    titoos declares no client capabilities, so the only request a server may
    legitimately send is ``ping``; anything else is refused.
    """
    if request.get("method") == "ping":
        return {"jsonrpc": "2.0", "id": request.get("id"), "result": {}}
    return {
        "jsonrpc": "2.0",
        "id": request.get("id"),
        "error": {"code": -32601, "message": f"method not supported: {request.get('method')}"},
    }


class _Pending:
    __slots__ = ("event", "message", "failure")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.message: dict[str, Any] | None = None
        self.failure: str | None = None


class _StdioTransport:
    """A server process spoken to with newline-delimited JSON-RPC.

    A reader thread routes responses to waiting callers by request id, so
    agents running on a thread pool can share one server concurrently.
    """

    kind = "stdio"

    def __init__(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        cwd: str | None,
        max_bytes: int,
        label: str,
    ) -> None:
        self._label = label
        self._max_bytes = max_bytes
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._pending: dict[Any, _Pending] = {}
        self._stderr = bytearray()
        self._failure: str | None = None
        self._closed = False
        try:
            self._process = subprocess.Popen(  # noqa: S603 - argv[0] resolved at construction, shell=False
                list(argv),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=dict(env),
                cwd=cwd,
                shell=False,
                # Its own process group, so shutdown reaches the server's
                # children too (npx and uvx start the real server as one).
                start_new_session=os.name != "nt",
            )
        except OSError as exc:
            raise _TransportError(f"cannot start {argv[0]!r}: {exc}") from exc
        self._readers = [
            threading.Thread(target=self._read_stdout, name=f"mcp-{label}-stdout", daemon=True),
            threading.Thread(target=self._read_stderr, name=f"mcp-{label}-stderr", daemon=True),
        ]
        for reader in self._readers:
            reader.start()

    @property
    def alive(self) -> bool:
        return self._failure is None and not self._closed and self._process.poll() is None

    def begin_legacy(self, version: str) -> None:
        """Nothing to remember: stdio has no headers to version."""

    def exchange(
        self,
        message: dict[str, Any],
        *,
        headers: Mapping[str, str],
        deadline: float,
        cancel_on_timeout: bool = True,
    ) -> _Reply:
        request_id = message["id"]
        pending = _Pending()
        with self._lock:
            failure = self._failure
            if failure is None:
                self._pending[request_id] = pending
        if failure is not None:
            raise _TransportError(failure + self._stderr_hint())
        try:
            self._write(message, deadline=deadline)
            if not pending.event.wait(max(deadline - time.monotonic(), 0.0)):
                if cancel_on_timeout:
                    self._write_quietly(
                        {
                            "jsonrpc": "2.0",
                            "method": "notifications/cancelled",
                            "params": {"requestId": request_id, "reason": "timed out"},
                        },
                        deadline=time.monotonic() + _CANCEL_WRITE_TIMEOUT,
                    )
                raise _TransportTimeout("the server did not answer in time")
            if pending.failure is not None:
                raise _TransportError(pending.failure + self._stderr_hint())
            return _Reply(200, pending.message)
        finally:
            with self._lock:
                self._pending.pop(request_id, None)

    def notify(self, message: dict[str, Any], *, headers: Mapping[str, str], deadline: float) -> None:
        self._write(message, deadline=deadline)

    def _write(self, message: dict[str, Any], *, deadline: float) -> None:
        # json.dumps escapes newlines inside strings, so one message is one line.
        data = json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n"
        if not self._write_lock.acquire(timeout=max(deadline - time.monotonic(), 0.0)):
            self._signal(_SIGKILL)
            raise _TransportTimeout("the server did not read the request in time")
        try:
            with self._lock:
                if self._closed:
                    raise _TransportError("the server process is closed")
            completed = threading.Event()
            errors: list[BaseException] = []

            def write() -> None:
                try:
                    stdin = self._process.stdin
                    assert stdin is not None
                    stdin.write(data)
                    stdin.flush()
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    completed.set()

            threading.Thread(target=write, name=f"mcp-{self._label}-writer", daemon=True).start()
            if not completed.wait(max(deadline - time.monotonic(), 0.0)):
                self._signal(_SIGKILL)
                completed.wait(_DRAIN_GRACE)
                raise _TransportTimeout("the server did not read the request in time")
            if errors:
                exc = errors[0]
                if isinstance(exc, (OSError, ValueError)):
                    raise _TransportError(
                        f"cannot write to the server: {exc}{self._stderr_hint()}"
                    ) from exc
                raise exc
        finally:
            self._write_lock.release()

    def _write_quietly(self, message: dict[str, Any], *, deadline: float | None = None) -> None:
        try:
            self._write(
                message,
                deadline=deadline or time.monotonic() + _COURTESY_TIMEOUT,
            )
        except _TransportError as exc:
            logger.debug("MCP server %r: %s", self._label, exc)

    def _read_stdout(self) -> None:
        stream = self._process.stdout
        assert stream is not None
        reason = None
        try:
            while True:
                line = stream.readline(self._max_bytes + 1)
                if not line:
                    break
                if len(line) > self._max_bytes:
                    reason = f"the server sent a message larger than max_bytes ({self._max_bytes} bytes)"
                    self._signal(_SIGKILL)
                    break
                self._receive(line)
        except (OSError, ValueError):  # the pipe was closed under us
            pass
        finally:
            self._fail_pending(reason)

    def _read_stderr(self) -> None:
        stream = self._process.stderr
        assert stream is not None
        try:
            while True:
                chunk = stream.read1(_CHUNK)
                if not chunk:
                    break
                with self._lock:
                    self._stderr += chunk
                    del self._stderr[:-_STDERR_TAIL]
        except (OSError, ValueError):
            pass

    def _receive(self, line: bytes) -> None:
        text = line.strip()
        if not text:
            return
        try:
            payload = json.loads(text)
        except ValueError:
            # Servers are not supposed to print anything else on stdout, but
            # a stray log line should not take the connection down.
            logger.debug("MCP server %r wrote non-JSON output: %.200r", self._label, text)
            return
        for item in payload if isinstance(payload, list) else (payload,):
            if not isinstance(item, dict):
                continue
            if "method" in item:
                if "id" in item:
                    # Answered off this thread: blocking on a full stdin pipe
                    # here would stop the responses that would drain it.
                    threading.Thread(
                        target=self._write_quietly,
                        args=(_answer_for(item),),
                        kwargs={"deadline": time.monotonic() + _COURTESY_TIMEOUT},
                        daemon=True,
                    ).start()
                continue
            request_id = item.get("id")
            if isinstance(request_id, bool) or not isinstance(request_id, (int, str)):
                continue
            with self._lock:
                pending = self._pending.get(request_id)
            if pending is not None and ("result" in item or "error" in item):
                pending.message = item
                pending.event.set()

    def _fail_pending(self, reason: str | None) -> None:
        if reason is None:
            try:
                code = self._process.wait(timeout=_DRAIN_GRACE)
            except subprocess.TimeoutExpired:
                code = None
            reason = "the server closed its output" + (
                f" and exited with code {code}" if code is not None else ""
            )
            # Let the last words on stderr arrive, so the error can quote them.
            self._readers[1].join(timeout=_DRAIN_GRACE)
        with self._lock:
            if self._failure is None:
                self._failure = reason
            pending = list(self._pending.values())
        for waiter in pending:
            waiter.failure = reason
            waiter.event.set()

    def _stderr_hint(self) -> str:
        with self._lock:
            tail = bytes(self._stderr).decode("utf-8", errors="replace").strip()
        return f"; stderr: {tail[-1000:]}" if tail else ""

    def _signal(self, sig: Any) -> None:
        process = self._process
        if process.poll() is not None:
            return
        try:
            if os.name == "nt":
                process.kill() if sig is _SIGKILL else process.terminate()
            else:
                os.killpg(process.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        except OSError:
            process.kill()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        process = self._process
        # Close stdin gracefully only when no writer is blocked in the pipe.
        if self._write_lock.acquire(timeout=_CANCEL_WRITE_TIMEOUT):
            try:
                try:
                    assert process.stdin is not None
                    process.stdin.close()
                except (OSError, ValueError):
                    pass
            finally:
                self._write_lock.release()
        else:
            self._signal(_SIGTERM)
        for sig in (None, _SIGTERM, _SIGKILL):
            if sig is not None:
                self._signal(sig)
            try:
                process.wait(timeout=_EXIT_GRACE)
                break
            except subprocess.TimeoutExpired:
                continue
        if self._write_lock.acquire(timeout=_DRAIN_GRACE):
            try:
                try:
                    assert process.stdin is not None
                    process.stdin.close()
                except (OSError, ValueError):
                    pass
            finally:
                self._write_lock.release()
        for reader in self._readers:
            reader.join(timeout=_DRAIN_GRACE)
        for stream in (process.stdout, process.stderr):
            if stream is not None and not any(r.is_alive() for r in self._readers):
                stream.close()


_SIGTERM = getattr(signal, "SIGTERM", None)
_SIGKILL = getattr(signal, "SIGKILL", object())


class _HttpTransport:
    """A Streamable HTTP endpoint: one POST per request, JSON or SSE back.

    Also carries what the legacy revisions add on top: the session id minted
    by ``initialize``, answers to pings arriving on a response stream, and
    resuming a stream the server closed early with ``Last-Event-ID``.
    """

    kind = "http"
    alive = True

    def __init__(
        self, url: str, *, headers: Mapping[str, str], max_bytes: int, label: str
    ) -> None:
        self.url = url
        self._headers = dict(headers)
        self._max_bytes = max_bytes
        self._label = label
        self.session_id: str | None = None
        #: The negotiated legacy revision, once ``initialize`` succeeded.
        self.legacy_version: str | None = None

    def begin_legacy(self, version: str) -> None:
        self.legacy_version = version

    def exchange(
        self,
        message: dict[str, Any],
        *,
        headers: Mapping[str, str],
        deadline: float,
        cancel_on_timeout: bool = True,
    ) -> _Reply:
        response = self._open("POST", message, headers, deadline, accept=_ACCEPT)
        try:
            status = response.getcode()
            if status == 404 and self.session_id is not None:
                raise _SessionExpired()
            if message.get("method") == "initialize" and 200 <= status < 300:
                self._capture_session(response)
            return self._reply(response, message["id"], headers, deadline)
        except TRANSPORT_ERRORS as exc:
            if not is_timeout(exc):
                raise _TransportError(f"{display_url(self.url)}: {exc}") from exc
            if cancel_on_timeout and self.legacy_version is not None:
                # Modern servers take a closed stream as the cancellation;
                # legacy ones expect to be told.
                self._post_quietly(
                    {
                        "jsonrpc": "2.0",
                        "method": "notifications/cancelled",
                        "params": {"requestId": message["id"], "reason": "timed out"},
                    },
                    headers,
                )
            raise _TransportTimeout(f"{display_url(self.url)} did not answer in time") from exc
        finally:
            response.close()

    def notify(self, message: dict[str, Any], *, headers: Mapping[str, str], deadline: float) -> None:
        response = self._open("POST", message, headers, deadline, accept=_ACCEPT)
        try:
            status = response.getcode()
        finally:
            response.close()
        if not 200 <= status < 300:
            raise _TransportError(f"the server answered {message['method']} with HTTP {status}")

    def close(self) -> None:
        session, self.session_id = self.session_id, None
        if session is None:
            return
        headers = {**self._headers, "Mcp-Session-Id": session}
        if self.legacy_version is not None:
            headers["MCP-Protocol-Version"] = self.legacy_version
        try:
            http_open("DELETE", self.url, headers=headers, timeout=_COURTESY_TIMEOUT).close()
        except TRANSPORT_ERRORS as exc:  # the server may not support DELETE
            logger.debug("MCP server %r: ending the session failed: %s", self._label, exc)

    def _open(
        self,
        method: str,
        message: dict[str, Any] | None,
        headers: Mapping[str, str],
        deadline: float,
        *,
        accept: str,
    ) -> Any:
        merged = {**self._headers, "Accept": accept}
        body = None
        if message is not None:
            merged["Content-Type"] = "application/json"
            body = json.dumps(message).encode("utf-8")
        merged.update(headers)
        if self.session_id is not None:
            merged["Mcp-Session-Id"] = self.session_id
        try:
            return http_open(
                method,
                self.url,
                headers=merged,
                body=body,
                timeout=max(deadline - time.monotonic(), 0.001),
            )
        except TRANSPORT_ERRORS as exc:
            if is_timeout(exc):
                raise _TransportTimeout(f"{display_url(self.url)} did not answer in time") from exc
            raise _TransportError(f"{display_url(self.url)}: {exc}") from exc

    def _post_quietly(self, message: dict[str, Any], headers: Mapping[str, str]) -> None:
        try:
            self._open(
                "POST", message, headers, time.monotonic() + _COURTESY_TIMEOUT, accept=_ACCEPT
            ).close()
        except _TransportError as exc:
            logger.debug("MCP server %r: %s", self._label, exc)

    def _capture_session(self, response: Any) -> None:
        value = response.headers.get("Mcp-Session-Id")
        if value is None:
            return
        if not value or not all("!" <= ch <= "~" for ch in value):
            raise _TransportError("the server sent a malformed Mcp-Session-Id")
        self.session_id = value

    def _reply(
        self, response: Any, request_id: Any, headers: Mapping[str, str], deadline: float
    ) -> _Reply:
        status = response.getcode()
        if 300 <= status < 400:
            location = response.headers.get("Location")
            target = display_url(urllib.parse.urljoin(self.url, location)) if location else "elsewhere"
            return _Reply(
                status,
                None,
                f"redirect to {target} not followed; point url= at the final address",
            )
        ok = 200 <= status < 300
        if ok and response.headers.get_content_type() == "text/event-stream":
            return _Reply(status, self._read_stream(response, request_id, headers, deadline))
        try:
            body = read_body(response, self._max_bytes)
        except ResponseTooLarge:
            if ok:
                raise
            body = b""
        found = None
        if body.strip():
            try:
                found = find_response(json.loads(body), request_id)
            except ValueError:
                found = None
        if ok and found is None:
            raise _TransportError(
                f"HTTP {status} carried no JSON-RPC response"
                + (f": {snippet(body)}" if body.strip() else "")
            )
        return _Reply(status, found, snippet(body))

    def _read_stream(
        self, response: Any, request_id: Any, headers: Mapping[str, str], deadline: float
    ) -> dict[str, Any]:
        stream = SseStream(response, max_bytes=self._max_bytes, deadline=deadline)
        current = response
        resumes = 0
        try:
            while True:
                for event in stream:
                    found = self._on_event(event, request_id, headers)
                    if found is not None:
                        return found
                # Only legacy streams can be resumed, and only past an id.
                if self.legacy_version is None or stream.last_event_id is None:
                    raise _TransportError("the server closed the event stream before responding")
                resumes += 1
                if resumes > _MAX_RESUMES:
                    raise _TransportError(
                        f"the event stream closed {_MAX_RESUMES} times before the response arrived"
                    )
                delay = stream.retry if stream.retry is not None else 1.0
                if time.monotonic() + delay >= deadline:
                    raise _TransportTimeout("the event stream closed and the response was not resent in time")
                time.sleep(delay)
                if current is not response:
                    current.close()
                current = self._open(
                    "GET",
                    None,
                    {**headers, "Last-Event-ID": stream.last_event_id},
                    deadline,
                    accept="text/event-stream",
                )
                status = current.getcode()
                if not 200 <= status < 300 or current.headers.get_content_type() != "text/event-stream":
                    # Not a session-expired retry: the server may already
                    # have acted on the request, so it must not be resent.
                    raise _TransportError(f"resuming the event stream failed with HTTP {status}")
                stream = SseStream(
                    current,
                    max_bytes=stream.remaining,
                    deadline=deadline,
                    last_event_id=stream.last_event_id,
                )
        finally:
            if current is not response:
                current.close()

    def _on_event(
        self, event: SseEvent, request_id: Any, headers: Mapping[str, str]
    ) -> dict[str, Any] | None:
        if event.event != "message" or not event.data:
            return None  # e.g. the empty event a server sends to prime resumption
        try:
            payload = json.loads(event.data)
        except ValueError:
            logger.debug("MCP server %r sent a non-JSON event: %.200r", self._label, event.data)
            return None
        for item in payload if isinstance(payload, list) else (payload,):
            if not isinstance(item, dict):
                continue
            if "method" in item:
                # Modern servers must not send requests on a response stream;
                # legacy ones may, and expect the answer as a POST.
                if "id" in item and self.legacy_version is not None:
                    self._post_quietly(_answer_for(item), headers)
                continue
            if item.get("id") == request_id and ("result" in item or "error" in item):
                return item
        return None


# --- x-mcp-header ------------------------------------------------------------

_HEADER_TOKEN = frozenset("!#$%&'*+-.^_`|~0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
_HEADER_TYPES = frozenset({"string", "integer", "boolean"})
#: Keywords whose values are data, not subschemas, so never walked.
_DATA_KEYWORDS = frozenset({"const", "default", "enum", "examples", "description", "title", "$comment"})
#: Keywords whose values map names to subschemas.
_SCHEMA_MAPS = frozenset({"$defs", "definitions", "patternProperties", "dependentSchemas"})
_MISSING = object()


def _header_params(schema: Any) -> dict[str, tuple[str, ...]]:
    """Map each ``x-mcp-header`` in a tool's input schema to its property path.

    Raises :class:`ValueError` naming the first rule an annotation breaks, in
    which case the specification requires the tool to be dropped: the header
    must be a token, unique ignoring case, on a string, integer or boolean
    property, reachable from the root through ``properties`` keys alone.
    """
    found: dict[str, tuple[str, ...]] = {}
    seen: set[str] = set()

    def visit(node: Any, path: tuple[str, ...] | None) -> None:
        if isinstance(node, list):
            for item in node:
                visit(item, None)
            return
        if not isinstance(node, dict):
            return
        if "x-mcp-header" in node:
            header = node["x-mcp-header"]
            if not isinstance(header, str) or not header or not set(header) <= _HEADER_TOKEN:
                raise ValueError(f"x-mcp-header {header!r} is not a valid header name")
            if not path:  # () is the root itself; None is unreachable
                raise ValueError(
                    f"x-mcp-header {header!r} is not on a property reachable "
                    "through 'properties' alone"
                )
            if header.lower() in seen:
                raise ValueError(f"x-mcp-header {header!r} is used more than once")
            declared = node.get("type")
            types = (
                {declared}
                if isinstance(declared, str)
                else set(declared)
                if isinstance(declared, list) and all(isinstance(t, str) for t in declared)
                else set()
            )
            # "null" may join the union: a null argument just omits the header.
            if not types - {"null"} or not types - {"null"} <= _HEADER_TYPES:
                raise ValueError(
                    f"x-mcp-header {header!r} is on a parameter of type "
                    f"{declared!r}; only string, integer and boolean are allowed"
                )
            seen.add(header.lower())
            found[header] = path
        for key, value in node.items():
            if key in _DATA_KEYWORDS or key == "x-mcp-header":
                continue
            if key == "properties" and isinstance(value, dict):
                for name, sub in value.items():
                    visit(sub, None if path is None else (*path, name))
            elif key in _SCHEMA_MAPS and isinstance(value, dict):
                for sub in value.values():
                    visit(sub, None)
            else:
                visit(value, None)

    visit(schema, ())
    return found


def _header_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        value = int(value)  # servers compare integers numerically: 42.0 == 42
    if isinstance(value, int):
        if abs(value) > _SAFE_INTEGER:
            raise ValueError("integer is outside the range a header may carry (±2**53-1)")
        return str(value)
    if isinstance(value, str):
        return value
    raise ValueError(f"a {type(value).__name__} cannot be mirrored into a header")


def _header_value(value: str) -> str:
    """``value`` as a header value, Base64-wrapped when it is not plain ASCII.

    Wrapped as well: values with leading or trailing spaces, and values that
    already look like the wrapper, so the server can always decode
    unambiguously.
    """
    plain = (
        all(" " <= ch <= "~" for ch in value)
        and value == value.strip(" ")
        and not (value.startswith("=?base64?") and value.endswith("?="))
    )
    if plain:
        return value
    return "=?base64?" + base64.b64encode(value.encode("utf-8")).decode("ascii") + "?="


# --- the integration ---------------------------------------------------------


@dataclass(frozen=True)
class McpToolResult:
    """The outcome of an MCP tool call, as a plain serializable record.

    A tool that ran and failed is reported here with :attr:`is_error` set,
    not raised: like a non-zero exit status, it is an answer the agent
    usually wants to read. Pass ``check=True`` to raise instead.
    """

    tool: str
    #: Content blocks: ``{"type": "text", "text": ...}``, images, resources...
    content: tuple[dict[str, Any], ...] = ()
    structured_content: Any = None
    is_error: bool = False

    @property
    def ok(self) -> bool:
        return not self.is_error

    @property
    def text(self) -> str:
        """The text content blocks, joined by newlines."""
        return "\n".join(
            block["text"]
            for block in self.content
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "content": [copy.deepcopy(block) for block in self.content],
            "structured_content": copy.deepcopy(self.structured_content),
            "is_error": self.is_error,
        }


@dataclass(eq=False)
class _Session:
    """One negotiated conversation with the server."""

    transport: Any
    era: str  # "modern" or "legacy"
    version: str
    server_info: dict[str, Any]
    capabilities: dict[str, Any]
    instructions: str | None
    #: Tools by name, cached for their x-mcp-header annotations.
    tools: dict[str, dict[str, Any]] | None = None
    tool_headers: dict[str, dict[str, tuple[str, ...]]] = field(default_factory=dict)
    rejected_tools: dict[str, str] = field(default_factory=dict)


def _tool_allowlist(names: Iterable[str] | None) -> frozenset[str] | None:
    if names is None:
        return None
    if isinstance(names, (str, bytes)):
        raise ValueError("allowed_tools must be a list of tool names, not a string")
    allowed = set()
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"allowed_tools entries must be non-empty strings: {name!r}")
        allowed.add(name.strip())
    if not allowed:
        raise ValueError(
            "allowed_tools must list at least one tool; omit it to allow every "
            "tool the server offers"
        )
    return frozenset(allowed)


class McpIntegration(Integration):
    """Use the tools, resources and prompts of one MCP server.

    Exactly one of ``command`` (start a local server and speak stdio) or
    ``url`` (speak Streamable HTTP to a remote one) names the server. That
    server is the grant: agents reach it and nothing else through this
    integration, and ``allowed_tools`` narrows the grant to the tools listed
    — tool names are case-sensitive. Install one integration per server, each
    under its own ``name``.

    ``command`` is an argument vector, never a shell string. Its program is
    resolved to an absolute path here, against the ``PATH`` the server will
    see. The server inherits only a few variables from the kernel's
    environment (``HOME``, ``PATH``, ``USER`` and the like) plus ``env``.

    ``headers`` — typically ``Authorization`` — go with every HTTP request and
    are redacted from :meth:`describe`. Redirects are never followed, so they
    only ever reach ``url``'s server.

    The connection opens on first use and reopens after a stdio server exits.
    A request that was in flight when the server died raises rather than
    being retried: a tool call may already have had its effect. ``timeout``
    bounds each request; for an SSE response it is checked between events,
    so one read can overrun it by at most the same amount again.
    """

    name = "mcp"
    operations = (
        "list_tools",
        "call_tool",
        "list_resources",
        "list_resource_templates",
        "read_resource",
        "list_prompts",
        "get_prompt",
        "server_info",
        "describe",
    )

    def __init__(
        self,
        *,
        command: Sequence[str] | None = None,
        url: str | None = None,
        name: str | None = None,
        allowed_tools: Iterable[str] | None = None,
        env: Mapping[str, str] | None = None,
        cwd: "str | os.PathLike[str] | None" = None,
        headers: Mapping[str, str] | None = None,
        protocol_version: str | None = None,
        timeout: float = 30.0,
        max_bytes: int = 8 << 20,
    ) -> None:
        super().__init__(name)
        if (command is None) == (url is None):
            raise ValueError("pass exactly one of command= (stdio) or url= (HTTP)")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if protocol_version is not None and protocol_version not in SUPPORTED_PROTOCOL_VERSIONS:
            raise ValueError(
                f"unsupported protocol_version {protocol_version!r}; choose one of "
                f"{', '.join(SUPPORTED_PROTOCOL_VERSIONS)}"
            )
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.protocol_version = protocol_version
        self.allowed_tools = _tool_allowlist(allowed_tools)
        self.transport = "stdio" if command is not None else "http"
        self.url: str | None = None
        self.headers: dict[str, str] = {}
        self.cwd: str | None = None
        self._argv: tuple[str, ...] = ()
        self._env: dict[str, str] = {}
        if command is not None:
            if headers:
                raise ValueError("headers= applies to url= servers; pass env= to a command")
            self._argv, self._env, self.cwd = _stdio_launch(command, env, cwd)
        else:
            if env is not None or cwd is not None:
                raise ValueError("env= and cwd= apply to command= servers only")
            self.url = check_url(url, "url")
            self.headers = check_headers(headers)
        self._lock = threading.RLock()
        self._session: _Session | None = None
        self._ids = itertools.count(1)

    # --- operations --------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """A safe summary: no arguments, environment or credentials."""
        summary: dict[str, Any] = {"name": self.name, "transport": self.transport}
        if self.transport == "stdio":
            # Only the program: arguments are where tokens get passed.
            summary["command"] = os.path.basename(self._argv[0])
            summary["cwd"] = self.cwd
        else:
            assert self.url is not None
            summary["url"] = display_url(self.url)
            summary["headers"] = redact_headers(self.headers)
        summary["allowed_tools"] = sorted(self.allowed_tools) if self.allowed_tools is not None else None
        summary["protocol_version"] = self.protocol_version
        summary["timeout"] = self.timeout
        return summary

    def server_info(self) -> dict[str, Any]:
        """What the server said about itself when the connection was made."""
        session = self._connect("server_info")
        return {
            "name": self.name,
            "transport": self.transport,
            "era": session.era,
            "protocol_version": session.version,
            "server": copy.deepcopy(session.server_info),
            "capabilities": copy.deepcopy(session.capabilities),
            "instructions": session.instructions,
        }

    def list_tools(self) -> list[dict[str, Any]]:
        """The tools an agent may call: name, description, input schema."""
        tools = self._load_tools("list_tools")
        return [copy.deepcopy(tool) for name, tool in tools.items() if self._tool_allowed(name)]

    def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        check: bool = False,
    ) -> McpToolResult:
        """Call the tool ``name`` with ``arguments`` and return its result."""
        if not isinstance(name, str) or not name:
            raise self._fail("tool name must be a non-empty string", "call_tool")
        if not self._tool_allowed(name):
            assert self.allowed_tools is not None
            raise self._fail(
                f"tool {name!r} is not in allowed_tools "
                f"({', '.join(sorted(self.allowed_tools))})",
                "call_tool",
            )
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, Mapping):
            raise self._fail("arguments must be a mapping of names to values", "call_tool")
        result, _ = self._request(
            "tools/call", {"name": name, "arguments": dict(arguments)}, "call_tool"
        )
        content = result.get("content", [])
        if not isinstance(content, list):
            raise self._fail(f"tool {name!r} returned malformed content", "call_tool")
        outcome = McpToolResult(
            tool=name,
            content=tuple(content),
            structured_content=result.get("structuredContent"),
            is_error=result.get("isError") is True,
        )
        if check and outcome.is_error:
            raise self._fail(
                f"tool {name!r} reported an error: {outcome.text or 'no details'}", "call_tool"
            )
        return outcome

    def list_resources(self) -> list[dict[str, Any]]:
        """The resources the server exposes: uri, name, mimeType..."""
        return self._list("resources/list", "resources", "list_resources")[0]

    def list_resource_templates(self) -> list[dict[str, Any]]:
        """Parameterized resources, as RFC 6570 URI templates."""
        return self._list(
            "resources/templates/list", "resourceTemplates", "list_resource_templates"
        )[0]

    def read_resource(self, uri: str) -> list[dict[str, Any]]:
        """The contents of the resource at ``uri``: text or base64 blobs."""
        if not isinstance(uri, str) or not uri:
            raise self._fail("uri must be a non-empty string", "read_resource")
        result, _ = self._request("resources/read", {"uri": uri}, "read_resource")
        contents = result.get("contents", [])
        if not isinstance(contents, list):
            raise self._fail(f"resource {uri!r} returned malformed contents", "read_resource")
        return contents

    def list_prompts(self) -> list[dict[str, Any]]:
        """The prompt templates the server offers."""
        return self._list("prompts/list", "prompts", "list_prompts")[0]

    def get_prompt(
        self, name: str, arguments: Mapping[str, str] | None = None
    ) -> dict[str, Any]:
        """Render the prompt ``name``: its description and messages."""
        if not isinstance(name, str) or not name:
            raise self._fail("prompt name must be a non-empty string", "get_prompt")
        params: dict[str, Any] = {"name": name}
        if arguments:
            if not isinstance(arguments, Mapping) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in arguments.items()
            ):
                raise self._fail("prompt arguments must map strings to strings", "get_prompt")
            params["arguments"] = dict(arguments)
        result, _ = self._request("prompts/get", params, "get_prompt")
        return {key: value for key, value in result.items() if key != "resultType"}

    def close(self) -> None:
        with self._lock:
            session, self._session = self._session, None
        if session is not None:
            session.transport.close()

    # --- requests ----------------------------------------------------------

    def _tool_allowed(self, name: str) -> bool:
        return self.allowed_tools is None or name in self.allowed_tools

    def _list(
        self, method: str, key: str, operation: str
    ) -> tuple[list[dict[str, Any]], _Session]:
        items: list[dict[str, Any]] = []
        cursor: str | None = None
        seen: set[str] = set()
        for _ in range(_MAX_PAGES):
            result, session = self._request(
                method, {} if cursor is None else {"cursor": cursor}, operation
            )
            page = result.get(key, [])
            if not isinstance(page, list):
                raise self._fail(f"{method} returned a malformed {key!r} list", operation)
            items.extend(page)
            cursor = result.get("nextCursor")
            if cursor is None or cursor == "":
                return items, session
            if not isinstance(cursor, str):
                raise self._fail(f"{method} returned a malformed cursor", operation)
            if cursor in seen:
                raise self._fail(f"{method} repeated a pagination cursor", operation)
            seen.add(cursor)
        raise self._fail(f"{method} returned more than {_MAX_PAGES} pages", operation)

    def _load_tools(self, operation: str) -> dict[str, dict[str, Any]]:
        listed, session = self._list("tools/list", "tools", operation)
        # Only HTTP mirrors arguments into headers, so only HTTP clients
        # must screen the annotations (stdio may ignore them).
        screen = session.era == "modern" and self.transport == "http"
        tools: dict[str, dict[str, Any]] = {}
        headers: dict[str, dict[str, tuple[str, ...]]] = {}
        rejected: dict[str, str] = {}
        for tool in listed:
            name = tool.get("name") if isinstance(tool, dict) else None
            if not isinstance(name, str) or not name:
                logger.warning("MCP server %r listed a tool without a name; ignoring it", self.name)
                continue
            if screen:
                try:
                    headers[name] = _header_params(tool.get("inputSchema"))
                except ValueError as exc:
                    logger.warning("MCP server %r: excluding tool %r: %s", self.name, name, exc)
                    rejected[name] = str(exc)
                    continue
            tools[name] = tool
        with self._lock:
            session.tools, session.tool_headers, session.rejected_tools = tools, headers, rejected
        return tools

    def _request(
        self, method: str, params: dict[str, Any], operation: str
    ) -> tuple[dict[str, Any], _Session]:
        """Send ``method`` and return its result, retrying where it is safe.

        Retried: a refusal of the protocol version or an expired legacy
        session (once each, both before the server acted), a header mismatch
        on a tool call after refreshing the tool list (once), and an
        ``input_required`` result that asks for nothing but a retry.
        """
        session = self._connect(operation)
        params = dict(params)
        renegotiated = refreshed = False
        rounds = 0
        while True:
            try:
                response = self._send(session, method, params, operation)
            except _SessionExpired:
                if renegotiated:
                    raise self._fail(
                        f"{method} failed: the server dropped a session it had just opened",
                        operation,
                    ) from None
                renegotiated = True
                session = self._renegotiate(session, operation)
                continue
            if "error" in response:
                code, message, data = error_parts(response["error"])
                if (
                    code == _UNSUPPORTED_PROTOCOL_VERSION
                    and session.era == "modern"
                    and not renegotiated
                ):
                    renegotiated = True
                    session = self._renegotiate(session, operation)
                    continue
                if code == _HEADER_MISMATCH and method == "tools/call" and not refreshed:
                    refreshed = True
                    self._load_tools(operation)
                    session = self._connect(operation)
                    continue
                raise JsonRpcError(
                    f"MCP server {self.name!r} refused {method}: {message} (code {code})",
                    code=code,
                    data=data,
                    integration=self.name,
                    operation=operation,
                )
            result = response.get("result")
            if not isinstance(result, dict):
                raise self._fail(f"{method} returned a malformed result", operation)
            kind = result.get("resultType", "complete") if session.era == "modern" else "complete"
            if kind == "complete":
                return result, session
            if kind != "input_required":
                raise self._fail(f"{method} returned an unknown resultType {kind!r}", operation)
            wanted = result.get("inputRequests")
            if wanted:
                names = ", ".join(sorted(map(str, wanted))) if isinstance(wanted, dict) else "?"
                raise self._fail(
                    f"{method} needs input titoos cannot provide ({names}): it "
                    "offers no elicitation, sampling or roots",
                    operation,
                )
            rounds += 1
            if rounds > _MAX_INPUT_ROUNDS:
                raise self._fail(
                    f"{method} still asked for a retry after {_MAX_INPUT_ROUNDS} attempts",
                    operation,
                )
            params.pop("requestState", None)
            if "requestState" in result:
                params["requestState"] = result["requestState"]

    def _send(
        self, session: _Session, method: str, params: dict[str, Any], operation: str
    ) -> dict[str, Any]:
        body = dict(params)
        if session.era == "modern":
            meta = body.get("_meta")
            body["_meta"] = {**(meta if isinstance(meta, dict) else {}), **_client_meta(session.version)}
        message = {"jsonrpc": "2.0", "id": next(self._ids), "method": method, "params": body}
        try:
            json.dumps(body)
        except (TypeError, ValueError) as exc:
            raise self._fail(f"{method} arguments are not JSON-serializable: {exc}", operation) from exc
        headers = self._protocol_headers(session, method, params, operation)
        reply = self._exchange(session.transport, message, headers, operation)
        return self._response(reply, method, operation)

    def _exchange(
        self,
        transport: Any,
        message: dict[str, Any],
        headers: Mapping[str, str],
        operation: str,
        *,
        cancel_on_timeout: bool = True,
    ) -> _Reply:
        method = message["method"]
        try:
            return transport.exchange(
                message,
                headers=headers,
                deadline=time.monotonic() + self.timeout,
                cancel_on_timeout=cancel_on_timeout,
            )
        except _TransportTimeout as exc:
            raise self._fail(
                f"MCP server {self.name!r} did not answer {method} within {self.timeout}s: {exc}",
                operation,
            ) from exc
        except _TransportError as exc:
            raise self._fail(f"MCP server {self.name!r} failed {method}: {exc}", operation) from exc

    def _response(self, reply: _Reply, method: str, operation: str) -> dict[str, Any]:
        if reply.status in (401, 403):
            raise self._fail(
                f"MCP server {self.name!r} refused {method} with HTTP {reply.status}: "
                "pass credentials in headers=, e.g. {'Authorization': 'Bearer ...'}",
                operation,
            )
        if reply.message is None:
            raise self._fail(
                f"MCP server {self.name!r} failed {method}: HTTP {reply.status}"
                + (f": {reply.detail}" if reply.detail else ""),
                operation,
            )
        return reply.message

    def _protocol_headers(
        self, session: _Session, method: str, params: dict[str, Any], operation: str
    ) -> dict[str, str]:
        if self.transport != "http":
            return {}
        headers = {"MCP-Protocol-Version": session.version}
        if session.era != "modern":
            return headers
        headers["Mcp-Method"] = method
        if method in ("tools/call", "prompts/get"):
            headers["Mcp-Name"] = _header_value(params["name"])
        elif method == "resources/read":
            headers["Mcp-Name"] = _header_value(params["uri"])
        if method == "tools/call":
            headers.update(self._param_headers(session, params, operation))
        return headers

    def _param_headers(
        self, session: _Session, params: dict[str, Any], operation: str
    ) -> dict[str, str]:
        """The ``Mcp-Param-*`` headers a tool's schema asks to be mirrored."""
        name = params["name"]
        if session.tools is None:
            self._load_tools(operation)
            if session.tools is None:
                # Listing reconnected, so the tools landed on a new session:
                # start the call over on that one.
                raise _SessionExpired()
        reason = session.rejected_tools.get(name)
        if reason is not None:
            raise self._fail(f"tool {name!r} was excluded: {reason}", operation)
        headers: dict[str, str] = {}
        for header, path in session.tool_headers.get(name, {}).items():
            value: Any = params["arguments"]
            for key in path:
                value = value.get(key, _MISSING) if isinstance(value, Mapping) else _MISSING
                if value is _MISSING:
                    break
            if value is _MISSING or value is None:
                continue  # no value at that path: the header is omitted
            try:
                headers[f"Mcp-Param-{header}"] = _header_value(_header_text(value))
            except ValueError as exc:
                raise self._fail(
                    f"argument mirrored into header {header!r} of tool {name!r}: {exc}",
                    operation,
                ) from exc
        return headers

    # --- connecting --------------------------------------------------------

    def _connect(self, operation: str) -> _Session:
        with self._lock:
            session = self._session
            if session is not None and session.transport.alive:
                return session
            self._session = None
            if session is not None:
                session.transport.close()
            transport = self._open_transport(operation)
            try:
                session = self._negotiate(transport, operation)
            except BaseException:
                transport.close()
                raise
            self._session = session
            return session

    def _renegotiate(self, stale: _Session, operation: str) -> _Session:
        """Negotiate again on the same transport, unless another call did."""
        with self._lock:
            current = self._session
            if current is not None and current is not stale and current.transport.alive:
                return current
            self._session = None
            transport = stale.transport
            if not transport.alive:
                transport.close()
                return self._connect(operation)
            if isinstance(transport, _HttpTransport):
                transport.session_id = None
                transport.legacy_version = None
            try:
                session = self._negotiate(transport, operation)
            except BaseException:
                transport.close()
                raise
            self._session = session
            return session

    def _open_transport(self, operation: str) -> Any:
        if self.transport == "http":
            assert self.url is not None
            return _HttpTransport(
                self.url, headers=self.headers, max_bytes=self.max_bytes, label=self.name
            )
        try:
            return _StdioTransport(
                self._argv,
                env=self._env,
                cwd=self.cwd,
                max_bytes=self.max_bytes,
                label=self.name,
            )
        except _TransportError as exc:
            raise self._fail(f"MCP server {self.name!r}: {exc}", operation) from exc

    def _negotiate(self, transport: Any, operation: str) -> _Session:
        """Find the era and version to speak, per the compatibility rules."""
        pinned = self.protocol_version
        if pinned in LEGACY_PROTOCOL_VERSIONS:
            return self._initialize(transport, pinned, operation)
        version = pinned or LATEST_PROTOCOL_VERSION
        tried: set[str] = set()
        while True:
            tried.add(version)
            outcome, value = self._probe(transport, version, operation)
            if outcome == "legacy":
                if pinned:
                    raise self._fail(
                        f"MCP server {self.name!r} does not speak protocol_version "
                        f"{pinned!r}: it does not answer server/discover",
                        operation,
                    )
                return self._initialize(transport, LEGACY_PROTOCOL_VERSIONS[0], operation)
            supported = value.get("supportedVersions") if outcome == "modern" else value
            offered = [v for v in supported if isinstance(v, str)] if isinstance(supported, list) else []
            if outcome == "modern" and version in offered:
                meta = value.get("_meta") if isinstance(value.get("_meta"), dict) else {}
                info = meta.get("io.modelcontextprotocol/serverInfo")
                capabilities = value.get("capabilities")
                instructions = value.get("instructions")
                return _Session(
                    transport,
                    "modern",
                    version,
                    server_info=info if isinstance(info, dict) else {},
                    capabilities=capabilities if isinstance(capabilities, dict) else {},
                    instructions=instructions if isinstance(instructions, str) else None,
                )
            choice = next(
                (v for v in ((pinned,) if pinned else SUPPORTED_PROTOCOL_VERSIONS) if v in offered),
                None,
            )
            if choice is None or choice in tried:
                raise self._fail(
                    f"MCP server {self.name!r} supports protocol versions "
                    f"{', '.join(offered) or 'none'}; titoos needs one of "
                    f"{', '.join((pinned,) if pinned else SUPPORTED_PROTOCOL_VERSIONS)}",
                    operation,
                )
            if choice in LEGACY_PROTOCOL_VERSIONS:
                return self._initialize(transport, choice, operation)
            version = choice

    def _probe(self, transport: Any, version: str, operation: str) -> tuple[str, Any]:
        """Send ``server/discover`` and classify the answer.

        Returns ``("modern", result)``, ``("unsupported", supported)`` for a
        modern server refusing ``version``, or ``("legacy", None)``. Legacy
        servers answer an unknown pre-``initialize`` request with whatever
        error they like, or not at all, so any error that is not a modern one
        — and, on stdio, silence — means legacy.
        """
        message = {
            "jsonrpc": "2.0",
            "id": next(self._ids),
            "method": "server/discover",
            "params": {"_meta": _client_meta(version)},
        }
        headers = (
            {"MCP-Protocol-Version": version, "Mcp-Method": "server/discover"}
            if self.transport == "http"
            else {}
        )
        try:
            # No cancellation on timeout: a legacy server expects nothing but
            # initialize before initialize.
            reply = transport.exchange(
                message,
                headers=headers,
                deadline=time.monotonic() + self.timeout,
                cancel_on_timeout=False,
            )
        except _TransportTimeout as exc:
            if transport.kind == "stdio":
                return "legacy", None
            raise self._fail(
                f"MCP server {self.name!r} did not answer server/discover within {self.timeout}s",
                operation,
            ) from exc
        except _TransportError as exc:
            raise self._fail(f"MCP server {self.name!r}: {exc}", operation) from exc
        if reply.status in (401, 403):
            self._response(reply, "server/discover", operation)
        response = reply.message
        if response is None:
            if reply.status in (400, 404, 405):
                return "legacy", None
            self._response(reply, "server/discover", operation)
        assert response is not None
        if "error" in response:
            code, text, data = error_parts(response["error"])
            if code == _UNSUPPORTED_PROTOCOL_VERSION:
                supported = data.get("supported") if isinstance(data, dict) else None
                return "unsupported", supported
            if code in (_HEADER_MISMATCH, _MISSING_CLIENT_CAPABILITY) or reply.status >= 500:
                raise JsonRpcError(
                    f"MCP server {self.name!r} refused server/discover: {text} (code {code})",
                    code=code,
                    data=data,
                    integration=self.name,
                    operation=operation,
                )
            return "legacy", None
        result = response.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("supportedVersions"), list):
            return "legacy", None
        return "modern", result

    def _initialize(self, transport: Any, version: str, operation: str) -> _Session:
        """The legacy handshake: ``initialize``, then ``initialized``."""
        message = {
            "jsonrpc": "2.0",
            "id": next(self._ids),
            "method": "initialize",
            "params": {
                "protocolVersion": version,
                "capabilities": {},
                "clientInfo": _client_info(),
            },
        }
        # The specification forbids cancelling initialize.
        reply = self._exchange(transport, message, {}, operation, cancel_on_timeout=False)
        if reply.message is None and reply.status in (404, 405):
            raise self._fail(
                f"MCP server {self.name!r} rejected initialize with HTTP {reply.status}; "
                "is url= the server's Streamable HTTP endpoint? (the deprecated "
                "HTTP+SSE transport is not supported)",
                operation,
            )
        response = self._response(reply, "initialize", operation)
        if "error" in response:
            code, text, data = error_parts(response["error"])
            raise JsonRpcError(
                f"MCP server {self.name!r} refused initialize: {text} (code {code})",
                code=code,
                data=data,
                integration=self.name,
                operation=operation,
            )
        result = response.get("result")
        if not isinstance(result, dict):
            raise self._fail(f"MCP server {self.name!r} returned a malformed initialize result", operation)
        negotiated = result.get("protocolVersion")
        if negotiated not in LEGACY_PROTOCOL_VERSIONS:
            raise self._fail(
                f"MCP server {self.name!r} chose protocol version {negotiated!r}, "
                "which titoos does not support",
                operation,
            )
        if self.protocol_version is not None and negotiated != self.protocol_version:
            raise self._fail(
                f"MCP server {self.name!r} chose protocol version {negotiated!r}, "
                f"not the pinned version {self.protocol_version!r}",
                operation,
            )
        transport.begin_legacy(negotiated)
        headers = {"MCP-Protocol-Version": negotiated} if self.transport == "http" else {}
        try:
            transport.notify(
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                headers=headers,
                deadline=time.monotonic() + self.timeout,
            )
        except _TransportError as exc:
            raise self._fail(f"MCP server {self.name!r}: {exc}", operation) from exc
        info = result.get("serverInfo")
        capabilities = result.get("capabilities")
        instructions = result.get("instructions")
        return _Session(
            transport,
            "legacy",
            negotiated,
            server_info=info if isinstance(info, dict) else {},
            capabilities=capabilities if isinstance(capabilities, dict) else {},
            instructions=instructions if isinstance(instructions, str) else None,
        )
