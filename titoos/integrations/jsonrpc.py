"""JSON-RPC 2.0 over HTTP, the plumbing shared by the MCP and A2A drivers.

Both protocols wrap JSON-RPC in an HTTP POST, and both may answer with a
Server-Sent Events stream instead of a single JSON body. This module keeps
what they share, using nothing but the standard library: an error type that
preserves the remote error code, URL and header checks, an opener that never
follows redirects, and an SSE parser with a byte budget and a deadline.
"""

from __future__ import annotations

import http.client
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

from .base import IntegrationError
from .http import SECRET_HEADERS

#: Exceptions meaning "the exchange broke", as opposed to "the peer said no".
TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (OSError, http.client.HTTPException)

_TOKEN = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")
#: Header names that look like credentials even when they are not in
#: :data:`~titoos.integrations.http.SECRET_HEADERS`, e.g. ``X-Goog-Api-Key``.
_SECRET_HINT = re.compile(r"auth|token|secret|key|cookie|session|password", re.I)


class JsonRpcError(IntegrationError):
    """A JSON-RPC error returned by a remote MCP server or A2A agent.

    The peer understood the request and refused it, so :attr:`code` and
    :attr:`data` are kept for callers that branch on them (``-32001`` is
    A2A's "task not found", ``-32602`` is "invalid params", ...).
    """

    def __init__(
        self,
        message: str,
        *,
        code: int | None,
        data: Any = None,
        integration: str = "",
        operation: str = "",
    ) -> None:
        super().__init__(message, integration=integration, operation=operation)
        self.code = code
        self.data = data


class ResponseTooLarge(OSError):
    """A response grew past the caller's ``max_bytes`` budget."""


def check_url(url: Any, what: str) -> str:
    """Return ``url`` without its fragment, or raise :class:`ValueError`.

    Only absolute ``http``/``https`` URLs with a host are accepted, so a
    configured endpoint can never name a local file or another scheme.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError(f"{what} must be a non-empty URL")
    try:
        parsed = urllib.parse.urlsplit(url.strip())
        parsed.port  # noqa: B018 - raises ValueError for a malformed port
    except ValueError as exc:
        raise ValueError(f"malformed {what} {url!r}: {exc}") from exc
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"{what} must be an http or https URL, got {url!r}")
    if not parsed.hostname:
        raise ValueError(f"{what} has no host: {url!r}")
    return urllib.parse.urlunsplit((scheme, parsed.netloc, parsed.path, parsed.query, ""))


def display_url(url: str) -> str:
    """``url`` without credentials or a query string, safe to log or show."""
    try:
        parsed = urllib.parse.urlsplit(url)
        host = parsed.hostname or ""
        port = parsed.port
    except ValueError:
        return "<malformed URL>"
    if ":" in host:
        host = f"[{host}]"
    netloc = f"{host}:{port}" if port else host
    return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def origin(url: str) -> tuple[str, str, int]:
    """The ``(scheme, host, port)`` origin of ``url``, with default ports."""
    parsed = urllib.parse.urlsplit(url)
    scheme = parsed.scheme.lower()
    return scheme, (parsed.hostname or "").lower(), parsed.port or (443 if scheme == "https" else 80)


def check_headers(headers: Mapping[str, str] | None, what: str = "headers") -> dict[str, str]:
    """Validate operator-supplied headers up front, or raise :class:`ValueError`.

    Failing at construction beats failing at the first call: a header value
    with a line break in it is refused by :mod:`http.client` anyway, but only
    once an agent is already mid-task.
    """
    checked: dict[str, str] = {}
    for key, value in (headers or {}).items():
        if not isinstance(key, str) or not _TOKEN.fullmatch(key):
            raise ValueError(f"invalid header name in {what}: {key!r}")
        if not isinstance(value, str) or any(c in value for c in "\r\n\0"):
            raise ValueError(f"invalid value for header {key!r} in {what}")
        checked[key] = value
    return checked


def redact_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    """Copy ``headers`` with anything credential-like replaced by ``***``."""
    secret = {name.lower() for name in SECRET_HEADERS}
    return {
        key: "***" if key.lower() in secret or _SECRET_HINT.search(key) else value
        for key, value in (headers or {}).items()
    }


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Turn every redirect into an error response instead of following it.

    Following one would replay the operator's headers, tokens included, at
    whatever address the server named, so a 3xx comes back to the caller as
    an answer to report rather than a hop to take.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


def http_open(
    method: str,
    url: str,
    *,
    headers: Mapping[str, str],
    body: bytes | None = None,
    timeout: float,
) -> Any:
    """Send one request and return the response, whatever its status.

    Error statuses (and redirects, which are never followed) are returned as
    :class:`urllib.error.HTTPError`, which reads like a response. Transport
    failures raise one of :data:`TRANSPORT_ERRORS`.
    """
    request = urllib.request.Request(url, data=body, method=method)
    for key, value in headers.items():
        request.add_header(key, value)
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        return opener.open(request, timeout=timeout)  # noqa: S310 - callers pass only checked http(s) URLs
    except urllib.error.HTTPError as exc:  # an answer, not a failure
        return exc


def is_timeout(exc: BaseException) -> bool:
    """True if ``exc`` is a timeout, including one wrapped by :mod:`urllib`."""
    return isinstance(exc, TimeoutError) or isinstance(
        getattr(exc, "reason", None), TimeoutError
    )


def read_body(response: Any, max_bytes: int) -> bytes:
    """Read the whole body of ``response``, refusing more than ``max_bytes``."""
    body = response.read(max_bytes + 1)
    if len(body) > max_bytes:
        raise ResponseTooLarge(f"response exceeded max_bytes ({max_bytes} bytes)")
    return body


def find_response(payload: Any, request_id: Any) -> dict[str, Any] | None:
    """The JSON-RPC response to ``request_id`` in ``payload``, if there is one.

    ``payload`` may be one message or a batch. An error response with a null
    ``id`` also counts: it is how a peer reports a request it could not parse
    far enough to read the id.
    """
    for item in payload if isinstance(payload, list) else (payload,):
        if not isinstance(item, dict) or "method" in item:
            continue
        if "result" not in item and "error" not in item:
            continue
        if item.get("id") == request_id or (item.get("id") is None and "error" in item):
            return item
    return None


def error_parts(error: Any) -> tuple[int | None, str, Any]:
    """Split a JSON-RPC error object into ``(code, message, data)``."""
    if not isinstance(error, dict):
        return None, str(error), None
    code = error.get("code")
    if not isinstance(code, int) or isinstance(code, bool):
        code = None
    message = error.get("message")
    return code, message if isinstance(message, str) and message else "unknown error", error.get("data")


def snippet(body: bytes, limit: int = 200) -> str:
    """The start of a response body, for an error message."""
    text = body[:limit].decode("utf-8", errors="replace").strip()
    return text + ("..." if len(body) > limit else "")


@dataclass(frozen=True)
class SseEvent:
    """One dispatched Server-Sent Event."""

    event: str
    data: str
    id: str | None


class SseStream:
    """Parse a ``text/event-stream`` body incrementally, within limits.

    Follows the WHATWG parsing rules that matter here: ``\\n``, ``\\r\\n`` and
    ``\\r`` line endings, comments, multi-line ``data``, and ``id``/``retry``
    bookkeeping that survives events without data (which is how a server
    primes a client to resume). Reading stops with :class:`ResponseTooLarge`
    once ``max_bytes`` have been consumed, and with :class:`TimeoutError`
    once ``deadline`` (a :func:`time.monotonic` value) has passed — checked
    between lines, since a single read is bounded by the socket timeout.
    """

    def __init__(
        self,
        response: Any,
        *,
        max_bytes: int,
        deadline: float | None = None,
        last_event_id: str | None = None,
    ) -> None:
        self._response = response
        self._deadline = deadline
        #: Bytes still allowed before the stream is refused.
        self.remaining = max_bytes
        #: The last ``id`` field seen, used to resume an interrupted stream.
        self.last_event_id = last_event_id
        #: The server's requested reconnection delay, in seconds.
        self.retry: float | None = None

    def __iter__(self) -> Iterator[SseEvent]:
        data: list[str] = []
        event = ""
        first = True
        while True:
            if self._deadline is not None and time.monotonic() > self._deadline:
                raise TimeoutError("timed out reading the event stream")
            raw = self._response.readline(self.remaining + 1)
            if not raw:
                return  # end of stream: an unterminated event is discarded
            if len(raw) > self.remaining:
                raise ResponseTooLarge("event stream exceeded max_bytes")
            self.remaining -= len(raw)
            text = raw.decode("utf-8", errors="replace")
            if first:
                text = text.removeprefix("\ufeff")
                first = False
            text = text.removesuffix("\n").removesuffix("\r")
            # readline() splits on "\n" only, so a lone "\r" left inside is
            # itself a line ending.
            for line in text.split("\r"):
                if not line:
                    if data:
                        yield SseEvent(event or "message", "\n".join(data), self.last_event_id)
                    data, event = [], ""
                    continue
                if line.startswith(":"):
                    continue
                name, colon, value = line.partition(":")
                if colon and value.startswith(" "):
                    value = value[1:]
                if name == "data":
                    data.append(value)
                elif name == "event":
                    event = value
                elif name == "id" and "\0" not in value:
                    self.last_event_id = value
                elif name == "retry" and value.isascii() and value.isdigit():
                    self.retry = int(value) / 1000
