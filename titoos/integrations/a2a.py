"""Agent2Agent (A2A) client: send work to remote agents over JSON-RPC.

An A2A agent is a service another team runs, built with any framework, that
publishes an *agent card* describing itself and accepts messages over HTTP.
:class:`A2AIntegration` holds a fixed address book of such agents — the
operator names each peer and its URL — and lets kernel agents message them,
follow the tasks they start, and cancel them.

It speaks the JSON-RPC binding of A2A 1.0 and of A2A 0.3, picking whichever
the peer's card advertises (1.0 first). gRPC and HTTP+JSON bindings, streaming
and push notifications are not used: one request, one answer, with polling
for tasks that are still running.
"""

from __future__ import annotations

import base64
import copy
import itertools
import json
import threading
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, NamedTuple, Sequence

from .base import Integration
from .jsonrpc import (
    TRANSPORT_ERRORS,
    JsonRpcError,
    ResponseTooLarge,
    SseStream,
    check_headers,
    check_url,
    display_url,
    error_parts,
    find_response,
    http_open,
    is_timeout,
    origin,
    read_body,
    redact_headers,
    snippet,
)

#: Protocol versions this client speaks, in order of preference.
A2A_PROTOCOL_VERSIONS: tuple[str, ...] = ("1.0", "0.3")
#: Where an agent publishes its card, relative to the peer's URL.
AGENT_CARD_PATH = "/.well-known/agent-card.json"

_METHODS = {
    "1.0": {"send": "SendMessage", "get": "GetTask", "cancel": "CancelTask"},
    "0.3": {"send": "message/send", "get": "tasks/get", "cancel": "tasks/cancel"},
}
_ERROR_NAMES = {
    -32700: "JSONParseError",
    -32600: "InvalidRequestError",
    -32601: "MethodNotFoundError",
    -32602: "InvalidParamsError",
    -32603: "InternalError",
    -32001: "TaskNotFoundError",
    -32002: "TaskNotCancelableError",
    -32003: "PushNotificationNotSupportedError",
    -32004: "UnsupportedOperationError",
    -32005: "ContentTypeNotSupportedError",
    -32006: "InvalidAgentResponseError",
    -32007: "ExtendedAgentCardNotConfiguredError",
    -32008: "ExtensionSupportRequiredError",
    -32009: "VersionNotSupportedError",
}
# A2A 1.0 serializes TaskState by name; the numbers are accepted too.
_STATE_NUMBERS = {
    0: "unspecified",
    1: "submitted",
    2: "working",
    3: "completed",
    4: "failed",
    5: "canceled",
    6: "input_required",
    7: "rejected",
    8: "auth_required",
}
#: States after which a task never changes again.
TERMINAL_STATES = frozenset({"completed", "failed", "canceled", "rejected"})
#: States in which a task is still running, so worth polling.
_ACTIVE_STATES = frozenset({"submitted", "working"})
_POLL_START = 0.25
_POLL_MAX = 2.0
_PART_CONTENT = ("text", "raw", "url", "data")


def _state(value: Any) -> str:
    """A task state in one spelling: ``TASK_STATE_INPUT_REQUIRED`` (1.0),
    ``input-required`` (0.3) and ``6`` all become ``input_required``."""
    if isinstance(value, int) and not isinstance(value, bool):
        return _STATE_NUMBERS.get(value, "unknown")
    if not isinstance(value, str) or not value.strip():
        return "unknown"
    state = value.strip().lower().replace("-", "_").removeprefix("task_state_")
    return "canceled" if state == "cancelled" else state


def _family(version: Any) -> str | None:
    """The protocol this client would speak to an interface of ``version``."""
    if not isinstance(version, str):
        return None
    major = version.strip().split(".", 1)[0]
    return {"1": "1.0", "0": "0.3"}.get(major)


@dataclass(frozen=True, repr=False)
class A2APeer:
    """How to reach one remote agent.

    ``url`` is where the agent lives: its card is read from
    ``{url}/.well-known/agent-card.json``, or from ``url`` itself when it
    names a ``.json`` file. The card says where to send requests; titoos only
    follows it to the same origin (scheme, host and port) as the card, so a
    card cannot redirect ``headers`` — typically an ``Authorization`` token —
    to another server. Pass ``endpoint=`` to name the JSON-RPC endpoint
    yourself and skip the card, and ``protocol_version=`` (``"1.0"`` or
    ``"0.3"``) to pin the protocol.
    """

    url: str
    headers: Mapping[str, str] = field(default_factory=dict, kw_only=True, hash=False)
    endpoint: str | None = field(default=None, kw_only=True)
    protocol_version: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "url", check_url(self.url, "A2A peer url"))
        object.__setattr__(
            self, "headers", MappingProxyType(check_headers(self.headers, "A2A peer headers"))
        )
        if self.endpoint is not None:
            object.__setattr__(self, "endpoint", check_url(self.endpoint, "A2A peer endpoint"))
        if self.protocol_version is not None and self.protocol_version not in A2A_PROTOCOL_VERSIONS:
            raise ValueError(
                f"unsupported A2A protocol_version {self.protocol_version!r}; choose one "
                f"of {', '.join(A2A_PROTOCOL_VERSIONS)}"
            )

    @property
    def card_url(self) -> str:
        """Where this peer's agent card is read from."""
        parts = urllib.parse.urlsplit(self.url)
        if parts.path.endswith(".json"):
            return self.url
        path = parts.path.rstrip("/") + AGENT_CARD_PATH
        return urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))

    def describe(self) -> dict[str, Any]:
        return {
            "url": display_url(self.url),
            "endpoint": display_url(self.endpoint) if self.endpoint else None,
            "protocol_version": self.protocol_version,
            "headers": redact_headers(self.headers),
        }

    def __repr__(self) -> str:  # never the headers: they carry credentials
        return f"A2APeer({display_url(self.url)!r})"


@dataclass(frozen=True)
class A2AResult:
    """A remote agent's answer: a finished message, or a task and its state.

    :attr:`state` is ``None`` for a plain message, otherwise one of
    ``submitted``, ``working``, ``input_required``, ``auth_required``,
    ``completed``, ``failed``, ``canceled`` or ``rejected``. :attr:`parts`
    are in the A2A 1.0 shape (``{"text": ...}``, ``{"data": ...}``,
    ``{"url": ...}``, ``{"raw": <base64>}``) whichever version the peer
    speaks, and :attr:`text` joins the text parts.
    """

    peer: str
    kind: str
    state: str | None
    task_id: str | None
    context_id: str | None
    text: str
    parts: tuple[dict[str, Any], ...]
    #: The task or message exactly as the peer sent it.
    raw: dict[str, Any] = field(repr=False)

    @property
    def ok(self) -> bool:
        """The peer answered, or finished the task successfully."""
        return self.kind == "message" or self.state == "completed"

    @property
    def done(self) -> bool:
        """Nothing further will happen: a message, or a task in a final state."""
        return self.kind == "message" or self.state in TERMINAL_STATES

    @property
    def needs_input(self) -> bool:
        """The task is waiting for a reply: send one with this ``task_id``."""
        return self.state == "input_required"

    def to_dict(self) -> dict[str, Any]:
        return {
            "peer": self.peer,
            "kind": self.kind,
            "state": self.state,
            "task_id": self.task_id,
            "context_id": self.context_id,
            "text": self.text,
            "parts": copy.deepcopy(list(self.parts)),
        }


class _Route(NamedTuple):
    url: str
    version: str
    tenant: str | None
    #: The version was assumed rather than read from a card or a pin.
    guessed: bool


# --- parts -------------------------------------------------------------------


def _normalize_part(part: Any) -> dict[str, Any] | None:
    """A part from either protocol version, in the 1.0 shape."""
    if not isinstance(part, dict):
        return None
    kind = part.get("kind")
    out: dict[str, Any] = {}
    if kind == "text":
        out["text"] = part.get("text", "")
    elif kind == "file":
        file = part.get("file") if isinstance(part.get("file"), dict) else {}
        if "bytes" in file:
            out["raw"] = file["bytes"]
        elif "uri" in file:
            out["url"] = file["uri"]
        if file.get("name"):
            out["filename"] = file["name"]
        if file.get("mimeType"):
            out["mediaType"] = file["mimeType"]
    elif kind == "data":
        out["data"] = part.get("data")
    elif kind is None:
        out = {k: v for k, v in part.items() if k in (*_PART_CONTENT, "filename", "mediaType")}
    else:
        return dict(part)  # a kind this client does not know: pass it through
    if isinstance(part.get("metadata"), dict):
        out["metadata"] = part["metadata"]
    return out


def _parts_text(parts: Sequence[dict[str, Any]]) -> str:
    return "\n".join(p["text"] for p in parts if isinstance(p.get("text"), str))


def _message_parts(message: Any) -> list[dict[str, Any]]:
    if not isinstance(message, dict) or not isinstance(message.get("parts"), list):
        return []
    return [p for p in map(_normalize_part, message["parts"]) if p is not None]


def _is_agent(message: Any) -> bool:
    role = message.get("role") if isinstance(message, dict) else None
    return role == 2 or (isinstance(role, str) and role.lower() in ("agent", "role_agent"))


def _task_parts(task: dict[str, Any]) -> list[dict[str, Any]]:
    """What a task has to say: its artifacts, else its status message, else
    the agent's last message in the history."""
    parts: list[dict[str, Any]] = []
    for artifact in task.get("artifacts") or ():
        parts.extend(_message_parts(artifact))
    if parts:
        return parts
    status = task.get("status") if isinstance(task.get("status"), dict) else {}
    parts = _message_parts(status.get("message"))
    if parts:
        return parts
    history = task.get("history") if isinstance(task.get("history"), list) else []
    for message in reversed(history):
        if _is_agent(message):
            return _message_parts(message)
    return []


def _outgoing_part(part: Any, version: str) -> dict[str, Any]:
    """One part a caller passed, in the shape ``version`` expects.

    A part is a string (text), bytes (raw content), or a mapping in either
    protocol's shape: ``{"text": ...}``, ``{"data": ...}``, ``{"url": ...,
    "mediaType": ...}``, ``{"raw": b"..."}``, or 0.3's ``{"kind": ...}``.
    """
    if isinstance(part, str):
        part = {"text": part}
    elif isinstance(part, (bytes, bytearray)):
        part = {"raw": bytes(part)}
    if not isinstance(part, Mapping):
        raise ValueError(f"a part must be a string, bytes or a mapping, not {type(part).__name__}")
    normalized = _normalize_part(dict(part))
    assert normalized is not None
    content = [key for key in _PART_CONTENT if key in normalized]
    if len(content) != 1:
        raise ValueError("a part needs exactly one of 'text', 'raw', 'url' or 'data'")
    key = content[0]
    if isinstance(normalized.get("raw"), (bytes, bytearray)):
        normalized["raw"] = base64.b64encode(normalized["raw"]).decode("ascii")
    if key in ("text", "raw", "url") and not isinstance(normalized[key], str):
        raise ValueError(f"part {key!r} must be a string")
    if version == "1.0":
        return normalized
    converted: dict[str, Any]
    if key == "text":
        converted = {"kind": "text", "text": normalized["text"]}
    elif key == "data":
        if not isinstance(normalized["data"], Mapping):
            raise ValueError("A2A 0.3 data parts must be objects (mappings)")
        converted = {"kind": "data", "data": normalized["data"]}
    else:
        file = {"bytes" if key == "raw" else "uri": normalized[key]}
        if normalized.get("filename"):
            file["name"] = normalized["filename"]
        if normalized.get("mediaType"):
            file["mimeType"] = normalized["mediaType"]
        converted = {"kind": "file", "file": file}
    if "metadata" in normalized:
        converted["metadata"] = normalized["metadata"]
    return converted


# --- the client --------------------------------------------------------------


class A2AClient:
    """The protocol logic for one peer, shared by the integration and the
    :class:`~titoos.platforms.a2a.A2AAdapter`.

    ``fail(message, operation)`` builds the exception to raise, so each
    caller reports failures in its own error type.
    """

    def __init__(
        self,
        peer_name: str,
        peer: A2APeer,
        *,
        owner: str,
        timeout: float,
        max_bytes: int,
        fail: Callable[[str, str], Exception],
    ) -> None:
        self.peer_name = peer_name
        self.peer = peer
        self.timeout = timeout
        self.max_bytes = max_bytes
        self._owner = owner
        self._fail = fail
        self._lock = threading.RLock()
        self._card: dict[str, Any] | None = None
        self._route: _Route | None = None
        self._ids = itertools.count(1)

    # --- discovery -----------------------------------------------------------

    def agent_card(self, operation: str) -> dict[str, Any]:
        with self._lock:
            if self._card is None:
                self._card = self._fetch_card(operation)
            return copy.deepcopy(self._card)

    def _fetch_card(self, operation: str) -> dict[str, Any]:
        url = self.peer.card_url
        shown = display_url(url)
        deadline = time.monotonic() + self.timeout
        headers = {**self.peer.headers, "Accept": "application/json"}
        status, body, _ = self._exchange("GET", url, headers, None, deadline, operation, "agent card")
        if not 200 <= status < 300:
            raise self._fail(
                f"A2A peer {self.peer_name!r}: fetching the agent card from {shown} "
                f"failed with HTTP {status}{self._status_hint(status)}"
                + (f": {snippet(body)}" if body.strip() else ""),
                operation,
            )
        try:
            card = json.loads(body)
        except ValueError:
            card = None
        if not isinstance(card, dict):
            raise self._fail(
                f"A2A peer {self.peer_name!r}: {shown} did not return an agent card (a JSON object)",
                operation,
            )
        return card

    def route(self, operation: str) -> _Route:
        with self._lock:
            if self._route is None:
                self._route = self._choose_route(operation)
            return self._route

    def _choose_route(self, operation: str) -> _Route:
        pinned = self.peer.protocol_version
        if self.peer.endpoint is not None:
            return _Route(self.peer.endpoint, pinned or "1.0", None, guessed=pinned is None)
        card = self.agent_card(operation)
        card_url = self.peer.card_url
        offers: list[tuple[Any, str, Any]] = []
        for interface in card.get("supportedInterfaces") or ():  # 1.0 cards
            if isinstance(interface, dict) and str(interface.get("protocolBinding", "")).upper() == "JSONRPC":
                family = _family(interface.get("protocolVersion"))
                if family is not None:
                    offers.append((interface.get("url"), family, interface.get("tenant")))
        if "url" in card and str(card.get("preferredTransport") or "JSONRPC").upper() == "JSONRPC":
            offers.append((card["url"], "0.3", None))  # 0.3 cards
        for interface in card.get("additionalInterfaces") or ():
            if isinstance(interface, dict) and str(interface.get("transport", "")).upper() == "JSONRPC":
                offers.append((interface.get("url"), "0.3", None))
        if pinned is not None:
            offers = [offer for offer in offers if offer[1] == pinned]
        offers.sort(key=lambda offer: A2A_PROTOCOL_VERSIONS.index(offer[1]))
        foreign: list[str] = []
        for url, family, tenant in offers:
            if not isinstance(url, str) or not url:
                continue
            try:
                resolved = check_url(urllib.parse.urljoin(card_url, url), "endpoint")
            except ValueError:
                continue
            if origin(resolved) != origin(card_url):
                foreign.append(display_url(resolved))
                continue
            return _Route(resolved, family, tenant if isinstance(tenant, str) and tenant else None, False)
        wanted = f"A2A {pinned}" if pinned else "A2A 1.0 or 0.3"
        if foreign:
            raise self._fail(
                f"A2A peer {self.peer_name!r}: the agent card only offers {wanted} over "
                f"JSON-RPC at another origin ({', '.join(dict.fromkeys(foreign))}); if you "
                "trust it with this peer's headers, pass that URL as A2APeer(endpoint=...)",
                operation,
            )
        raise self._fail(
            f"A2A peer {self.peer_name!r}: the agent card at {display_url(card_url)} offers "
            f"no {wanted} JSON-RPC interface (titoos does not speak gRPC or HTTP+JSON)",
            operation,
        )

    # --- requests ------------------------------------------------------------

    def send_message(
        self,
        operation: str,
        *,
        text: str | None = None,
        data: Any = None,
        parts: Sequence[Any] | None = None,
        context_id: str | None = None,
        task_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        wait: bool = True,
        history_length: int | None = None,
    ) -> A2AResult:
        deadline = time.monotonic() + self.timeout
        route = self.route(operation)
        message: dict[str, Any] = {
            "messageId": str(uuid.uuid4()),
            "role": "ROLE_USER" if route.version == "1.0" else "user",
            "parts": self._parts(text, data, parts, route.version, operation),
        }
        if route.version == "0.3":
            message["kind"] = "message"
        for key, value in (("contextId", context_id), ("taskId", task_id)):
            if value is not None:
                if not isinstance(value, str) or not value:
                    raise self._fail(f"{key} must be a non-empty string", operation)
                message[key] = value
        if metadata is not None:
            if not isinstance(metadata, Mapping):
                raise self._fail("metadata must be a mapping", operation)
            message["metadata"] = dict(metadata)
        configuration: dict[str, Any] = (
            {"returnImmediately": not wait} if route.version == "1.0" else {"blocking": bool(wait)}
        )
        if history_length is not None:
            configuration["historyLength"] = self._history_length(history_length, operation)
        payload = self._rpc(
            "send", {"message": message, "configuration": configuration}, deadline, operation
        )
        result = self._result(payload, operation)
        delay = _POLL_START
        # A blocking request may still come back early, e.g. from a server
        # that does not block: poll until the task settles or time runs out.
        while wait and result.kind == "task" and result.state in _ACTIVE_STATES and result.task_id:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, _POLL_MAX)
            if deadline - time.monotonic() <= 0:
                break
            latest = self._rpc("get", {"id": result.task_id}, deadline, operation)
            result = self._result(latest, operation)
        return result

    def get_task(self, operation: str, task_id: str, *, history_length: int | None = None) -> A2AResult:
        if not isinstance(task_id, str) or not task_id:
            raise self._fail("task_id must be a non-empty string", operation)
        params: dict[str, Any] = {"id": task_id}
        if history_length is not None:
            params["historyLength"] = self._history_length(history_length, operation)
        deadline = time.monotonic() + self.timeout
        return self._result(self._rpc("get", params, deadline, operation), operation)

    def cancel_task(self, operation: str, task_id: str) -> A2AResult:
        if not isinstance(task_id, str) or not task_id:
            raise self._fail("task_id must be a non-empty string", operation)
        deadline = time.monotonic() + self.timeout
        return self._result(self._rpc("cancel", {"id": task_id}, deadline, operation), operation)

    def _history_length(self, value: Any, operation: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise self._fail("history_length must be a non-negative integer", operation)
        return value

    def _parts(
        self, text: Any, data: Any, parts: Any, version: str, operation: str
    ) -> list[dict[str, Any]]:
        given: list[Any] = []
        if text is not None:
            if not isinstance(text, str):
                raise self._fail("text must be a string", operation)
            given.append({"text": text})
        if data is not None:
            given.append({"data": data})
        if parts is not None:
            if isinstance(parts, (str, bytes, Mapping)) or not isinstance(parts, Sequence):
                raise self._fail("parts must be a list of parts", operation)
            given.extend(parts)
        if not given:
            raise self._fail("send_message needs text, data or parts", operation)
        try:
            return [_outgoing_part(part, version) for part in given]
        except ValueError as exc:
            raise self._fail(str(exc), operation) from exc

    def _result(self, payload: Any, operation: str) -> A2AResult:
        if not isinstance(payload, dict):
            raise self._fail(f"A2A peer {self.peer_name!r} returned a malformed result", operation)
        if isinstance(payload.get("task"), dict):  # 1.0 SendMessageResponse
            kind, body = "task", payload["task"]
        elif isinstance(payload.get("message"), dict) and "messageId" not in payload:
            kind, body = "message", payload["message"]
        elif payload.get("kind") == "message" or ("messageId" in payload and "status" not in payload):
            kind, body = "message", payload
        elif payload.get("kind") == "task" or "status" in payload:
            kind, body = "task", payload
        else:
            raise self._fail(
                f"A2A peer {self.peer_name!r} returned neither a task nor a message", operation
            )
        if kind == "message":
            parts = _message_parts(body)
            state = None
            task_id, context_id = body.get("taskId"), body.get("contextId")
        else:
            parts = _task_parts(body)
            status = body.get("status") if isinstance(body.get("status"), dict) else {}
            state = _state(status.get("state"))
            task_id, context_id = body.get("id"), body.get("contextId")
        return A2AResult(
            peer=self.peer_name,
            kind=kind,
            state=state,
            task_id=task_id if isinstance(task_id, str) and task_id else None,
            context_id=context_id if isinstance(context_id, str) and context_id else None,
            text=_parts_text(parts),
            parts=tuple(parts),
            raw=body,
        )

    def _rpc(self, method_key: str, params: dict[str, Any], deadline: float, operation: str) -> Any:
        route = self.route(operation)
        method = _METHODS[route.version][method_key]
        if route.tenant is not None:
            params = {"tenant": route.tenant, **params}
        request_id = next(self._ids)
        message = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        headers = {
            **self.peer.headers,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if route.version == "1.0":
            headers["A2A-Version"] = "1.0"
        try:
            body = json.dumps(message).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise self._fail(f"{method} arguments are not JSON-serializable: {exc}", operation) from exc
        status, raw, found = self._exchange(
            "POST", route.url, headers, body, deadline, operation, method, request_id
        )
        if found is None:
            detail = f": {snippet(raw)}" if raw.strip() else ""
            raise self._fail(
                f"A2A peer {self.peer_name!r} answered {method} with HTTP {status} and no "
                f"JSON-RPC response{self._status_hint(status)}{detail}",
                operation,
            )
        if "error" in found:
            code, text, data = error_parts(found["error"])
            name = _ERROR_NAMES.get(code) if code is not None else None
            hint = ""
            if code in (-32601, -32009) and route.guessed:
                hint = "; if the agent speaks A2A 0.3, pass A2APeer(protocol_version='0.3')"
            raise JsonRpcError(
                f"A2A peer {self.peer_name!r} refused {method}: "
                f"{name + ': ' if name else ''}{text} (code {code}){hint}",
                code=code,
                data=data,
                integration=self._owner,
                operation=operation,
            )
        return found.get("result")

    def _exchange(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        deadline: float,
        operation: str,
        what: str,
        request_id: Any = None,
    ) -> tuple[int, bytes, dict[str, Any] | None]:
        """One HTTP exchange: ``(status, body, JSON-RPC response or None)``."""
        shown = display_url(url)
        try:
            response = http_open(
                method,
                url,
                headers=headers,
                body=body,
                timeout=max(deadline - time.monotonic(), 0.001),
            )
            try:
                status = response.getcode()
                if 300 <= status < 400:
                    location = response.headers.get("Location")
                    target = display_url(urllib.parse.urljoin(url, location)) if location else "elsewhere"
                    raise self._fail(
                        f"A2A peer {self.peer_name!r}: {shown} redirected {what} to {target}; "
                        "redirects are not followed, so configure the final URL",
                        operation,
                    )
                if request_id is not None and response.headers.get_content_type() == "text/event-stream":
                    for event in SseStream(response, max_bytes=self.max_bytes, deadline=deadline):
                        try:
                            found = find_response(json.loads(event.data), request_id)
                        except ValueError:
                            continue
                        if found is not None:
                            return status, b"", found
                    return status, b"", None
                raw = read_body(response, self.max_bytes)
            finally:
                response.close()
        except ResponseTooLarge as exc:
            raise self._fail(
                f"A2A peer {self.peer_name!r}: {what} from {shown} exceeded max_bytes "
                f"({self.max_bytes} bytes)",
                operation,
            ) from exc
        except TRANSPORT_ERRORS as exc:
            if is_timeout(exc):
                hint = (
                    "; for long tasks, pass wait=False and follow up with get_task"
                    if what == _METHODS["1.0"]["send"] or what == _METHODS["0.3"]["send"]
                    else ""
                )
                raise self._fail(
                    f"A2A peer {self.peer_name!r} did not answer {what} within "
                    f"{self.timeout}s{hint}",
                    operation,
                ) from exc
            raise self._fail(f"A2A peer {self.peer_name!r}: {what} via {shown} failed: {exc}", operation) from exc
        found = None
        if request_id is not None and raw.strip():
            try:
                found = find_response(json.loads(raw), request_id)
            except ValueError:
                found = None
        return status, raw, found

    @staticmethod
    def _status_hint(status: int) -> str:
        if status in (401, 403):
            return " (pass credentials in A2APeer(headers={'Authorization': ...}))"
        return ""


# --- the integration ---------------------------------------------------------


def _peer_book(peers: Any) -> dict[str, A2APeer]:
    if not isinstance(peers, Mapping) or not peers:
        raise ValueError("peers must be a non-empty mapping of names to URLs or A2APeer objects")
    book: dict[str, A2APeer] = {}
    for name, peer in peers.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"peer names must be non-empty strings: {name!r}")
        if isinstance(peer, str):
            peer = A2APeer(peer)
        if not isinstance(peer, A2APeer):
            raise ValueError(f"peer {name!r} must be a URL or an A2APeer, not {type(peer).__name__}")
        book[name] = peer
    return book


class A2AIntegration(Integration):
    """Message remote A2A agents from the operator's address book.

    ``peers`` maps a name agents use to each remote agent: a URL, or an
    :class:`A2APeer` carrying credentials and pins. Agents can reach these
    peers and no others; names are case-sensitive.

    :meth:`send_message` returns an :class:`A2AResult`. By default it waits
    for the task to settle, within ``timeout`` (blocking on the peer, then
    polling it); with ``wait=False`` it returns as soon as the task exists,
    and :meth:`get_task` follows it up. A task that wants more input (state
    ``input_required``) continues with another :meth:`send_message` carrying
    its ``task_id``; sending ``context_id`` too keeps a conversation going.
    A peer that refuses a request raises :class:`~titoos.integrations.jsonrpc.JsonRpcError`
    with A2A's error ``code``.
    """

    name = "a2a"
    operations = ("send_message", "get_task", "cancel_task", "agent_card", "describe")

    def __init__(
        self,
        peers: Mapping[str, str | A2APeer],
        *,
        name: str | None = None,
        timeout: float = 60.0,
        max_bytes: int = 8 << 20,
    ) -> None:
        super().__init__(name)
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.peers = MappingProxyType(_peer_book(peers))
        self._clients = {
            peer_name: A2AClient(
                peer_name,
                peer,
                owner=self.name,
                timeout=timeout,
                max_bytes=max_bytes,
                fail=self._fail,
            )
            for peer_name, peer in self.peers.items()
        }

    def send_message(
        self,
        peer: str,
        text: str | None = None,
        *,
        data: Any = None,
        parts: Sequence[Any] | None = None,
        context_id: str | None = None,
        task_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        wait: bool = True,
        history_length: int | None = None,
    ) -> A2AResult:
        """Send ``text``, ``data`` (a JSON value) and/or ``parts`` to ``peer``."""
        return self._client(peer, "send_message").send_message(
            "send_message",
            text=text,
            data=data,
            parts=parts,
            context_id=context_id,
            task_id=task_id,
            metadata=metadata,
            wait=wait,
            history_length=history_length,
        )

    def get_task(self, peer: str, task_id: str, *, history_length: int | None = None) -> A2AResult:
        """The current state of a task ``peer`` is running."""
        return self._client(peer, "get_task").get_task(
            "get_task", task_id, history_length=history_length
        )

    def cancel_task(self, peer: str, task_id: str) -> A2AResult:
        """Ask ``peer`` to cancel a task; returns the task as it now stands."""
        return self._client(peer, "cancel_task").cancel_task("cancel_task", task_id)

    def agent_card(self, peer: str) -> dict[str, Any]:
        """``peer``'s agent card: its name, skills, capabilities, interfaces."""
        return self._client(peer, "agent_card").agent_card("agent_card")

    def describe(self) -> dict[str, Any]:
        """The address book, with credentials redacted."""
        return {
            "name": self.name,
            "peers": {name: peer.describe() for name, peer in self.peers.items()},
            "timeout": self.timeout,
        }

    def _client(self, peer: Any, operation: str) -> A2AClient:
        client = self._clients.get(peer) if isinstance(peer, str) else None
        if client is None:
            raise self._fail(
                f"unknown A2A peer {peer!r}; configured peers: {', '.join(self._clients)}",
                operation,
            )
        return client
