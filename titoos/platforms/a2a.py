"""Run a remote A2A agent as a kernel agent.

:class:`A2AAdapter` is the inbound direction of A2A: a remote agent joins the
bus under a local name, and the agents around it message it like any other.
The outbound direction — an agent deciding to call a remote one — is
:class:`~titoos.integrations.a2a.A2AIntegration`.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Mapping, Sequence

from ..integrations.a2a import A2AClient, A2APeer, A2AResult
from .base import Outbound, PlatformAdapter, PlatformTurn

#: Task states that end a turn with an error rather than a reply.
_FAILURES = frozenset({"failed", "rejected", "canceled", "auth_required"})
#: Task states that mean the peer had not finished when time ran out.
_ACTIVE = frozenset({"submitted", "working"})


class A2AAdapter(PlatformAdapter):
    """Put a remote A2A agent on the bus as if it were a local one.

    Each turn's mail goes to ``peer`` as one message — strings as text parts,
    bytes as raw parts, mappings as data parts, anything else as JSON text —
    and the answer goes back to whoever wrote: the reply's text, or the whole
    :meth:`~titoos.integrations.a2a.A2AResult.to_dict` when it carries more
    than text. Replies carry ``a2a_state``, ``a2a_task_id`` and
    ``a2a_context_id`` metadata where the peer supplied them.

    All turns share one A2A context, so the peer sees a single conversation,
    and when it asks for more input (``a2a_state == "input_required"``) the
    next turn answers that task. A task that fails, is rejected or canceled,
    needs authentication, or is still running after ``timeout`` raises
    :class:`~titoos.platforms.base.PlatformError`, and the kernel's
    supervision decides what happens next. The context survives snapshots.
    """

    name = "a2a"

    def __init__(
        self,
        peer: str | A2APeer,
        *,
        name: str | None = None,
        timeout: float = 60.0,
        max_bytes: int = 8 << 20,
    ) -> None:
        super().__init__(name)
        self._state_lock = threading.Lock()
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if isinstance(peer, str):
            peer = A2APeer(peer)
        if not isinstance(peer, A2APeer):
            raise ValueError(f"peer must be a URL or an A2APeer, not {type(peer).__name__}")
        self.peer = peer
        self.timeout = timeout
        self._client = A2AClient(
            self.name,
            peer,
            owner=self.name,
            timeout=timeout,
            max_bytes=max_bytes,
            fail=self._fail,
        )
        #: The conversation every turn continues, once the peer started one.
        self.context_id: str | None = None
        #: The task waiting for input, which the next turn answers.
        self.task_id: str | None = None

    def to_platform(self, turn: PlatformTurn) -> list[dict[str, Any]]:
        payloads = turn.payloads or (() if turn.seed is None else (turn.seed,))
        return [self._part(payload) for payload in payloads]

    @staticmethod
    def _part(payload: Any) -> dict[str, Any]:
        if isinstance(payload, str):
            return {"text": payload}
        if isinstance(payload, (bytes, bytearray)):
            return {"raw": bytes(payload)}
        if isinstance(payload, Mapping):
            return {"data": dict(payload)}
        try:
            return {"text": json.dumps(payload)}
        except (TypeError, ValueError):
            return {"text": str(payload)}

    def invoke(self, request: Any, turn: PlatformTurn) -> A2AResult:
        with self._state_lock:
            if not request:
                raise self._fail(f"agent {turn.agent!r} has nothing to send", "invoke")
            result = self._client.send_message(
                "invoke", parts=request, context_id=self.context_id, task_id=self.task_id
            )
            if result.context_id:
                self.context_id = result.context_id
            self.task_id = result.task_id if result.needs_input else None
            if result.state in _FAILURES:
                detail = f": {result.text}" if result.text else ""
                raise self._fail(
                    f"A2A peer {self.name!r} ended task {result.task_id} as {result.state}{detail}",
                    "invoke",
                )
            if result.state in _ACTIVE:
                raise self._fail(
                    f"A2A peer {self.name!r} is still running task {result.task_id} "
                    f"({result.state}) after {self.timeout}s; raise timeout= for slow peers",
                    "invoke",
                )
            if not (result.done or result.needs_input):
                raise self._fail(
                    f"A2A peer {self.name!r} left task {result.task_id} in state {result.state!r}",
                    "invoke",
                )
            return result

    def from_platform(self, result: Any, turn: PlatformTurn) -> Sequence[Outbound]:
        if not isinstance(result, A2AResult):
            return super().from_platform(result, turn)
        plain = bool(result.parts) and all(set(part) <= {"text", "metadata"} for part in result.parts)
        metadata = {
            key: value
            for key, value in (
                ("a2a_state", result.state),
                ("a2a_task_id", result.task_id),
                ("a2a_context_id", result.context_id),
            )
            if value is not None
        }
        return (Outbound(payload=result.text if plain else result.to_dict(), metadata=metadata),)

    def save_state(self) -> dict[str, Any]:
        with self._state_lock:
            return {"context_id": self.context_id, "task_id": self.task_id}

    def load_state(self, data: dict[str, Any]) -> None:
        context_id, task_id = data.get("context_id"), data.get("task_id")
        with self._state_lock:
            self.context_id = context_id if isinstance(context_id, str) and context_id else None
            self.task_id = task_id if isinstance(task_id, str) and task_id else None
