"""Messages exchanged between agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

BROADCAST = "*"

#: Shared read-only mapping reused by every message without metadata, which is
#: the common case; it saves a dict plus a proxy allocation per message.
_EMPTY_METADATA: Mapping[str, Any] = MappingProxyType({})


@dataclass(frozen=True)
class Message:
    """A single message routed by the kernel's message bus.

    A ``to`` value of :data:`BROADCAST` delivers the message to every
    registered agent except the sender.
    """

    sender: str
    to: str
    payload: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        metadata = self.metadata
        if not metadata:
            object.__setattr__(self, "metadata", _EMPTY_METADATA)
        else:
            object.__setattr__(self, "metadata", MappingProxyType(dict(metadata)))

    @property
    def is_broadcast(self) -> bool:
        return self.to == BROADCAST

    def to_dict(self) -> dict[str, Any]:
        """A plain, JSON-friendly representation of this message."""
        return {
            "sender": self.sender,
            "to": self.to,
            "payload": self.payload,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Message":
        return cls(
            sender=data["sender"],
            to=data["to"],
            payload=data.get("payload"),
            metadata=data.get("metadata") or {},
        )
