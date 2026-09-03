"""Messages exchanged between agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

BROADCAST = "*"


@dataclass(frozen=True)
class Message:
    """A single message routed by the kernel's message bus.

    A ``to`` value of :data:`BROADCAST` delivers the message to every
    registered agent except the sender.
    """

    sender: str
    to: str
    payload: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_broadcast(self) -> bool:
        return self.to == BROADCAST
