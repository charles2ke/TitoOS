"""Messages exchanged between agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

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
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def is_broadcast(self) -> bool:
        return self.to == BROADCAST
