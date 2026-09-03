"""In-memory message bus used by the kernel to route agent messages."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Iterable

from .message import BROADCAST, Message


class MessageBus:
    """Routes :class:`~titoos.message.Message` objects between mailboxes."""

    def __init__(self) -> None:
        self._mailboxes: dict[str, deque[Message]] = defaultdict(deque)

    def register(self, name: str) -> None:
        """Create an (empty) mailbox for ``name`` if it does not exist yet."""
        self._mailboxes[name]  # noqa: B018 - defaultdict creates the mailbox

    def unregister(self, name: str) -> None:
        self._mailboxes.pop(name, None)

    def send(self, message: Message) -> None:
        """Deliver ``message`` to its recipient, or to everyone if broadcast."""
        if message.is_broadcast:
            for name, mailbox in self._mailboxes.items():
                if name != message.sender:
                    mailbox.append(message)
            return
        if message.to not in self._mailboxes:
            raise KeyError(f"unknown recipient: {message.to!r}")
        self._mailboxes[message.to].append(message)

    def post(self, sender: str, to: str, payload: Any = None, **metadata: Any) -> Message:
        """Build and send a message in one call."""
        message = Message(sender=sender, to=to, payload=payload, metadata=dict(metadata))
        self.send(message)
        return message

    def broadcast(self, sender: str, payload: Any = None, **metadata: Any) -> Message:
        return self.post(sender, BROADCAST, payload, **metadata)

    def receive(self, name: str) -> list[Message]:
        """Drain and return every pending message for ``name``."""
        mailbox = self._mailboxes.get(name)
        if not mailbox:
            return []
        messages = list(mailbox)
        mailbox.clear()
        return messages

    def pending(self, name: str) -> int:
        return len(self._mailboxes.get(name, ()))

    def mailboxes(self) -> Iterable[str]:
        return tuple(self._mailboxes)

    def has_traffic(self) -> bool:
        return any(self._mailboxes.values())
