"""In-memory message bus used by the kernel to route agent messages.

All operations are thread-safe so agents running concurrently in a worker
pool can send and receive messages without external locking.
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from typing import Any, Iterable, Sequence

from .message import BROADCAST, Message, _make_message


class MessageBus:
    """Routes :class:`~titoos.message.Message` objects between mailboxes."""

    def __init__(self) -> None:
        self._mailboxes: dict[str, deque[Message]] = defaultdict(deque)
        self._lock = threading.RLock()

    def register(self, name: str) -> None:
        """Create an (empty) mailbox for ``name`` if it does not exist yet."""
        with self._lock:
            self._mailboxes[name]  # noqa: B018 - defaultdict creates the mailbox

    def unregister(self, name: str) -> None:
        with self._lock:
            self._mailboxes.pop(name, None)

    def send(self, message: Message) -> None:
        """Deliver ``message`` to its recipient, or to everyone if broadcast."""
        to = message.to
        with self._lock:
            if to == BROADCAST:
                sender = message.sender
                for name, mailbox in self._mailboxes.items():
                    if name != sender:
                        mailbox.append(message)
                return
            mailbox = self._mailboxes.get(to)
            if mailbox is None:
                known = ", ".join(sorted(self._mailboxes)) or "none"
                raise KeyError(
                    f"unknown recipient: {to!r}; mailboxes: {known}"
                )
            mailbox.append(message)

    def post(self, sender: str, to: str, payload: Any = None, **metadata: Any) -> Message:
        """Build and send a message in one call."""
        message = _make_message(sender, to, payload, metadata)
        self.send(message)
        return message

    def broadcast(self, sender: str, payload: Any = None, **metadata: Any) -> Message:
        return self.post(sender, BROADCAST, payload, **metadata)

    def receive(self, name: str) -> list[Message]:
        """Drain and return every pending message for ``name``."""
        with self._lock:
            mailbox = self._mailboxes.get(name)
            if not mailbox:
                return []
            messages = list(mailbox)
            mailbox.clear()
            return messages

    def pending(self, name: str) -> int:
        with self._lock:
            return len(self._mailboxes.get(name, ()))

    def receive_many(self, names: Sequence[str]) -> list[list[Message]]:
        """Drain the mailboxes of ``names`` in one pass, in the same order.

        Equivalent to calling :meth:`receive` on each name, but takes the lock
        once for the whole batch, which is what the kernel needs at a tick
        boundary.
        """
        with self._lock:
            get = self._mailboxes.get
            drained: list[list[Message]] = []
            append = drained.append
            for name in names:
                mailbox = get(name)
                if mailbox:
                    append(list(mailbox))
                    mailbox.clear()
                else:
                    append([])
            return drained

    def any_pending(self, names: Iterable[str]) -> bool:
        """True if at least one of ``names`` has a pending message."""
        with self._lock:
            get = self._mailboxes.get
            for name in names:
                if get(name):
                    return True
            return False

    def mailboxes(self) -> Iterable[str]:
        with self._lock:
            return tuple(self._mailboxes)

    def requeue(self, name: str, messages: Sequence[Message]) -> None:
        """Put ``messages`` back at the front of ``name``'s mailbox.

        Used when an agent fails mid-tick and is restarted: the messages it
        was handed must be redelivered rather than lost.
        """
        if not messages:
            return
        with self._lock:
            mailbox = self._mailboxes.get(name)
            if mailbox is None:
                return
            mailbox.extendleft(reversed(messages))

    def dump(self) -> dict[str, list[Message]]:
        """A copy of every mailbox, without consuming anything."""
        with self._lock:
            return {name: list(box) for name, box in self._mailboxes.items()}

    def load(self, mailboxes: dict[str, list[Message]]) -> None:
        """Replace all mailboxes with ``mailboxes``."""
        with self._lock:
            self._mailboxes.clear()
            for name, messages in mailboxes.items():
                self._mailboxes[name] = deque(messages)

    def has_traffic(self) -> bool:
        with self._lock:
            return any(self._mailboxes.values())
