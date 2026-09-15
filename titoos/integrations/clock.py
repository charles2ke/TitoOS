"""Clock integration: the one source of time agents should use.

Reading the clock directly inside ``step()`` makes an agent impossible to test
and impossible to replay. Going through an integration means a test can install
a frozen clock and get the same run every time.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

from .base import Integration


class ClockIntegration(Integration):
    """Wall-clock and monotonic time, plus a bounded sleep.

    ``max_sleep`` caps how long a single :meth:`sleep` may block: a tick is a
    barrier, so an agent sleeping for an hour stalls every other agent running
    on the same worker.
    """

    name = "clock"
    operations = ("now", "timestamp", "monotonic", "sleep", "describe")

    def __init__(
        self,
        *,
        name: str | None = None,
        max_sleep: float = 5.0,
        fixed: datetime | None = None,
    ) -> None:
        super().__init__(name)
        if max_sleep < 0:
            raise ValueError("max_sleep must not be negative")
        self.max_sleep = max_sleep
        if fixed is not None:
            if fixed.tzinfo is None or fixed.utcoffset() is None:
                raise ValueError(
                    "fixed must be timezone-aware; a naive datetime has no "
                    "defined instant"
                )
            fixed = fixed.astimezone(timezone.utc)
        #: When set, the clock is frozen at this instant and never sleeps.
        self.fixed = fixed
        #: Deterministic stand-in for the process clock while frozen. Advanced
        #: by :meth:`sleep`, so measuring a slept duration still works.
        self._elapsed = 0.0
        self._lock = threading.Lock()

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "max_sleep": self.max_sleep,
            "fixed": self.fixed.isoformat() if self.fixed else None,
        }

    def now(self) -> datetime:
        """The current time as a timezone-aware UTC ``datetime``."""
        if self.fixed is not None:
            return self.fixed
        return datetime.now(timezone.utc)

    def timestamp(self) -> float:
        """Seconds since the epoch."""
        return self.now().timestamp()

    def monotonic(self) -> float:
        """A monotonic clock reading, suitable for measuring durations.

        While frozen this counts the time the clock was asked to sleep rather
        than the host's process clock, so a fixed run stays reproducible.
        """
        if self.fixed is not None:
            with self._lock:
                return self._elapsed
        return time.monotonic()

    def sleep(self, seconds: float) -> float:
        """Sleep for ``seconds``, capped by ``max_sleep``. Returns the delay."""
        if seconds < 0:
            raise self._fail("cannot sleep for a negative duration", "sleep")
        delay = min(seconds, self.max_sleep)
        if self.fixed is not None:
            with self._lock:
                self._elapsed += delay
        elif delay:
            time.sleep(delay)
        return delay
