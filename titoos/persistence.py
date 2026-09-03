"""Persistence: capture and restore a kernel at a tick boundary.

A tick boundary is the only instant at which no agent is mid-execution and no
message is in flight, which makes it the natural consistent checkpoint. The
kernel refuses to snapshot at any other time.

Agents opt in by implementing :meth:`~titoos.agent.Agent.save_state` and
:meth:`~titoos.agent.Agent.load_state`. Anything they do not put in that dict
is not preserved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .agent import Agent, AgentState
from .message import Message

#: Builds a bare agent of a given kind for a given name. The kernel restores
#: its state afterwards, so the factory must not need the saved data.
AgentFactory = Callable[[str], Agent]

SNAPSHOT_VERSION = 1


class FinishedAgent(Agent):
    """Inert stand-in for an agent that had already finished when saved.

    A DONE or FAILED agent will never be scheduled again, so its behaviour
    does not need to be reconstructible. Restoring it as a placeholder keeps
    the run's history — its name, outcome, parent link, restart count and
    saved data — without forcing callers to supply a factory for it.
    """

    def __init__(self, name: str, kind: str = "FinishedAgent") -> None:
        super().__init__(name)
        self.kind = kind
        self.saved_state: dict[str, Any] = {}

    def step(self, ctx: Any) -> None:  # pragma: no cover - never scheduled
        raise RuntimeError(f"finished agent {self.name!r} cannot run again")

    def save_state(self) -> dict[str, Any]:
        return dict(self.saved_state)

    def load_state(self, data: dict[str, Any]) -> None:
        self.saved_state = dict(data)


@dataclass(frozen=True)
class AgentRecord:
    """The persisted form of a single agent."""

    name: str
    kind: str
    state: str
    parent: str | None = None
    restarts: int = 0
    data: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "state": self.state,
            "parent": self.parent,
            "restarts": self.restarts,
            "data": dict(self.data),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentRecord":
        return cls(
            name=data["name"],
            kind=data["kind"],
            state=data["state"],
            parent=data.get("parent"),
            restarts=data.get("restarts", 0),
            data=data.get("data") or {},
        )


@dataclass(frozen=True)
class Snapshot:
    """A consistent, serializable picture of a kernel between two ticks."""

    tick: int
    agents: tuple[AgentRecord, ...]
    mailboxes: Mapping[str, tuple[Message, ...]] = field(default_factory=dict)
    version: int = SNAPSHOT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tick": self.tick,
            "agents": [record.to_dict() for record in self.agents],
            "mailboxes": {
                name: [message.to_dict() for message in messages]
                for name, messages in self.mailboxes.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Snapshot":
        version = data.get("version", SNAPSHOT_VERSION)
        if version != SNAPSHOT_VERSION:
            raise ValueError(
                f"unsupported snapshot version {version!r}; "
                f"this build reads version {SNAPSHOT_VERSION}"
            )
        return cls(
            tick=data["tick"],
            agents=tuple(AgentRecord.from_dict(r) for r in data["agents"]),
            mailboxes={
                name: tuple(Message.from_dict(m) for m in messages)
                for name, messages in (data.get("mailboxes") or {}).items()
            },
            version=version,
        )

    def to_json(self, **kwargs: Any) -> str:
        """Serialize to JSON. Agent state and payloads must be JSON-able."""
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, text: str) -> "Snapshot":
        return cls.from_dict(json.loads(text))

    def agent_states(self) -> dict[str, AgentState]:
        return {record.name: AgentState(record.state) for record in self.agents}
