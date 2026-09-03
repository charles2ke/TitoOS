"""TitoOS - a minimal operating system for agents."""

from .agent import Agent, AgentState, Context, FunctionAgent, RestartPolicy
from .backends import AsyncBackend, ExecutionBackend, SerialBackend, ThreadBackend
from .bus import MessageBus
from .kernel import CHILD_FAILED, Kernel, StopReason, TickReport
from .message import BROADCAST, Message
from .persistence import AgentFactory, AgentRecord, Snapshot

__all__ = [
    "BROADCAST",
    "CHILD_FAILED",
    "Agent",
    "AgentFactory",
    "AgentRecord",
    "AgentState",
    "AsyncBackend",
    "Context",
    "ExecutionBackend",
    "FunctionAgent",
    "Kernel",
    "Message",
    "MessageBus",
    "RestartPolicy",
    "SerialBackend",
    "Snapshot",
    "StopReason",
    "ThreadBackend",
    "TickReport",
]

__version__ = "0.2.0"
