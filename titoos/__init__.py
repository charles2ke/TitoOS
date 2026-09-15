"""TitoOS - a minimal operating system for agents."""

from .agent import Agent, AgentState, Context, FunctionAgent, RestartPolicy
from .backends import AsyncBackend, ExecutionBackend, SerialBackend, ThreadBackend
from .bus import MessageBus
from .integrations import (
    ClockIntegration,
    CommandResult,
    FileSystemIntegration,
    HttpIntegration,
    HttpResponse,
    Integration,
    IntegrationError,
    IntegrationRegistry,
    ShellIntegration,
)
from .kernel import CHILD_FAILED, Kernel, StopReason, TickReport
from .message import BROADCAST, Message
from .persistence import AgentFactory, AgentRecord, Snapshot

__all__ = [
    "Agent",
    "AgentFactory",
    "AgentRecord",
    "AgentState",
    "AsyncBackend",
    "BROADCAST",
    "CHILD_FAILED",
    "ClockIntegration",
    "CommandResult",
    "Context",
    "ExecutionBackend",
    "FileSystemIntegration",
    "FunctionAgent",
    "HttpIntegration",
    "HttpResponse",
    "Integration",
    "IntegrationError",
    "IntegrationRegistry",
    "Kernel",
    "Message",
    "MessageBus",
    "RestartPolicy",
    "SerialBackend",
    "ShellIntegration",
    "Snapshot",
    "StopReason",
    "ThreadBackend",
    "TickReport",
]

__version__ = "0.2.0"
