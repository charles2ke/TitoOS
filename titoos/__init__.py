"""TitoOS - a minimal operating system for agents."""

from .agent import Agent, AgentState, Context, FunctionAgent
from .bus import MessageBus
from .kernel import Kernel, StopReason, TickReport
from .message import BROADCAST, Message

__all__ = [
    "BROADCAST",
    "Agent",
    "AgentState",
    "Context",
    "FunctionAgent",
    "Kernel",
    "Message",
    "MessageBus",
    "StopReason",
    "TickReport",
]

__version__ = "0.1.0"
