"""Integrations: the drivers agents use to reach the real world.

Install them on the kernel and call them from an agent::

    from titoos import Kernel
    from titoos.integrations import HttpIntegration

    kernel = Kernel()
    kernel.install(HttpIntegration(allowed_hosts=["api.example.com"]))

    def fetcher(ctx):
        response = ctx.call("http", "get", "https://api.example.com/health")
        ctx.send("sink", response.json())
        ctx.exit()
"""

from .base import (
    Integration,
    IntegrationError,
    IntegrationRegistry,
    normalize_allowlist,
    redact,
)
from .a2a import A2AIntegration, A2APeer, A2AResult
from .clock import ClockIntegration
from .files import FileSystemIntegration
from .http import HttpIntegration, HttpResponse
from .jsonrpc import JsonRpcError
from .mcp import McpIntegration, McpToolResult
from .shell import CommandResult, ShellIntegration

__all__ = [
    "A2AIntegration",
    "A2APeer",
    "A2AResult",
    "ClockIntegration",
    "CommandResult",
    "FileSystemIntegration",
    "HttpIntegration",
    "HttpResponse",
    "Integration",
    "IntegrationError",
    "IntegrationRegistry",
    "JsonRpcError",
    "McpIntegration",
    "McpToolResult",
    "ShellIntegration",
    "normalize_allowlist",
    "redact",
]
