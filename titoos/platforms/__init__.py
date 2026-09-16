"""Platform adapters: running agents from other frameworks on this kernel.

Everything exported here is dependency-free. Adapters for real frameworks are
published in :data:`~titoos.platforms.base.registry` by name and imported only
when one is built, so importing this package never imports an SDK::

    from titoos import Kernel
    from titoos.platforms import CallableAdapter

    kernel = Kernel()
    adapter = CallableAdapter("echo", lambda request, turn: f"echo:{request}")
    kernel.register(adapter.agent("assistant"))

Write your own by subclassing
:class:`~titoos.platforms.base.PlatformAdapter` and, if it should be
selectable by name, publishing it with
:func:`~titoos.platforms.base.register_platform`.
"""

from .base import (
    AdapterFactory,
    AgentPlatform,
    AsyncPlatformAgent,
    CallableAdapter,
    MissingDependency,
    Outbound,
    PlatformAdapter,
    PlatformAgent,
    PlatformError,
    PlatformRegistry,
    PlatformTurn,
    available_platforms,
    create_platform,
    register_platform,
    registry,
    require,
)

__all__ = [
    "AdapterFactory",
    "AgentPlatform",
    "AsyncPlatformAgent",
    "CallableAdapter",
    "MissingDependency",
    "Outbound",
    "PlatformAdapter",
    "PlatformAgent",
    "PlatformError",
    "PlatformRegistry",
    "PlatformTurn",
    "available_platforms",
    "create_platform",
    "register_platform",
    "registry",
    "require",
]
