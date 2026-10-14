"""A browser-based GUI layer for TitoOS.

Dependency-free like the rest of the package: :class:`KernelGUI` serves a
dashboard and a small JSON API from the standard library's HTTP server.
``python -m titoos.gui`` starts it on a demo kernel.
"""

from .server import GUIError, KernelGUI

__all__ = ["GUIError", "KernelGUI"]
