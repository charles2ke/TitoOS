"""Filesystem integration confined to a single directory."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base import Integration


class FileSystemIntegration(Integration):
    """Read and write files below one directory and nowhere else.

    Every path is resolved and checked against ``root``, so ``"../../etc"`` or
    a symlink pointing outside the sandbox is rejected rather than followed.
    Paths are always interpreted relative to ``root``; absolute paths are an
    error, not an escape hatch.
    """

    name = "files"
    operations = (
        "read_text",
        "write_text",
        "append_text",
        "list_dir",
        "exists",
        "delete",
        "describe",
    )

    def __init__(
        self,
        root: "str | os.PathLike[str]",
        *,
        name: str | None = None,
        read_only: bool = False,
        max_bytes: int = 1 << 20,
    ) -> None:
        super().__init__(name)
        resolved = Path(root).expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"root is not an existing directory: {resolved}")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.root = resolved
        self.read_only = read_only
        self.max_bytes = max_bytes

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "root": str(self.root),
            "read_only": self.read_only,
            "max_bytes": self.max_bytes,
        }

    def _resolve(self, path: str, operation: str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute() or candidate.drive or candidate.root:
            raise self._fail(f"path must be relative to the root: {path!r}", operation)
        resolved = (self.root / candidate).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise self._fail(
                f"path escapes the sandbox root {self.root}: {path!r}", operation
            )
        return resolved

    def _writable(self, path: str, operation: str) -> Path:
        if self.read_only:
            raise self._fail(f"integration {self.name!r} is read-only", operation)
        return self._resolve(path, operation)

    def read_text(self, path: str, *, encoding: str = "utf-8") -> str:
        """Return the contents of ``path``."""
        target = self._resolve(path, "read_text")
        try:
            if target.stat().st_size > self.max_bytes:
                raise self._fail(
                    f"{path!r} is larger than max_bytes ({self.max_bytes} bytes)",
                    "read_text",
                )
            return target.read_text(encoding=encoding)
        except OSError as exc:
            raise self._fail(f"cannot read {path!r}: {exc}", "read_text") from exc

    def write_text(self, path: str, content: str, *, encoding: str = "utf-8") -> int:
        """Write ``content`` to ``path``, creating parent directories."""
        target = self._writable(path, "write_text")
        if len(content.encode(encoding)) > self.max_bytes:
            raise self._fail(
                f"content is larger than max_bytes ({self.max_bytes} bytes)",
                "write_text",
            )
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            return target.write_text(content, encoding=encoding)
        except OSError as exc:
            raise self._fail(f"cannot write {path!r}: {exc}", "write_text") from exc

    def append_text(self, path: str, content: str, *, encoding: str = "utf-8") -> int:
        """Append ``content`` to ``path``, creating it if needed."""
        target = self._writable(path, "append_text")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding=encoding) as handle:
                return handle.write(content)
        except OSError as exc:
            raise self._fail(f"cannot append to {path!r}: {exc}", "append_text") from exc

    def list_dir(self, path: str = ".") -> list[str]:
        """List the entries of a directory, as paths relative to the root."""
        target = self._resolve(path, "list_dir")
        try:
            return sorted(
                str(entry.relative_to(self.root)) for entry in target.iterdir()
            )
        except OSError as exc:
            raise self._fail(f"cannot list {path!r}: {exc}", "list_dir") from exc

    def exists(self, path: str) -> bool:
        return self._resolve(path, "exists").exists()

    def delete(self, path: str, *, missing_ok: bool = True) -> bool:
        """Delete a file. Directories are not removed."""
        target = self._writable(path, "delete")
        try:
            target.unlink()
            return True
        except FileNotFoundError:
            if missing_ok:
                return False
            raise self._fail(f"no such file: {path!r}", "delete") from None
        except OSError as exc:
            raise self._fail(f"cannot delete {path!r}: {exc}", "delete") from exc
