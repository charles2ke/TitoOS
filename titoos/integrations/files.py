"""Filesystem integration confined to a single directory."""

from __future__ import annotations

import errno
import os
import stat
from contextlib import contextmanager
from pathlib import Path, PurePath
from typing import Any, Iterator

from .base import Integration

_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_SANDBOX_SUPPORTED = (
    {os.open, os.stat, os.mkdir, os.unlink} <= os.supports_dir_fd
    and hasattr(os, "O_NOFOLLOW")
    and hasattr(os, "O_DIRECTORY")
)


class FileSystemIntegration(Integration):
    """Read and write files below one directory and nowhere else.

    Paths are walked one component at a time, each opened relative to the file
    descriptor of its parent and never following symlinks, so ``"../../etc"``,
    a symlink pointing outside the sandbox, or a directory swapped for a
    symlink between the check and the read is rejected rather than followed.
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
        if not _SANDBOX_SUPPORTED:
            raise ValueError(
                "this platform lacks directory descriptors or O_NOFOLLOW, so "
                "the sandbox cannot be enforced safely"
            )
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

    def _parts(self, path: str, operation: str) -> list[str]:
        """Split a caller path into components, rejecting escapes up front."""
        candidate = PurePath(path)
        if candidate.is_absolute() or candidate.drive or candidate.root:
            raise self._fail(f"path must be relative to the root: {path!r}", operation)
        parts = [part for part in candidate.parts if part not in (".", "")]
        if any(part == ".." for part in parts):
            raise self._fail(
                f"path escapes the sandbox root {self.root}: {path!r}", operation
            )
        return parts

    def _writable(self, path: str, operation: str) -> list[str]:
        if self.read_only:
            raise self._fail(f"integration {self.name!r} is read-only", operation)
        return self._parts(path, operation)

    @contextmanager
    def _walk(
        self,
        parts: list[str],
        path: str,
        operation: str,
        *,
        create: bool = False,
        include_last: bool = False,
    ) -> Iterator[int]:
        """Yield a descriptor for the directory holding ``parts[-1]``.

        Each component is opened relative to the descriptor of its parent with
        ``O_NOFOLLOW``, so the sandbox cannot be escaped by swapping a checked
        directory for a symlink between the check and the operation. With
        ``include_last`` the final component is opened as a directory too.
        """
        flags = os.O_RDONLY | _O_DIRECTORY | _O_CLOEXEC
        components = parts if include_last else parts[:-1]
        fd = os.open(self.root, flags)
        try:
            for part in components:
                if create:
                    try:
                        os.mkdir(part, dir_fd=fd)
                    except FileExistsError:
                        pass
                try:
                    nxt = os.open(part, flags | _O_NOFOLLOW, dir_fd=fd)
                except OSError as exc:
                    raise self._describe_oserror(
                        exc, path, operation, part=part, dir_fd=fd
                    ) from exc
                os.close(fd)
                fd = nxt
            yield fd
        finally:
            os.close(fd)

    def _describe_oserror(
        self,
        exc: OSError,
        path: str,
        operation: str,
        *,
        part: str | None = None,
        dir_fd: int | None = None,
    ) -> Exception:
        """Translate a no-follow failure into a sandbox error, else pass through."""
        escaped = exc.errno == errno.ELOOP
        if not escaped and exc.errno == errno.ENOTDIR and part is not None:
            # Linux reports ENOTDIR rather than ELOOP when O_DIRECTORY and
            # O_NOFOLLOW meet a symlink; tell the two cases apart.
            try:
                escaped = stat.S_ISLNK(
                    os.stat(part, dir_fd=dir_fd, follow_symlinks=False).st_mode
                )
            except OSError:
                escaped = False
        if escaped:
            return self._fail(
                f"path escapes the sandbox root {self.root}: {path!r}", operation
            )
        return exc

    @contextmanager
    def _open(
        self,
        path: str,
        operation: str,
        flags: int,
        mode: str,
        *,
        encoding: str,
        writable: bool = False,
    ) -> Iterator[Any]:
        """Open a sandboxed file without ever following a symlink."""
        if writable:
            parts = self._writable(path, operation)
        else:
            parts = self._parts(path, operation)
        if not parts:
            raise self._fail(f"{path!r} is not a file", operation)
        with self._walk(parts, path, operation, create=writable) as dir_fd:
            try:
                fd = os.open(
                    parts[-1], flags | _O_NOFOLLOW | _O_CLOEXEC, 0o600, dir_fd=dir_fd
                )
            except OSError as exc:
                raise self._describe_oserror(exc, path, operation) from exc
            try:
                handle = os.fdopen(fd, mode, encoding=encoding)
            except Exception:
                os.close(fd)
                raise
            with handle:
                yield handle

    def read_text(self, path: str, *, encoding: str = "utf-8") -> str:
        """Return the contents of ``path``."""
        try:
            with self._open(
                path, "read_text", os.O_RDONLY, "r", encoding=encoding
            ) as handle:
                if os.fstat(handle.fileno()).st_size > self.max_bytes:
                    raise self._fail(
                        f"{path!r} is larger than max_bytes ({self.max_bytes} bytes)",
                        "read_text",
                    )
                return handle.read()
        except OSError as exc:
            raise self._fail(f"cannot read {path!r}: {exc}", "read_text") from exc

    def write_text(self, path: str, content: str, *, encoding: str = "utf-8") -> int:
        """Write ``content`` to ``path``, creating parent directories."""
        self._writable(path, "write_text")
        if len(content.encode(encoding)) > self.max_bytes:
            raise self._fail(
                f"content is larger than max_bytes ({self.max_bytes} bytes)",
                "write_text",
            )
        try:
            with self._open(
                path,
                "write_text",
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                "w",
                encoding=encoding,
                writable=True,
            ) as handle:
                return handle.write(content)
        except OSError as exc:
            raise self._fail(f"cannot write {path!r}: {exc}", "write_text") from exc

    def append_text(self, path: str, content: str, *, encoding: str = "utf-8") -> int:
        """Append ``content`` to ``path``, creating it if needed."""
        try:
            with self._open(
                path,
                "append_text",
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                "a",
                encoding=encoding,
                writable=True,
            ) as handle:
                return handle.write(content)
        except OSError as exc:
            raise self._fail(f"cannot append to {path!r}: {exc}", "append_text") from exc

    def list_dir(self, path: str = ".") -> list[str]:
        """List the entries of a directory, as paths relative to the root."""
        parts = self._parts(path, "list_dir")
        prefix = PurePath(*parts) if parts else None
        try:
            with self._walk(parts, path, "list_dir", include_last=True) as dir_fd:
                entries = os.listdir(dir_fd)
        except OSError as exc:
            raise self._fail(f"cannot list {path!r}: {exc}", "list_dir") from exc
        return sorted(
            str(prefix / entry) if prefix is not None else entry for entry in entries
        )

    def exists(self, path: str) -> bool:
        parts = self._parts(path, "exists")
        if not parts:
            return True
        try:
            with self._walk(parts, path, "exists") as dir_fd:
                info = os.stat(parts[-1], dir_fd=dir_fd, follow_symlinks=False)
        except (FileNotFoundError, NotADirectoryError):
            return False
        except OSError as exc:
            raise self._fail(f"cannot stat {path!r}: {exc}", "exists") from exc
        if stat.S_ISLNK(info.st_mode):
            raise self._fail(
                f"path escapes the sandbox root {self.root}: {path!r}", "exists"
            )
        return True

    def delete(self, path: str, *, missing_ok: bool = True) -> bool:
        """Delete a file. Directories are not removed."""
        parts = self._writable(path, "delete")
        if not parts:
            raise self._fail(f"{path!r} is not a file", "delete")
        try:
            with self._walk(parts, path, "delete") as dir_fd:
                os.unlink(parts[-1], dir_fd=dir_fd)
        except (FileNotFoundError, NotADirectoryError):
            if missing_ok:
                return False
            raise self._fail(f"no such file: {path!r}", "delete") from None
        except OSError as exc:
            raise self._fail(f"cannot delete {path!r}: {exc}", "delete") from exc
        return True
