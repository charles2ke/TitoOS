"""Subprocess integration limited to an explicit list of executables."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .base import Integration, normalize_allowlist

#: Environment handed to a subprocess when the operator names none. Deliberately
#: minimal: whatever secrets live in the kernel's own environment are not the
#: agent's to read.
DEFAULT_ENV: Mapping[str, str] = {
    "PATH": os.pathsep.join(part for part in os.defpath.split(os.pathsep) if part)
}

#: Size of a single read while draining a child's output.
_CHUNK = 8192

#: How long to wait for a drain thread after the child is gone. A grandchild
#: that inherited the pipes can hold them open forever, and the timeout of
#: :meth:`ShellIntegration.run` has to mean something.
_DRAIN_GRACE = 1.0


@dataclass(frozen=True)
class CommandResult:
    """The outcome of running a command."""

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": list(self.command),
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


class ShellIntegration(Integration):
    """Run a fixed set of programs, with arguments passed as a list.

    There is no shell: commands are given as argument vectors and executed
    directly, so quoting and metacharacters in an agent-produced argument are
    data rather than syntax. Only the executables named in ``allowed_commands``
    can be started, matched on the program's base name.
    """

    name = "shell"
    operations = ("run", "describe")

    def __init__(
        self,
        allowed_commands: Sequence[str],
        *,
        name: str | None = None,
        cwd: "str | os.PathLike[str] | None" = None,
        timeout: float = 30.0,
        env: Mapping[str, str] | None = None,
        max_output: int = 1 << 20,
    ) -> None:
        super().__init__(name)
        self.allowed_commands = normalize_allowlist(
            allowed_commands, "allowed_commands"
        )
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_output <= 0:
            raise ValueError("max_output must be positive")
        self.cwd = Path(cwd).expanduser().resolve() if cwd is not None else None
        if self.cwd is not None and not self.cwd.is_dir():
            raise ValueError(f"cwd is not an existing directory: {self.cwd}")
        self.timeout = timeout
        self.max_output = max_output
        # An explicit environment keeps the agent's subprocesses from
        # inheriting whatever secrets happen to live in the parent's: an
        # omitted env means a minimal one, never the kernel's own.
        self.env = dict(env) if env is not None else dict(DEFAULT_ENV)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "allowed_commands": sorted(self.allowed_commands),
            "cwd": str(self.cwd) if self.cwd else None,
            "timeout": self.timeout,
        }

    def _resolve(self, command: Sequence[str]) -> list[str]:
        if isinstance(command, (str, bytes)):
            raise self._fail(
                "command must be a list of arguments, not a string; there is no "
                "shell to split it",
                "run",
            )
        argv = [str(part) for part in command]
        if not argv:
            raise self._fail("command must not be empty", "run")
        program = Path(argv[0]).name.lower()
        if program not in self.allowed_commands:
            raise self._fail(
                f"command {program!r} is not in the allowlist "
                f"({', '.join(sorted(self.allowed_commands))})",
                "run",
            )
        executable = shutil.which(argv[0])
        if executable is None:
            raise self._fail(f"executable not found: {argv[0]!r}", "run")
        return [executable, *argv[1:]]

    def run(
        self,
        command: Sequence[str],
        *,
        stdin: str | None = None,
        timeout: float | None = None,
        check: bool = False,
    ) -> CommandResult:
        """Run ``command`` and return its result.

        A non-zero exit status is reported in the result rather than raised,
        unless ``check`` is set: an agent usually wants to inspect stderr.
        """
        argv = self._resolve(command)
        limit = timeout or self.timeout
        try:
            process = subprocess.Popen(  # noqa: S603 - argv is allowlisted and shell=False
                argv,
                stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.cwd) if self.cwd else None,
                env=self.env,
                shell=False,
            )
        except OSError as exc:
            raise self._fail(f"cannot run {argv[0]!r}: {exc}", "run") from exc
        timed_out = False
        with process:
            # Drained on separate threads into buffers that stop growing at
            # max_output, so a chatty command cannot exhaust this process
            # while the rest of its output is still discarded, not buffered.
            readers = [
                _BoundedReader(process.stdout, self.max_output),
                _BoundedReader(process.stderr, self.max_output),
            ]
            for reader in readers:
                reader.start()
            try:
                if stdin is not None and process.stdin is not None:
                    try:
                        process.stdin.write(stdin)
                    except OSError:  # the child exited before reading it all
                        pass
                    finally:
                        process.stdin.close()
                process.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                process.wait()
            finally:
                for reader in readers:
                    reader.join(timeout=_DRAIN_GRACE)
        if timed_out:
            raise self._fail(
                f"command timed out after {limit}s: {' '.join(argv)}", "run"
            )
        stdout, stderr = (reader.text for reader in readers)
        result = CommandResult(
            command=tuple(argv),
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
        )
        if check and not result.ok:
            raise self._fail(
                f"command failed with exit code {result.returncode}: "
                f"{result.stderr.strip() or ' '.join(argv)}",
                "run",
            )
        return result


class _BoundedReader(threading.Thread):
    """Drains a child's pipe, keeping at most ``limit`` characters.

    The pipe has to be read to the end or the child blocks once its buffer
    fills, but nothing past the limit is kept, so the cap bounds this process's
    memory rather than merely truncating what was already buffered.
    """

    def __init__(self, stream: Any, limit: int) -> None:
        super().__init__(daemon=True)
        self._stream = stream
        self._limit = limit
        self._chunks: list[str] = []
        self._kept = 0
        # A join that times out leaves this thread running, so the buffer is
        # shared with the caller reading `text`.
        self._lock = threading.Lock()

    def run(self) -> None:
        try:
            while True:
                chunk = self._stream.read(_CHUNK)
                if not chunk:
                    break
                with self._lock:
                    room = self._limit - self._kept
                    if room > 0:
                        kept = chunk[:room]
                        self._chunks.append(kept)
                        self._kept += len(kept)
        except (OSError, ValueError):  # the pipe was closed under us
            pass

    @property
    def text(self) -> str:
        """What was captured so far, capped at ``limit`` characters."""
        with self._lock:
            return "".join(list(self._chunks))
