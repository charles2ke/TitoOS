"""Subprocess integration limited to an explicit list of executables."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .base import Integration, normalize_allowlist


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
        # inheriting whatever secrets happen to live in the parent's.
        self.env = dict(env) if env is not None else None

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
        try:
            completed = subprocess.run(  # noqa: S603 - argv is allowlisted and shell=False
                argv,
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout or self.timeout,
                cwd=str(self.cwd) if self.cwd else None,
                env=self.env,
                shell=False,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise self._fail(
                f"command timed out after {exc.timeout}s: {' '.join(argv)}", "run"
            ) from exc
        except OSError as exc:
            raise self._fail(f"cannot run {argv[0]!r}: {exc}", "run") from exc
        result = CommandResult(
            command=tuple(argv),
            returncode=completed.returncode,
            stdout=(completed.stdout or "")[: self.max_output],
            stderr=(completed.stderr or "")[: self.max_output],
        )
        if check and not result.ok:
            raise self._fail(
                f"command failed with exit code {result.returncode}: "
                f"{result.stderr.strip() or ' '.join(argv)}",
                "run",
            )
        return result
