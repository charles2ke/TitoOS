"""``python -m titoos.gui`` - serve the dashboard for a small demo kernel."""

from __future__ import annotations

import argparse
from typing import Sequence

from ..agent import Agent, Context, RestartPolicy
from ..kernel import Kernel
from .server import KernelGUI


class Dispatcher(Agent):
    """Hands each job it receives to the workers in turn."""

    def __init__(self, name: str, workers: Sequence[str]) -> None:
        super().__init__(name)
        self.workers = tuple(workers)
        self.next = 0

    def step(self, ctx: Context) -> None:
        if ctx.tick == 1:
            for job in ("index", "resize", "upload"):
                ctx.send(self.name, job)
        for message in ctx.inbox:
            ctx.send(self.workers[self.next % len(self.workers)], message.payload)
            self.next += 1
        ctx.wait()


class Worker(Agent):
    """Processes a job and reports the result to the collector."""

    restart_policy = RestartPolicy.ON_FAILURE

    def __init__(self, name: str, flaky: bool = False) -> None:
        super().__init__(name)
        self.flaky = flaky

    def step(self, ctx: Context) -> None:
        for message in ctx.inbox:
            if self.flaky:
                self.flaky = False
                raise RuntimeError(f"transient failure on {message.payload!r}")
            ctx.send("collector", f"{message.payload}: done by {self.name}")
        ctx.wait()


class Collector(Agent):
    """Keeps every result it is sent."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.results: list[object] = []

    def step(self, ctx: Context) -> None:
        self.results.extend(m.payload for m in ctx.inbox)
        ctx.wait()


def demo_kernel() -> Kernel:
    """A dispatcher, two workers (one fails once and is restarted) and a collector."""
    kernel = Kernel()
    kernel.register(Dispatcher("dispatcher", ["worker-1", "worker-2"]))
    kernel.register(Worker("worker-1"), parent="dispatcher")
    kernel.register(Worker("worker-2", flaky=True), parent="dispatcher")
    kernel.register(Collector("collector"))
    return kernel


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m titoos.gui", description="Serve the TitoOS dashboard for a demo kernel."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    with KernelGUI(demo_kernel(), args.host, args.port) as gui:
        print(f"TitoOS GUI on {gui.url} - Ctrl+C to stop")
        try:
            gui.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":  # pragma: no cover
    main()
