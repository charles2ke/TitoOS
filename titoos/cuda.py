"""CUDA execution: agents that compute on GPUs, under the same tick barrier.

The kernel's own operations — scheduling a tick, routing a message — are a few
hundred nanoseconds of Python object handling each, less than launching one
CUDA kernel, so a GPU cannot make the kernel itself faster. What it can make
faster is what agents *compute*: model inference, embeddings, vector search.
:class:`CudaBackend` is the execution backend for such agents::

    from titoos import CudaBackend, Kernel

    with Kernel(backend=CudaBackend()) as kernel:
        kernel.spawn("embedder", embedder)
        kernel.run()

It adds three things and changes nothing else about scheduling:

* **Placement.** Each agent gets one CUDA device the first time it is
  scheduled and keeps it while it is alive — through waits and restarts — so
  whatever it keeps on the GPU stays where it put it. An agent can pin itself
  with a ``cuda_device`` attribute; the others go to the device with the
  fewest live agents, ties going to the one listed first.
* **Activation.** While an agent's ``step()`` runs, its device is the current
  CUDA device of the thread running it, so ``torch.ones(3, device="cuda")`` or
  a CuPy allocation lands there without the agent naming a device.
* **The barrier reaches the GPU.** A CUDA call returns once its work is
  queued, so a step can end long before its GPU work does. Before the tick
  ends the backend waits for every device used during it: a tensor sent in a
  message is fully computed before its recipient runs on the next tick, while
  within the tick queued work overlaps across devices and with the CPU. An
  asynchronous CUDA error surfaces in that wait and fails the agents that ran
  on the device during the tick, which are then supervised like any failure.

Agents on one device share their library's current stream there instead of
getting a stream each. With a stream per agent, the caching allocators of
PyTorch and CuPy could reuse a tensor's memory on the stream that allocated it
while another agent's stream is still reading it — exactly the situation a
tensor passed in a message creates.

The GPU library is imported when a backend is built, never by ``import
titoos``, which stays dependency-free.
"""

from __future__ import annotations

import operator
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ContextManager,
    Iterable,
    Protocol,
    Sequence,
)

from .agent import Agent
from .backends import ExecutionBackend
from .message import Message

if TYPE_CHECKING:  # pragma: no cover - typing only
    from types import ModuleType

    from .kernel import Kernel


class CudaUnavailableError(RuntimeError):
    """Raised when a :class:`CudaBackend` is built but no CUDA device is usable.

    Catch it to fall back to a CPU backend::

        try:
            backend = CudaBackend()
        except CudaUnavailableError:
            backend = ThreadBackend(8)
    """


class CudaDriver(Protocol):
    """What :class:`CudaBackend` needs from a GPU library.

    :class:`TorchDriver` and :class:`CupyDriver` implement it; anything else
    with the same methods can be passed as ``CudaBackend(driver=...)``.
    """

    #: Short name of the library, e.g. ``"torch"``.
    name: str

    def device_count(self) -> int:
        """How many CUDA devices the library can see; ``0`` if none."""
        ...

    def use_device(self, index: int) -> ContextManager[Any]:
        """Make device ``index`` current for the calling thread, then restore."""
        ...

    def synchronize(self, index: int) -> None:
        """Block until all work queued on device ``index`` has finished."""
        ...

    def describe(self, index: int) -> dict[str, Any]:
        """Plain facts about device ``index``, e.g. its name and memory."""
        ...


def _require(module: str, library: str, install: str) -> ModuleType:
    try:
        return import_module(module)
    except Exception as exc:  # noqa: BLE001 - a broken install is as unusable as a missing one
        raise CudaUnavailableError(
            f"{library} is not usable ({exc}); install it with: {install}"
        ) from exc


class TorchDriver:
    """Runs agents on the CUDA devices PyTorch sees."""

    name = "torch"

    def __init__(self) -> None:
        self._cuda = _require("torch", "PyTorch", "pip install torch").cuda

    def device_count(self) -> int:
        return self._cuda.device_count() if self._cuda.is_available() else 0

    def use_device(self, index: int) -> ContextManager[Any]:
        return self._cuda.device(index)

    def synchronize(self, index: int) -> None:
        self._cuda.synchronize(index)

    def describe(self, index: int) -> dict[str, Any]:
        properties = self._cuda.get_device_properties(index)
        return {
            "name": properties.name,
            "memory": properties.total_memory,
            "capability": f"{properties.major}.{properties.minor}",
        }


class CupyDriver:
    """Runs agents on the CUDA devices CuPy sees."""

    name = "cupy"

    def __init__(self) -> None:
        self._cuda = _require(
            "cupy", "CuPy", "pip install cupy-cuda12x (match your CUDA version)"
        ).cuda

    def device_count(self) -> int:
        try:
            return self._cuda.runtime.getDeviceCount()
        except self._cuda.runtime.CUDARuntimeError:
            return 0

    def use_device(self, index: int) -> ContextManager[Any]:
        return self._cuda.Device(index)

    def synchronize(self, index: int) -> None:
        self._cuda.Device(index).synchronize()

    def describe(self, index: int) -> dict[str, Any]:
        properties = self._cuda.runtime.getDeviceProperties(index)
        name = properties["name"]
        return {
            "name": name.decode() if isinstance(name, bytes) else name,
            "memory": properties["totalGlobalMem"],
            "capability": f"{properties['major']}.{properties['minor']}",
        }


#: Drivers selectable by name, in the order ``driver="auto"`` tries them.
DRIVERS: dict[str, Callable[[], CudaDriver]] = {
    "torch": TorchDriver,
    "cupy": CupyDriver,
}


_DRIVER_API = ("name", "device_count", "use_device", "synchronize", "describe")


def _resolve_driver(driver: str | CudaDriver) -> CudaDriver:
    if not isinstance(driver, str):
        missing = [attr for attr in _DRIVER_API if not hasattr(driver, attr)]
        if missing:
            raise TypeError(f"CUDA driver {driver!r} lacks: {', '.join(missing)}")
        return driver
    if driver != "auto":
        factory = DRIVERS.get(driver)
        if factory is None:
            raise ValueError(
                f"unknown CUDA driver {driver!r}; choose from: auto, {', '.join(DRIVERS)}"
            )
        return factory()
    reasons = []
    for name, factory in DRIVERS.items():
        try:
            candidate = factory()
        except CudaUnavailableError as exc:
            reasons.append(f"{name}: {exc}")
            continue
        if candidate.device_count() > 0:
            return candidate
        reasons.append(f"{name}: no CUDA device visible")
    raise CudaUnavailableError(
        "no CUDA device is available:" + "".join(f"\n  {r}" for r in reasons)
    )


def _device_index(value: object) -> int | None:
    """``value`` as a device index, or ``None`` if it is not an integer."""
    if isinstance(value, bool):
        return None
    try:
        return operator.index(value)
    except TypeError:
        return None


class CudaBackend(ExecutionBackend):
    """Runs each agent with its own CUDA device current, and waits for the GPUs.

    ``devices`` lists the device indices agents may be placed on, every
    visible device by default. ``driver`` names the GPU library — ``"torch"``,
    ``"cupy"``, or ``"auto"`` for the first of those that sees a device — and
    should be the library the agents use.

    ``max_workers > 1`` runs the agents of a tick in a thread pool, as
    :class:`~titoos.backends.ThreadBackend` does, so their CPU-side work
    overlaps too. Agents on a device then still share one stream unless the
    library gives each thread its own default stream, so leave that off (for
    CuPy, ``CUPY_CUDA_PER_THREAD_DEFAULT_STREAM``).

    Raises :class:`CudaUnavailableError` if no CUDA device can be used.
    """

    def __init__(
        self,
        devices: Iterable[int] | None = None,
        *,
        driver: str | CudaDriver = "auto",
        max_workers: int = 1,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        self.driver = _resolve_driver(driver)
        count = self.driver.device_count()
        if count < 1:
            raise CudaUnavailableError(f"{self.driver.name} sees no CUDA device")
        listed = tuple(range(count)) if devices is None else tuple(devices)
        if not listed:
            raise ValueError("devices must list at least one CUDA device")
        indices = [_device_index(d) for d in listed]
        invalid = [d for d, i in zip(listed, indices) if i is None or not 0 <= i < count]
        if invalid:
            raise ValueError(
                f"not a visible CUDA device: {', '.join(map(repr, invalid))}; "
                f"{self.driver.name} sees devices 0-{count - 1}"
            )
        if len(set(indices)) != len(indices):
            raise ValueError("devices must not repeat")
        #: The device indices agents are placed on, in preference order.
        self.devices: tuple[int, ...] = tuple(indices)  # type: ignore[arg-type]
        self.max_workers = max_workers
        self._pool: ThreadPoolExecutor | None = None
        #: Agent name -> (the agent, its device). The agent is kept so that a
        #: different agent registered later under the same name is placed anew.
        self._placement: dict[str, tuple[Agent, int]] = {}
        self._lock = threading.Lock()

    def device_of(self, name: str) -> int | None:
        """The device the live agent named ``name`` runs on, if it was placed."""
        with self._lock:
            entry = self._placement.get(name)
        if entry is None or not entry[0].is_alive:
            return None
        return entry[1]

    def describe(self) -> dict[str, Any]:
        """The driver, and each device with the live agents placed on it."""
        with self._lock:
            entries = tuple(self._placement.items())
        agents: dict[int, list[str]] = {device: [] for device in self.devices}
        for name, (agent, device) in entries:
            if agent.is_alive:
                agents[device].append(name)
        return {
            "driver": self.driver.name,
            "max_workers": self.max_workers,
            "devices": [
                {"index": d, **self.driver.describe(d), "agents": agents[d]}
                for d in self.devices
            ],
        }

    def run_tick(
        self,
        kernel: Kernel,
        agents: Sequence[Agent],
        inboxes: Sequence[list[Message]],
        tick: int,
    ) -> list[bool]:
        self._reject_async(agents)
        devices = self._place(kernel, agents)
        # An agent that could not be placed has already been failed.
        outcomes = [False] * len(agents)
        jobs = [i for i, device in enumerate(devices) if device is not None]
        if self.max_workers > 1 and len(jobs) > 1:
            pool = self._executor()
            futures = [
                pool.submit(self._run, kernel, agents[i], inboxes[i], devices[i], tick)
                for i in jobs
            ]
            # Collecting in submission order keeps the outcomes aligned with
            # registration order regardless of which worker finishes first.
            for i, future in zip(jobs, futures):
                outcomes[i] = future.result()
        else:
            for i in jobs:
                outcomes[i] = self._run(kernel, agents[i], inboxes[i], devices[i], tick)
        self._synchronize(kernel, agents, devices, outcomes)
        return outcomes

    def _place(self, kernel: Kernel, agents: Sequence[Agent]) -> list[int | None]:
        """Each agent's device, placing the agents scheduled for the first time.

        Runs at the tick barrier in registration order, so placement does not
        depend on worker timing. An agent pinned to a device this backend does
        not run on is failed instead, and gets ``None``.
        """
        devices: list[int | None] = []
        rejected: list[tuple[Agent, ValueError]] = []
        with self._lock:
            placement = self._placement
            load: Counter[int] | None = None
            for agent in agents:
                entry = placement.get(agent.name)
                if entry is not None and entry[0] is agent:
                    devices.append(entry[1])
                    continue
                if load is None:
                    load = self._release(kernel)
                try:
                    device = self._choose(agent, load)
                except ValueError as exc:
                    rejected.append((agent, exc))
                    devices.append(None)
                    continue
                placement[agent.name] = (agent, device)
                load[device] += 1
                devices.append(device)
        for agent, exc in rejected:
            kernel._record_failure(agent, exc)
        return devices

    def _release(self, kernel: Kernel) -> Counter[int]:
        """Forget agents that are gone and count the rest per device.

        Called with the lock held, and only when an agent needs a device:
        placing is rare, so it is also when finished, failed, unregistered or
        replaced agents give their device back.
        """
        current = {agent.name: agent for agent in kernel.live_agents()}
        placement = self._placement
        for name, (agent, _) in tuple(placement.items()):
            if current.get(name) is not agent:
                del placement[name]
        return Counter(device for _, device in placement.values())

    def _choose(self, agent: Agent, load: Counter[int]) -> int:
        pinned = getattr(agent, "cuda_device", None)
        if pinned is None:
            # min() keeps the first of equally loaded devices.
            return min(self.devices, key=load.__getitem__)
        index = _device_index(pinned)
        if index is None or index not in self.devices:
            raise ValueError(
                f"agent {agent.name!r} is pinned to CUDA device {pinned!r}, but "
                f"this backend runs on devices {', '.join(map(str, self.devices))}"
            )
        return index

    def _run(
        self, kernel: Kernel, agent: Agent, inbox: list[Message], device: int, tick: int
    ) -> bool:
        outcome: bool | None = None
        try:
            with self.driver.use_device(device):
                outcome = kernel._run_agent(agent, tick, inbox)
        except Exception as exc:  # noqa: BLE001 - a device that cannot be selected fails the agent, not the OS
            if outcome is False:
                return False  # the step's own failure is already recorded
            return kernel._record_failure(agent, exc)
        return outcome

    def _synchronize(
        self,
        kernel: Kernel,
        agents: Sequence[Agent],
        devices: Sequence[int | None],
        outcomes: list[bool],
    ) -> None:
        """Wait for every device used this tick, in the order they are listed.

        A device that fails to synchronize reports an error from work queued
        during the tick, which cannot be traced to one agent, so every agent
        that ran on it and had not failed already is failed with that error.
        """
        used = set(devices)
        for device in self.devices:
            if device not in used:
                continue
            try:
                self.driver.synchronize(device)
            except Exception as exc:  # noqa: BLE001 - an asynchronous CUDA error fails agents, not the OS
                for i, placed in enumerate(devices):
                    if placed == device and outcomes[i]:
                        outcomes[i] = kernel._record_failure(agents[i], exc)

    def _executor(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=self.max_workers, thread_name_prefix="titoos-cuda"
            )
        return self._pool

    def shutdown(self, wait: bool = True) -> None:
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=wait)
