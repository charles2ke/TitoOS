import contextlib
import subprocess
import sys
import threading
import weakref

import pytest

from titoos import (
    Agent,
    AgentState,
    CudaBackend,
    CudaUnavailableError,
    Kernel,
    RestartPolicy,
    cuda,
)


class FakeDriver:
    """A CUDA library without a GPU: a current device per thread, and work
    that is queued when launched and only runs when its device synchronizes."""

    name = "fake"

    def __init__(self, count=2):
        self.count = count
        self.synced = []
        #: Devices whose next synchronize raises, as an asynchronous error would.
        self.fail_sync = set()
        self._queued = {device: [] for device in range(count)}
        self._local = threading.local()
        self._lock = threading.Lock()

    def device_count(self):
        return self.count

    def current(self):
        return getattr(self._local, "device", None)

    @contextlib.contextmanager
    def use_device(self, index):
        previous = self.current()
        self._local.device = index
        try:
            yield
        finally:
            self._local.device = previous

    def launch(self, work):
        """Queue ``work`` on the current device, as a kernel launch does."""
        device = self.current()
        assert device is not None, "no CUDA device is current"
        with self._lock:
            self._queued[device].append(work)

    def synchronize(self, index):
        self.synced.append(index)
        with self._lock:
            work, self._queued[index] = self._queued[index], []
        for item in work:
            item()
        if index in self.fail_sync:
            self.fail_sync.discard(index)
            raise RuntimeError(f"CUDA error: illegal memory access on device {index}")

    def describe(self, index):
        return {"name": f"Fake GPU {index}", "memory": 2**30, "capability": "9.0"}


class Recorder(Agent):
    """Records the current device on every step; waits, or exits after ``steps``."""

    def __init__(self, name, driver, steps=None, cuda_device=None):
        super().__init__(name)
        self.driver = driver
        self.steps = steps
        self.seen = []
        if cuda_device is not None:
            self.cuda_device = cuda_device

    def step(self, ctx):
        self.seen.append((self.driver.current(), [m.payload for m in ctx.inbox]))
        if self.steps is None:
            ctx.wait()
        elif len(self.seen) >= self.steps:
            ctx.exit()


def make_kernel(driver, **options):
    return Kernel(backend=CudaBackend(driver=driver, **options))


def test_agents_are_spread_over_devices_and_keep_them():
    driver = FakeDriver(2)
    kernel = make_kernel(driver)
    agents = [kernel.register(Recorder(name, driver, steps=3)) for name in "abc"]

    kernel.run()

    assert [{device for device, _ in agent.seen} for agent in agents] == [{0}, {1}, {0}]
    assert driver.current() is None


def test_placement_prefers_the_least_loaded_then_the_first_listed_device():
    driver = FakeDriver(3)
    kernel = make_kernel(driver)
    backend = kernel.backend
    kernel.register(Recorder("pinned", driver, cuda_device=0))
    kernel.register(Recorder("b", driver))
    kernel.register(Recorder("c", driver))
    kernel.register(Recorder("d", driver))

    kernel.step()

    assert [backend.device_of(name) for name in ("pinned", "b", "c", "d")] == [0, 1, 2, 0]
    assert backend.device_of("nobody") is None
    described = backend.describe()
    assert described["driver"] == "fake"
    assert [d["agents"] for d in described["devices"]] == [["pinned", "d"], ["b"], ["c"]]
    assert described["devices"][1]["name"] == "Fake GPU 1"


def test_device_order_sets_the_preference():
    driver = FakeDriver(2)
    kernel = make_kernel(driver, devices=(1, 0))
    kernel.register(Recorder("a", driver))

    kernel.step()

    assert kernel.backend.device_of("a") == 1


def test_agents_that_are_gone_give_their_device_back():
    driver = FakeDriver(2)
    kernel = make_kernel(driver)
    backend = kernel.backend
    kernel.register(Recorder("done", driver, steps=1))
    kernel.register(Recorder("stays", driver))
    kernel.register(Recorder("replaced", driver))
    kernel.step()
    assert kernel.get("done").seen == [(0, [])]
    assert [backend.device_of(n) for n in ("done", "stays", "replaced")] == [None, 1, 0]

    kernel.unregister("replaced")
    replacement = kernel.register(Recorder("replaced", driver, steps=1))
    kernel.register(Recorder("new", driver, steps=1))
    kernel.step()

    # "done" and the old "replaced" are gone, which frees device 0 again.
    assert replacement.seen == [(0, [])]
    assert kernel.get("new").seen == [(0, [])]
    assert [d["agents"] for d in backend.describe()["devices"]] == [[], ["stays"]]


@pytest.mark.parametrize("keep_waiting_agent", [False, True])
def test_empty_ticks_release_unregistered_agents(keep_waiting_agent):
    driver = FakeDriver(2)
    kernel = make_kernel(driver)
    backend = kernel.backend
    removed = kernel.register(Recorder("removed", driver))
    reference = weakref.ref(removed)
    if keep_waiting_agent:
        kernel.register(Recorder("waiting", driver))
    kernel.step()
    kernel.unregister("removed")
    del removed

    report = kernel.step()

    assert report.ran == ()
    assert backend.device_of("removed") is None
    assert reference() is None
    assert [d["agents"] for d in backend.describe()["devices"]] == [
        [], ["waiting"] if keep_waiting_agent else []
    ]
    if keep_waiting_agent:
        assert backend.device_of("waiting") == 1


@pytest.mark.parametrize("max_workers", [1, 2])
@pytest.mark.parametrize("wait", [False, True])
def test_shutdown_releases_agent_references(max_workers, wait):
    driver = FakeDriver(2)
    kernel = make_kernel(driver, max_workers=max_workers)
    backend = kernel.backend
    removed = kernel.register(Recorder("removed", driver))
    reference = weakref.ref(removed)
    kernel.register(Recorder("waiting", driver))
    kernel.step()
    kernel.unregister("removed")
    del removed

    backend.shutdown(wait=wait)

    assert backend._pool is None
    assert reference() is None
    assert backend.device_of("removed") is None
    assert backend.device_of("waiting") is None
    assert [d["agents"] for d in backend.describe()["devices"]] == [[], []]
    backend.shutdown(wait=wait)


def test_gpu_work_finishes_before_the_next_tick():
    driver = FakeDriver(2)
    kernel = make_kernel(driver)
    received = []

    def producer(ctx):
        tensor = []
        driver.launch(lambda: tensor.extend([1, 2, 3]))
        ctx.send("consumer", tensor)
        ctx.exit()

    def consumer(ctx):
        received.extend(list(m.payload) for m in ctx.inbox)
        if received:
            ctx.exit()
        else:
            ctx.wait()

    kernel.spawn("producer", producer)
    kernel.spawn("consumer", consumer)
    kernel.run()

    assert received == [[1, 2, 3]]
    # Tick 1 used both devices; tick 2 only the consumer's.
    assert driver.synced == [0, 1, 1]


def test_an_asynchronous_error_fails_the_agents_that_ran_on_the_device():
    driver = FakeDriver(2)
    kernel = make_kernel(driver)

    def broken(ctx):
        raise ValueError("own failure")

    kernel.register(Recorder("a", driver))
    kernel.register(Recorder("b", driver))
    kernel.spawn("broken", broken).cuda_device = 1
    driver.fail_sync = {1}

    report = kernel.step()

    assert report.ran == ("a",)
    assert report.failed == ("b", "broken")
    assert kernel.get("b").state is AgentState.FAILED
    # The agent that failed by itself keeps its own error.
    assert [(name, type(exc)) for name, exc in kernel.errors] == [
        ("broken", ValueError),
        ("b", RuntimeError),
    ]


def test_a_restarted_agent_keeps_its_device_and_gets_its_messages_back():
    driver = FakeDriver(2)
    kernel = make_kernel(driver)

    def sender(ctx):
        ctx.send("worker", "job")
        ctx.exit()

    worker = Recorder("worker", driver, cuda_device=1)
    worker.restart_policy = RestartPolicy.ON_FAILURE
    kernel.spawn("sender", sender)
    kernel.register(worker)

    kernel.step()
    driver.fail_sync = {1}
    failed = kernel.step()
    kernel.step()

    assert failed.failed == failed.restarted == ("worker",)
    assert worker.seen == [(1, []), (1, ["job"]), (1, ["job"])]
    assert kernel.backend.device_of("worker") == 1


@pytest.mark.parametrize("pin", [2, True, "0", 0.5])
def test_a_pin_the_backend_cannot_honour_fails_only_that_agent(pin):
    driver = FakeDriver(3)
    kernel = make_kernel(driver, devices=(0, 1))
    kernel.register(Recorder("pinned", driver, cuda_device=pin))
    kernel.register(Recorder("other", driver))

    report = kernel.step()

    assert report.failed == ("pinned",)
    assert report.ran == ("other",)
    ((name, error),) = kernel.errors
    assert name == "pinned" and isinstance(error, ValueError)
    assert "'pinned'" in str(error) and "devices 0, 1" in str(error)
    assert kernel.get("pinned").seen == []


def test_a_device_that_cannot_be_selected_or_restored_fails_the_agent_once():
    class Faulty(FakeDriver):
        @contextlib.contextmanager
        def use_device(self, index):
            if index == 1:
                raise RuntimeError("cannot select device 1")
            with super().use_device(index):
                yield
            if index == 2:
                raise RuntimeError("cannot restore after device 2")

    def broken(ctx):
        raise ValueError("own failure")

    driver = Faulty(3)
    kernel = make_kernel(driver)
    for name in "abc":
        kernel.register(Recorder(name, driver))
    kernel.spawn("broken", broken).cuda_device = 2

    report = kernel.step()

    assert report.ran == ("a",)
    assert report.failed == ("b", "c", "broken")
    assert [(name, str(exc)) for name, exc in kernel.errors] == [
        ("b", "cannot select device 1"),
        ("c", "cannot restore after device 2"),
        ("broken", "own failure"),
    ]


def test_threaded_ticks_match_serial_ones():
    def run(max_workers):
        driver = FakeDriver(2)
        kernel = make_kernel(driver, max_workers=max_workers)
        received, threads = {}, set()
        names = ["a", "b", "c", "d"]

        def make(name, target):
            def fn(ctx):
                threads.add(threading.current_thread().name)
                received.setdefault(name, []).extend(m.payload for m in ctx.inbox)
                ctx.send(target, (name, driver.current(), ctx.tick))
                if ctx.tick == 3:
                    ctx.exit()

            return fn

        for name, target in zip(names, names[1:] + names[:1]):
            kernel.spawn(name, make(name, target))
        with kernel:
            reports = kernel.run()
        return reports, received, driver.synced, threads

    serial, threaded = run(1), run(4)

    assert threaded[:3] == serial[:3]
    assert threaded[1]["b"] == [("a", 0, 1), ("a", 0, 2)]
    assert threaded[1]["c"] == [("b", 1, 1), ("b", 1, 2)]
    assert threaded[2] == [0, 1] * 3
    assert all(name.startswith("titoos-cuda") for name in threaded[3])


def test_async_agents_are_rejected():
    kernel = make_kernel(FakeDriver(1))

    async def step(ctx):
        ctx.exit()

    kernel.spawn("async", step)

    with pytest.raises(TypeError, match="AsyncBackend"):
        kernel.step()


class Index:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


def test_devices_accept_any_integer_index():
    assert CudaBackend(devices=[Index(1)], driver=FakeDriver(2)).devices == (1,)


@pytest.mark.parametrize(
    "options, error, message",
    [
        ({"max_workers": 0}, ValueError, "max_workers"),
        ({"devices": []}, ValueError, "at least one"),
        ({"devices": [2]}, ValueError, "not a visible CUDA device: 2"),
        ({"devices": [True]}, ValueError, "not a visible CUDA device: True"),
        ({"devices": [0, 0]}, ValueError, "repeat"),
        ({"driver": "nope"}, ValueError, "choose from: auto, torch, cupy"),
        ({"driver": object()}, TypeError, "lacks: name, device_count"),
    ],
)
def test_backend_options_are_validated(options, error, message):
    options = {"driver": FakeDriver(2), **options}
    with pytest.raises(error, match=message):
        CudaBackend(**options)


def test_a_driver_without_devices_is_unavailable():
    with pytest.raises(CudaUnavailableError, match="fake sees no CUDA device"):
        CudaBackend(driver=FakeDriver(0))


def test_auto_picks_the_first_library_that_sees_a_device(monkeypatch):
    found = FakeDriver(1)
    monkeypatch.setattr(
        cuda, "DRIVERS", {"none": lambda: FakeDriver(0), "found": lambda: found}
    )

    assert CudaBackend().driver is found


def test_auto_explains_why_no_library_is_usable(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "cupy", None)

    with pytest.raises(CudaUnavailableError) as excinfo:
        CudaBackend()

    message = str(excinfo.value)
    assert "torch: PyTorch is not usable" in message
    assert "pip install torch" in message
    assert "cupy: CuPy is not usable" in message
    assert "pip install cupy-cuda12x" in message
    assert isinstance(excinfo.value, RuntimeError)


def test_importing_titoos_does_not_import_a_gpu_library():
    code = (
        "import sys, titoos;"
        "assert titoos.CudaBackend;"
        "assert not {'torch', 'cupy'} & set(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def _gpu_library(name):
    library = pytest.importorskip(name)
    try:
        if name == "torch":
            count = library.cuda.device_count() if library.cuda.is_available() else 0
        else:
            count = library.cuda.runtime.getDeviceCount()
    except Exception:  # noqa: BLE001 - e.g. no CUDA driver
        count = 0
    if count < 1:
        pytest.skip(f"{name} sees no CUDA device")
    return library


def test_torch_agents_compute_on_their_devices_across_the_barrier():
    torch = _gpu_library("torch")
    kernel = make_kernel("torch")
    devices = kernel.backend.devices
    seen = []

    def producer(ctx):
        ctx.send("consumer", torch.arange(1_000_000, device="cuda") * 2)
        ctx.exit()

    def consumer(ctx):
        if not ctx.inbox:
            ctx.wait()
            return
        (tensor,) = (m.payload for m in ctx.inbox)
        local = tensor.to("cuda")
        seen.append((tensor.device.index, local.device.index, int(local.sum())))
        ctx.exit()

    kernel.spawn("producer", producer)
    kernel.spawn("consumer", consumer)
    with kernel:
        kernel.run()

    assert seen == [(devices[0], devices[1 % len(devices)], 999_999 * 1_000_000)]


def test_cupy_agents_compute_on_their_devices_across_the_barrier():
    cupy = _gpu_library("cupy")
    kernel = make_kernel("cupy")
    devices = kernel.backend.devices
    seen = []

    def producer(ctx):
        ctx.send("consumer", cupy.arange(1_000_000, dtype=cupy.int64) * 2)
        ctx.exit()

    def consumer(ctx):
        if not ctx.inbox:
            ctx.wait()
            return
        (array,) = (m.payload for m in ctx.inbox)
        local = array.copy()  # onto the current device
        seen.append((array.device.id, local.device.id, int(local.sum())))
        ctx.exit()

    kernel.spawn("producer", producer)
    kernel.spawn("consumer", consumer)
    with kernel:
        kernel.run()

    assert seen == [(devices[0], devices[1 % len(devices)], 999_999 * 1_000_000)]
