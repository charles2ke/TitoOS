# TitoOS

Operating System for Agents.

TitoOS is a tiny, dependency-free kernel that runs agents cooperatively: it
registers them, schedules them round-robin, and routes messages between them.

## Install

```bash
pip install -e ".[test]"
```

## Usage

```python
from titoos import Kernel

kernel = Kernel()

def worker(ctx):
    for message in ctx.inbox:
        ctx.send(message.sender, f"done:{message.payload}")
        ctx.exit()

def boss(ctx):
    if ctx.tick == 1:
        ctx.broadcast("task-1")
    for message in ctx.inbox:
        print(message.payload)  # done:task-1
        ctx.exit()

kernel.spawn("boss", boss)
kernel.spawn("worker", worker)
kernel.run(max_ticks=10)
```

Agents can also subclass `Agent` and implement `step(ctx)`:

```python
from titoos import Agent

class Counter(Agent):
    def __init__(self, name, limit):
        super().__init__(name)
        self.limit = limit
        self.count = 0

    def step(self, ctx):
        self.count += 1
        if self.count >= self.limit:
            ctx.exit()
```

## Concepts

| Piece | Role |
| --- | --- |
| `Kernel` | Registers agents and runs one `step()` per runnable agent per tick. |
| `Agent` / `FunctionAgent` | A unit of work; implement `step(ctx)` or wrap a callable. |
| `Context` | Per-tick handle: `tick`, `inbox`, `send`, `broadcast`, `spawn`, `wait`, `exit`. |
| `MessageBus` | Mailboxes and routing, including `BROADCAST` delivery. |
| `TickReport` | Which agents ran, failed, waited or were restarted during a tick. |
| `StopReason` | Why `run()` stopped: `FINISHED`, `QUIESCENT`, or `MAX_TICKS`. |
| `ExecutionBackend` | How a tick's agents run: `SerialBackend`, `ThreadBackend`, `AsyncBackend`. |
| `RestartPolicy` | What happens when an agent raises: `NEVER` or `ON_FAILURE`. |
| `Snapshot` | A consistent, serializable picture of the kernel between ticks. |

## Blocking and quiescence

An agent with nothing to do calls `ctx.wait()`. It stays alive but is skipped
on later ticks until a message lands in its mailbox, so an idle agent costs
neither a scheduler slot nor a worker thread:

```python
def worker(ctx):
    if not ctx.inbox:
        ctx.wait()      # skipped until someone sends to us
        return
    handle(ctx.inbox)
    ctx.exit()
```

Because waiting agents make no progress on their own, the kernel can tell the
difference between work that is *done* and work that is *stuck*. `run()` stops
as soon as every live agent is waiting with an empty mailbox, and records why
in `kernel.stop_reason`:

| `StopReason` | Meaning |
| --- | --- |
| `FINISHED` | Every agent reached `DONE` or `FAILED`. |
| `QUIESCENT` | Live agents remain, but all are waiting on messages that will never come — a completed workflow or a deadlock. |
| `MAX_TICKS` | The tick budget ran out while agents were still runnable. |

`TickReport.waiting` lists the agents skipped on a given tick.

## Multi-threading

By default the kernel runs agents one after another on the calling thread.
Pass `max_workers` to run the agents of each tick concurrently in a thread
pool — useful when agents are I/O bound, e.g. waiting on model or tool calls:

```python
with Kernel(max_workers=8) as kernel:
    for i in range(8):
        kernel.spawn(f"worker-{i}", slow_io_agent)
    kernel.run(max_ticks=10)
```

Using the kernel as a context manager shuts the pool down at the end; you can
also call `kernel.shutdown()` explicitly. The pool is created lazily, so a
default kernel never starts a thread.

Threading does not change results. The scheduler guarantees:

- **Ticks are barriers.** Every agent of a tick finishes before the next tick
  starts, and mailboxes are drained *before* any agent runs — so a message
  sent during a tick is always delivered in the following one, never
  opportunistically within the same tick.
- **Wake-ups are deterministic.** A waiting agent runs on a tick if and only if
  its mailbox was non-empty at that tick's boundary, never depending on which
  worker happened to run first.
- **Reports are deterministic.** `TickReport.ran` and `.failed` list agents in
  registration order, not completion order.
- **Failures stay isolated.** An agent raising in a worker thread is marked
  `FAILED` and recorded in `kernel.errors` like in serial mode.

Known limitation: an agent spawned mid-tick registers its mailbox immediately,
so whether a `ctx.broadcast()` made during that same tick reaches it depends on
worker timing. Spawn and broadcast in the same tick are therefore not yet
deterministic; sequence them across ticks if it matters.

`MessageBus` is thread-safe, and `ctx.send`, `ctx.broadcast`, and `ctx.spawn`
may all be called from worker threads. Agents that share their own mutable
state with each other are still responsible for their own locking.

An agent that raises is marked `AgentState.FAILED` and removed from the
schedule; the exception is recorded in `kernel.errors` so one bad agent cannot
take down the rest of the system. `kernel.run()` stops once no agent is alive
or `max_ticks` is reached.

## Async execution

The backend decides *how* a tick's agents run; the kernel still decides *which*
agents run and what they see. `AsyncBackend` runs them on an event loop and
supports `async def step(ctx)`:

```python
from titoos import AsyncBackend, Agent, Kernel

class Fetcher(Agent):
    async def step(self, ctx):
        ctx.send("sink", await fetch_something())
        ctx.exit()

with Kernel(backend=AsyncBackend()) as kernel:
    kernel.register(Fetcher("fetcher"))
    kernel.run()
```

`SerialBackend` (the default) and `ThreadBackend` raise a `TypeError` if given
an async agent rather than silently skipping it — including a plain `async def`
callable handed to `kernel.spawn()`. `AsyncBackend` also runs
ordinary sync agents, but a blocking one stalls the whole tick — use
`ThreadBackend` for those. `max_workers=N` remains a shorthand for
`ThreadBackend(N)`.

All three backends produce identical results for the same program.

## Supervision

`ctx.spawn()` records the spawning agent as the child's `parent`, and
`ctx.children()` lists them. Agents opt into restarts:

```python
from titoos import Agent, RestartPolicy

class Worker(Agent):
    restart_policy = RestartPolicy.ON_FAILURE
    max_restarts = 3

    def on_restart(self):
        self.partial_work = None   # reset anything the failed step left behind

    def step(self, ctx):
        ...
```

A failing agent is reset to `READY` and run again on the next tick, keeping its
mailbox. The messages delivered to the step that failed are put back at the
front of that mailbox, so a crash while handling a job redelivers the job
rather than losing it. Once `max_restarts` is exhausted it stays `FAILED`, and its parent
receives a message whose payload is `CHILD_FAILED` with `child` and `restarts`
metadata — which also wakes a supervisor that was waiting. `TickReport.restarted`
lists the agents restarted on a tick.

Restart decisions are made at the tick barrier in registration order, so they
never depend on worker timing. All restarts for a tick are applied before any
escalation is sent, so a parent that failed and was restarted in the same tick
as its child is still notified.

## Persistence

A tick boundary is the only moment when no agent is mid-execution and no message
is in flight, so that is where the kernel checkpoints. Agents opt in:

```python
class Accumulator(Agent):
    def save_state(self):
        return {"total": self.total}

    def load_state(self, data):
        self.total = data["total"]
```

```python
text = kernel.snapshot().to_json()
...
restored = Kernel.restore(Snapshot.from_json(text), {"Accumulator": Accumulator})
restored.run()
```

A snapshot holds the tick number, every agent's lifecycle state, parent link,
restart count and saved data, plus all pending messages. `snapshot()` raises if
called while agents are running, e.g. from inside a `step()`.

Behaviour cannot be serialized, so `restore()` takes factories mapping each
recorded `kind` (the class name) to a callable building a bare agent of that
kind. A factory is required for every agent that is still alive; missing ones
are an error rather than a silent drop. Agents that had already finished are
restored as inert `FinishedAgent` placeholders that preserve their name,
outcome, parent and saved data.

Anything not returned by `save_state()` is not preserved — the default saves
nothing.

## Tests

```bash
python -m pytest
```
