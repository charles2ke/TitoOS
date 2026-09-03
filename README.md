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
| `Kernel` | Registers agents and runs one `step()` per agent per tick. |
| `Agent` / `FunctionAgent` | A unit of work; implement `step(ctx)` or wrap a callable. |
| `Context` | Per-tick handle: `tick`, `inbox`, `send`, `broadcast`, `spawn`, `exit`. |
| `MessageBus` | Mailboxes and routing, including `BROADCAST` delivery. |
| `TickReport` | Which agents ran or failed during a tick. |

An agent that raises is marked `AgentState.FAILED` and removed from the
schedule; the exception is recorded in `kernel.errors` so one bad agent cannot
take down the rest of the system. `kernel.run()` stops once no agent is alive
or `max_ticks` is reached.

## Tests

```bash
python -m pytest
```
