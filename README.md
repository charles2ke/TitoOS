# TitoOS

Operating System for Agents.

TitoOS is a tiny, dependency-free kernel that runs agents cooperatively: it
registers them, schedules them round-robin, and routes messages between them.

Agents are ordinary Python objects or functions. The kernel gives them the
things a multi-agent system otherwise has to reinvent:

- **A deterministic scheduler.** One `step()` per runnable agent per tick, with
  the tick as a barrier — serial, threaded and async runs give the same result.
- **Message passing.** Mailboxes, direct sends and broadcasts, routed by the
  kernel rather than by agents holding references to each other.
- **Supervision.** A failing agent is isolated, optionally restarted with its
  mail intact, and its parent is told when it gives up.
- **Persistence.** Snapshot the kernel's tick and lifecycle state, pending
  mailboxes and opted-in agent data at a tick boundary, and restore it later.
- **Sandboxed side effects.** HTTP, files, shell and clock drivers that are
  default-deny and installed by the operator, not by the agent.
- **Foreign agents.** Adapters run agents written for other frameworks as
  ordinary agents here.
- **MCP servers and A2A agents.** Agents use the tools, resources and prompts
  of Model Context Protocol servers and message remote agents over
  Agent2Agent (A2A), and a remote A2A agent can join the bus as a local one.
- **GPU agents.** A CUDA backend gives each agent a device and makes the tick
  barrier wait for the GPU, using PyTorch or CuPy when one is installed.
- **A dashboard.** A browser GUI to watch agents, mailboxes and ticks, and to
  step, run and message the kernel by hand.

Requires Python 3.10+ and nothing else; the test extra pulls in `pytest`, and
the CUDA backend uses PyTorch or CuPy only if you install one.

## Contents

- [Install](#install) · [Usage](#usage) · [Concepts](#concepts)
- [Blocking and quiescence](#blocking-and-quiescence)
- [Multi-threading](#multi-threading) · [Async execution](#async-execution) · [CUDA](#cuda)
- [Supervision](#supervision) · [Persistence](#persistence)
- [Integrations](#integrations) · [MCP servers](#mcp-servers) · [A2A agents](#a2a-agents)
- [Agent platforms](#agent-platforms) · [Remote A2A agents](#remote-a2a-agents)
- [GUI](#gui)
- [Troubleshooting](#troubleshooting)
- [Layout](#layout) · [Tests](#tests) · [License](#license)

## Install

```bash
pip install -e .            # the library
pip install -e ".[test]"    # plus pytest, to run the suite
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
reports = kernel.run(max_ticks=10)
```

The run above prints `done:task-1` and stops after three ticks. `run()`
returns one `TickReport` per tick and sets `kernel.stop_reason`, so a run can
be inspected afterwards without instrumenting the agents:

```python
print(kernel.stop_reason)        # StopReason.FINISHED
print(reports[-1].ran)           # ('boss',)
print(kernel.errors)             # [] - (agent name, exception) for each failure
print(kernel)                    # <Kernel tick=3 agents=0/2 live ...>
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
| `ExecutionBackend` | How a tick's agents run: `SerialBackend`, `ThreadBackend`, `AsyncBackend`, `CudaBackend`. |
| `RestartPolicy` | What happens when an agent raises: `NEVER` or `ON_FAILURE`. |
| `Snapshot` | A consistent, serializable picture of the kernel between ticks. |
| `Integration` | A driver for the world outside the kernel: HTTP, files, processes, time, MCP servers, A2A agents. |
| `PlatformAdapter` | A bridge in the other direction: an agent from another framework, scheduled here. |

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

Every backend produces the same results for the same program.

## CUDA

A GPU cannot make the kernel itself faster — scheduling a tick or routing a
message costs less than launching one CUDA kernel — but it can make what agents
compute faster: inference, embeddings, vector search. `CudaBackend` runs such
agents with PyTorch or CuPy:

```python
import torch
from titoos import Agent, CudaBackend, Kernel

class Embedder(Agent):
    model = None

    def step(self, ctx):
        if self.model is None:
            self.model = load_model().to("cuda")  # onto this agent's GPU
        for message in ctx.inbox:
            batch = torch.as_tensor(message.payload, device="cuda")
            ctx.send("indexer", self.model(batch))
        ctx.wait()

with Kernel(backend=CudaBackend()) as kernel:
    kernel.register(Embedder("embedder"))
    kernel.spawn("indexer", indexer)
    kernel.run()
```

- **Placement.** Each agent gets a device the first time it is scheduled and
  keeps it while it is alive, through waits and restarts. Give an agent a
  `cuda_device` attribute to pin it; the others go to the device with the
  fewest live agents. `backend.device_of(name)` and `backend.describe()` show
  where everything runs.
- **Activation.** While an agent's `step()` runs, its device is the current
  CUDA device, so `device="cuda"` and CuPy allocations land there without the
  agent naming one.
- **The barrier reaches the GPU.** CUDA calls return once work is queued, so
  before a tick ends the backend waits for every device used in it: a tensor
  sent in a message is complete when its recipient runs, while within a tick
  work on different devices overlaps. An asynchronous CUDA error surfacing
  then fails the agents that ran on that device, and supervision takes over.

`CudaBackend(devices=[0, 1])` limits the devices, `driver="torch"` or
`"cupy"` picks the library — by default the first that sees a GPU; pick the
one your agents use — and `max_workers=N` runs each tick's agents in a thread
pool as `ThreadBackend` does. Without a usable GPU the constructor raises
`CudaUnavailableError`, so a program can fall back:

```python
try:
    backend = CudaBackend()
except CudaUnavailableError:
    backend = ThreadBackend(8)
```

Agents on a device share the library's current stream there instead of
getting one each: with a stream per agent, the caching allocators of PyTorch
and CuPy could hand a tensor's memory to new work while another agent's stream
still reads it. For the same reason leave CuPy's per-thread default stream
(`CUPY_CUDA_PER_THREAD_DEFAULT_STREAM`) off when `max_workers > 1`, and an
agent that creates its own streams must follow its library's rules for sharing
tensors across them (`Tensor.record_stream` in PyTorch). The barrier waits only
for the devices agents are placed on, so an agent that also queues work on
another GPU should synchronize it before its step returns. Neither library is a
dependency, and `import titoos` never imports one.

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

## Integrations

The kernel routes messages between agents; an `Integration` is how an agent
touches anything else. Integrations are installed on the kernel, so the same
agent code runs against a real endpoint or a stub depending on what the
operator installed, and every side effect of a tick is attributable to a named
agent and a named driver:

```python
from titoos import Kernel
from titoos.integrations import ClockIntegration, FileSystemIntegration, HttpIntegration

kernel = Kernel()
kernel.install(HttpIntegration(allowed_hosts=["api.example.com"]))
kernel.install(FileSystemIntegration("/var/lib/titoos/work"))
kernel.install(ClockIntegration())

def reporter(ctx):
    response = ctx.call("http", "get", "https://api.example.com/status")
    ctx.call("files", "write_text", "status.json", response.body)
    ctx.exit()

kernel.spawn("reporter", reporter)
kernel.run()
```

| Integration | `ctx.call(...)` operations |
| --- | --- |
| `HttpIntegration` | `get`, `post_json`, `request`, `describe` |
| `FileSystemIntegration` | `read_text`, `write_text`, `append_text`, `list_dir`, `exists`, `delete`, `describe` |
| `ShellIntegration` | `run`, `describe` |
| `ClockIntegration` | `now`, `timestamp`, `monotonic`, `sleep`, `describe` |
| `McpIntegration` | `list_tools`, `call_tool`, `list_resources`, `list_resource_templates`, `read_resource`, `list_prompts`, `get_prompt`, `server_info`, `describe` |
| `A2AIntegration` | `send_message`, `get_task`, `cancel_task`, `agent_card`, `describe` |

An integration call happens inside the running `step()`, so it is covered by
the tick barrier like any other work: results only reach other agents through
messages, on the next tick. A call that cannot be completed raises
`IntegrationError`, which propagates out of `step()` like any exception — the
agent is marked `FAILED` and its restart policy decides what happens next.

### Default-deny

The drivers that reach dangerous resources have no "allow everything" mode:

- `HttpIntegration` requires `allowed_hosts`, accepts only `http`/`https`,
  re-checks the allowlist on every redirect hop, and caps the response size.
  HTTP error statuses are returned as responses, not raised — a 404 is an
  answer, not a broken integration.
- `FileSystemIntegration` confines every path to one `root`; absolute paths,
  `..` traversal and symlinks pointing outside are rejected. `read_only=True`
  disables the writing operations.
- `ShellIntegration` requires `allowed_commands`, never uses a shell, and takes
  argument vectors, so an agent-produced argument is data rather than syntax.
  Each allowed command must be a bare name and is resolved to an absolute
  executable when the integration is built — an entry that does not resolve is
  an error there, not at the first call — and a program given with a path
  separator is refused, so an agent cannot aim an allowed name at a binary it
  wrote itself. Subprocesses get an explicit minimal environment — the kernel's
  own is never inherited — and their output is streamed into buffers bounded by
  `max_output`.
- `ClockIntegration` caps a single `sleep()`, since a tick is a barrier. A
  `fixed=` clock must be timezone-aware and also freezes `monotonic()`, so a
  fixed run is reproducible.
- `McpIntegration` reaches the one server it was built for. A `command=` is an
  argument vector, never a shell string, and its program is resolved to an
  absolute path when the integration is built; the server inherits only a
  minimal environment (`HOME`, `PATH`, `USER` and the like) plus `env=`.
  `allowed_tools=` narrows which tools agents may list and call. HTTP
  `headers=` are redacted from `describe()` and redirects are never followed,
  so credentials only ever reach `url`.
- `A2AIntegration` reaches only the peers in its address book. An agent card
  is followed only to endpoints at the card's own origin, so a card cannot
  point a peer's `headers` at another server; `A2APeer(endpoint=...)` names an
  endpoint explicitly. Redirects are never followed and every response is
  capped at `max_bytes`.

Only the operations an integration lists in `operations` are callable, so its
surface is exactly what it advertises — including through
`ctx.integration("http")`, which hands back a restricted handle exposing those
operations as methods and nothing else.

### MCP servers

`McpIntegration` connects to one [Model Context Protocol](https://modelcontextprotocol.io)
server — a program it starts and talks to over stdio, or a Streamable HTTP
endpoint — and lets agents use its tools, resources and prompts. Install one
per server, each under its own name:

```python
from titoos import Kernel, McpIntegration

kernel = Kernel()
kernel.install(McpIntegration(command=["python", "-m", "notes_server"], name="notes"))
kernel.install(
    McpIntegration(
        url="https://mcp.example.com/mcp",
        headers={"Authorization": "Bearer ..."},
        allowed_tools=["search"],
        name="search",
    )
)

def researcher(ctx):
    print([tool["name"] for tool in ctx.call("search", "list_tools")])   # ['search']
    result = ctx.call("search", "call_tool", "search", {"query": "titoos"})
    if result.ok:
        ctx.call("notes", "call_tool", "append", {"text": result.text})
    ctx.exit()
```

- **Versions.** The client speaks the 2026-07-28 revision, where every request
  carries its own version, and falls back to the `initialize` handshake of
  2025-11-25 back to 2024-11-05. It detects which on first use;
  `protocol_version=` pins one. The deprecated HTTP+SSE transport is not
  supported.
- **Results.** `call_tool` returns an `McpToolResult` — `content`,
  `structured_content`, `is_error`, plus `text`, `ok` and `to_dict()`. A tool
  that ran and failed is an answer, not an exception; pass `check=True` to
  raise instead. A server that refuses a request raises `JsonRpcError`, an
  `IntegrationError` carrying the JSON-RPC `code`.
- **Connections.** The connection opens on first use, and a stdio server that
  exits is started again on the next call. A request that was in flight when
  it died raises rather than being retried, since a tool call may already have
  had its effect. `timeout` bounds each request and `max_bytes` each message.
- **What titoos does not do.** It declares no client capabilities, so it
  answers a server's `ping` and refuses anything else a server asks of it —
  sampling, elicitation, roots. Over HTTP, a 2026-07-28 server's tools whose
  `x-mcp-header` annotations are invalid are left out of `list_tools`, as the
  specification requires.

### A2A agents

`A2AIntegration` lets agents message remote agents over
[Agent2Agent (A2A)](https://a2a-protocol.org), from an address book the
operator fixes when installing it:

```python
from titoos import A2AIntegration, A2APeer, Kernel

kernel = Kernel()
kernel.install(
    A2AIntegration(
        {
            "translator": "https://agents.example.com/translator",
            "billing": A2APeer("https://billing.example.com", headers={"Authorization": "Bearer ..."}),
        }
    )
)

def clerk(ctx):
    answer = ctx.call("a2a", "send_message", "translator", "Translate to French: good morning")
    if answer.needs_input:                      # the task asked a question back
        answer = ctx.call(
            "a2a", "send_message", "translator", "formal, please",
            task_id=answer.task_id, context_id=answer.context_id,
        )
    print(answer.state, answer.text)
    ctx.exit()
```

- **Discovery.** Each peer's agent card is read on first use from
  `{url}/.well-known/agent-card.json`, or from `url` itself when it names a
  `.json` file. The client speaks the JSON-RPC binding of A2A 1.0 and 0.3,
  choosing what the card offers (1.0 first); `A2APeer(protocol_version=...)`
  pins one. gRPC, HTTP+JSON, streaming and push notifications are not used.
- **Messages.** `send_message` takes `text`, `data=` (a JSON value) and
  `parts=` (strings, bytes, or part mappings such as `{"url": ...,
  "mediaType": ...}`), plus `context_id=`, `task_id=` and `metadata=`. It
  returns an `A2AResult`: `kind` (`"message"` or `"task"`), `state`, `text`,
  `parts` (always in the 1.0 shape), `task_id`, `context_id`, and `ok`,
  `done` and `needs_input`.
- **Tasks.** By default `send_message` waits for a task to settle within
  `timeout`, polling while it is `submitted` or `working`. `wait=False`
  returns as soon as the task exists; `get_task` and `cancel_task` follow it
  up. A failed task is an answer (`state == "failed"`); a peer that refuses a
  request raises `JsonRpcError` with A2A's error `code`, such as `-32001` for
  `TaskNotFoundError`.

### Writing your own

Subclass `Integration`, name it, and list its operations. Anything an agent
should reach — a model API, a queue, a database — becomes one:

```python
from titoos import Integration

class Slack(Integration):
    name = "slack"
    operations = ("post_message",)

    def post_message(self, channel: str, text: str) -> str:
        ...
```

Because integrations live on the kernel, tests install a stub under the same
name and the agents under test never notice. Integrations are runtime
resources, not state: they are not captured by `snapshot()`, so a restored
kernel is installed with the drivers it should use. `kernel.shutdown()` — and
the context-manager form — closes them once the backend's workers have
finished, so a step still inside a call never meets a closed driver.

## Agent platforms

An `Integration` is what an agent *calls out* through. A `PlatformAdapter` is
the other direction: it takes an agent that already exists somewhere else — a
LangGraph graph, an OpenAI Agents SDK agent, a CrewAI crew, a remote A2A peer
— and runs it as an ordinary agent here, with a mailbox, supervision, restarts
and snapshots:

```python
from titoos import Kernel
from titoos.platforms import Outbound, PlatformAdapter

class MyFramework(PlatformAdapter):
    name = "myframework"

    def __init__(self, agent):
        super().__init__()
        self.agent = agent                      # the foreign object

    def invoke(self, request, turn):
        return self.agent.run(request)          # one turn of the platform

kernel = Kernel()
kernel.register(MyFramework(their_agent).agent("assistant"))
```

For something that is already a plain callable, `CallableAdapter` skips the
subclass:

```python
from titoos.platforms import CallableAdapter

kernel.register(CallableAdapter("echo", lambda request, turn: f"echo:{request}").agent("assistant"))
```

One tick is one platform turn. The adapter never decides when it runs or what
it sees: the kernel fixes both at the tick barrier, hands it that turn's mail,
and puts whatever comes back on the bus — which is what keeps a run
reproducible even though the reasoning happens in someone else's library.

An adapter answers three questions, and only `invoke` is usually needed:

| Hook | Question |
| --- | --- |
| `to_platform(turn)` | What does this turn's mail look like to the platform? |
| `invoke(request, turn)` / `ainvoke(...)` | How is the platform actually run? |
| `from_platform(result, turn)` | Which messages should the result put on the bus? |

The defaults pass a lone payload through untouched, hand a multi-message turn
over as a list, and reply to the sender with whatever came back. Return
`None` to say nothing, or one or more `Outbound(payload, to=..., metadata=...,
broadcast=...)` to address replies yourself.

`PlatformAgent` options: `seed=` gives the platform something to act on before
anyone has written to it, `reply_to=` or `broadcast=True` sets where unaddressed
replies go, and `max_turns=` ends the agent after N turns. With an empty mailbox
the agent waits instead of invoking anything, so an idle foreign agent never
costs a model call.

Anything the SDK raises is normalised into `PlatformError` — a subclass of
`IntegrationError` — so restart policies and `CHILD_FAILED` escalation work
exactly as they do for any other agent. `save_state()`/`load_state()` on the
adapter map onto the platform's conversation state where it has one; the
default saves nothing rather than restoring into a subtly different
conversation.

### Async SDKs

Set `is_async = True` and implement `ainvoke`; `adapter.agent(...)` then builds
an `AsyncPlatformAgent`, which needs `AsyncBackend`. The other backends refuse
it through the same `async def step` check that applies to any agent. A sync
adapter whose SDK blocks belongs on `ThreadBackend`.

### Selecting a platform by name

Adapters can be published in a registry and built from configuration. A
registered target may be a `"module:attribute"` string, imported only when
something actually builds it — so nothing you did not install is ever imported:

```python
from titoos.platforms import create_platform, register_platform

register_platform("myframework", "mypackage.adapters:MyFramework")
adapter = create_platform("myframework", their_agent)
```

Adapters for real frameworks import their SDK lazily through
`require("package", extra="...")`, which raises `MissingDependency` naming the
extra to install. TitoOS itself stays dependency-free, and `import titoos`
never imports the platform layer at all.

### Remote A2A agents

`A2AAdapter` puts a remote A2A agent on the bus under a local name, so the
agents around it message it like any other. It needs no SDK, and is
registered as `"a2a"`, so `create_platform("a2a", url, name=...)` builds the
same thing from configuration:

```python
from titoos.platforms import A2AAdapter

kernel.register(A2AAdapter("https://agents.example.com/translator", name="translator").agent("translator"))
```

Each turn's mail goes to the peer as one message — strings as text parts,
bytes as raw parts, mappings as data parts — and the answer goes back to the
sender: its text, or the whole `A2AResult.to_dict()` when it carries more than
text, with `a2a_state`, `a2a_task_id` and `a2a_context_id` metadata. All turns
share one A2A context, so the peer sees a single conversation; when it asks
for more input (`a2a_state == "input_required"`), the next turn answers that
task. Both survive snapshots. A task that fails, is rejected or canceled,
needs authentication, or is still running after `timeout` raises
`PlatformError`, and supervision takes it from there.

## GUI

`titoos.gui` is a browser dashboard for a kernel, served by the standard
library's HTTP server — still no dependencies. Try it on a demo kernel (a
dispatcher, two workers of which one fails once and is restarted, and a
collector):

```bash
python -m titoos.gui            # http://127.0.0.1:8765/ ; --host/--port to change
```

Or attach it to your own kernel:

```python
from titoos.gui import KernelGUI

with KernelGUI(kernel, port=8765).start() as gui:   # port=0 picks a free port
    print(gui.url)
    input("press Enter to stop\n")
```

The page shows the tick, stop reason, every agent's state, parent, restarts and
pending mail, the contents of each mailbox, recent tick reports, `kernel.errors`
and the installed integrations. Its controls step the kernel, run it for N
ticks, and post a message (or a `*` broadcast) onto the bus. Payloads typed as
JSON are sent as JSON; anything else is sent as text.

The page is a thin client over a JSON API that scripts can use too:

| Endpoint | Does |
| --- | --- |
| `GET /api/state` | Everything the page shows. |
| `GET /api/snapshot` | `kernel.snapshot().to_json()`. |
| `POST /api/step` | Runs one tick; returns its `TickReport`. |
| `POST /api/run` | Body `{"max_ticks": N}` (at most 10 000); returns the reports and stop reason. |
| `POST /api/send` | Body `{"to": "...", "payload": ..., "metadata": {...}}`. |

Things to know:

- **The GUI drives the kernel.** Ticks it runs are serialized under one lock
  and state is read between ticks. Do not also call `step()`/`run()` from
  another thread while it is attached. Tick reports are only recorded for ticks
  the GUI itself ran.
- **Messages come from `gui`** (change with `sender=`). That name has no
  mailbox, so an agent replying to it fails with `unknown recipient` unless you
  register an agent under that name to receive replies.
- **It is local by default.** It binds to `127.0.0.1`, answers only to the
  host names it was bound under, and refuses `POST`s that are not JSON or that
  carry a foreign `Origin`, so another web page cannot drive your kernel. There
  is no authentication: binding to another interface (with `allowed_hosts=`)
  lets anyone who can reach it control the kernel.
- Payloads that are not JSON-serializable are displayed by their `repr`.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| An agent never runs again | It called `ctx.wait()` and its mailbox stayed empty. It wakes on the first tick a message is pending; `TickReport.waiting` lists who was skipped. |
| `run()` returns early, `stop_reason` is `QUIESCENT` | Every live agent is waiting for a message nobody will send — a finished workflow, or a deadlock. |
| `stop_reason` is `MAX_TICKS` | The budget ran out while agents were still runnable; raise `max_ticks`. |
| An agent stopped mid-run and nothing was raised | It raised inside `step()`, so it is `FAILED` and isolated. The exception is in `kernel.errors` and the name in `TickReport.failed`; set `restart_policy = RestartPolicy.ON_FAILURE` to retry it. |
| `KeyError: "unknown recipient: 'x'; mailboxes: a"` | No agent of that name is registered, or it already reached `DONE`/`FAILED` and its mailbox was reclaimed. The message lists the mailboxes that do exist. |
| A reply is not in `ctx.inbox` in the same tick | Ticks are barriers: a message sent during a tick is delivered on the next one. |
| `TypeError: ... defines an async step() but SerialBackend cannot await it` | Run the kernel with `Kernel(backend=AsyncBackend())`. |
| `IntegrationError: no integration installed named 'http'` | Install the driver on the kernel (`kernel.install(...)`) before the agents run. |
| `RuntimeError: cannot snapshot while agents are running` | `snapshot()` belongs between ticks, never inside a `step()`. |
| `KeyError: no factory for agent kind(s): ...` | `Kernel.restore()` needs a factory for every live agent's class name. |
| `CudaUnavailableError: no CUDA device is available` | Neither PyTorch nor CuPy is installed with GPU support, or no GPU or driver is visible; the message gives the reason per library. Catch it and fall back to another backend. |
| `Expected all tensors to be on the same device` | An agent used a tensor or model that lives on another agent's GPU — or was created at import time, on device 0. Create GPU state inside `step()`, move tensors that arrive in messages with `.to("cuda")`, or pin the agents to one device with `cuda_device`. |
| Every agent on one GPU failed in the same tick | Waiting for that device raised: an asynchronous CUDA error from work queued during the tick, which cannot be traced to one agent. The error is in `kernel.errors`. |
| The first call to a stdio MCP server takes `timeout` seconds | The server ignored the 2026-07-28 `server/discover` probe instead of answering it with an error, so titoos waited it out before falling back to `initialize`. Pin the revision it speaks, e.g. `McpIntegration(..., protocol_version="2025-11-25")`. |
| `MCP server 'x' rejected initialize with HTTP 404` (or `405`) | `url=` is not a Streamable HTTP endpoint — often the server only speaks the deprecated HTTP+SSE transport, which titoos does not support. Point `url=` at its Streamable HTTP endpoint (commonly `/mcp`), or run it with `command=`. |
| `tool 'x' was excluded: ...` | The server annotated that tool's arguments with an invalid `x-mcp-header`, so the tool is left out of `list_tools`; the reason names the problem in the server's schema. |
| `... redirect to ... not followed` / `redirects are not followed` | MCP and A2A never follow redirects, so credentials cannot leak to another server. Configure the final URL. |
| `the agent card only offers ... at another origin (...)` | The peer's card points requests at another host, and titoos will not send the peer's `headers` there on the card's say-so. If you trust that host, pass it as `A2APeer(url, endpoint=...)`. |
| `... if the agent speaks A2A 0.3, pass A2APeer(protocol_version='0.3')` | With `endpoint=` there is no card to read the version from, so titoos assumed A2A 1.0 and the peer refused it. Add `protocol_version="0.3"` for a 0.3 agent. |
| `PlatformError: A2A peer 'x' is still running task ...` | The remote task outlasted the adapter's `timeout`; raise it with `A2AAdapter(..., timeout=...)`. |

## Layout

| Path | Contents |
| --- | --- |
| `titoos/kernel.py` | The scheduler: ticks, supervision, integrations, snapshots. |
| `titoos/agent.py` | `Agent`, `FunctionAgent`, `Context`, `AgentState`, `RestartPolicy`. |
| `titoos/bus.py`, `titoos/message.py` | Mailboxes, routing and the `Message` type. |
| `titoos/backends.py` | `SerialBackend`, `ThreadBackend`, `AsyncBackend`. |
| `titoos/cuda.py` | `CudaBackend`: device placement and the tick barrier on the GPU. |
| `titoos/persistence.py` | `Snapshot` and the records it serializes. |
| `titoos/integrations/` | HTTP, filesystem, shell, clock, MCP and A2A drivers, and the JSON-RPC plumbing MCP and A2A share. |
| `titoos/platforms/` | `PlatformAdapter`, the platform registry and `A2AAdapter`. |
| `titoos/gui/` | `KernelGUI`: the browser dashboard and its JSON API. |
| `tests/` | Pytest suite, one module per area. |

## Tests

```bash
pip install -e ".[test]"
python -m pytest
```

## License

[Apache 2.0](LICENSE).
