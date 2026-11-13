"""A scripted MCP server for the tests, spoken to over stdio.

Run as ``python fake_mcp_server.py MODE``, where MODE is ``modern`` (answers
``server/discover``), ``legacy`` (refuses it and wants ``initialize``) or
``silent`` (a legacy server that ignores it). Every message received is
appended as a JSON line to the file named by ``FAKE_MCP_LOG``, after a
``{"started": pid}`` line per process.
"""

import json
import os
import sys

MODE = sys.argv[1]
LOG = os.environ["FAKE_MCP_LOG"]
MODERN = "2026-07-28"
LEGACY = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
META_VERSION = "io.modelcontextprotocol/protocolVersion"

TOOLS = [
    {
        "name": "echo",
        "description": "Repeat the text back.",
        "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}},
    },
    {"name": "fail", "inputSchema": {"type": "object"}},
    {"name": "retry", "inputSchema": {"type": "object"}},
    {"name": "elicit", "inputSchema": {"type": "object"}},
    {"name": "die", "inputSchema": {"type": "object"}},
    {"name": "huge", "inputSchema": {"type": "object"}},
    {"name": "ping_me", "inputSchema": {"type": "object"}},
    {"name": "noisy", "inputSchema": {"type": "object"}},
    {"name": "getenv", "inputSchema": {"type": "object"}},
]


def log(message):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(message) + "\n")


def send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def receive():
    line = sys.stdin.readline()
    if not line:
        log({"exited": os.getpid()})
        sys.exit(0)
    message = json.loads(line)
    log(message)
    return message


def text(value):
    return {"content": [{"type": "text", "text": value}]}


def call_tool(params):
    name = params["name"]
    args = params.get("arguments") or {}
    if name == "echo":
        return text(args.get("text", ""))
    if name == "fail":
        return {"content": [{"type": "text", "text": "it broke"}], "isError": True}
    if name == "retry":
        if params.get("requestState") == "state-1":
            return text("retried")
        return {"resultType": "input_required", "requestState": "state-1"}
    if name == "elicit":
        return {
            "resultType": "input_required",
            "inputRequests": {"confirm": {"method": "elicitation/create", "params": {}}},
        }
    if name == "die":
        sys.stderr.write("fatal: asked to die\n")
        sys.stderr.flush()
        os._exit(3)
    if name == "huge":
        return text("x" * 10_000)
    if name == "ping_me":
        send({"jsonrpc": "2.0", "id": "server-ping", "method": "ping"})
        while True:
            reply = receive()
            if reply.get("id") == "server-ping":
                return text("pong" if reply.get("result") == {} else "no pong")
    if name == "noisy":
        sys.stdout.write("this line is not JSON\n")
        return text("still fine")
    if name == "getenv":
        return text(os.environ.get(args["name"], "<unset>"))
    raise LookupError(name)


def handle(method, params):
    if method == "tools/list":
        if params.get("cursor") == "page-2":
            return {"tools": TOOLS[3:]}
        return {"tools": TOOLS[:3], "nextCursor": "page-2"}
    if method == "tools/call":
        return call_tool(params)
    if method == "resources/list":
        return {"resources": [{"uri": "mem://notes", "name": "notes"}]}
    if method == "resources/read":
        return {"contents": [{"uri": params["uri"], "text": "remember the milk"}]}
    if method == "prompts/get":
        who = (params.get("arguments") or {}).get("who", "")
        return {
            "description": "A greeting",
            "messages": [{"role": "user", "content": {"type": "text", "text": f"Hello, {who}"}}],
        }
    raise LookupError(method)


def main():
    log({"started": os.getpid()})
    initialized = False
    while True:
        message = receive()
        method, request_id = message.get("method"), message.get("id")
        params = message.get("params") or {}
        if method == "notifications/initialized":
            initialized = True
        if method is None or request_id is None:
            continue  # notifications, and answers to requests we did not wait on
        if method == "server/discover":
            if MODE == "silent":
                continue
            if MODE == "legacy":
                send({"jsonrpc": "2.0", "id": request_id,
                      "error": {"code": -32601, "message": "Method not found"}})
                continue
            send({"jsonrpc": "2.0", "id": request_id, "result": {
                "resultType": "complete",
                "supportedVersions": [MODERN],
                "capabilities": {"tools": {}},
                "instructions": "Be brief.",
                "_meta": {"io.modelcontextprotocol/serverInfo": {"name": "fake-modern", "version": "1.0"}},
            }})
            continue
        if method == "initialize":
            asked = params.get("protocolVersion")
            send({"jsonrpc": "2.0", "id": request_id, "result": {
                "protocolVersion": asked if asked in LEGACY else LEGACY[0],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake-legacy", "version": "0.9"},
            }})
            continue
        if MODE == "modern":
            version = (params.get("_meta") or {}).get(META_VERSION)
            if version != MODERN:
                send({"jsonrpc": "2.0", "id": request_id, "error": {
                    "code": -32022, "message": "Unsupported protocol version",
                    "data": {"supported": [MODERN], "requested": version},
                }})
                continue
        elif not initialized:
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32600, "message": "not initialized"}})
            continue
        try:
            result = handle(method, params)
        except LookupError as exc:
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32601, "message": f"unknown: {exc}"}})
            continue
        if MODE == "modern":
            result.setdefault("resultType", "complete")
        send({"jsonrpc": "2.0", "id": request_id, "result": result})


main()
