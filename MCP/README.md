# MCP: Model Context Protocol, FastMCP & Multi-Server Agent Architecture

Welcome to the **MCP (Model Context Protocol)** module. The Model Context Protocol is an open standard designed by Anthropic that standardizes how AI models securely connect to external tools, data sources, and services.

This directory provides a practical, full-stack implementation of MCP:
1. **Math Server (`mathserver.py`)**: Built with FastMCP over **`stdio`** (standard input/output).
2. **Weather Server (`weather.py`)**: Built with FastMCP over **`streamable-http`** (HTTP streaming).
3. **Multi-Server Client (`client.py`)**: Built with `MultiServerMCPClient` (from `langchain-mcp-adapters`), orchestrating tools across multiple servers into a unified **LangGraph ReAct agent** powered by **Groq**.

---

## Table of Contents
1. [What is Model Context Protocol (MCP)?](#what-is-model-context-protocol-mcp)
2. [Why MCP Matters (The USB-C of Agentic AI)](#why-mcp-matters-the-usb-c-of-agentic-ai)
3. [Architecture & Transport Protocols](#architecture--transport-protocols)
   - [`stdio` Transport (Process-to-Process)](#stdio-transport-process-to-process)
   - [`streamable-http` / SSE Transport (Network Services)](#streamable-http--sse-transport-network-services)
4. [File Breakdown & Implementation](#file-breakdown--implementation)
   - [`mathserver.py`](#1-mathserverpy)
   - [`weather.py`](#2-weatherpy)
   - [`client.py`](#3-clientpy)
5. [How `MultiServerMCPClient` Works](#how-multiservermcpclient-works)
6. [How to Run the Servers and Client](#how-to-run-the-servers-and-client)
7. [Interview Masterclass: MCP Questions & Answers](#interview-masterclass-mcp-questions--answers)

---

## What is Model Context Protocol (MCP)?

The **Model Context Protocol (MCP)** is an open-source standard for AI systems. It decouples the AI model/client from tool implementations by defining a unified protocol for:
- **Tools**: Executable functions that can perform actions (e.g. calculation, database mutation, API fetching).
- **Resources**: Read-only contextual data (e.g. file contents, documentation, system metrics).
- **Prompts**: Pre-engineered prompt templates exposed by the server.

```
┌────────────────────────────────────────────────────────┐
│                   LLM / Agent Host                     │
│    (LangGraph Agent / Claude Desktop / IDE Extension)   │
└──────────────────────────┬─────────────────────────────┘
                           │
             MCP Client / Protocol Layer
                           │
             ┌─────────────┴─────────────┐
             │                           │
       stdio transport           HTTP / SSE transport
             │                           │
             ▼                           ▼
┌─────────────────────────┐ ┌─────────────────────────┐
│     MCP Server A        │ │     MCP Server B        │
│    (e.g. MathServer)    │ │   (e.g. WeatherServer)  │
│  - add()                │ │  - get_weather()        │
│  - multiply()           │ │                         │
└─────────────────────────┘ └─────────────────────────┘
```

---

## Why MCP Matters (The USB-C of Agentic AI)

Before MCP, every AI framework had its own proprietary tool abstraction:
- LangChain Tools (`BaseTool`)
- OpenAI Function Calling schemas
- LlamaIndex Tools
- AutoGen Tools

If you built 50 database integration tools for LangChain, you had to re-write or wrap them for other frameworks.
**With MCP:**
- Tools are written **once** as an MCP server.
- Any MCP-compliant client (LangGraph, Claude Desktop, Cursor, Zed, custom apps) can consume them instantly.
- Tools can run in isolated environments (containers, remote servers, local sub-processes) with sandboxed permissions.

---

## Architecture & Transport Protocols

MCP defines standard transports for client-server communication:

### `stdio` Transport (Process-to-Process)
- **Mechanism**: The client launches the server as a child process and communicates via JSON-RPC 2.0 over standard input (`stdin`) and standard output (`stdout`).
- **Use Cases**: Local CLI tools, filesystem access, local utility scripts.
- **Benefits**: Zero network configuration, no port conflicts, secure by default (process isolation).

### `streamable-http` / SSE Transport (Network Services)
- **Mechanism**: Communication happens over HTTP with Server-Sent Events (SSE) for streaming updates and POST requests for client commands.
- **Use Cases**: Remote microservices, centralized enterprise tool servers, cloud databases.
- **Benefits**: Can be accessed over LAN or WAN, supports multi-client connections.

---

## File Breakdown & Implementation

### 1. `mathserver.py`
A lightweight math server implemented using `FastMCP`:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """multiply two numbers"""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")
```
- `@mcp.tool()` parses the Python docstrings and type annotations to automatically generate JSON Schema definitions for function calling.
- `mcp.run(transport="stdio")` starts the JSON-RPC listener on standard I/O.

---

### 2. `weather.py`
An asynchronous weather server running on HTTP streaming:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    return f"The current weather in {location} is sunny with a temperature of 25°C."

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```
- Listens on `http://127.0.0.1:8000/mcp` by default.
- Uses async handlers for non-blocking I/O.

---

### 3. `client.py`
The orchestrator that bridges MCP servers to LangGraph and Groq:

```python
import asyncio
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

load_dotenv()

async def main():
    # 1. Configure multiple MCP server connections
    client = MultiServerMCPClient(
        {
            "math": {
                "command": r"C:\Users\Rishabh Pandey\Desktop\AgenticAIWorkSpace\.venv\Scripts\python.exe",
                "args": [
                    r"C:\Users\Rishabh Pandey\Desktop\AgenticAIWorkSpace\MCP\mathserver.py"
                ],
                "transport": "stdio"
            },
            "weather": {
                "url": "http://127.0.0.1:8000/mcp",
                "transport": "streamable-http"
            }
        }
    )

    # 2. Discover and convert tools dynamically to LangChain tools
    tools = await client.get_tools()

    # 3. Initialize LLM
    model = ChatGroq(model="openai/gpt-oss-20b")

    # 4. Build LangGraph ReAct agent
    agent = create_react_agent(model, tools)

    # 5. Execute user query
    math_response = await agent.ainvoke(
        {"messages": [("user", "What is the sum of 5 and 10?")]}
    )
    print("Math Response:", math_response["messages"][-1].content)

asyncio.run(main())
```

---

## How `MultiServerMCPClient` Works

1. **Connection Initialization**:
   - For `"math"`, it spawns a Python subprocess running `mathserver.py` and opens stdin/stdout pipes.
   - For `"weather"`, it establishes an HTTP SSE connection to `http://127.0.0.1:8000/mcp`.
2. **Handshake & Capability Exchange**:
   - Client and server exchange supported protocol versions and capabilities.
3. **Tool Discovery (`client.get_tools()`)**:
   - Queries both servers for available tools.
   - Converts MCP tool schemas into standard LangChain `StructuredTool` objects.
4. **Invocation (`agent.ainvoke()`)**:
   - LLM decides to call `add(a=5, b=10)`.
   - The adapter routes the JSON-RPC call to the `"math"` server process.
   - The server computes the result and sends it back to the client.

---

## How to Run the Servers and Client

### Terminal 1: Run Weather Server (streamable-http)
```bash
python MCP/weather.py
```
*(Starts listening on http://127.0.0.1:8000)*

### Terminal 2: Run Client
```bash
python MCP/client.py
```
*(The client automatically spawns `mathserver.py` via `stdio`, queries `weather.py` via HTTP, discovers tools, and runs the agent!)*

---

## Interview Masterclass: MCP Questions & Answers

### Q1: What is the Model Context Protocol (MCP) and what problem does it solve?
**Answer:**
MCP is an open standard created by Anthropic that standardizes communication between LLM clients and external tools, databases, and resources using JSON-RPC 2.0. Before MCP, every tool required custom integration for each framework. MCP solves the M×N integration problem by acting as an open, standardized interface: developers build an MCP server once, and any MCP-compliant client or agent can interact with it regardless of the host application or programming language.

---

### Q2: Compare `stdio` transport vs `streamable-http` / `sse` transport in MCP.
**Answer:**
- **`stdio`**:
  - The client launches the server as a local child process.
  - Messages are sent via standard input and received via standard output.
  - Zero network overhead, completely sandboxed to the local machine, ideal for local scripts and file access.
- **`streamable-http` / `sse`**:
  - Server runs as an independent network service.
  - Uses HTTP POST for requests and Server-Sent Events (SSE) for server-to-client streaming.
  - Ideal for shared enterprise tools, cloud microservices, and distributed architectures.

---

### Q3: How does `langchain-mcp-adapters` convert MCP tools into LangChain tools?
**Answer:**
`langchain-mcp-adapters` reads the tool descriptions and JSON schemas exposed by the MCP server during the `tools/list` protocol handshake. It dynamically creates LangChain `StructuredTool` instances where the args schema is converted from JSON Schema to Pydantic models. When the LangChain agent invokes the tool, the adapter translates the call into an MCP `tools/call` JSON-RPC message, sends it over the transport, awaits the response, and returns the result as a `ToolMessage`.

---

### Q4: What security advantages does MCP offer over traditional tool calling?
**Answer:**
1. **Process Isolation**: Tools run in separate processes or containers, preventing malicious code or prompt injection from compromising the agent's host memory.
2. **Granular Permissions & Consent**: MCP supports authorization gates where the client can prompt the user for confirmation before executing specific tool calls (e.g. file deletion or payments).
3. **Resource Boundaries**: MCP servers declare explicit URI templates for accessible resources, preventing unauthorized directory traversals.
