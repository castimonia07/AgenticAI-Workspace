# Debugging: LangSmith Observability, LangGraph Studio & Production Tool-Calling Agent

Welcome to the **Debugging** module. Developing agentic workflows requires observability, step-by-step state inspection, and time-travel debugging. This folder demonstrates how to build a robust tool-calling agent, trace execution runs with **LangSmith**, and configure **LangGraph Studio / CLI** for interactive local debugging.

---

## Table of Contents
1. [The Observability Challenge in Agentic AI](#the-observability-challenge-in-agentic-ai)
2. [File Overview](#file-overview)
   - [`agent.py`](#agentpy)
   - [`debugging.ipynb`](#debuggingipynb)
   - [`langgraph.json`](#langgraphjson)
   - [`.langgraph_api/`](#langgraph_api)
3. [Deep Dive: LangSmith Tracing & Telemetry](#deep-dive-langsmith-tracing--telemetry)
4. [Deep Dive: LangGraph Studio & Local Dev Server](#deep-dive-langgraph-studio--local-dev-server)
5. [Step-by-Step Code Walkthrough (`agent.py`)](#step-by-step-code-walkthrough-agentpy)
6. [Time-Travel Debugging & State Forking](#time-travel-debugging--state-forking)
7. [Interview Masterclass: Debugging & Observability Q&A](#interview-masterclass-debugging--observability-qa)

---

## The Observability Challenge in Agentic AI

Unlike traditional software where code execution is deterministic, LLM agents:
- Make non-deterministic decisions at runtime.
- Call external tools with dynamic arguments.
- Loop across multiple thought-action-observation cycles.
- Can fail silently (e.g. infinite loops, hallucinated tool arguments, or malformed JSON).

To debug production agents effectively, you need:
1. **Full Trace Visibility**: Seeing every LLM prompt, completion, token count, latency, and tool payload.
2. **Step-by-Step State Inspection**: Seeing the exact state dictionary at each node transition.
3. **Time Travel**: Rewinding an agent run to step 3, modifying the state or prompt, and re-running from that exact point.

---

## File Overview

### `agent.py`
A modular, production-ready script that builds and exports a tool-calling LangGraph agent. It configures LangSmith tracing environment variables, initializes `ChatGroq(model="openai/gpt-oss-20b")`, defines a typed tool `@tool def add(...)`, binds it to the model, wires the graph with `ToolNode` and `tools_condition`, and compiles the graph into `tool_agent`.

### `debugging.ipynb`
An interactive testing ground used to:
- Verify environment variables and API keys.
- Inspect state type definitions (`print(State.__annotations__)`).
- Test graph invocations with `HumanMessage`.
- Verify that LangSmith traces appear on the LangSmith cloud dashboard.

### `langgraph.json`
The configuration file consumed by the **LangGraph CLI** (`langgraph dev`):
```json
{
    "dependencies": ["."],
    "graphs": {
        "tool_agent": "../agent.py:tool_agent"
    },
    "env": "../.env"
}
```
This tells the LangGraph local server where to find the compiled graph (`tool_agent` in `agent.py`) and where to load environment variables.

### `.langgraph_api/`
The local storage directory generated automatically when running the LangGraph local development server. It contains serialized pickles (`.pckl`) of checkpoints and runs for local UI sessions.

---

## Deep Dive: LangSmith Tracing & Telemetry

LangSmith is the observability platform purpose-built for LLM applications. In `agent.py`, tracing is activated via environment variables:

```python
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"          # Enables telemetry hooks
os.environ["LANGSMITH_PROJECT"] = "TestProject"    # Groups traces into this project dashboard
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
```

### What LangSmith captures automatically:
- **Run Tree**: Visual hierarchy of parent graph runs, child node executions, LLM calls, and tool runs.
- **Latency & Tokens**: Exact time spent in LLM inference vs. tool execution, along with input/output token counts.
- **Payload Inspection**: The exact inputs and outputs of every single function, tool, and LLM call.
- **Errors & Exceptions**: Full Python tracebacks linked to the exact node that failed.

---

## Deep Dive: LangGraph Studio & Local Dev Server

### What is LangGraph Studio?
LangGraph Studio is a specialized IDE for agent development. It visualizes the graph architecture, lets you step through nodes interactively, inspect checkpoints, edit messages, and rerun from any step.

### How to Run LangGraph Studio Locally:
Inside the `Debugging/` folder, run:
```bash
langgraph dev
```
This starts an in-memory FastAPI server hosting the graph endpoint configured in `langgraph.json` and opens the LangGraph Studio UI in your browser.

---

## Step-by-Step Code Walkthrough (`agent.py`)

```python
# 1. State Definition
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def make_tool_graph():
    # 2. Tool Definition
    @tool 
    def add(a: float, b: float) -> float:
        """add two numbers"""
        return a + b

    tools = [add]
    tool_node = ToolNode(tools)
    
    # 3. Model Tool Binding
    llm_with_tool = llm.bind_tools(tools)

    # 4. Node Logic
    def call_llm_model(state: State):
        return {"messages": [llm_with_tool.invoke(state['messages'])]}

    # 5. Graph Topology
    builder = StateGraph(State)
    builder.add_node("tool_calling_llm", call_llm_model)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges(
        "tool_calling_llm",
        tools_condition # Routes to "tools" if tool_calls present, else to END
    )
    builder.add_edge("tools", "tool_calling_llm")

    # 6. Compilation
    return builder.compile()

tool_agent = make_tool_graph()
```

---

## Time-Travel Debugging & State Forking

LangGraph's checkpointing architecture enables **Time Travel**:
1. **Inspecting History**: Retrieve all previous states of a thread:
   ```python
   for state in graph.get_state_history(config):
       print(state.config, state.values)
   ```
2. **Replaying / Forking from a Checkpoint**:
   You can pick any checkpoint ID from the past, update the state, and resume execution:
   ```python
   checkpoint_config = {"configurable": {"thread_id": "1", "checkpoint_id": "some_past_id"}}
   # Resume or fork:
   graph.invoke(None, config=checkpoint_config)
   ```
This allows developers to fix a bug in prompt or tool logic and see if the historical run would have succeeded without starting from scratch.

---

## Interview Masterclass: Debugging & Observability Q&A

### Q1: How do you trace and monitor a LangGraph application in production?
**Answer:**
1. Configure LangSmith by exporting `LANGSMITH_TRACING="true"`, `LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT`.
2. All graph runs, subgraphs, LLM calls, and tools are automatically instrumented via LangChain callbacks.
3. Tag runs dynamically using metadata:
   ```python
   config = {"metadata": {"user_id": "user_123", "environment": "production"}}
   graph.invoke(..., config=config)
   ```
4. Set up alerts and track latency, error rates, and token cost dashboards in LangSmith.

---

### Q2: What is the role of `langgraph.json`?
**Answer:**
`langgraph.json` is the manifest file for the LangGraph CLI (`langgraph dev` / `langgraph up`). It defines:
- The Python dependencies or project path (`"dependencies": ["."]`).
- The exported graphs and their module paths (`"graphs": {"tool_agent": "agent.py:tool_agent"}`).
- The location of the environment variables file (`"env": ".env"`).

---

### Q3: What is "Time Travel" in LangGraph, and how is it implemented?
**Answer:**
Time travel refers to inspecting, rewinding, or modifying past states of an agent run. It is implemented by LangGraph's checkpointer mechanism, which saves an immutable snapshot (checkpoint) of the state after each step. Each checkpoint has a unique ID. By passing `checkpoint_id` inside `config["configurable"]`, developers can load the exact memory state at that point in time, fork the thread, or test hypothetical fixes.

---

### Q4: If an LLM calls a tool with invalid arguments, how do you handle and debug it?
**Answer:**
1. In LangGraph, configure `ToolNode(tools, handle_tool_errors=True)`.
2. When an exception occurs, `ToolNode` catches it, formats the error as a `ToolMessage` (e.g. `Error: Invalid argument 'x'`), and passes it back to the LLM.
3. The LLM reads the error message in the next turn and self-corrects its tool call arguments.
4. Traces in LangSmith will clearly show the red error badge on the tool call and the subsequent correction step.
