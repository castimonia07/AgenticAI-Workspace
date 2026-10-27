# 1-BasicChatbot: LangGraph Fundamentals, ReAct, Memory & Human-in-the-Loop (HITL)

Welcome to the **1-BasicChatbot** module. This directory contains the foundational building blocks of Agentic AI using **LangGraph**, **LangChain**, and **Groq LLM**. It transitions from a simple conversational graph to a full ReAct agent with tools, conversation memory/checkpointing, streaming outputs, and Human-in-the-Loop (HITL) pause-and-resume workflows.

---

## Table of Contents
1. [Core Concepts & Architecture](#core-concepts--architecture)
2. [Deep Dive: Notebook Walkthroughs](#deep-dive-notebook-walkthroughs)
   - [BasicChatbot.ipynb](#1-basicchatbotipynb)
   - [humanintheloop.ipynb](#2-humanintheloopipynb)
3. [Component Breakdown](#component-breakdown)
   - [State & Reducers (`add_messages`)](#state--reducers-add_messages)
   - [Graph Topology: Nodes & Edges](#graph-topology-nodes--edges)
   - [ReAct Agent Architecture (Cyclical Tool Calling)](#react-agent-architecture-cyclical-tool-calling)
   - [Memory & Checkpointing (`InMemorySaver` / `MemorySaver`)](#memory--checkpointing-inmemorysaver--memorysaver)
   - [Streaming Paradigms (`updates`, `values`, `astream_events`)](#streaming-paradigms-updates-values-astream_events)
   - [Human-in-the-Loop (`interrupt` and `Command`)](#human-in-the-loop-interrupt-and-command)
4. [Step-by-Step Code Walkthrough](#step-by-step-code-walkthrough)
5. [Interview Masterclass: Questions, Answers & Deep Dives](#interview-masterclass-questions-answers--deep-dives)

---

## Core Concepts & Architecture

### What is LangGraph?
LangGraph is an orchestration framework by LangChain designed to model agentic workflows as **stateful, multi-actor applications with cyclic computational graphs**. Unlike linear chains (like LangChain Expression Language / LCEL), real agents require:
- **Loops / Cycles**: Try an action, observe output, reflect, and try again until a stopping condition is met.
- **Persistence (Checkpoints)**: Save every step to allow conversational memory, time-travel, fault tolerance, and human intervention.
- **Controllability**: Precise control over state transitions without black-box agent behavior.

```
       [START]
          │
          ▼
   ┌──────────────┐
   │  Chatbot /   │ ◄──────────────────┐
   │  Tool LLM   │                    │
   └──────┬───────┘                    │
          │ tools_condition            │ (Feedback loop)
     ┌────┴────────────┐               │
     │ Tool Call?      │               │
     ├────────┬────────┤               │
    YES       │        NO              │
     │        │        │               │
     ▼        │        ▼               │
┌─────────┐   │     [END]              │
│ ToolNode│   │                        │
└────┬────┘   │                        │
     └────────┴────────────────────────┘
```

---

## Deep Dive: Notebook Walkthroughs

### 1. `BasicChatbot.ipynb`
This notebook incrementally builds an enterprise-grade chatbot across 4 phases:

1. **Phase 1: Minimal Stateful Chatbot**
   - Defines a `State` TypedDict containing `messages: Annotated[list, add_messages]`.
   - Introduces `add_messages` as a reducer (instead of overwriting the message list, it appends or replaces messages matching an ID).
   - Defines a node `chatbot(state: State)` calling `llm.invoke(state['messages'])`.
   - Connects `START -> llmchatbot -> END`.
   - Compiles and executes via `.invoke()` and `.stream()`.

2. **Phase 2: Adding Tools (Tavily Search & Arithmetic)**
   - Integrates web search (`TavilySearch(max_results=2)`) and custom Python function tools (`multiply(a: int, b: int)`).
   - Binds tools to the Groq model using `llm.bind_tools(tools)`.
   - Sets up `ToolNode(tools)` to execute tool calls emitted by the model.
   - Utilizes `tools_condition` to route conditionally: if the LLM emitted a `tool_calls` request, route to `ToolNode`; otherwise route to `END`.

3. **Phase 3: The ReAct Cyclic Feedback Loop**
   - Closes the cycle by adding an edge from `tools` back to `tool_calling_llm`.
   - The LLM receives the tool output as a `ToolMessage` and synthesizes the final human-readable answer.

4. **Phase 4: Persistence & Memory Checkpointing**
   - Introduces `InMemorySaver`.
   - Compiles the graph with `checkpointer=memory`.
   - Passes a execution context `config = {"configurable": {"thread_id": "1"}}`.
   - Proves conversational context: The agent remembers "Hello, my name is Rishabh" when asked "What is my name?" in a subsequent turn.

5. **Phase 5: Streaming Strategies**
   - Compares `stream_mode="updates"` (streams only the state diff produced by each node) vs `stream_mode="values"` (streams the entire state after each node runs).
   - Demonstrates asynchronous event streaming with `.astream_events(..., version="v2")`.

---

### 2. `humanintheloop.ipynb`
Demonstrates **Human-in-the-Loop (HITL)** orchestration using LangGraph's modern primitives: `interrupt()` and `Command(resume=...)`.

#### Why Human-in-the-Loop?
Autonomous agents in production cannot be allowed to take destructive or high-stakes actions (sending funds, modifying production databases, answering critical compliance questions) without human supervision.

#### The HITL Workflow:
1. The agent decides it needs assistance or approval and calls a special tool:
   ```python
   @tool
   def human_assistance(query: str) -> str:
       """Request assistance from a human."""
       human_response = interrupt({"query": query})
       return human_response["data"]
   ```
2. When `interrupt()` is encountered:
   - LangGraph pauses execution right at that point.
   - Graph state is persisted to the checkpointer (`MemorySaver`).
   - The thread waits indefinitely without consuming active compute.
3. A human operator reviews the interrupted request.
4. The system resumes execution by sending a `Command`:
   ```python
   human_command = Command(resume={"data": human_response})
   graph.stream(human_command, config, stream_mode="values")
   ```
5. Execution resumes inside `human_assistance`, assigns `human_response["data"]` to the return value, and passes the result back to the LLM to complete its turn.

---

## Component Breakdown

### State & Reducers (`add_messages`)
In LangGraph, state is passed between nodes as a dictionary or `TypedDict`.

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
```

#### What does `Annotated[list, add_messages]` do?
- **Default behavior without reducer**: If a node returns `{"messages": [new_msg]}`, the key `messages` in the state is completely overwritten by `[new_msg]`.
- **Reducer behavior (`add_messages`)**: Tells LangGraph how to merge the node output with existing state:
  1. If a message has a new `id`, it is **appended** to the list.
  2. If a message has an existing `id`, it **overwrites/updates** that specific message (useful for editing messages or streaming tokens).
  3. Supports casting dicts (e.g. `{"role": "user", "content": "..."}`) into LangChain message objects (`HumanMessage`, `AIMessage`).

---

### ReAct Agent Architecture (Cyclical Tool Calling)

**ReAct** stands for **Reasoning + Acting**.
1. **Thought / Reason**: LLM inspects user input and decides whether to respond directly or invoke a tool.
2. **Action**: LLM outputs a structured `tool_calls` payload (function name & arguments).
3. **Observation**: LangGraph's `ToolNode` executes the tool and produces a `ToolMessage` containing the result.
4. **Reflection**: The `ToolMessage` is fed back into the LLM. The LLM checks if it has enough data to answer or if it needs another tool.

```python
builder = StateGraph(State)
builder.add_node("tool_calling_llm", call_llm_model)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    tools_condition
)
builder.add_edge("tools", "tool_calling_llm") # The ReAct Loop!
```

---

### Memory & Checkpointing (`InMemorySaver` / `MemorySaver`)

Without a checkpointer, every invocation of `graph.invoke()` starts with a blank slate.
- **Checkpointer**: Saves a snapshot of the `State` after every super-step.
- **Thread ID (`thread_id`)**: A unique partition key for a conversation session.

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "session_user_42"}}
graph.invoke({"messages": [HumanMessage(content="My secret is 12345")]}, config=config)
# Next turn, same thread_id:
graph.invoke({"messages": [HumanMessage(content="What is my secret?")]}, config=config)
# Agent remembers 12345!
```

---

### Streaming Paradigms (`updates`, `values`, `astream_events`)

LangGraph supports multiple streaming modes:
1. `stream_mode="values"`:
   - Emits the complete state dictionary after each step completes.
   - Ideal for UIs that re-render the entire message list.
2. `stream_mode="updates"`:
   - Emits only the changes/deltas produced by each specific node.
   - Example: `{'tool_calling_llm': {'messages': [AIMessage(...)]}}`.
   - Highly efficient; saves bandwidth.
3. `.astream_events(..., version="v2")`:
   - Fine-grained event-by-event streaming (token streaming from the LLM, tool start/end events, node start/end events).

---

## Interview Masterclass: Questions, Answers & Deep Dives

### Q1: What is the fundamental difference between LangChain (LCEL) and LangGraph?
**Answer:**
LCEL (LangChain Expression Language) is strictly a **Directed Acyclic Graph (DAG)** suited for linear chains or pipelines (e.g. Prompt -> LLM -> OutputParser). It cannot represent native loops, cycles, or state mutations across dynamic steps.
**LangGraph** is a generalized **cyclic computation graph**. It allows nodes to route back to previous nodes (enabling ReAct loops, multi-agent debates, validation retries), features built-in checkpointing for state persistence, and provides native Human-in-the-Loop breakpoints.

---

### Q2: How does `tools_condition` determine whether to route to `tools` or `END`?
**Answer:**
`tools_condition` is a prebuilt conditional edge function in LangGraph. It inspects the *last message* in `state["messages"]`:
1. It checks if the message is an `AIMessage` and whether `message.tool_calls` exists and is non-empty.
2. If `tool_calls` contains items, it returns the string `"tools"`.
3. If no `tool_calls` are present, it returns `END` (`"__end__"`).

---

### Q3: Explain how LangGraph handles Human-in-the-Loop (HITL) without losing execution context.
**Answer:**
LangGraph uses the `interrupt()` function paired with a checkpointer (`MemorySaver`, `PostgresSaver`, etc.):
1. When `interrupt(data)` is called inside a node or tool, LangGraph halts execution immediately.
2. The current graph state and the execution point are serialized and saved to the checkpointer under the specified `thread_id`.
3. The server can restart or handle other requests; state is completely safe in storage.
4. When human feedback is ready, the client calls `graph.stream(Command(resume=payload), config=config)`.
5. LangGraph loads the checkpoint for `thread_id`, restores the exact frame where it stopped, delivers `payload` as the return value of `interrupt()`, and resumes execution.

---

### Q4: What is a Reducer in LangGraph, and what happens if you omit `add_messages`?
**Answer:**
A reducer specifies how updates from a node are applied to a specific key in the State.
If you declare:
```python
class State(TypedDict):
    messages: list
```
And a node returns `{"messages": [new_msg]}`, LangGraph replaces the entire list of messages with `[new_msg]`. The conversation history is wiped out.
By declaring:
```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
```
LangGraph uses `add_messages` to append new messages, deduplicate existing messages by ID, and maintain full conversational history.

---

### Q5: How do you achieve LLM token-by-token streaming in LangGraph?
**Answer:**
Use the asynchronous `.astream_events(..., version="v2")` method:
```python
async for event in graph.astream_events(input_data, config, version="v2"):
    kind = event["event"]
    if kind == "on_chat_model_stream":
        content = event["data"]["chunk"].content
        print(content, end="", flush=True)
```
This listens directly to the chat model's chunk generation events while the graph runs.
