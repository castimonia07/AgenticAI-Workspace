from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import END, START
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
import os
from dotenv import load_dotenv
load_dotenv()


os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING"]="true"

os.environ["LANGSMITH_PROJECT"]="TestProject"

from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def make_tool_graph():
    ## Graph with tool call
    from langchain_core.tools import tool

    @tool 
    def add(a:float,b:float)->float:
        """add two numbers"""
        return a+b

    tools=[add]

    tool_node=ToolNode([add])

    llm_with_tool=llm.bind_tools([add])

    def call_llm_model(state:State):
        return {"messages":[llm_with_tool.invoke(state['messages'])]}

    ## Node definition
    def call_llm_model(state:State):
        return {"messages":[llm_with_tool.invoke(state['messages'])]}
    ## Grpah
    builder=StateGraph(State)
    builder.add_node("tool_calling_llm",call_llm_model)
    builder.add_node("tools",ToolNode(tools))

    ## Add Edges
    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges(
        "tool_calling_llm",
        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
        tools_condition
    )
    builder.add_edge("tools","tool_calling_llm")

    ## compile the graph
    graph=builder.compile()
    return graph


tool_agent=make_tool_graph()