from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

import asyncio

async def main():
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

    import os
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

    tools=await client.get_tools()
    model=ChatGroq(
        model="openai/gpt-oss-20b"
    )
    agent=create_react_agent(model,tools)

    math_response = await agent.ainvoke(
        {
            "messages": [
                ("user", "What is the sum of 5 and 10?")
            ]
        }
    )
    print("Math Response:",math_response["messages"][-1].content)


asyncio.run(main())