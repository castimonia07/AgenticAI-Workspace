from mcp.server.fastmcp import FastMCP

mcp=FastMCP("Math")

@mcp.tool()
def add(a:int,b:int)->int:
    """add two numbers"""
    return a+b


@mcp.tool()
def multiply(a:int,b:int)->int:
    """multiply two numbers"""
    return a*b

if __name__=="__main__":
    mcp.run(transport="stdio")

    
# The transport="stdio" argument tells the server to use standard input/output for communication, 
# which is useful for testing and debugging. You can change it to "http" or "websocket" for different transport methods.



# in short ishne ye kra ki server ko run karna hai with a specific transport method.