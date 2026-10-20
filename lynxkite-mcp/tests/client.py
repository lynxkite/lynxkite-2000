"""Test client for LynxKite MCP. Lists all available tools and resources from the server.
Run it from the root of the repository with:
    python lynxkite-mcp/tests/client.py
"""

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    # Define server parameters for stdio connection
    server_params = StdioServerParameters(
        command="python",
        args=[
            "lynxkite-mcp/src/lynxkite_mcp/assistant_mcp.py",
            "examples/Basic examples/Airlines demo.lynxkite.json",
        ],
    )

    # Start the stdio client and get the read/write streams for communication
    async with stdio_client(server_params) as (read, write):
        # Create a client session using the communication streams
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Client-Server interactions go here...

            # List all tools the server provides
            tools_response = await session.list_tools()
            print("Available tools:")
            for tool in tools_response.tools:
                print(f" - {tool.name}: {tool.description}")

            # List all resources
            resources_response = await session.list_resources()
            print("\nAvailable resources:")
            for res in resources_response.resources:
                print(f" - {res.uri}: {res.description}")


if __name__ == "__main__":
    asyncio.run(main())
