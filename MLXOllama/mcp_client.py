import asyncio
from fastmcp.client import Client
from mcp.shared.exceptions import McpError

class MCPService:
    def __init__(self, config):
        self.config = config
        # auto_initialize=False prevents the startup log warnings
        self._client = Client(self.config)
        self._initialized = False
        self._lock = asyncio.Lock()
        self.available_tools = set()

    async def get_client(self):
        """Ensures the client is connected before returning it."""
        async with self._lock:
            if not self._initialized:
                await self._client.__aenter__()
                self._initialized = True
            return self._client

    async def call_tool(self, tool_name, tool_args=None):
        client = await self.get_client()
        try:
            return await client.call_tool(tool_name, tool_args)
        except (McpError, ConnectionError, BrokenPipeError):
            # If a server (like HA) restarted, reset and try once more
            self._initialized = False 
            client = await self.get_client()
            return await client.call_tool(tool_name, tool_args)

    async def list_tools(self):
        client = await self.get_client()
        tools = await client.list_tools()
        self.available_tools = {tool.name for tool in tools}
        return tools

    async def close(self):
        if self._initialized:
            await self._client.__aexit__(None, None, None)
            await self._client.close()