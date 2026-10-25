import asyncio
from fastmcp.client import Client
from fastmcp.exceptions import MCPError, ClientError

class MCPService:
    def __init__(self, config):
        self.config = config
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

    async def reset(self):
        """Resets the client connection and state."""
        async with self._lock:
            if self._initialized:
                try:
                    await self._client.close()
                except Exception:
                    pass
                self._initialized = False
            self._client = Client(self.config)

    async def call_tool(self, tool_name, tool_args=None):
        try:
            client = await self.get_client()
            return await client.call_tool(tool_name, tool_args)
        except (MCPError, ClientError, ConnectionError, BrokenPipeError):
            # If server restarted or connection dropped, reset and retry once
            await self.reset()
            client = await self.get_client()
            return await client.call_tool(tool_name, tool_args)

    async def list_tools(self):
        client = await self.get_client()
        tools = await client.list_tools()
        self.available_tools = {tool.name for tool in tools}
        return tools

    async def close(self):
        async with self._lock:
            if self._initialized:
                await self._client.close()
                self._initialized = False