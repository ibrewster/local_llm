import asyncio
import fnmatch
import logging
from typing import List, Dict, Optional

from fastmcp.client import Client
from fastmcp.exceptions import MCPError, ClientError

from . import config, local_tools


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
            return await client.call_tool(tool_name, tool_args, raise_on_error=False)
        except (MCPError, ClientError, ConnectionError, BrokenPipeError):
            # If server restarted or connection dropped, reset and retry once
            await self.reset()
            client = await self.get_client()
            return await client.call_tool(tool_name, tool_args, raise_on_error=False)

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


# ---------------------------------------------------------------------------
# Configuration & Singleton
# ---------------------------------------------------------------------------

MCP_CLIENT_CONFIG = {
    "mcpServers": {
        "homeassistant": {
            "url": "http://10.27.81.207:8123/api/mcp",
            "headers": {
                "Authorization": f"Bearer {config.HA_TOKEN}"
            }
        },
        "HA-MCP": {
            "url": "http://10.27.81.207:9584/private_Mj4Aa5j0ba8NrDJJliaohw",
        },
        "web_fetch": {
            "command": "uvx",
            "args": ["mcp-server-fetch"]
        },
    }
}

MCP_CLIENT = MCPService(MCP_CLIENT_CONFIG)

# Standard MCP tools list — pre-computed at startup, used by merge_local_tools
STANDARD_MCP_TOOLS: List[Dict] = []

SKIPPED_MCP_TOOLS = {
    'homeassistant_GetLiveContext',
    'MCPAssist_perform_action',
    'homeassistant_HassMediaSearchAndPlay',
    'homeassistant_HassMediaPause',
    # Wildcard patterns (using glob syntax) to exclude entire tool groups
    'HA-MCP_*',
}


# ---------------------------------------------------------------------------
# Tool management
# ---------------------------------------------------------------------------

def _tool_should_skip(tool_name: str) -> bool:
    """Check if a tool name matches any skip pattern.
    
    Supports both exact matches and glob-style wildcards (e.g., 'HA-MCP_*').
    """
    for pattern in SKIPPED_MCP_TOOLS:
        if fnmatch.fnmatch(tool_name, pattern):
            return True
    return False


async def build_standard_mcp_tools() -> List[Dict]:
    """Fetch and transform MCP tools into OpenAI-compatible format.

    Returns a list of tool schemas with skipped tools filtered out.
    Used by ``mcp_tools_refresh_loop`` to periodically refresh
    ``STANDARD_MCP_TOOLS``.
    """
    raw = await MCP_CLIENT.list_tools()
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema,
            },
        }
        for tool in raw
        if not _tool_should_skip(tool.name)
    ]


async def mcp_tools_refresh_loop():
    """Background task that periodically refreshes the standard MCP tools list.

    Runs every ``config.MCP_TOOLS_REFRESH_INTERVAL`` seconds, fetching the
    latest MCP tools and updating ``STANDARD_MCP_TOOLS`` so that
    ``merge_local_tools`` always sees a recent list without any per-request
    async overhead.
    """
    while True:
        try:
            global STANDARD_MCP_TOOLS
            STANDARD_MCP_TOOLS = await build_standard_mcp_tools()
            logging.info(f"Refreshed {len(STANDARD_MCP_TOOLS)} standard MCP tools")
        except Exception as e:
            logging.error(f"MCP tools refresh failed: {e}", exc_info=True)
        await asyncio.sleep(config.MCP_TOOLS_REFRESH_INTERVAL)


def merge_local_tools(
    tools: Optional[List[Dict]],
    include: Optional[List[str]] = None,
) -> Optional[List[Dict]]:
    """Merge selected server-side tools into the client-provided tools list.

    ``include`` selects from ``"STANDARD_MCP_TOOLS"`` and ``"local_tools"``.
    By default, both are included to preserve the existing behavior. When the
    client already supplied a tools list, duplicates are skipped by tool name
    so the client's version takes precedence.

    Returns a new list (or ``None`` when no tools are involved).
    """
    available_tools = {
        "STANDARD_MCP_TOOLS": STANDARD_MCP_TOOLS,
        "local_tools": list(local_tools.LOCAL_TOOLS.values()),
    }
    if include is None:
        include = ["STANDARD_MCP_TOOLS", "local_tools"]
    unknown_includes = set(include) - available_tools.keys()
    if unknown_includes:
        sources = ", ".join(sorted(unknown_includes))
        raise ValueError(f"Unknown tool source(s): {sources}")
    all_server_tools = [
        schema for source in include for schema in available_tools[source]
    ]

    if tools is None:
        return list(all_server_tools)

    client_names = {t.get("function", {}).get("name") for t in tools}
    merged = list(tools)  # shallow copy to avoid mutating the caller
    for schema in all_server_tools:
        if schema["function"]["name"] not in client_names:
            merged.append(schema)
    return merged