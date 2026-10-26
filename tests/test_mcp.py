from fastmcp.client import Client
import asyncio

from MLXOllama import mcp_client
HA_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4NGQ0ZjE2MmY2YzI0OGNmYmE1YTA1YjdmNDQ3ZTRmNiIsImlhdCI6MTc3MzI0OTU4NCwiZXhwIjoyMDg4NjA5NTg0fQ.EIV_RM1hK5UeLAB8sdzPep2r1iuLxCu9D6gcTt-RNxA"

MCP_CLIENT_CONFIG = {
    "mcpServers": {
        # "MCPAssist": {
            # "url": "http://10.27.81.207:8090",
        # },
        "homeassistant": {
            "url": "http://10.27.81.207:8123/api/mcp",
            "headers": {
                "Authorization": f"Bearer {HA_TOKEN}"
            }
        },
        "web_fetch": {
            "command": "uvx",
            "args": ["mcp-server-fetch"]
        },
        # "searxng-http": {
            # "url": "http://10.27.81.60:3003/mcp",
            # "env": {
                # "SEARXNG_URL": "http://10.27.81.60:3002/search",
                # "MCP_HTTP_PORT": "3003"
            # }
        # },
        # "tavily": {
        #     "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={config.TAVILY_TOKEN}",
        # },
    }
}

client = mcp_client.MCPService(MCP_CLIENT_CONFIG)
result=asyncio.run(client.list_tools())
print(result)
