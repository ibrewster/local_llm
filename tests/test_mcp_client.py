"""Tests for MLXOllama.mcp_client.MCPService."""
import asyncio
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure MLXOllama package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastmcp import FastMCP
from fastmcp.exceptions import ClientError, MCPError
from MLXOllama.mcp_client import MCPService


class TestMCPService(unittest.IsolatedAsyncioTestCase):
    """Unit and integration tests for MCPService."""

    async def asyncSetUp(self):
        # Create an in-memory FastMCP server with test tools
        self.server = FastMCP("test-mcp-server")

        @self.server.tool()
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        @self.server.tool()
        def echo(message: str) -> str:
            """Echo a message."""
            return message

        self.service = MCPService(self.server)

    async def asyncTearDown(self):
        await self.service.close()

    async def test_get_client_and_initialization(self):
        """Test client initialization and idempotent get_client calls."""
        self.assertFalse(self.service._initialized)
        client = await self.service.get_client()
        self.assertTrue(self.service._initialized)
        self.assertIsNotNone(client)

        # Subsequent call returns the same client instance without re-initializing
        client2 = await self.service.get_client()
        self.assertIs(client, client2)

    async def test_list_tools(self):
        """Test discovery of tools and updating of available_tools set."""
        tools = await self.service.list_tools()
        tool_names = {t.name for t in tools}
        self.assertIn("add", tool_names)
        self.assertIn("echo", tool_names)
        self.assertEqual(self.service.available_tools, {"add", "echo"})

    async def test_call_tool(self):
        """Test invoking an MCP tool and receiving results."""
        result = await self.service.call_tool("add", {"a": 10, "b": 32})
        self.assertIsNotNone(result)
        self.assertEqual(result.data, 42)

    async def test_call_tool_retry_on_connection_error(self):
        """Test automatic reset and retry upon encountering connection errors."""
        mock_client = AsyncMock()
        # First call fails with ConnectionError, retry succeeds
        mock_client.call_tool.side_effect = [
            ConnectionError("Lost connection to server"),
            MagicMock(data="recovered_result", content=[]),
        ]
        mock_client.__aenter__.return_value = mock_client
        mock_client.close = AsyncMock()

        with patch("MLXOllama.mcp_client.Client", return_value=mock_client):
            service = MCPService({"mcpServers": {}})
            result = await service.call_tool("echo", {"message": "hello"})

            self.assertEqual(result.data, "recovered_result")
            self.assertEqual(mock_client.call_tool.call_count, 2)
            await service.close()

    async def test_call_tool_retry_on_mcp_and_client_errors(self):
        """Test automatic reset and retry on MCPError and ClientError."""
        test_errors = [
            MCPError(code=-32603, message="Internal error"),
            ClientError("Client connection failed"),
            BrokenPipeError("Pipe broken"),
        ]
        for error_inst in test_errors:
            mock_client = AsyncMock()
            mock_client.call_tool.side_effect = [
                error_inst,
                MagicMock(data="success", content=[]),
            ]
            mock_client.__aenter__.return_value = mock_client
            mock_client.close = AsyncMock()

            with patch("MLXOllama.mcp_client.Client", return_value=mock_client):
                service = MCPService({"dummy": "config"})
                result = await service.call_tool("add", {"a": 1, "b": 2})
                self.assertEqual(result.data, "success")
                self.assertEqual(mock_client.call_tool.call_count, 2)
                await service.close()

    async def test_reset(self):
        """Test reset closes active client and constructs a fresh client instance."""
        client_initial = await self.service.get_client()
        self.assertTrue(self.service._initialized)

        await self.service.reset()
        self.assertFalse(self.service._initialized)
        self.assertIsNot(self.service._client, client_initial)

    async def test_close(self):
        """Test closing the service cleanly tears down the client."""
        await self.service.get_client()
        self.assertTrue(self.service._initialized)

        await self.service.close()
        self.assertFalse(self.service._initialized)


def test_mcp_service():
    """Top-level test function executing end-to-end verification of MCPService."""
    async def _runner():
        server = FastMCP("smoke-test-server")

        @server.tool()
        def ping() -> str:
            return "pong"

        service = MCPService(server)
        try:
            tools = await service.list_tools()
            assert "ping" in service.available_tools, "Tool 'ping' was not registered in available_tools"

            result = await service.call_tool("ping")
            assert result.data == "pong", f"Expected 'pong', got {result.data}"

            await service.reset()
            assert not service._initialized, "Service should not be initialized after reset"

            # Re-call after reset to verify re-initialization
            result2 = await service.call_tool("ping")
            assert result2.data == "pong", f"Expected 'pong' after reset, got {result2.data}"

            print("✅ test_mcp_service passed successfully.")
        finally:
            await service.close()

    asyncio.run(_runner())


if __name__ == "__main__":
    print("Running standalone test function...")
    test_mcp_service()
    print("\nRunning unittest suite...")
    unittest.main()
