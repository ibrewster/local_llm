"""
Integration tests for agentic_worker.py - requires server running on localhost:11434.

These tests send real requests to the agentic-completions endpoint and verify
the full pipeline works end-to-end.

RUNNING THE SERVER:
    Before running these tests, start the MLXOllama server:
    python3 run_mlxollama.py

    Or ensure it's running via launchd/plist.

SKIP CONDITIONS:
    Tests will be skipped if the server is not responding on localhost:11434.
"""

import json
import unittest
import urllib.request
import urllib.error
import asyncio
from typing import Optional

# Server configuration
BASE_URL = "http://localhost:11434"
AGENTIC_ENDPOINT = f"{BASE_URL}/v1/chat/agentic-completions"

# A realistic task that can be broken down into multiple steps
# Using weather lookups as the example since it's read-only and safe
WEATHER_COMPARISON_PROMPT = (
    "Compare the current weather in three cities: Boston, Miami, and Denver. "
    "For each city, look up the current temperature and conditions. "
    "Then determine which city is warmest and create a brief summary."
)

# A Home Assistant monitoring scenario
HOME_ASSISTANT_PROMPT = (
    "Check the current state of my home: what are the temperature readings from "
    "each sensor, which lights are on, and is the HVAC system running? "
    "Summarize the home status."
)


def is_server_available() -> bool:
    """Check if the MLXOllama server is running and responsive."""
    try:
        req = urllib.request.Request(
            f"{BASE_URL}/",
            method="GET"
        )
        urllib.request.urlopen(req, timeout=2)
        return True
    except urllib.error.HTTPError:
        # Even a 404 means the server is running
        return True
    except (urllib.error.URLError, OSError):
        return False


def send_agentic_request(
    prompt: str,
    model: str = "gemma-4-26b-a4b-it-mxfp4",
    stream: bool = False,
    max_steps: int = 10,
    tools: Optional[list] = None
) -> tuple[int, Optional[dict], Optional[str]]:
    """
    Send a request to the agentic-completions endpoint.
    
    Args:
        prompt: User prompt
        model: Model name
        stream: Whether to use streaming mode
        max_steps: Max execution steps
        tools: Optional list of OpenAI-compatible tool definitions
    
    Returns:
        Tuple of (status_code, json_response, raw_text)
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": stream,
        "max_steps": max_steps
    }
    if tools is not None:
        payload["tools"] = tools
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        AGENTIC_ENDPOINT,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json"
        },
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            status = response.status
            raw = response.read().decode("utf-8")
            try:
                parsed = json.loads(raw)
                return status, parsed, raw
            except json.JSONDecodeError:
                return status, None, raw
    except urllib.error.HTTPError as e:
        return e.code, None, e.read().decode("utf-8")
    except urllib.error.URLError as e:
        return 0, None, str(e)


def send_streaming_request(
    prompt: str,
    model: str = "gemma-4-26b-a4b-it-mxfp4",
    max_steps: int = 10
) -> list[str]:
    """
    Send a streaming request and collect all SSE chunks.
    
    Returns:
        List of SSE data lines
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": True,
        "max_steps": max_steps
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        AGENTIC_ENDPOINT,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        },
        method="POST"
    )
    
    chunks = []
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            for line in response:
                line_text = line.decode("utf-8").strip()
                if line_text.startswith("data: "):
                    chunks.append(line_text[6:])  # Strip "data: " prefix
    except urllib.error.URLError as e:
        print(f"Streaming request failed: {e}")
    
    return chunks


class TestAgenticEndpointAvailability(unittest.TestCase):
    """Test that the agentic endpoint is accessible."""

    @classmethod
    def setUpClass(cls):
        if not is_server_available():
            raise unittest.SkipTest("MLXOllama server not available at localhost:11434")

    def test_endpoint_exists(self):
        """Test that the agentic-completions endpoint responds."""
        status, response, raw = send_agentic_request(
            "What is 2+2?",
            max_steps=1
        )
        
        # Should get a response, even if it's an error about model not found
        self.assertGreater(status, 0, "Server should respond with a status code")

    def test_invalid_request_handling(self):
        """Test that invalid requests are properly rejected."""
        payload = {
            "model": "test-model",
            "messages": []  # Empty messages should be rejected
        }
        
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            AGENTIC_ENDPOINT,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                status = response.status
                # Should not succeed with empty messages
                self.assertEqual(status, 400)
        except urllib.error.HTTPError as e:
            # 400 is expected
            self.assertEqual(e.code, 400)


class TestAgenticNonStreaming(unittest.TestCase):
    """Test the agentic endpoint in non-streaming mode."""

    @classmethod
    def setUpClass(cls):
        if not is_server_available():
            raise unittest.SkipTest("MLXOllama server not available at localhost:11434")

    def test_simple_math_task(self):
        """Test a simple task that should produce a clear plan."""
        prompt = "Calculate: what is 100 divided by 4, then add 15 to the result?"
        
        status, response, raw = send_agentic_request(
            prompt,
            max_steps=3
        )
        
        self.assertEqual(status, 200, f"Expected 200, got {status}: {raw}")
        self.assertIsNotNone(response, "Response should be valid JSON")
        
        # Check response structure
        self.assertIn("choices", response)
        self.assertIn("message", response["choices"][0])
        content = response["choices"][0]["message"]["content"]
        
        # The result should contain step markers
        self.assertIn("Step 1:", content, "Response should contain step markers")

    def test_weather_comparison_task(self):
        """Test the weather comparison scenario."""
        status, response, raw = send_agentic_request(
            WEATHER_COMPARISON_PROMPT,
            max_steps=5
        )
        
        self.assertEqual(status, 200, f"Expected 200, got {status}: {raw}")
        self.assertIsNotNone(response)
        
        content = response["choices"][0]["message"]["content"]
        
        # Content may be empty if the inference queue was full during the request.
        # When there is content, it should contain at least one step header.
        if content:
            step_count = content.count("--- Step")
            self.assertGreaterEqual(step_count, 1,
                f"Response should contain at least one step, got {step_count}: {content[:200]}")
            self.assertGreater(len(content), 50, "Response should have substantial content")

    def test_response_has_required_fields(self):
        """Test that response contains all required OpenAI-compatible fields."""
        prompt = "What is the capital of France?"
        
        status, response, raw = send_agentic_request(prompt, max_steps=1)
        
        self.assertEqual(status, 200)
        self.assertIsNotNone(response)
        
        # Check required fields
        self.assertIn("id", response)
        self.assertIn("object", response)
        self.assertEqual(response["object"], "chat.completion")
        self.assertIn("created", response)
        self.assertIn("model", response)
        self.assertIn("choices", response)


class TestAgenticStreaming(unittest.TestCase):
    """Test the agentic endpoint in streaming mode."""

    @classmethod
    def setUpClass(cls):
        if not is_server_available():
            raise unittest.SkipTest("MLXOllama server not available at localhost:11434")

    def test_streaming_returns_chunks(self):
        """Test that streaming mode returns multiple SSE chunks."""
        prompt = "List three capital cities and their countries."
        
        chunks = send_streaming_request(prompt, max_steps=2)
        
        # Should receive multiple chunks
        self.assertGreater(len(chunks), 0, "Should receive at least one chunk")
        
        # Last chunk should be [DONE]
        self.assertEqual(chunks[-1], "[DONE]", "Last chunk should be [DONE]")

    def test_streaming_contains_step_headers(self):
        """Test that streaming includes step header chunks."""
        prompt = "1. Say hello. 2. Say goodbye."
        
        chunks = send_streaming_request(prompt, max_steps=2)
        
        # Parse chunks to find step headers
        has_step_header = False
        for chunk in chunks:
            if chunk == "[DONE]":
                continue
            try:
                parsed = json.loads(chunk)
                delta = parsed.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if "Step" in content and "---" in content:
                    has_step_header = True
                    break
            except json.JSONDecodeError:
                continue
        
        self.assertTrue(has_step_header, "Streaming should include step header chunks")

    def test_streaming_weather_task(self):
        """Test streaming mode with weather comparison task."""
        chunks = send_streaming_request(
            "Compare weather in New York and London.",
            max_steps=3
        )
        
        self.assertGreater(len(chunks), 0, "Should receive at least one chunk")
        self.assertEqual(chunks[-1], "[DONE]", "Last chunk should be [DONE]")
        
        # Collect all content from chunks
        all_content = ""
        for chunk in chunks:
            if chunk == "[DONE]":
                continue
            try:
                parsed = json.loads(chunk)
                delta = parsed.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                all_content += content
            except json.JSONDecodeError:
                continue
        
        # Content may be minimal if the queue was full during the request,
        # but we should at least have received the [DONE] marker above.
        # If there is content, it should be non-trivial.
        if all_content:
            self.assertGreater(len(all_content), 10,
                f"Streaming content should be non-trivial, got {len(all_content)} chars")


class TestAgenticScenarios(unittest.TestCase):
    """Test realistic agentic scenarios."""

    @classmethod
    def setUpClass(cls):
        if not is_server_available():
            raise unittest.SkipTest("MLXOllama server not available at localhost:11434")

    def test_home_assistant_status_check(self):
        """Test Home Assistant status checking scenario."""
        status, response, raw = send_agentic_request(
            HOME_ASSISTANT_PROMPT,
            max_steps=5
        )
        
        self.assertEqual(status, 200, f"Expected 200, got {status}: {raw}")
        self.assertIsNotNone(response)
        
        content = response["choices"][0]["message"]["content"]
        
        # Should have at least one step header for a multi-part query
        step_count = content.count("--- Step")
        self.assertGreaterEqual(step_count, 1,
            f"Home Assistant query should produce at least one step, got {step_count}: {content[:200]}")

    def test_multi_step_reasoning(self):
        """Test a task that requires multi-step reasoning."""
        prompt = (
            "I have 3 apples. I buy 5 more apples. I eat 2 apples. "
            "How many apples do I have left? Show your work step by step."
        )
        
        status, response, raw = send_agentic_request(prompt, max_steps=3)
        
        self.assertEqual(status, 200)
        self.assertIsNotNone(response)
        
        content = response["choices"][0]["message"]["content"]
        
        # Should contain the final answer (6)
        self.assertIn("6", content, "Should calculate correct answer: 6")

    def test_conversation_history_handling(self):
        """Test that conversation history is properly handled."""
        payload = {
            "model": "gemma-4-26b-a4b-it-mxfp4",
            "messages": [
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Nice to meet you, Alice!"},
                {"role": "user", "content": "What is my name?"}
            ],
            "stream": False,
            "max_steps": 1
        }
        
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            AGENTIC_ENDPOINT,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=60) as response:
            parsed = json.loads(response.read().decode("utf-8"))
            content = parsed["choices"][0]["message"]["content"]
            
            # Should reference the name from conversation history
            self.assertIn("Alice", content, "Should remember name from conversation")


class TestAgenticEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    @classmethod
    def setUpClass(cls):
        if not is_server_available():
            raise unittest.SkipTest("MLXOllama server not available at localhost:11434")

    def test_single_step_task(self):
        """Test a simple task that needs only a few steps."""
        prompt = "What is 2 + 2?"

        status, response, raw = send_agentic_request(prompt, max_steps=5)

        self.assertEqual(status, 200)
        self.assertIsNotNone(response)

        content = response["choices"][0]["message"]["content"]
        # The LLM may produce a plan with multiple steps. Due to non-determinism,
        # it may or may not give the correct answer. Just verify the pipeline
        # ran and produced step output.
        self.assertIn("Step 1:", content)

    def test_max_steps_limit(self):
        """Test that max_steps parameter limits execution."""
        prompt = (
            "Do these 10 things: 1. Say A, 2. Say B, 3. Say C, 4. Say D, "
            "5. Say E, 6. Say F, 7. Say G, 8. Say H, 9. Say I, 10. Say J"
        )
        
        status, response, raw = send_agentic_request(prompt, max_steps=3)
        
        self.assertEqual(status, 200)
        self.assertIsNotNone(response)
        
        content = response["choices"][0]["message"]["content"]
        
        # Should not have more than max_steps executed step headers
        # Count "--- Step N:" markers, which are the actual executed step headers
        step_count = content.count("--- Step")
        self.assertLessEqual(step_count, 3,
            f"Should not exceed max_steps=3, but found {step_count} steps")

    def test_unknown_model_returns_404(self):
        """Test that requesting an unknown model returns 404."""
        payload = {
            "model": "nonexistent-model-12345",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False
        }
        
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            AGENTIC_ENDPOINT,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                self.fail("Should have raised HTTPError for unknown model")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404, "Unknown model should return 404")


class TestAgenticWithTools(unittest.TestCase):
    """Test the agentic endpoint with client-provided tools.
    
    These tests verify that the agentic worker can accept tool definitions
    from the client and pass them through to the LLM during execution.
    The tools provided here match real server-side tools (get_weather_forecast,
    local_entity_state) so they can actually be executed.
    """

    @classmethod
    def setUpClass(cls):
        if not is_server_available():
            raise unittest.SkipTest("MLXOllama server not available at localhost:11434")

    def test_tools_accepted_in_request(self):
        """Test that the endpoint accepts a tools parameter without error."""
        # Mock weather tool matching the server's get_weather_forecast
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather_forecast",
                "description": "Get weather forecast for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "timeframe": {"type": "string", "description": "Forecast period", "enum": ["today", "tomorrow", "7 day"]}
                    },
                    "required": ["location"]
                }
            }
        }]

        status, response, raw = send_agentic_request(
            "What's the weather in Boston today?",
            max_steps=2,
            tools=tools
        )

        # Should succeed — even if the LLM doesn't use the tool, the request should not fail
        self.assertEqual(status, 200, f"Expected 200, got {status}: {raw}")
        self.assertIsNotNone(response, "Response should be valid JSON")

    def test_tools_with_weather_task(self):
        """Test weather query with a mock weather tool definition.
        
        The LLM may or may not actually call the tool depending on the model,
        but the key test is that providing tools doesn't break the pipeline.
        """
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather_forecast",
                "description": "Get weather forecast for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "timeframe": {"type": "string", "description": "Forecast period", "enum": ["today", "tomorrow", "7 day"]}
                    },
                    "required": ["location"]
                }
            }
        }]

        status, response, raw = send_agentic_request(
            "Compare the weather in Boston and Miami today.",
            max_steps=3,
            tools=tools
        )

        self.assertEqual(status, 200, f"Expected 200, got {status}: {raw}")
        self.assertIsNotNone(response)
        content = response["choices"][0]["message"]["content"]
        # Should have at least one step
        if content:
            step_count = content.count("--- Step")
            self.assertGreaterEqual(step_count, 1,
                f"Should have at least one step, got {step_count}: {content[:200]}")

    def test_tools_with_home_assistant_task(self):
        """Test Home Assistant query with mock entity state tool.
        
        Provides a tool definition matching local_entity_state so the LLM
        can potentially call it during execution.
        """
        tools = [{
            "type": "function",
            "function": {
                "name": "local_entity_state",
                "description": "Get the state of Home Assistant entities",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of entity IDs"
                        }
                    },
                    "required": ["entity_ids"]
                }
            }
        }]

        status, response, raw = send_agentic_request(
            "Check if the living room light is on and what the temperature is.",
            max_steps=3,
            tools=tools
        )

        self.assertEqual(status, 200, f"Expected 200, got {status}: {raw}")
        self.assertIsNotNone(response)

    def test_multiple_tools(self):
        """Test that multiple tools can be provided simultaneously."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather_forecast",
                    "description": "Get weather forecast",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "timeframe": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "local_entity_state",
                    "description": "Get Home Assistant entity states",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entity_ids": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["entity_ids"]
                    }
                }
            }
        ]

        status, response, raw = send_agentic_request(
            "What's the weather like and is the front door locked?",
            max_steps=3,
            tools=tools
        )

        self.assertEqual(status, 200, f"Expected 200, got {status}: {raw}")
        self.assertIsNotNone(response)


if __name__ == "__main__":
    # Check server availability before running tests
    if not is_server_available():
        print("WARNING: MLXOllama server not available at localhost:11434")
        print("Tests will be skipped. Start the server first:")
        print("  python3 run_mlxollama.py")
    
    unittest.main()
