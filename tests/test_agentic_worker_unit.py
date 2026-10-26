"""
Unit tests for agentic_worker.py - no server required.

Tests the individual components like _parse_steps and plan_and_execute
with mocked LLM responses.

Note: _parse_steps is a pure function that can be tested directly.
plan_and_execute tests mock the worker.generate_stream function.
"""

import asyncio
import importlib.util
import json
import sys
import unittest
from unittest.mock import patch, AsyncMock, MagicMock

# Load agentic_worker directly from file to avoid the app.route decorator issue
# We'll read the source and extract just the functions we need to test
def _load_agentic_functions():
    """Load _parse_steps and plan_and_execute from the source file without triggering the route decorator."""
    spec = importlib.util.spec_from_file_location(
        "MLXOllama.agentic_worker",
        "MLXOllama/agentic_worker.py",
        submodule_search_locations=[]
    )
    
    # Create mock modules for dependencies
    mock_config = MagicMock()
    mock_config.MAIN_MODEL = 'test-model'
    mock_config.QUICK_MODEL = 'test-model'
    
    mock_worker = MagicMock()
    mock_utils = MagicMock()
    mock_common = MagicMock()
    mock_common.message_text = lambda x: x if isinstance(x, str) else (x[0]['text'] if isinstance(x, list) else str(x))
    mock_common.read_chat_request = AsyncMock(return_value=({}, []))
    
    mock_openai = MagicMock()
    def mock_format_tool_calls(tool_calls, for_stream=False):
        formatted = []
        for idx, tc in enumerate(tool_calls):
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            args_str = json.dumps(args) if isinstance(args, dict) else str(args or "{}")
            item = {
                "id": tc.get("id", f"call_{idx}"),
                "type": "function",
                "function": {
                    "name": fn.get("name", ""),
                    "arguments": args_str,
                },
            }
            if for_stream:
                item["index"] = idx
            formatted.append(item)
        return formatted
    mock_openai._format_tool_calls = mock_format_tool_calls

    mock_quart = MagicMock()
    mock_app = MagicMock()
    mock_app.route = MagicMock(return_value=lambda x: x)

    # Install mocks before loading
    sys.modules['quart'] = mock_quart
    sys.modules['MLXOllama.config'] = mock_config
    sys.modules['MLXOllama.worker'] = mock_worker
    sys.modules['MLXOllama.utils'] = mock_utils
    sys.modules['MLXOllama.common'] = mock_common
    sys.modules['MLXOllama.openai'] = mock_openai

    # Create a mock parent module
    mock_parent = MagicMock()
    mock_parent.app = mock_app
    mock_parent.config = mock_config
    mock_parent.worker = mock_worker
    mock_parent.utils = mock_utils
    sys.modules['MLXOllama'] = mock_parent
    
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "MLXOllama"
    sys.modules['MLXOllama.agentic_worker'] = module
    spec.loader.exec_module(module)
    
    return module

# Load the functions
agentic_module = _load_agentic_functions()
_parse_steps = agentic_module._parse_steps
plan_and_execute = agentic_module.plan_and_execute


class TestParseSteps(unittest.TestCase):
    """Test the _parse_steps function which extracts numbered steps from LLM output."""

    def test_standard_numbered_list(self):
        """Test parsing standard numbered list format: 1. 2. 3."""
        plan_text = """1. Check the current weather in New York
2. Compare it with the weather in London
3. Summarize the differences"""
        
        steps = _parse_steps(plan_text)
        
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0], "Check the current weather in New York")
        self.assertEqual(steps[1], "Compare it with the weather in London")
        self.assertEqual(steps[2], "Summarize the differences")

    def test_numbered_list_with_parentheses(self):
        """Test parsing numbered list with parentheses: 1) 2) 3)."""
        plan_text = """1) Get the temperature sensor reading
2) Check if heating is on
3) Report the status"""
        
        steps = _parse_steps(plan_text)
        
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0], "Get the temperature sensor reading")
        self.assertEqual(steps[1], "Check if heating is on")
        self.assertEqual(steps[2], "Report the status")

    def test_step_n_format(self):
        """Test parsing Step N: format."""
        plan_text = """Step 1: Retrieve weather data
Step 2: Analyze the data
Step 3: Generate report"""
        
        steps = _parse_steps(plan_text)
        
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0], "Retrieve weather data")
        self.assertEqual(steps[1], "Analyze the data")
        self.assertEqual(steps[2], "Generate report")

    def test_step_n_with_colon_and_space(self):
        """Test parsing Step N: format with extra spaces."""
        plan_text = """Step 1:  Check living room temperature
Step 2:  Check kitchen temperature
Step 3:  Compare and report"""
        
        steps = _parse_steps(plan_text)
        
        self.assertEqual(len(steps), 3)
        self.assertTrue("living room" in steps[0])
        self.assertTrue("kitchen" in steps[1])

    def test_paragraph_fallback(self):
        """Test fallback to paragraph splitting when no numbered format is found."""
        plan_text = """First, we need to check the weather API for current conditions.

Next, we should compare the temperatures between locations.

Finally, we'll summarize the findings for the user."""
        
        steps = _parse_steps(plan_text)
        
        self.assertEqual(len(steps), 3)
        self.assertTrue("weather API" in steps[0])
        self.assertTrue("compare" in steps[1])
        self.assertTrue("summarize" in steps[2])

    def test_mixed_content_with_numbered_list(self):
        """Test that numbered list is extracted even with surrounding text."""
        plan_text = """Here's my plan:

1. Check weather in Boston
2. Check weather in Miami
3. Determine which is warmer

Let me know if you want me to proceed."""
        
        steps = _parse_steps(plan_text)
        
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0], "Check weather in Boston")
        self.assertEqual(steps[1], "Check weather in Miami")
        self.assertEqual(steps[2], "Determine which is warmer")

    def test_indented_numbered_list(self):
        """Test parsing indented numbered list."""
        plan_text = """    1. First step
    2. Second step
    3. Third step"""
        
        steps = _parse_steps(plan_text)
        
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0], "First step")
        self.assertEqual(steps[1], "Second step")
        self.assertEqual(steps[2], "Third step")

    def test_empty_input(self):
        """Test that empty input returns empty list."""
        steps = _parse_steps("")
        self.assertEqual(steps, [])

    def test_single_step(self):
        """Test parsing a single step."""
        plan_text = "1. Only one step needed"
        
        steps = _parse_steps(plan_text)
        
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0], "Only one step needed")

    def test_steps_with_complex_content(self):
        """Test parsing steps that contain complex content like URLs or special characters."""
        plan_text = """1. Fetch data from http://api.weather.com/v1/forecast?loc=NYC
2. Parse JSON response looking for "temperature" and "humidity" keys
3. Format output as: "Temp: {temp}°F, Humidity: {humidity}%" """
        
        steps = _parse_steps(plan_text)
        
        self.assertEqual(len(steps), 3)
        self.assertIn("http://", steps[0])
        self.assertIn("JSON", steps[1])
        self.assertIn("Temp:", steps[2])


class TestPlanAndExecute(unittest.TestCase):
    """Test the plan_and_execute function with mocked LLM responses."""

    def _create_mock_stream(self, response_text: str):
        """Create a mock async generator that yields a JSON response."""
        async def mock_generator():
            # Simulate the format generate_stream yields when stream=False
            yield '{"message": {"role": "assistant", "content": "' + response_text + '"}, "done": true}'
        return mock_generator

    def test_plan_and_execute_basic_flow(self):
        """Test the complete plan-and-execute flow with mocked responses."""
        # Mock plan response
        plan_response = """1. Check the current temperature in New York
2. Check the current temperature in London
3. Compare and summarize the difference"""
        
        # Mock execution responses
        exec_responses = [
            "New York temperature: 72°F (22°C), partly cloudy",
            "London temperature: 59°F (15°C), overcast",
            "New York is 13°F (7°C) warmer than London today. NY is partly cloudy while London is overcast."
        ]
        
        async def run_test():
            # Track which response to return
            call_count = [0]
            
            async def mock_generate_stream(stream=False, model_info=None, msg_history=None, options=None, tools=None, think=True, state=None):
                if call_count[0] == 0:
                    response = plan_response
                    call_count[0] += 1
                else:
                    idx = min(call_count[0] - 1, len(exec_responses) - 1)
                    response = exec_responses[idx]
                    call_count[0] += 1
                
                # Yield dict as _collect_completion expects chunk.get("message")
                yield {"message": {"role": "assistant", "content": response}, "done": True}
            
            with patch.object(agentic_module.worker, "generate_stream", mock_generate_stream):
                result = await plan_and_execute(
                    "Compare weather in New York and London",
                    model_name="test-model"
                )
            
            return result
        
        result = asyncio.run(run_test())
        
        # plan_and_execute returns a dict with "content" key
        content = result["content"]
        # Verify the result contains all three steps
        self.assertIn("Step 1:", content)
        self.assertIn("Step 2:", content)
        self.assertIn("Step 3:", content)
        self.assertIn("72°F", content)
        self.assertIn("59°F", content)

    def test_plan_and_execute_with_on_step_callback(self):
        """Test that on_step callback is called for each step."""
        plan_response = """1. First step
2. Second step"""
        
        exec_responses = ["Result 1", "Result 2"]
        
        async def run_test():
            call_count = [0]
            captured_steps = []
            
            async def on_step(step_idx, instruction, result):
                captured_steps.append((step_idx, instruction, result))
            
            async def mock_generate_stream(stream=False, model_info=None, msg_history=None, options=None, tools=None, think=True, state=None):
                if call_count[0] == 0:
                    response = plan_response
                    call_count[0] += 1
                else:
                    idx = min(call_count[0] - 1, len(exec_responses) - 1)
                    response = exec_responses[idx]
                    call_count[0] += 1
                
                # Yield dict as _collect_completion expects chunk.get("message")
                yield {"message": {"role": "assistant", "content": response}, "done": True}
            
            with patch.object(agentic_module.worker, "generate_stream", mock_generate_stream):
                await plan_and_execute(
                    "Test prompt",
                    model_name="test-model",
                    on_step=on_step
                )
            
            return captured_steps
        
        captured = asyncio.run(run_test())
        
        self.assertEqual(len(captured), 2)
        self.assertEqual(captured[0][0], 1)  # step index
        self.assertEqual(captured[0][2], "Result 1")  # result
        self.assertEqual(captured[1][0], 2)
        self.assertEqual(captured[1][2], "Result 2")

    def test_plan_and_execute_max_steps_cap(self):
        """Test that max_steps parameter limits the number of executed steps."""
        plan_response = """1. Step one
2. Step two
3. Step three
4. Step four
5. Step five"""
        
        exec_responses = ["One", "Two", "Three", "Four", "Five"]
        
        async def run_test():
            call_count = [0]
            
            async def mock_generate_stream(stream=False, model_info=None, msg_history=None, options=None, tools=None, think=True, state=None):
                if call_count[0] == 0:
                    response = plan_response
                    call_count[0] += 1
                else:
                    idx = min(call_count[0] - 1, len(exec_responses) - 1)
                    response = exec_responses[idx]
                    call_count[0] += 1
                
                # Yield dict as _collect_completion expects chunk.get("message")
                yield {"message": {"role": "assistant", "content": response}, "done": True}
            
            with patch.object(agentic_module.worker, "generate_stream", mock_generate_stream):
                result = await plan_and_execute(
                    "Test prompt",
                    model_name="test-model",
                    max_steps=2  # Cap at 2 steps
                )
            
            return result
        
        result = asyncio.run(run_test())
        
        # Should only have 2 steps despite plan having 5
        content = result["content"]
        self.assertIn("Step 1:", content)
        self.assertIn("Step 2:", content)
        self.assertNotIn("Step 3:", content)

    def test_plan_and_execute_empty_plan(self):
        """Test behavior when LLM returns an empty plan."""
        async def run_test():
            async def mock_generate_stream(stream=False, model_info=None, msg_history=None, options=None, tools=None, think=True, state=None):
                # Yield dict as _collect_completion expects chunk.get("message")
                yield {"message": {"role": "assistant", "content": ""}, "done": True}
            
            with patch.object(agentic_module.worker, "generate_stream", mock_generate_stream):
                result = await plan_and_execute(
                    "Test prompt",
                    model_name="test-model"
                )
            
            return result
        
        result = asyncio.run(run_test())
        
        # When no steps are extracted, should return the raw plan output (empty in this case)
        self.assertEqual(result["content"], "")

    def test_plan_and_execute_custom_prompts(self):
        """Test that custom system prompts are used."""
        custom_plan_prompt = "You are a planning assistant."
        custom_exec_prompt = "You are an execution assistant."
        
        plan_response = "1. Single step"
        exec_response = "Done"
        
        async def run_test():
            call_count = [0]
            prompts_used = []
            
            async def mock_generate_stream(stream=False, model_info=None, msg_history=None, options=None, tools=None, think=True, state=None):
                # Capture the system prompt used
                for msg in msg_history:
                    if msg.get("role") == "system":
                        prompts_used.append(msg.get("content", ""))
                        break
                
                if call_count[0] == 0:
                    response = plan_response
                    call_count[0] += 1
                else:
                    response = exec_response
                    call_count[0] += 1
                
                # Yield dict as _collect_completion expects chunk.get("message")
                yield {"message": {"role": "assistant", "content": response}, "done": True}
            
            with patch.object(agentic_module.worker, "generate_stream", mock_generate_stream):
                await plan_and_execute(
                    "Test prompt",
                    model_name="test-model",
                    plan_system_prompt=custom_plan_prompt,
                    execute_system_prompt=custom_exec_prompt
                )
            
            return prompts_used
        
        prompts_used = asyncio.run(run_test())
        
        self.assertIn(custom_plan_prompt, prompts_used)
        self.assertIn(custom_exec_prompt, prompts_used)


class TestPlanAndExecuteWithTools(unittest.TestCase):
    """Test plan_and_execute with tool support."""

    def test_tools_passed_to_generate_stream(self):
        """Test that tools are passed through to generate_stream during planning and execution."""
        plan_response = "1. Get the weather\n2. Summarize"
        exec_responses = ["Sunny, 75°F", "The weather is sunny and warm."]

        mock_tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}
            }
        }]

        async def run_test():
            call_count = [0]
            captured_tools = []

            async def mock_generate_stream(stream=False, model_info=None, msg_history=None, options=None, tools=None, think=True, state=None):
                captured_tools.append(tools)
                if call_count[0] == 0:
                    response = plan_response
                    call_count[0] += 1
                else:
                    idx = min(call_count[0] - 1, len(exec_responses) - 1)
                    response = exec_responses[idx]
                    call_count[0] += 1

                yield {"message": {"role": "assistant", "content": response}, "done": True}

            with patch.object(agentic_module.worker, "generate_stream", mock_generate_stream):
                await plan_and_execute(
                    "What's the weather?",
                    model_name="test-model",
                    tools=mock_tools,
                )

            return captured_tools

        captured_tools = asyncio.run(run_test())

        # Plan phase SHOULD now have tools (Bug #5 fix)
        self.assertEqual(captured_tools[0], mock_tools)
        # Execution phases SHOULD also have tools
        self.assertEqual(captured_tools[1], mock_tools)
        self.assertEqual(captured_tools[2], mock_tools)

    def test_tool_calls_forwarded_to_client(self):
        """Test that tool calls not found locally are forwarded to the caller."""
        plan_response = "1. Check temperature"

        mock_tools = [{
            "type": "function",
            "function": {
                "name": "get_temperature",
                "description": "Get current temperature",
                "parameters": {"type": "object", "properties": {}, "required": []}
            }
        }]

        async def run_test():
            call_count = [0]

            async def mock_generate_stream(stream=False, model_info=None, msg_history=None, options=None, tools=None, think=True, state=None):
                call_count[0] += 1
                if call_count[0] == 1:
                    # Planning phase
                    yield {"message": {"role": "assistant", "content": plan_response}, "done": True}
                else:
                    # Execution phase — LLM calls a tool the server cannot execute
                    yield {"message": {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "call_123", "function": {"name": "get_temperature", "arguments": "{}"}}
                    ]}, "done": True, "done_reason": "tool_call"}

            with patch.object(agentic_module.worker, "generate_stream", mock_generate_stream):
                result = await plan_and_execute(
                    "What's the temperature?",
                    model_name="test-model",
                    tools=mock_tools,
                )

            return result

        result = asyncio.run(run_test())
        # Tool calls should be present so the caller can forward them to the client
        self.assertIn("tool_calls", result)
        self.assertEqual(result["done_reason"], "tool_call")
        self.assertEqual(result["tool_calls"][0]["function"]["name"], "get_temperature")

    def test_tools_passed_during_planning(self):
        """Test that tools ARE passed during the planning phase so the LLM can plan tool-aware steps."""
        plan_response = "1. Step one"
        exec_responses = ["Done"]

        mock_tools = [{"type": "function", "function": {"name": "dummy", "description": "Dummy tool", "parameters": {}}}]

        async def run_test():
            call_count = [0]
            tools_per_call = []

            async def mock_generate_stream(stream=False, model_info=None, msg_history=None, options=None, tools=None, think=True, state=None):
                tools_per_call.append(tools)
                if call_count[0] == 0:
                    response = plan_response
                    call_count[0] += 1
                else:
                    response = exec_responses[0]
                    call_count[0] += 1

                yield {"message": {"role": "assistant", "content": response}, "done": True}

            with patch.object(agentic_module.worker, "generate_stream", mock_generate_stream):
                await plan_and_execute(
                    "Test",
                    model_name="test-model",
                    tools=mock_tools,
                )

            return tools_per_call

        tools_per_call = asyncio.run(run_test())
        # First call (planning) should now have tools (Bug #5 fix)
        self.assertEqual(tools_per_call[0], mock_tools)
        # Second call (execution) should also have tools
        self.assertEqual(tools_per_call[1], mock_tools)


class TestToolCallingRoundTrip(unittest.TestCase):
    """Test the full tool-calling round-trip: plan → tool call → client executes → resume → final answer."""

    def test_session_created_on_tool_call(self):
        """Test that a session is created and returned when the LLM calls a tool."""
        plan_response = "1. Get weather for Boston"

        async def run_test():
            call_count = [0]

            async def mock_generate_stream(stream=False, model_info=None, msg_history=None, options=None, tools=None, think=True, state=None):
                call_count[0] += 1
                if call_count[0] == 1:
                    yield {"message": {"role": "assistant", "content": plan_response}, "done": True}
                else:
                    yield {"message": {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "call_1", "function": {"name": "get_weather", "arguments": '{"location": "Boston"}'}}
                    ]}, "done": True, "done_reason": "tool_call"}

            with patch.object(agentic_module.worker, "generate_stream", mock_generate_stream):
                result = await plan_and_execute(
                    "What's the weather in Boston?",
                    model_name="test-model",
                    tools=[{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {}}}],
                )

            return result

        result = asyncio.run(run_test())

        self.assertIn("tool_calls", result)
        self.assertIn("session_id", result)
        self.assertEqual(result["done_reason"], "tool_call")
        # Session should be saved
        session = agentic_module._get_session(result["session_id"])
        self.assertIsNotNone(session)
        self.assertEqual(session.current_step_index, 1)
        self.assertEqual(len(session.steps), 1)

    def test_resume_after_tool_result(self):
        """Test resuming a session after the client executes the tool and sends back the result."""
        plan_response = "1. Get weather for Boston\n2. Summarize the result"

        async def run_test():
            call_count = [0]

            async def mock_generate_stream(stream=False, model_info=None, msg_history=None, options=None, tools=None, think=True, state=None):
                call_count[0] += 1
                if call_count[0] == 1:
                    # Planning phase
                    yield {"message": {"role": "assistant", "content": plan_response}, "done": True}
                elif call_count[0] == 2:
                    # Execution step 1 — LLM calls the tool
                    yield {"message": {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "call_1", "function": {"name": "get_weather", "arguments": '{"location": "Boston"}'}}
                    ]}, "done": True, "done_reason": "tool_call"}
                elif call_count[0] == 3:
                    # Resumed execution step 1 — LLM gets the tool result and produces output
                    yield {"message": {"role": "assistant", "content": "Boston: 72°F, sunny"}, "done": True}
                elif call_count[0] == 4:
                    # Execution step 2 — summarization
                    yield {"message": {"role": "assistant", "content": "Boston is sunny and warm at 72°F."}, "done": True}

            with patch.object(agentic_module.worker, "generate_stream", mock_generate_stream):
                # Phase 1: Initial request — gets tool call
                result1 = await plan_and_execute(
                    "What's the weather in Boston?",
                    model_name="test-model",
                    tools=[{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {}}}],
                )

                session_id = result1["session_id"]
                session = agentic_module._get_session(session_id)

                # Simulate client executing the tool and injecting the result
                session.execute_messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": session.pending_tool_calls,
                })
                session.execute_messages.append({
                    "role": "tool",
                    "content": "72°F, sunny",
                    "tool_call_id": "call_1",
                })

                # Phase 2: Resume with tool result
                result2 = await plan_and_execute(
                    "What's the weather in Boston?",
                    model_name="test-model",
                    session_id=session_id,
                )

            return result2

        result = asyncio.run(run_test())

        # Should have completed all steps
        content = result["content"]
        self.assertIn("Step 1:", content)
        self.assertIn("Step 2:", content)
        self.assertNotIn("tool_calls", result)
        # Session should be cleaned up
        self.assertIsNone(agentic_module._get_session("test-session"))

    def test_full_round_trip_with_http_helpers(self):
        """Test helper functions used by the HTTP endpoint for continuation detection."""
        # Test _has_tool_results
        messages_with_tool = [
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "function": {"name": "get_weather", "arguments": "{}"}}]},
            {"role": "tool", "content": "72°F", "tool_call_id": "call_1"},
        ]
        messages_without_tool = [
            {"role": "user", "content": "What's the weather?"},
        ]

        self.assertTrue(agentic_module._has_tool_results(messages_with_tool))
        self.assertFalse(agentic_module._has_tool_results(messages_without_tool))

        # Test _extract_session_id
        data_with_session = {"session_id": "abc123"}
        data_with_metadata = {"metadata": {"agentic_session_id": "def456"}}
        data_without_session = {}

        self.assertEqual(agentic_module._extract_session_id([], data_with_session), "abc123")
        self.assertEqual(agentic_module._extract_session_id([], data_with_metadata), "def456")
        self.assertIsNone(agentic_module._extract_session_id([], data_without_session))

    def test_inject_tool_results_into_session(self):
        """Test that tool results are properly injected into a session's execute messages."""
        session = agentic_module.AgenticSession(
            session_id="test-session",
            prompt="What's the weather?",
            model_name="test-model",
            plan_system_prompt="Plan prompt",
            execute_system_prompt="Execute prompt",
            max_steps=5,
            tools=[],
            steps=["Get weather", "Summarize"],
            current_step_index=1,
            step_outputs=[],
            accumulated="",
            execute_messages=[
                {"role": "system", "content": "Execute prompt"},
                {"role": "user", "content": "Execute this step:\n**Step 1: Get weather**"},
            ],
            pending_tool_calls=[
                {"id": "call_1", "function": {"name": "get_weather", "arguments": '{"location": "Boston"}'}}
            ],
        )

        messages = [
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "function": {"name": "get_weather", "arguments": "{}"}}]},
            {"role": "tool", "content": "72°F, sunny", "tool_call_id": "call_1"},
        ]

        agentic_module._inject_tool_results_into_session(session, messages)

        # Should have appended assistant tool call message and tool result message
        self.assertEqual(len(session.execute_messages), 4)
        self.assertEqual(session.execute_messages[2]["role"], "assistant")
        self.assertEqual(session.execute_messages[2]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(session.execute_messages[3]["role"], "tool")
        self.assertEqual(session.execute_messages[3]["content"], "72°F, sunny")
        # Bug #4 fix: pending_tool_calls should be cleared after injection
        self.assertEqual(session.pending_tool_calls, [])


class TestSessionCleanup(unittest.TestCase):
    """Test session TTL and cleanup functionality."""

    def test_cleanup_expired_sessions(self):
        """Test that expired sessions are removed by cleanup."""
        import time

        # Create an expired session (created 1 hour ago)
        expired_session = agentic_module.AgenticSession(
            session_id="expired-session",
            prompt="Test",
            model_name="test-model",
            plan_system_prompt="Plan",
            execute_system_prompt="Execute",
            max_steps=5,
            tools=[],
            steps=["Step 1"],
            current_step_index=1,
            step_outputs=[],
            accumulated="",
            execute_messages=[],
            pending_tool_calls=[],
            created_at=time.time() - 3600,  # 1 hour ago
        )
        agentic_module._save_session(expired_session)

        # Create a fresh session
        fresh_session = agentic_module.AgenticSession(
            session_id="fresh-session",
            prompt="Test",
            model_name="test-model",
            plan_system_prompt="Plan",
            execute_system_prompt="Execute",
            max_steps=5,
            tools=[],
            steps=["Step 1"],
            current_step_index=1,
            step_outputs=[],
            accumulated="",
            execute_messages=[],
            pending_tool_calls=[],
            created_at=time.time(),  # just now
        )
        agentic_module._save_session(fresh_session)

        # Run cleanup
        agentic_module._cleanup_expired_sessions()

        # Expired session should be removed
        self.assertIsNone(agentic_module._get_session("expired-session"))
        # Fresh session should still exist
        self.assertIsNotNone(agentic_module._get_session("fresh-session"))

        # Clean up
        agentic_module._remove_session("fresh-session")

    def test_session_ttl_constant(self):
        """Test that SESSION_TTL_SECONDS is set to 30 minutes."""
        self.assertEqual(agentic_module.SESSION_TTL_SECONDS, 1800)


class TestMaxTokens(unittest.TestCase):
    """Test that max_tokens is properly passed through."""

    def test_max_tokens_default(self):
        """Test that default max_tokens is 4096."""
        plan_response = "1. Step one"

        async def run_test():
            captured_options = []

            async def mock_generate_stream(stream=False, model_info=None, msg_history=None, options=None, tools=None, think=True, state=None):
                captured_options.append(options)
                yield {"message": {"role": "assistant", "content": plan_response}, "done": True}

            with patch.object(agentic_module.worker, "generate_stream", mock_generate_stream):
                await plan_and_execute(
                    "Test",
                    model_name="test-model",
                )

            return captured_options

        options_list = asyncio.run(run_test())
        # Default max_tokens should be 4096
        self.assertEqual(options_list[0]["max_tokens"], 4096)

    def test_max_tokens_custom(self):
        """Test that custom max_tokens is passed through."""
        plan_response = "1. Step one"

        async def run_test():
            captured_options = []

            async def mock_generate_stream(stream=False, model_info=None, msg_history=None, options=None, tools=None, think=True, state=None):
                captured_options.append(options)
                yield {"message": {"role": "assistant", "content": plan_response}, "done": True}

            with patch.object(agentic_module.worker, "generate_stream", mock_generate_stream):
                await plan_and_execute(
                    "Test",
                    model_name="test-model",
                    max_tokens=8192,
                )

            return captured_options

        options_list = asyncio.run(run_test())
        # Custom max_tokens should be 8192
        self.assertEqual(options_list[0]["max_tokens"], 8192)


class TestHomeAssistantScenario(unittest.TestCase):
    """Test a realistic Home Assistant scenario."""

    def test_home_assistant_plan_parsing(self):
        """Test parsing a typical Home Assistant monitoring plan."""
        plan_text = """1. Check the current state of all temperature sensors in the house
2. Verify which lights are currently on in each room
3. Check if the HVAC system is running and what mode it's in
4. Summarize the home status report"""
        
        steps = _parse_steps(plan_text)
        
        self.assertEqual(len(steps), 4)
        self.assertIn("temperature sensors", steps[0])
        self.assertIn("lights", steps[1])
        self.assertIn("HVAC", steps[2])
        self.assertTrue("summarize" in steps[3].lower(), 
            f"Expected 'summarize' in step 4, got: {steps[3]}")

    def test_weather_comparison_scenario(self):
        """Test a weather comparison scenario with multiple locations."""
        plan_text = """1. Look up the current weather in Boston, MA
2. Look up the current weather in Miami, FL
3. Look up the current weather in Denver, CO
4. Compare temperatures and determine which city is warmest
5. Create a summary table showing all three locations"""
        
        steps = _parse_steps(plan_text)
        
        self.assertEqual(len(steps), 5)
        self.assertIn("Boston", steps[0])
        self.assertIn("Miami", steps[1])
        self.assertIn("Denver", steps[2])
        self.assertIn("warmest", steps[3])
        self.assertIn("summary", steps[4])

    def test_weather_tool_definition(self):
        """Test a mock weather tool definition for integration tests."""
        # This is the tool definition that test clients would provide
        weather_tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name or coordinates"}
                    },
                    "required": ["location"]
                }
            }
        }
        self.assertEqual(weather_tool["function"]["name"], "get_weather")
        self.assertIn("location", weather_tool["function"]["parameters"]["properties"])

    def test_home_assistant_tool_definition(self):
        """Test a mock Home Assistant tool definition for integration tests."""
        ha_tool = {
            "type": "function",
            "function": {
                "name": "get_entity_state",
                "description": "Get the state of a Home Assistant entity",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_id": {"type": "string", "description": "Entity ID like light.living_room"}
                    },
                    "required": ["entity_id"]
                }
            }
        }
        self.assertEqual(ha_tool["function"]["name"], "get_entity_state")
        self.assertIn("entity_id", ha_tool["function"]["parameters"]["required"])


if __name__ == "__main__":
    unittest.main()
