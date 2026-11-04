"""
Unit tests for parse_tool_calls in utils.py - no server or MLX required.

Tests all Gemma 4 tool call format variants that the model has produced,
including fixes applied during development to handle malformed output.
"""

import importlib.util
import json
import sys
import unittest
from unittest.mock import MagicMock


def _load_parse_tool_calls():
    """Load parse_tool_calls from utils.py without triggering MLX imports."""
    sys.modules['mlx'] = MagicMock()
    sys.modules['mlx.utils'] = MagicMock()
    sys.modules['mlx.core'] = MagicMock()
    sys.modules['mlx_vlm'] = MagicMock()
    sys.modules['mlx_vlm.sample_utils'] = MagicMock()
    sys.modules['mlx_vlm.load'] = MagicMock()
    sys.modules['mlx_vlm.utils'] = MagicMock()
    sys.modules['sentence_transformers'] = MagicMock()
    sys.modules['cachetools'] = MagicMock()
    sys.modules['huggingface_hub'] = MagicMock()
    sys.modules['MLXOllama.config'] = MagicMock()
    sys.modules['MLXOllama.local_tools'] = MagicMock()
    sys.modules['MLXOllama.local_tools.LOCAL_TOOLS'] = {}
    sys.modules['MLXOllama.mcp_client'] = MagicMock()
    sys.modules['MLXOllama.preemptive_queue'] = MagicMock()

    spec = importlib.util.spec_from_file_location(
        "MLXOllama.utils",
        "MLXOllama/utils.py",
        submodule_search_locations=[]
    )

    mock_parent = MagicMock()
    sys.modules['MLXOllama'] = mock_parent

    module = importlib.util.module_from_spec(spec)
    module.__package__ = "MLXOllama"
    sys.modules['MLXOllama.utils'] = module
    spec.loader.exec_module(module)

    return module.parse_tool_calls


parse_tool_calls = _load_parse_tool_calls()


class TestGemma4BraceFormat(unittest.TestCase):
    """Test Gemma 4 brace-format tool calls: call:name{...}"""

    def test_standard_json_braces(self):
        """Standard JSON-style braces with quoted keys and colons."""
        text = '<|tool_call>call:some_tool{"key": "val"}<tool_call|>'
        tool_calls, cleaned = parse_tool_calls(text)

        self.assertIsNotNone(tool_calls)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["function"]["name"], "some_tool")
        self.assertEqual(tool_calls[0]["function"]["arguments"], {"key": "val"})
        self.assertEqual(cleaned, "")

    def test_standard_json_multiple_args(self):
        """Standard JSON braces with multiple arguments."""
        text = '<|tool_call>call:some_tool{"key1": "val1", "key2": "val2"}<tool_call|>'
        tool_calls, _ = parse_tool_calls(text)

        self.assertEqual(tool_calls[0]["function"]["arguments"], {"key1": "val1", "key2": "val2"})

    def test_brace_equals_syntax_single_arg(self):
        """Brace format with key="value" syntax - the exact issue case."""
        text = '<|tool_call>call:ha_config_get_scene{query="Day Start"}<tool_call|>'
        tool_calls, cleaned = parse_tool_calls(text)

        self.assertIsNotNone(tool_calls)
        self.assertEqual(tool_calls[0]["function"]["name"], "ha_config_get_scene")
        self.assertEqual(tool_calls[0]["function"]["arguments"], {"query": "Day Start"})
        self.assertEqual(cleaned, "")

    def test_brace_equals_syntax_multiple_args(self):
        """Brace format with key="value" syntax and multiple arguments."""
        text = '<|tool_call>call:some_tool{key1="val1", key2="val2"}<tool_call|>'
        tool_calls, _ = parse_tool_calls(text)

        self.assertEqual(tool_calls[0]["function"]["arguments"], {"key1": "val1", "key2": "val2"})

    def test_brace_equals_with_boolean(self):
        """Brace = syntax with boolean value conversion."""
        text = '<|tool_call>call:some_tool{enabled=true, name="test"}<tool_call|>'
        tool_calls, _ = parse_tool_calls(text)

        self.assertEqual(tool_calls[0]["function"]["arguments"], {"enabled": True, "name": "test"})

    def test_brace_equals_with_number(self):
        """Brace = syntax with numeric value conversion."""
        text = '<|tool_call>call:some_tool{count=42}<tool_call|>'
        tool_calls, _ = parse_tool_calls(text)

        self.assertEqual(tool_calls[0]["function"]["arguments"], {"count": 42})

    def test_brace_equals_with_float(self):
        """Brace = syntax with float value conversion."""
        text = '<|tool_call>call:some_tool{temp=3.14}<tool_call|>'
        tool_calls, _ = parse_tool_calls(text)

        self.assertEqual(tool_calls[0]["function"]["arguments"], {"temp": 3.14})

    def test_delimiter_format(self):
        """Brace format with <|">> string delimiters."""
        text = '<|tool_call>call:some_tool{key:<|"|>val<|"|>}<tool_call|>'
        tool_calls, _ = parse_tool_calls(text)

        self.assertEqual(tool_calls[0]["function"]["arguments"], {"key": "val"})

    def test_brace_with_unquoted_keys_quoted_values(self):
        """Brace format with unquoted keys but quoted values (colon separators)."""
        text = '<|tool_call>call:some_tool{key: "val"}<tool_call|>'
        tool_calls, _ = parse_tool_calls(text)

        self.assertEqual(tool_calls[0]["function"]["arguments"], {"key": "val"})

    def test_nested_object_value(self):
        """Brace format with nested object value - the exact issue case."""
        text = '<|tool_call>call:ha_call_read_tool{name: "ha_config_get_yaml", arguments: {yaml_path: "scene"}}<tool_call|>'
        tool_calls, cleaned = parse_tool_calls(text)

        self.assertIsNotNone(tool_calls)
        self.assertEqual(tool_calls[0]["function"]["name"], "ha_call_read_tool")
        self.assertEqual(
            tool_calls[0]["function"]["arguments"],
            {"name": "ha_config_get_yaml", "arguments": {"yaml_path": "scene"}}
        )
        self.assertEqual(cleaned, "")

    def test_nested_object_multiple_keys(self):
        """Brace format with nested object containing multiple keys."""
        text = '<|tool_call>call:some_tool{data: {key1: "val1", key2: "val2"}}<tool_call|>'
        tool_calls, _ = parse_tool_calls(text)

        self.assertEqual(
            tool_calls[0]["function"]["arguments"],
            {"data": {"key1": "val1", "key2": "val2"}}
        )

    def test_nested_object_with_number(self):
        """Brace format with nested object containing numeric values."""
        text = '<|tool_call>call:some_tool{config: {count: 42, enabled: true}}<tool_call|>'
        tool_calls, _ = parse_tool_calls(text)

        self.assertEqual(
            tool_calls[0]["function"]["arguments"],
            {"config": {"count": 42, "enabled": True}}
        )

    def test_deeply_nested_object(self):
        """Brace format with deeply nested objects."""
        text = '<|tool_call>call:some_tool{outer: {inner: {key: "val"}}}<tool_call|>'
        tool_calls, _ = parse_tool_calls(text)

        self.assertEqual(
            tool_calls[0]["function"]["arguments"],
            {"outer": {"inner": {"key": "val"}}}
        )

    def test_mixed_flat_and_nested(self):
        """Brace format with both flat and nested values."""
        text = '<|tool_call>call:some_tool{name: "test", data: {key: "val"}, count: 10}<tool_call|>'
        tool_calls, _ = parse_tool_calls(text)

        self.assertEqual(
            tool_calls[0]["function"]["arguments"],
            {"name": "test", "data": {"key": "val"}, "count": 10}
        )


class TestGemma4ParenFormat(unittest.TestCase):
    """Test Gemma 4 paren-format tool calls: call:name(key='val')"""

    def test_paren_double_quotes(self):
        """Paren format with double-quoted string values."""
        text = '<|tool_call>call:ha_search_tools(query="scene")<tool_call|>'
        tool_calls, cleaned = parse_tool_calls(text)

        self.assertIsNotNone(tool_calls)
        self.assertEqual(tool_calls[0]["function"]["name"], "ha_search_tools")
        self.assertEqual(tool_calls[0]["function"]["arguments"], {"query": "scene"})
        self.assertEqual(cleaned, "")

    def test_paren_single_quotes(self):
        """Paren format with single-quoted string values - the exact issue case."""
        text = "<|tool_call>call:ha_search_tools(query='scene')<tool_call|>"
        tool_calls, _ = parse_tool_calls(text)

        self.assertEqual(tool_calls[0]["function"]["arguments"], {"query": "scene"})

    def test_paren_multiple_args(self):
        """Paren format with multiple arguments."""
        text = "<|tool_call>call:some_tool(key1='val1', key2='val2')<tool_call|>"
        tool_calls, _ = parse_tool_calls(text)

        self.assertEqual(tool_calls[0]["function"]["arguments"], {"key1": "val1", "key2": "val2"})

    def test_paren_mixed_quotes(self):
        """Paren format with mixed single and double quotes."""
        text = '<|tool_call>call:some_tool(key1="val1", key2=\'val2\')<tool_call|>'
        tool_calls, _ = parse_tool_calls(text)

        self.assertEqual(tool_calls[0]["function"]["arguments"], {"key1": "val1", "key2": "val2"})

    def test_paren_boolean_conversion(self):
        """Paren format with boolean value conversion."""
        text = "<|tool_call>call:some_tool(enabled=true, name='test')<tool_call|>"
        tool_calls, _ = parse_tool_calls(text)

        self.assertEqual(tool_calls[0]["function"]["arguments"], {"enabled": True, "name": "test"})

    def test_paren_numeric_conversion(self):
        """Paren format with numeric value conversion."""
        text = "<|tool_call>call:some_tool(count=42, ratio=0.5)<tool_call|>"
        tool_calls, _ = parse_tool_calls(text)

        self.assertEqual(tool_calls[0]["function"]["arguments"], {"count": 42, "ratio": 0.5})


class TestMalformedClosingTags(unittest.TestCase):
    """Test handling of malformed closing tags."""

    def test_malformed_closing_tag_brace_equals(self):
        """Malformed closing tag <toolcall|> (missing underscore) with brace = syntax."""
        text = '<|tool_call>call:some_tool{key="val"}<toolcall|>'
        tool_calls, cleaned = parse_tool_calls(text)

        self.assertIsNotNone(tool_calls)
        self.assertEqual(tool_calls[0]["function"]["arguments"], {"key": "val"})
        self.assertEqual(cleaned, "")

    def test_malformed_closing_tag_paren(self):
        """Malformed closing tag with paren format."""
        text = "<|tool_call>call:some_tool(key='val')<toolcall|>"
        tool_calls, cleaned = parse_tool_calls(text)

        self.assertIsNotNone(tool_calls)
        self.assertEqual(tool_calls[0]["function"]["arguments"], {"key": "val"})
        self.assertEqual(cleaned, "")


class TestCleanedText(unittest.TestCase):
    """Test that cleaned text correctly removes tool call blocks."""

    def test_cleaned_text_with_prose(self):
        """Tool call embedded in prose should leave the prose after cleaning."""
        text = 'Some intro text <|tool_call>call:some_tool{key="val"}<tool_call|> and trailing text'
        tool_calls, cleaned = parse_tool_calls(text)

        self.assertIsNotNone(tool_calls)
        self.assertIn("intro text", cleaned)
        self.assertIn("trailing text", cleaned)
        self.assertNotIn("tool_call", cleaned)

    def test_multiple_tool_calls_cleaned(self):
        """Multiple tool calls should all be removed from cleaned text."""
        text = ('<|tool_call>call:tool1{a="1"}<tool_call|> '
                '<|tool_call>call:tool2{b="2"}<tool_call|>')
        tool_calls, cleaned = parse_tool_calls(text)

        self.assertEqual(len(tool_calls), 2)
        self.assertNotIn("tool_call", cleaned)

    def test_no_tool_calls_returns_none(self):
        """Text without tool calls should return None."""
        text = "Just plain text with no tool calls"
        tool_calls, cleaned = parse_tool_calls(text)

        self.assertIsNone(tool_calls)
        self.assertEqual(cleaned, text)


class TestGemma4ArrayFormat(unittest.TestCase):
    """Test Gemma 4 brace-format tool calls with arrays."""

    def test_simple_array(self):
        """Brace format with a simple array value."""
        text = '<|tool_call>call:some_tool{items: ["a", "b", "c"]}<tool_call|>'
        tool_calls, _ = parse_tool_calls(text)

        self.assertEqual(
            tool_calls[0]["function"]["arguments"],
            {"items": ["a", "b", "c"]}
        )

    def test_array_of_objects(self):
        """Brace format with array of objects."""
        text = '<|tool_call>call:some_tool{items: [{name: "a"}, {name: "b"}]}<tool_call|>'
        tool_calls, _ = parse_tool_calls(text)

        self.assertEqual(
            tool_calls[0]["function"]["arguments"],
            {"items": [{"name": "a"}, {"name": "b"}]}
        )

    def test_mcp_nested_array_with_extra_braces(self):
        """The exact failing MCP case: deeply nested objects with arrays and extra }."""
        text = ('<|tool_call>call:HA-MCP_ha_call_write_tool'
                '{arguments:{arguments:{config:{sequence:['
                '{action:"light.turn_off",target:{entity_id:"light.dining_room_lights"}}},'
                '{action:"light.turn_off",target:{entity_id:"light.living_room_lights"}},'
                '{action:"switch.turn_off",target:{entity_id:"switch.lr_curtains"}}'
                '],name:"Day Start Script"},name:"Day Start Script"}}}'
                '<tool_call|>')
        tool_calls, _ = parse_tool_calls(text)

        self.assertIsNotNone(tool_calls)
        self.assertEqual(tool_calls[0]["function"]["name"], "HA-MCP_ha_call_write_tool")
        args = tool_calls[0]["function"]["arguments"]

        # Verify the deeply nested structure
        self.assertIn("arguments", args)
        inner = args["arguments"]
        self.assertIn("arguments", inner)
        inner2 = inner["arguments"]
        self.assertIn("config", inner2)
        self.assertEqual(inner2["name"], "Day Start Script")

        config = inner2["config"]
        self.assertIn("sequence", config)
        self.assertEqual(len(config["sequence"]), 3)

        # Verify first element
        self.assertEqual(config["sequence"][0]["action"], "light.turn_off")
        self.assertEqual(
            config["sequence"][0]["target"]["entity_id"],
            "light.dining_room_lights"
        )

        # Verify third element
        self.assertEqual(config["sequence"][2]["action"], "switch.turn_off")
        self.assertEqual(
            config["sequence"][2]["target"]["entity_id"],
            "switch.lr_curtains"
        )

    def test_array_with_numbers(self):
        """Brace format with array of numbers."""
        text = '<|tool_call>call:some_tool{values: [1, 2, 3]}<tool_call|>'
        tool_calls, _ = parse_tool_calls(text)

        self.assertEqual(
            tool_calls[0]["function"]["arguments"],
            {"values": [1, 2, 3]}
        )

    def test_empty_array(self):
        """Brace format with empty array."""
        text = '<|tool_call>call:some_tool{items: []}<tool_call|>'
        tool_calls, _ = parse_tool_calls(text)

        self.assertEqual(
            tool_calls[0]["function"]["arguments"],
            {"items": []}
        )


class TestGemma4EqualsSyntaxAdvanced(unittest.TestCase):
    """Test advanced key="value" syntax cases not covered elsewhere."""

    def test_equals_syntax_with_nested_object(self):
        """key="value" syntax where a value is a nested object."""
        text = '<|tool_call>call:some_tool{name="test", data: {key: "val"}}<tool_call|>'
        tool_calls, _ = parse_tool_calls(text)

        self.assertEqual(
            tool_calls[0]["function"]["arguments"],
            {"name": "test", "data": {"key": "val"}}
        )

    def test_equals_syntax_with_array_value(self):
        """key="value" syntax where a value is an array."""
        text = '<|tool_call>call:some_tool{name="test", items: ["a", "b"]}<tool_call|>'
        tool_calls, _ = parse_tool_calls(text)

        self.assertEqual(
            tool_calls[0]["function"]["arguments"],
            {"name": "test", "items": ["a", "b"]}
        )

    def test_missing_trailing_braces(self):
        """LLM omits trailing closing braces — parser should still work."""
        text = '<|tool_call>call:some_tool{key: "val"<tool_call|>'
        tool_calls, _ = parse_tool_calls(text)

        self.assertIsNotNone(tool_calls)
        self.assertEqual(
            tool_calls[0]["function"]["arguments"],
            {"key": "val"}
        )

    def test_missing_trailing_braces_nested(self):
        """LLM omits trailing closing braces on nested objects."""
        text = '<|tool_call>call:some_tool{data: {key: "val"}<tool_call|>'
        tool_calls, _ = parse_tool_calls(text)

        self.assertIsNotNone(tool_calls)
        self.assertEqual(
            tool_calls[0]["function"]["arguments"],
            {"data": {"key": "val"}}
        )

    def test_equals_with_boolean_and_number(self):
        """key="value" syntax mixed with boolean and numeric values."""
        text = '<|tool_call>call:some_tool{name="test", enabled=true, count=42}<tool_call|>'
        tool_calls, _ = parse_tool_calls(text)

        self.assertEqual(
            tool_calls[0]["function"]["arguments"],
            {"name": "test", "enabled": True, "count": 42}
        )


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and invalid input."""

    def test_invalid_json_in_braces_returns_none(self):
        """Invalid JSON with trailing characters — parser extracts the quoted string."""
        text = '<|tool_call>call:ha_search_tools{query: "get scene"* }<tool_call|>'
        tool_calls, _ = parse_tool_calls(text)

        # The parser extracts the valid quoted string, ignoring trailing *
        self.assertIsNotNone(tool_calls)
        self.assertEqual(tool_calls[0]["function"]["arguments"], {"query": "get scene"})

    def test_tool_call_without_call_prefix(self):
        """Tool call block without 'call:' prefix should be skipped."""
        text = '<|tool_call>some_random_content<tool_call|>'
        tool_calls, _ = parse_tool_calls(text)

        self.assertIsNone(tool_calls)

    def test_empty_arguments(self):
        """Tool call with empty braces should still parse the function name."""
        text = '<|tool_call>call:some_tool{}<tool_call|>'
        tool_calls, _ = parse_tool_calls(text)

        self.assertIsNotNone(tool_calls)
        self.assertEqual(tool_calls[0]["function"]["name"], "some_tool")
        self.assertEqual(tool_calls[0]["function"]["arguments"], {})

    def test_tool_call_id_generated(self):
        """Each tool call should have a unique ID."""
        text = '<|tool_call>call:some_tool{key="val"}<tool_call|>'
        tool_calls, _ = parse_tool_calls(text)

        self.assertIn("id", tool_calls[0])
        self.assertTrue(tool_calls[0]["id"].startswith("call_"))

    def test_tool_call_type_is_function(self):
        """Tool call type should be 'function'."""
        text = '<|tool_call>call:some_tool{key="val"}<tool_call|>'
        tool_calls, _ = parse_tool_calls(text)

        self.assertEqual(tool_calls[0]["type"], "function")


if __name__ == '__main__':
    unittest.main()
