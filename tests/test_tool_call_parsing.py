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


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and invalid input."""

    def test_invalid_json_in_braces_returns_none(self):
        """Invalid JSON that doesn't match any fallback pattern should return None."""
        text = '<|tool_call>call:ha_search_tools{query: "get scene"* }<tool_call|>'
        tool_calls, _ = parse_tool_calls(text)

        self.assertIsNone(tool_calls)

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
