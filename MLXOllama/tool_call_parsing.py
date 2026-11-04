"""Standalone tool-call parser for LLM responses.

Supports two formats:
- Qwen3.5-style XML blocks (``<tool_call>...</tool_call>``)
- Gemma 4 native format (``<|tool_call>...<tool_call|>``)

Only standard-library dependencies.
"""

import json
import logging
import re
import uuid

from typing import Dict, List, Optional


def _split_on_commas(s: str, start_bracket_depth: int = 0):
    """Split a string on commas, respecting quotes, braces, and brackets.

    Tracks the minimum combined depth reached so that structurally unbalanced
    LLM output (e.g. extra ``}`` inside nested objects) doesn't cause spurious
    splits.  Only splits at commas where the current depth equals the minimum.

    ``start_bracket_depth`` lets callers pretend the content is already inside
    that many levels of ``[]`` (used when splitting array element content).
    """
    parts = []
    current = ''
    brace_depth = 0
    bracket_depth = start_bracket_depth
    min_depth = 0
    in_quote = None
    for ch in s:
        if ch in ("'", '"') and in_quote is None:
            in_quote = ch
            current += ch
        elif ch == in_quote:
            in_quote = None
            current += ch
        elif ch == '{':
            brace_depth += 1
            current += ch
        elif ch == '}':
            brace_depth -= 1
            min_depth = min(min_depth, brace_depth)
            current += ch
        elif ch == '[':
            bracket_depth += 1
            current += ch
        elif ch == ']':
            bracket_depth -= 1
            current += ch
        elif ch == ',':
            if in_quote is None and brace_depth == min_depth and bracket_depth == start_bracket_depth:
                parts.append(current.strip())
                current = ''
                continue
            current += ch
        else:
            current += ch
    if current.strip():
        parts.append(current.strip())
    return parts


def _parse_brace_value(val_str: str):
    """Parse a single value from a brace-format argument string.
    
    Handles nested objects, arrays, strings, booleans, and numbers.
    """
    val_str = val_str.strip()
    
    # Nested object
    if val_str.startswith('{'):
        # Try JSON first (handles properly quoted keys from _quote_bare_keys)
        try:
            return json.loads(val_str)
        except (json.JSONDecodeError, ValueError):
            pass

        # Try balancing trailing braces before extraction
        balanced_val = _balance_braces(val_str)
        try:
            return json.loads(balanced_val)
        except (json.JSONDecodeError, ValueError):
            pass

        # Find the matching closing brace, respecting nesting and quotes
        inner, _rest = _extract_balanced(balanced_val, '{', '}')
        if inner is not None:
            inner = inner.strip()
            if not inner:
                return {}
            nested = {}
            pairs = _split_on_commas(inner)
            for pair in pairs:
                if ':' in pair:
                    key, _, val = pair.partition(':')
                    key = key.strip().strip('"').strip("'")
                    val = _parse_brace_value(val.strip())
                    nested[key] = val
            return nested
        # Unbalanced braces — treat as plain string
        return val_str
    
    # Array
    if val_str.startswith('['):
        inner, _rest = _extract_balanced(val_str, '[', ']')
        if inner is not None:
            inner = inner.strip()
            if not inner:
                return []
            items = []
            # Parse array elements by finding top-level commas (not inside {} or [])
            # Track the minimum depth reached so that extra } inside elements
            # don't cause spurious splits.
            elements = []
            current = ''
            depth = 0
            min_depth = 0
            in_quote = None
            for ch in inner:
                if ch in ("'", '"') and in_quote is None:
                    in_quote = ch
                    current += ch
                elif ch == in_quote:
                    in_quote = None
                    current += ch
                elif ch in ('{', '['):
                    depth += 1
                    current += ch
                elif ch in ('}', ']'):
                    depth -= 1
                    min_depth = min(min_depth, depth)
                    current += ch
                elif ch == ',' and in_quote is None and depth == min_depth:
                    elements.append(current.strip())
                    current = ''
                else:
                    current += ch
            if current.strip():
                elements.append(current.strip())

            for elem in elements:
                elem = elem.strip()
                if elem:
                    items.append(_parse_brace_value(elem))
            return items
        # Unbalanced brackets — treat as plain string
        return val_str
    
    # Quoted string (possibly followed by trailing structural chars like }})
    if len(val_str) >= 2 and val_str[0] in ("'", '"'):
        quote = val_str[0]
        # Find the matching closing quote
        end = val_str.find(quote, 1)
        if end != -1:
            return val_str[1:end]
    
    # Boolean
    if val_str.lower() == 'true':
        return True
    if val_str.lower() == 'false':
        return False
    
    # Number
    if val_str.replace('.', '', 1).replace('-', '', 1).isdigit():
        return float(val_str) if '.' in val_str else int(val_str)
    
    # Plain string
    return val_str


def _extract_balanced(s: str, open_ch: str, close_ch: str):
    """Extract content between matching open/close characters.

    Returns (inner_content, rest_after_close) or (None, None) if unbalanced.
    Respects quotes.  When looking for a matching ``}``, ignores ``}`` that sit
    inside ``[...]`` so that structurally unbalanced LLM output with extra
    closing braces inside arrays still produces a usable result.
    """
    if not s.startswith(open_ch):
        return None, None

    brace_depth = 0
    bracket_depth = 0
    in_quote = None
    for i, ch in enumerate(s):
        if ch in ("'", '"') and in_quote is None:
            in_quote = ch
        elif ch == in_quote:
            in_quote = None
        elif ch == '{':
            # Only count { when NOT inside an array
            if bracket_depth == 0:
                brace_depth += 1
        elif ch == '}':
            # Only count } when NOT inside an array
            if bracket_depth == 0:
                brace_depth -= 1
        elif ch == '[':
            bracket_depth += 1
        elif ch == ']':
            bracket_depth -= 1

        # Check for matching close
        if ch == close_ch:
            if open_ch == '{' and brace_depth <= 0 and bracket_depth == 0:
                inner = s[1:i]
                rest = s[i+1:]
                return inner, rest
            elif open_ch == '[' and bracket_depth <= 0:
                inner = s[1:i]
                rest = s[i+1:]
                return inner, rest

    # If we reach here, the string is unbalanced.  If brace_depth is small
    # and positive, the LLM simply omitted trailing } — treat the end of
    # string as the match point so the fallback parser can still work.
    if open_ch == '{' and brace_depth > 0 and bracket_depth == 0:
        return s[1:], ''
    if open_ch == '[' and bracket_depth > 0:
        return s[1:], ''

    return None, None


def _quote_bare_keys(s: str) -> str:
    """Quote unquoted identifier keys before colons, without consuming structural chars.

    Uses a lookbehind for ``{`` so that array elements like ``},{key:`` keep
    their opening brace intact.  Commas are captured and re-emitted.
    """
    # Keys after { — lookbehind so we don't consume the {
    s = re.sub(r'(?<={)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'"\1":', s)
    # Keys after , — capture and re-emit the comma + whitespace
    s = re.sub(r'(\s*,\s*)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', s)
    return s


def _balance_braces(s: str) -> str:
    """Balance unmatched braces/brackets by removing excess closers or adding missing ones.

    The LLM sometimes emits extra ``}`` or ``]`` at the end, or omits closing
    characters.  Remove excess trailing closers, then append any missing ones.
    """
    s = s.rstrip()
    depth_brace = s.count('{') - s.count('}')
    depth_bracket = s.count('[') - s.count(']')

    # Remove excess trailing }
    while depth_brace < 0 and s.endswith('}'):
        s = s[:-1]
        depth_brace += 1

    # Remove excess trailing ]
    while depth_bracket < 0 and s.endswith(']'):
        s = s[:-1]
        depth_bracket += 1

    # Append missing closers
    if depth_bracket > 0:
        s += ']' * depth_bracket
    if depth_brace > 0:
        s += '}' * depth_brace

    return s


def _json_loads_first_wins(s: str) -> Dict:
    """Parse JSON, keeping the **first** value when duplicate keys appear.

    Python's standard ``json.loads`` keeps the last value for duplicate keys,
    but the LLM often emits duplicate keys (e.g. two ``name`` fields) where the
    **first** one is the meaningful value.  This wrapper detects duplicates and
    preserves the first occurrence.
    """
    # Quick path: standard parse
    obj = json.loads(s)

    # Check for duplicate keys by scanning the raw string
    # If no duplicates, return the standard result
    if _has_duplicate_keys(s) is False:
        return obj

    # Re-parse with object_pairs_hook to keep first value
    pairs = json.loads(s, object_pairs_hook=lambda pairs: {
        k: v for k, v in pairs if k not in {kk for kk, _vv in pairs[:pairs.index((k, v))]}
    })
    return pairs


def _has_duplicate_keys(s: str) -> bool:
    """Check whether a JSON string contains duplicate keys at any level.

    Returns ``True`` if duplicates are found, ``False`` otherwise.
    Returns ``None`` if the check cannot be determined (falls back to
    re-parsing with first-wins semantics).
    """
    try:
        pairs_list = json.loads(s, object_pairs_hook=lambda pairs: pairs)
        return _pairs_has_duplicate(pairs_list)
    except json.JSONDecodeError:
        return False


def _pairs_has_duplicate(pairs) -> bool:
    """Recursively check a list of (key, value) pairs for duplicate keys."""
    if not isinstance(pairs, list):
        return False
    # Filter out array elements (they don't have key-value structure)
    kv_pairs = [p for p in pairs if isinstance(p, (list, tuple)) and len(p) == 2]
    if not kv_pairs:
        # Check nested values recursively
        return any(_pairs_has_duplicate(p) if isinstance(p, (list, dict)) else False
                   for p in pairs)
    keys = [k for k, _ in kv_pairs]
    seen = set()
    for key in keys:
        if key in seen:
            return True
        seen.add(key)
    # Check nested values recursively
    for _, val in kv_pairs:
        if isinstance(val, list):
            if _pairs_has_duplicate(val):
                return True
    return False


def _flatten_single_key_wrapper(result: Dict) -> Dict:
    """Unwrap a single-top-level-key result when the LLM misplaced a closing brace.

    When the LLM emits an extra ``}`` it can wrap every real parameter inside one
    top-level key (usually ``arguments``).  Detect that pattern and promote the
    inner keys to the top level so the downstream tool sees the expected flat
    parameter dict.

    Only triggers when **all** of these conditions hold:

    1. The parsed dict has exactly **one** top-level key named ``"arguments"``.
    2. Its value is a dict with **more than one** key.
    3. That inner dict also contains an ``"arguments"`` key (the real nested
       payload), proving the outer wrapper is an artifact of a misplaced brace.
    4. At least one inner value is a simple scalar (str, int, float, bool),
       confirming the inner dict is a mixed parameter set rather than a pure
       nested structure.

    This is narrow enough that legitimate single-param calls like
    ``{"data": {...}}`` or ``{"config": {...}}`` are never touched.
    """
    if len(result) != 1:
        return result

    only_key = next(iter(result))

    # Only consider the well-known misplaced-brace wrapper key
    if only_key != "arguments":
        return result

    inner = result[only_key]

    if not isinstance(inner, dict) or len(inner) <= 1:
        return result

    # The inner dict must also contain an "arguments" key — this is the
    # smoking gun that the outer wrapper is an LLM artifact.
    if "arguments" not in inner:
        return result

    # At least one inner value must be a simple scalar to confirm this is
    # a mixed parameter set, not a purely nested structure.
    has_scalar = any(
        not isinstance(v, (dict, list))
        for v in inner.values()
    )
    if not has_scalar:
        return result

    # The inner dict looks like the real parameter set — flatten it.
    return inner


def _parse_brace_args(args_str: str, func_name: str) -> Optional[Dict]:
    """Parse brace-format arguments, handling nested objects and arrays.

    Returns a dict of arguments or None if parsing fails.
    """
    # Replace <|">> string delimiters
    args_str = args_str.replace('<|"|>', '"')

    # Try standard JSON first (first-wins for duplicate keys)
    try:
        return _flatten_single_key_wrapper(_json_loads_first_wins(args_str))
    except json.JSONDecodeError:
        pass

    # Quote bare keys for JSON-like syntax (fixed: doesn't consume { in arrays)
    quoted = _quote_bare_keys(args_str)

    # Try JSON again after quoting keys (first-wins for duplicate keys)
    try:
        return _flatten_single_key_wrapper(_json_loads_first_wins(quoted))
    except json.JSONDecodeError:
        pass

    # LLM sometimes emits extra trailing } or ] — balance before retrying
    balanced = _balance_braces(quoted)
    try:
        return _flatten_single_key_wrapper(_json_loads_first_wins(balanced))
    except json.JSONDecodeError:
        pass

    # Fallback: manual recursive descent parser with nested object & array support
    # Use the quoted string (not balanced) — _extract_balanced handles extra
    # } inside arrays by ignoring them, so the original structure is preserved.
    args = {}
    inner, _rest = _extract_balanced(quoted, '{', '}')
    if inner is None:
        # No matching braces at all — try to parse what we can from strip
        inner = quoted.strip('{}').strip()
    if not inner:
        return {}

    pairs = _split_on_commas(inner)

    for pair in pairs:
        pair = pair.strip()
        if not pair:
            continue
        # Try colon separator first (JSON-like)
        if ':' in pair:
            key, _, val = pair.partition(':')
            key = key.strip().strip('"').strip("'")
            # The value may have trailing } from nested object closers.
            # _parse_brace_value handles this by trying to parse the value
            # and ignoring trailing structural characters.
            val = _parse_brace_value(val.strip())
            # Keep first value for duplicate keys (LLM often emits 'name' twice)
            if key not in args:
                args[key] = val
        # Try = separator (key="value" syntax)
        elif '=' in pair:
            key, _, val = pair.partition('=')
            key = key.strip()
            val = val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                val = val[1:-1]
            elif val.lower() == 'true':
                val = True
            elif val.lower() == 'false':
                val = False
            elif val.replace('.', '', 1).replace('-', '', 1).isdigit():
                val = float(val) if '.' in val else int(val)
            # Keep first value for duplicate keys
            if key not in args:
                args[key] = val

    if not args:
        logging.warning(f"Gemma 4 tool call JSON parse failed for function '{func_name}': args_str='{args_str}'")
        return None
    return _flatten_single_key_wrapper(args)


def parse_tool_calls(text: str) -> tuple[Optional[List[Dict]], str]:
    """
    Parse tool calls and return both:
    - list of tool call dicts (or None)
    - cleaned text (original text with tool call blocks removed)
    """
    tool_calls = []
    cleaned_text = text  # start with full text

    # ────────────────────────────────────────────────
    # Qwen3.5-style XML parsing (primary path)
    # ────────────────────────────────────────────────
    xml_pattern = r'<tool_call>(.*?)</tool_call>'
    xml_blocks = re.findall(xml_pattern, text, re.DOTALL | re.IGNORECASE)

    for block in xml_blocks:
        block_full = f'<tool_call>{block}</tool_call>'  # reconstruct for removal
        # remove this block from cleaned text
        cleaned_text = cleaned_text.replace(block_full, '', 1).strip()

        # parse the function + parameters (same as before)
        func_match = re.search(r'<function\s*=\s*([^>]+)>', block, re.IGNORECASE)
        if not func_match:
            continue
        func_name = func_match.group(1).strip()

        params = {}
        param_pattern = r'<parameter\s*=\s*([^>]+)>(.*?)</parameter>'
        for m in re.finditer(param_pattern, block, re.DOTALL | re.IGNORECASE):
            key = m.group(1).strip()
            val_str = m.group(2).strip()

            try:
                if val_str.startswith('[') and val_str.endswith(']'):
                    val = json.loads(val_str)
                elif val_str.lower() in ('true', 'false'):
                    val = val_str.lower() == 'true'
                elif val_str.replace('.', '', 1).replace('-', '', 1).isdigit():
                    val = float(val_str) if '.' in val_str else int(val_str)
                else:
                    val = val_str
            except (json.JSONDecodeError, ValueError) as _e:
                val = val_str
            params[key] = val

        if func_name:
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": params
                }
            })

    # If we found XML tool calls → return them + cleaned text
    if tool_calls:
        return tool_calls, cleaned_text.strip()

    # ────────────────────────────────────────────────
    # Gemma 4 native format
    # <|tool_call>call:name{key:<|"|>val<|"|>}<tool_call|>
    # Also supports: <|tool_call>call:name(key='val')<tool_call|>
    # Also accept malformed closing tags like <toolcall|> (missing underscore)
    # ────────────────────────────────────────────────
    gemma4_pattern = r'<\|tool_call>(.*?)(?:<tool_call\|>|<toolcall\|>)'
    gemma4_blocks = re.findall(gemma4_pattern, text, re.DOTALL)

    for block in gemma4_blocks:
        # Try both closing tag variants for removal from cleaned text
        block_full = f'<|tool_call>{block}<tool_call|>'
        if block_full in cleaned_text:
            cleaned_text = cleaned_text.replace(block_full, '', 1).strip()
        else:
            # Malformed closing tag (e.g. <toolcall|> without underscore)
            block_full_alt = f'<|tool_call>{block}<toolcall|>'
            cleaned_text = cleaned_text.replace(block_full_alt, '', 1).strip()

        body = block.strip()
        if not body.startswith('call:'):
            continue
        body = body[len('call:'):]

        # Try brace format first: call:name{...}
        brace_idx = body.find('{')
        if brace_idx != -1:
            func_name = body[:brace_idx].strip()
            args_str = body[brace_idx:]

            args = _parse_brace_args(args_str, func_name)
            if args is None:
                continue

        # Try paren format: call:name(key='val', key2="val2")
        elif '(' in body:
            paren_idx = body.find('(')
            func_name = body[:paren_idx].strip()
            args_str = body[paren_idx + 1:].rstrip(')').strip()

            # Parse Python-style keyword arguments
            args = {}
            if args_str:
                # Split on commas, but respect quoted strings
                pairs = []
                current = ''
                in_quote = None
                for ch in args_str:
                    if ch in ("'", '"') and in_quote is None:
                        in_quote = ch
                        current += ch
                    elif ch == in_quote:
                        in_quote = None
                        current += ch
                    elif ch == ',' and in_quote is None:
                        pairs.append(current.strip())
                        current = ''
                    else:
                        current += ch
                if current.strip():
                    pairs.append(current.strip())

                for pair in pairs:
                    if '=' in pair:
                        key, _, val = pair.partition('=')
                        key = key.strip()
                        val = val.strip()
                        # Remove surrounding quotes
                        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                            val = val[1:-1]
                        # Try to convert numeric values
                        elif val.lower() == 'true':
                            val = True
                        elif val.lower() == 'false':
                            val = False
                        elif val.replace('.', '', 1).replace('-', '', 1).isdigit():
                            val = float(val) if '.' in val else int(val)
                        args[key] = val

        else:
            continue

        if func_name:
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": args
                }
            })

    if tool_calls:
        return tool_calls, cleaned_text.strip()

    # No tool calls found → return None + original text
    return None, text.strip()
