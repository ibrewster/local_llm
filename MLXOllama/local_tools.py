import json
import inspect
import logging
import fnmatch
import os
import re
import types
import time
import stat
from pathlib import Path

from datetime import datetime, timedelta, date
from enum import Enum
from typing import (
    Any,
    Annotated,
    Literal,
    Mapping,
    Sequence,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)
from uuid import UUID
from urllib.parse import quote

import httpx
import pytz

from rapidfuzz import process, fuzz, utils

from . import config
from .async_ttl_cache import async_ttl_cache
from .config import BRAVE_SEARCH_API_KEY

LOCAL_TOOLS = {}

MAX_LOCAL_FILE_READ_BYTES = 1_048_576
MAX_LOCAL_FILE_WRITE_BYTES = 1_048_576
MAX_LOCAL_DIRECTORY_ENTRIES = 500

SENSITIVE_LOCAL_FILE_PATTERNS = (
    ".env",
    ".env.*",
    ".git-credentials",
    ".netrc",
    ".npmrc",
    ".ssh",
    "id_rsa",
    "id_rsa.*",
    "id_ed25519",
    "id_ed25519.*",
    "id_ecdsa",
    "id_ecdsa.*",
    "id_dsa",
    "id_dsa.*",
    "authorized_keys",
    "*.pem",
    "*.key",
    "*.p8",
    "*.p12",
    "*.pfx",
    "*.jks",
    "credentials",
    "credentials.*",
    "application_default_credentials.json",
    "secrets.json",
)


def _local_files_root() -> Path:
    root = Path(getattr(config, "LOCAL_FILES_ROOT", os.getcwd())).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"Local files root is not a directory: {root}")
    return root


def _local_file_policy() -> dict[str, list[str]]:
    """Load the sibling policy JSON (or LOCAL_FILES_POLICY_PATH) per tool call.

    Include entries are directory roots; relative deny globs apply across roots.
    Built-in sensitive-file patterns cannot be removed.
    """
    configured_policy_path = os.environ.get("LOCAL_FILES_POLICY_PATH") or getattr(
        config, "LOCAL_FILES_POLICY_PATH", None
    )
    policy_path = Path(
        configured_policy_path or Path(__file__).with_name("local_files_policy.json")
    ).expanduser()
    if not policy_path.exists():
        if configured_policy_path is not None:
            raise FileNotFoundError(f"Local file policy does not exist: {policy_path}")
        return {
            "include": [str(_local_files_root())],
            "deny": [],
            "sensitive": [],
        }

    with policy_path.open(encoding="utf-8") as policy_file:
        policy = json.load(policy_file)
    if not isinstance(policy, dict) or set(policy) - {"include", "deny", "sensitive"}:
        raise ValueError(
            "local file policy must be an object containing only include, deny, and sensitive"
        )

    normalized = {}
    for key in ("include", "deny", "sensitive"):
        value = policy.get(key, [])
        if not isinstance(value, list) or not all(
            isinstance(rule, str) and rule for rule in value
        ):
            raise ValueError(f"local file policy '{key}' must be a list of non-empty strings")
        normalized[key] = value
    if not normalized["include"]:
        raise ValueError("local file policy 'include' must contain at least one path")
    return normalized


def _absolute_policy_rule(rule: str, base: Path) -> str:
    rule_path = Path(rule).expanduser()
    if not rule_path.is_absolute():
        rule_path = base / rule_path
    if any(character in os.fspath(rule_path) for character in "*?["):
        return os.path.abspath(os.fspath(rule_path))
    return os.fspath(rule_path.resolve(strict=False))


def _path_matches_rule(
    path: Path,
    rule: str,
    base: Path,
    *,
    match_relative_anywhere: bool = False,
) -> bool:
    rule_path = Path(rule).expanduser()
    if match_relative_anywhere and not rule_path.is_absolute():
        rule_text = rule.replace(os.sep, "/")
        path_parts = path.parts[1:] if path.is_absolute() else path.parts
        suffixes = ["/".join(path_parts[index:]) for index in range(len(path_parts))]
        if any(character in rule_text for character in "*?["):
            return any(fnmatch.fnmatchcase(suffix, rule_text) for suffix in suffixes)
        return any(
            suffix == rule_text or suffix.startswith(rule_text.rstrip("/") + "/")
            for suffix in suffixes
        )

    path_text = os.fspath(path)
    rule_text = _absolute_policy_rule(rule, base)
    if any(character in rule_text for character in "*?["):
        return fnmatch.fnmatchcase(path_text, rule_text)
    return path_text == rule_text or path_text.startswith(rule_text.rstrip(os.sep) + os.sep)


def _is_sensitive_path(path: Path, policy: dict[str, list[str]], base: Path) -> bool:
    for rule in (*SENSITIVE_LOCAL_FILE_PATTERNS, *policy["sensitive"]):
        if "/" in rule or os.path.isabs(rule):
            if _path_matches_rule(
                path, rule, base, match_relative_anywhere=not os.path.isabs(rule)
            ):
                return True
        elif any(fnmatch.fnmatchcase(part.lower(), rule.lower()) for part in path.parts):
            return True
    return False


def _check_local_file_access(
    path: str | Path,
    *,
    action: str,
    must_exist: bool,
    policy: dict[str, list[str]] | None = None,
) -> tuple[Path, Path]:
    """Resolve and authorize a file path; all local file tools use this check."""
    if not isinstance(path, (str, Path)) or not os.fspath(path):
        raise ValueError("path must be a non-empty path")
    base = _local_files_root()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    requested_path = Path(os.path.abspath(os.fspath(candidate)))
    resolved = candidate.resolve(strict=must_exist)
    policy = policy if policy is not None else _local_file_policy()

    for checked_path in {requested_path, resolved}:
        if _is_sensitive_path(checked_path, policy, base):
            raise PermissionError(f"{action} of sensitive local file is blocked: {path}")
        if any(
            _path_matches_rule(
                checked_path, rule, base, match_relative_anywhere=True
            )
            for rule in policy["deny"]
        ):
            raise PermissionError(f"{action} of local path is denied: {path}")
    if not any(_path_matches_rule(resolved, rule, base) for rule in policy["include"]):
        raise PermissionError(f"{action} of local path is not included: {path}")
    return base, resolved


def _included_search_roots(
    policy: dict[str, list[str]] | None = None,
) -> list[Path]:
    base = _local_files_root()
    policy = policy if policy is not None else _local_file_policy()
    roots = []
    seen = set()
    for rule in policy["include"]:
        if any(character in rule for character in "*?["):
            continue
        root = Path(_absolute_policy_rule(rule, base)).resolve(strict=False)
        if root.is_dir() and os.fspath(root) not in seen:
            try:
                _check_local_file_access(
                    root, action="search", must_exist=True, policy=policy
                )
            except PermissionError:
                continue
            roots.append(root)
            seen.add(os.fspath(root))
    return roots


###################### UTILITY FUNCTIONS/DECORATORS ############################
def _json_schema(annotation: Any) -> dict[str, Any]:
    """Generate a JSON Schema dict from a Python type annotation.

    Supported:
      - str, int, float, bool, None
      - Optional[T] / T | None
      - Union[T1, T2, ...]
      - Literal[...]
      - list[T], set[T], tuple[T], tuple[T, ...]
      - dict[K, V], Mapping[K, V]
      - Sequence[T]
      - Annotated[T, ...]
      - Enum subclasses
      - NewType
      - Any
      - date, datetime, UUID

    The generated schemas intentionally use a conservative subset of JSON
    Schema suitable for LLM function-calling backends.
    """
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}

    # Resolve typing.NewType(...)
    # NewType objects expose their underlying type as __supertype__.
    if hasattr(annotation, "__supertype__"):
        return _json_schema(annotation.__supertype__)

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Annotated[T, ...] -> schema for T.
    if origin is Annotated:
        return _json_schema(args[0])

    # Literal[...]
    if origin is Literal:
        literal_values = list(args)
        schema: dict[str, Any] = {"enum": literal_values}

        # Only add "type" when every literal has the same exact Python type.
        if literal_values:
            first_type = type(literal_values[0])
            if all(type(value) is first_type for value in literal_values):
                if first_type is str:
                    schema["type"] = "string"
                elif first_type is bool:
                    schema["type"] = "boolean"
                elif first_type is int:
                    schema["type"] = "integer"
                elif first_type is float:
                    schema["type"] = "number"
                elif first_type is type(None):
                    schema["type"] = "null"

        return schema

    # Optional[T], T | None, or a general Union.
    if origin in (Union, types.UnionType):
        non_none = [arg for arg in args if arg is not type(None)]
        has_none = len(non_none) != len(args)

        # Optional[T] / T | None.
        #
        # Preserve nullability explicitly when None is actually part of the
        # allowed value set. This is different from requiredness, which is
        # determined separately from the function parameter's default value.
        if len(non_none) == 1:
            base_schema = _json_schema(non_none[0])

            if has_none:
                # JSON Schema can express this cleanly with anyOf.
                #
                # Example:
                #   str | None ->
                #   {"anyOf": [{"type": "string"}, {"type": "null"}]}
                return {
                    "anyOf": [
                        base_schema,
                        {"type": "null"},
                    ]
                }

            return base_schema

        # General Union.
        union_schema = {
            "anyOf": [_json_schema(arg) for arg in non_none]
        }

        if has_none:
            union_schema["anyOf"].append({"type": "null"})

        return union_schema

    # Enum classes.
    if inspect.isclass(annotation) and issubclass(annotation, Enum):
        values = [member.value for member in annotation]

        schema = {"enum": values}

        if values:
            first_type = type(values[0])
            if all(type(value) is first_type for value in values):
                if first_type is str:
                    schema["type"] = "string"
                elif first_type is bool:
                    schema["type"] = "boolean"
                elif first_type is int:
                    schema["type"] = "integer"
                elif first_type is float:
                    schema["type"] = "number"

        return schema

    # list[T] / set[T] / Sequence[T]
    if origin in (list, set, Sequence):
        schema = {"type": "array"}

        if args:
            schema["items"] = _json_schema(args[0])

        return schema

    # tuple[T, ...]
    #
    # Intentionally do not use prefixItems for fixed heterogeneous tuples.
    # That is valid modern JSON Schema, but not worth the compatibility risk
    # in an LLM tool-calling stack.
    if origin is tuple:
        schema = {"type": "array"}

        if args:
            if len(args) == 2 and args[1] is Ellipsis:
                schema["items"] = _json_schema(args[0])
            else:
                # Conservative fallback for tuple[T1, T2, ...].
                schema["items"] = {}

        return schema

    # Bare containers.
    if annotation in (list, set, tuple, Sequence):
        return {"type": "array"}

    if annotation in (dict, Mapping):
        return {"type": "object"}

    # dict[K, V] / Mapping[K, V]
    if origin in (dict, Mapping):
        schema = {"type": "object"}

        if len(args) == 2:
            key_type, value_type = args

            # JSON object keys are strings. For dict[str, V], describe the
            # values precisely via additionalProperties.
            if key_type is str:
                schema["additionalProperties"] = _json_schema(value_type)
            else:
                # Python allows non-string dict keys, but JSON objects don't.
                # Don't pretend the key constraint is enforceable here.
                schema["additionalProperties"] = _json_schema(value_type)

        return schema

    # Bare scalar types.
    if annotation is str:
        return {"type": "string"}

    if annotation is bool:
        return {"type": "boolean"}

    if annotation is int:
        return {"type": "integer"}

    if annotation is float:
        return {"type": "number"}

    if annotation is type(None):
        return {"type": "null"}

    # Date/time-like values.
    #
    # Keep these as plain strings rather than relying on JSON Schema "format",
    # since format validation is not consistently honored by tool backends.
    if annotation in (date, datetime, UUID):
        return {"type": "string"}

    # Unknown types: don't lie.
    #
    # {} means "unconstrained value", which is preferable to claiming that
    # an arbitrary Python type is a string.
    return {}

_PARAM_REGEX = re.compile(r"^\s{0,4}(\w+)\s*(?:\([^)]*\))?:\s*(.*)$")
def _parameter_descriptions(docstring: str) -> dict[str, str]:
    """Extract parameter descriptions from an Args/Parameters docstring section."""
    descriptions = {}
    in_parameters = False
    current = None

    for line in inspect.cleandoc(docstring or "").splitlines():
        clean_line = line.strip().lower().rstrip(":")

        # Detect the start of the parameters section
        if clean_line in {"args", "arguments", "parameters"}:
            in_parameters = True
            continue

        if in_parameters:
            # Break on known next-sections
            if clean_line in {"returns", "raises", "yields"}:
                break
            # Break on unknown next-sections (unindented text ending in a colon)
            if line and not line[0].isspace() and line.rstrip().endswith(":"):
                break

            match = _PARAM_REGEX.match(line)
            if match:
                current = match.group(1)
                descriptions[current] = match.group(2).strip()
            elif current and line.strip():
                descriptions[current] += f" {line.strip()}"

    return descriptions


def local_tool(func):
    """Register a local function as an OpenAI-compatible tool."""
    signature = inspect.signature(func)
    descriptions = _parameter_descriptions(func.__doc__)

    # Resolve postponed annotations / forward references.
    #
    # Falls back to the raw signature annotation if resolution fails, so one
    # problematic annotation doesn't prevent the tool from registering.
    try:
        type_hints = get_type_hints(func)
    except (NameError, TypeError, ValueError):
        type_hints = {}

    properties = {}
    required = []

    for name, parameter in signature.parameters.items():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        # Prefer resolved type hints, falling back to the raw annotation.
        annotation = type_hints.get(name, parameter.annotation)

        # Handle missing type hints explicitly.
        if annotation is inspect.Parameter.empty:
            annotation = str

        schema = _json_schema(annotation)

        if name in descriptions:
            schema["description"] = descriptions[name]

        if parameter.default is not inspect.Parameter.empty:
            if parameter.default is not None:
                schema["default"] = parameter.default
        else:
            required.append(name)

        properties[name] = schema

    # Ensure there is always a description.
    doc_text = inspect.cleandoc(func.__doc__ or "")
    description_part = re.split(
        r"(?i)\n(?:parameters|args|arguments|returns|yields|raises)"
        r"\s*(?:\n[-=]+)?\s*\n",
        doc_text,
    )[0]
    description = " ".join(description_part.split())

    if not description:
        description = f"Executes the {func.__name__} function."

    LOCAL_TOOLS[func.__name__] = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }

    return func

############################ TOOLS #############################
@local_tool
async def find_local_files(
    name_pattern: str,
    max_depth: int = 5,
    max_results: int = 50,
) -> str:
    """Find files by filename pattern inside the configured local files workspace.

    The pattern is matched against each file's name using glob syntax (for example,
    "*.py" or "notes-*.txt"). Results under LOCAL_FILES_ROOT are relative;
    results from other included roots are absolute. Search does not follow
    directory symlinks and has bounded depth and result count.

    PARAMETERS
    ----------
    name_pattern (str): Glob pattern matched against file names.
    max_depth (int): Maximum directory depth to search, from 0 to 10. The root is depth 0.
    max_results (int): Maximum matching files to return, from 1 to 100.
    """
    if not isinstance(name_pattern, str) or not name_pattern:
        raise ValueError("name_pattern must be a non-empty glob pattern")
    if not 0 <= max_depth <= 10:
        raise ValueError("max_depth must be between 0 and 10")
    if not 1 <= max_results <= 100:
        raise ValueError("max_results must be between 1 and 100")

    matches = []
    truncated = False
    seen_files = set()
    base = _local_files_root()
    policy = _local_file_policy()
    for root in _included_search_roots(policy):
        for current, dirs, files in os.walk(root, followlinks=False):
            current_path = Path(current)
            depth = len(current_path.relative_to(root).parts)
            dirs[:] = [
                directory for directory in dirs
                if not (current_path / directory).is_symlink()
            ]
            allowed_dirs = []
            for directory in dirs:
                try:
                    _check_local_file_access(
                        current_path / directory,
                        action="search",
                        must_exist=True,
                        policy=policy,
                    )
                except PermissionError:
                    continue
                allowed_dirs.append(directory)
            dirs[:] = allowed_dirs
            if depth >= max_depth:
                dirs.clear()
            for filename in files:
                if not fnmatch.fnmatchcase(filename, name_pattern):
                    continue
                file_path = current_path / filename
                try:
                    _, resolved_path = _check_local_file_access(
                        file_path, action="search", must_exist=True, policy=policy
                    )
                    info = file_path.stat(follow_symlinks=False)
                except (FileNotFoundError, PermissionError):
                    continue
                if not stat.S_ISREG(info.st_mode) or resolved_path in seen_files:
                    continue
                if len(matches) == max_results:
                    truncated = True
                    break
                seen_files.add(resolved_path)
                try:
                    display_path = resolved_path.relative_to(base).as_posix()
                except ValueError:
                    display_path = str(resolved_path)
                matches.append({
                    "path": display_path,
                    "size_bytes": info.st_size,
                })
            if truncated:
                break
        if truncated:
            break
    return json.dumps({"matches": matches, "truncated": truncated})


@local_tool
async def list_local_directory(path: str = ".", max_entries: int = 100) -> str:
    """List the immediate contents of an included local directory.

    Paths may be relative to LOCAL_FILES_ROOT or absolute when included by policy.
    Results are limited to at most 500 entries.

    PARAMETERS
    ----------
    path (str): Directory path relative to the default root or an included absolute path.
    max_entries (int): Maximum number of entries to return, from 1 to 500.
    """
    if not 1 <= max_entries <= MAX_LOCAL_DIRECTORY_ENTRIES:
        raise ValueError(f"max_entries must be between 1 and {MAX_LOCAL_DIRECTORY_ENTRIES}")

    policy = _local_file_policy()
    _, directory = _check_local_file_access(
        path, action="list", must_exist=True, policy=policy
    )
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    entries = []
    truncated = False
    for entry in directory.iterdir():
        try:
            _check_local_file_access(
                entry, action="list", must_exist=False, policy=policy
            )
            info = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode):
                entry_type = "symlink"
            elif stat.S_ISDIR(info.st_mode):
                entry_type = "directory"
            elif stat.S_ISREG(info.st_mode):
                entry_type = "file"
            else:
                entry_type = "other"
            item = {"name": entry.name, "type": entry_type}
            if entry_type == "file":
                item["size_bytes"] = info.st_size
            if len(entries) == max_entries:
                truncated = True
                break
            entries.append(item)
        except FileNotFoundError:
            continue
        except PermissionError:
            continue

    return json.dumps({
        "path": path,
        "entries": entries,
        "truncated": truncated,
    })


@local_tool
async def read_local_file(path: str, max_bytes: int = 262_144) -> str:
    """Read an included UTF-8 text file.

    Paths may be relative to LOCAL_FILES_ROOT or absolute when included by policy.
    Files larger than the requested limit are rejected; reads are capped at 1 MiB.

    PARAMETERS
    ----------
    path (str): File path relative to the default root or an included absolute path.
    max_bytes (int): Maximum file size to read, from 1 to 1048576 bytes.
    """
    if not 1 <= max_bytes <= MAX_LOCAL_FILE_READ_BYTES:
        raise ValueError(f"max_bytes must be between 1 and {MAX_LOCAL_FILE_READ_BYTES}")
    _, file_path = _check_local_file_access(
        path, action="read", must_exist=True, policy=_local_file_policy()
    )
    file_descriptor = os.open(file_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(file_descriptor, "rb") as file:
        info = os.fstat(file.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("path must refer to a regular file")
        if info.st_size > max_bytes:
            raise ValueError(f"file exceeds the {max_bytes}-byte read limit")
        content = file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError(f"file exceeds the {max_bytes}-byte read limit")
    return content.decode("utf-8")


@local_tool
async def write_local_file(path: str, content: str) -> str:
    """Create a new UTF-8 text file at an included path.

    Paths may be relative to LOCAL_FILES_ROOT or absolute when included by policy.
    Existing files are never overwritten, parent directories must already exist, and content
    is limited to 1 MiB.

    PARAMETERS
    ----------
    path (str): Destination path relative to the default root or an included absolute path.
    content (str): Text to write to the new file.
    """
    if not isinstance(content, str):
        raise ValueError("content must be a string")
    data = content.encode("utf-8")
    if len(data) > MAX_LOCAL_FILE_WRITE_BYTES:
        raise ValueError(f"content exceeds the {MAX_LOCAL_FILE_WRITE_BYTES}-byte write limit")

    relative = Path(path).expanduser() if isinstance(path, str) else None
    if relative is None or not path or relative.name in {"", ".", ".."}:
        raise ValueError("path must be a file path")
    base = _local_files_root()
    destination_path = relative if relative.is_absolute() else base / relative
    parent = destination_path.parent.resolve(strict=True)
    if not parent.is_dir():
        raise NotADirectoryError(f"Parent directory does not exist: {relative.parent}")

    _, destination = _check_local_file_access(
        destination_path,
        action="write",
        must_exist=False,
        policy=_local_file_policy(),
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    file_descriptor = os.open(destination, flags, 0o600)
    try:
        with os.fdopen(file_descriptor, "wb") as file:
            file.write(data)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return f"Created {path} ({len(data)} bytes)"


@local_tool
async def get_current_time(timezone: str | None = None) -> str:
    """
    Get the current date and time in the user's timezone. Always use this when the query involves now, today, tomorrow, current time, scheduling, deadlines, recency, etc.

    PARAMETERS
    ----------
    timezone (str | None): Optional IANA timezone name e.g. 'America/Anchorage'. Defaults to the user's known timezone.

    RETURNS
    -------
    Formatted string like: "2026-03-17 14:55 AKDT"
    """
    # Default to system local time
    tz = datetime.now().astimezone().tzinfo
    if timezone:
        try:
            tz = pytz.timezone(timezone)
        except pytz.exceptions.UnknownTimeZoneError:
            pass

    now = datetime.now(tz)

    # Most readable format for agents & humans
    return now.strftime("%Y-%m-%d %-I:%M:%S %p %Z")


@local_tool
async def get_notifications() -> str:
    """Retrieves unread smart home notifications, alerts, or messages. Use this tool whenever the user asks for their notifications, alerts, or messages (e.g., "What are my notifications?", "Read my messages", "Do I have any alerts?").

    CRITICAL: The returned text in the 'data' field already has a specific theatrical persona applied. You MUST output this exact 'data' string to the user verbatim without summarizing, adding introductory text, or applying any additional personality.

    PARAMETERS
    ----------
    None.

    RETURNS
    -------
    str: A JSON-encoded string containing the notification status and pre-formatted text payload.
    """

    # Secrets and config are handled by the host environment, completely hidden from the LLM
    ha_base_url = config.HA_URL
    access_token = config.HA_TOKEN

    endpoint = f"{ha_base_url.rstrip('/')}/api/services/script/read_and_clear_notifications?return_response=true"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            endpoint,
            headers=headers,
            json={}
        )
        response.raise_for_status()

        # Depending on your HA setup, you may need to extract the exact nested path
        # from the HA API response to match the clean schema returned to the LLM.
        raw_ha_response = response.json()

        result = raw_ha_response.get("service_response") or raw_ha_response.get("response", {
            "status": "error",
            "data": "Failed to parse the notification payload."
        })

        return json.dumps(result)


@local_tool
async def brave_search(query: str) -> str:
    """
    Query the Brave Answers AI to research current events, recent information, or factual questions.
    Pass full, natural language questions as the argument. The tool will return a concise, synthesized
    summary of the most up-to-date web intelligence.

    PARAMETERS
    ----------
    query (str): The question or search query to research on the internet.

    RETURNS
    -------
    A JSON-encoded Brave Answers response containing the generated answer and supporting search context.
    """
    if not BRAVE_SEARCH_API_KEY:
        return json.dumps({
            "error": "BRAVE_SEARCH_API_KEY is not configured."
        })

    system_instructions= ("Answer with the bare minimum needed to answer the user's "
                    "query accurately. Be extremely concise: no introduction, "
                    "repetition, explanation, formatting, or conversational "
                    "filler. Include only essential facts and necessary "
                    "qualifiers.")

    if not query.endswith("?"):
        query += "?"
    payload = {
        "stream": False,
        "messages": [
            {"role": "user", "content": f"{query}\n\nQuery: {system_instructions}"},
        ],
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-Subscription-Token": BRAVE_SEARCH_API_KEY,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.search.brave.com/res/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            return json.dumps(response.json())
    except httpx.HTTPStatusError as exc:
        return json.dumps({
            "error": "Brave Answers request failed.",
            "status_code": exc.response.status_code,
            "details": exc.response.text,
        })
    except httpx.RequestError as exc:
        return json.dumps({
            "error": "Unable to reach Brave Answers.",
            "details": str(exc),
        })


@local_tool
async def local_entity_state(entity_ids: list[str]) -> str:
    """Retrieve the current state and key attributes of one or more entities. Use this tool whenever a question involves the current status, value, location, health, on/off state, brightness, temperature, position, or any real-time property of an entity. Prefer fetching multiple relevant entities in a single call. NOTE: This tool name DOES NOT have a prefix. Call it exactly as 'local_entity_state'

    PARAMETERS
    ----------
    entity_ids (list[str]): List of entity IDs to fetch (e.g. ['light.kitchen_ceiling', 'sensor.backyard_temperature']). Use a list even for one entity.

    RETURNS
    -------
    A compact JSON object containing the current state of each entity.
    """
    from . import cache_utils

    if isinstance(entity_ids, str):
        entity_ids = [entity_ids]

    result:dict[str,Any] = {}
    missing = []

    for eid in entity_ids:
        state = cache_utils.state_cache.get(eid)
        if state is not None:
            result[eid] = state
        else:
            missing.append(eid)

    # Include missing entities so LLM knows what failed
    if missing:
        result["_missing"] = missing

    # Return compact JSON (LLMs prefer this over pretty-printed)
    return json.dumps(result, separators=(",", ":"))


### Utilities for weather fetching####
@async_ttl_cache(ttl=None, key_fn=lambda location, client: location)
async def _geocode(location:str, client: httpx.AsyncClient)->tuple[float, float]:
    """Geocode a location string to latitude and longitude."""
    logging.warning(f"Geocode cache miss for {location}")
    geo_url = f"https://nominatim.openstreetmap.org/search?q={location}&format=json&limit=1"
    response = await client.get(geo_url)
    response.raise_for_status()
    data = response.json()
    if not data:
        raise ValueError(f"Location not found: {location}")

    return float(data[0]["lat"]), float(data[0]["lon"])

@async_ttl_cache(ttl=None, key_fn=lambda lat, lon, client: f"{lat:.3f},{lon:.3f}")
async def _grid_forecast_url(lat: float, lon: float, client: httpx.AsyncClient) -> str:
    logging.warning(f"Grid forecast cache miss for {lat:.3f},{lon:.3f}")
    res = await client.get(f"https://api.weather.gov/points/{lat:.3f},{lon:.3f}", follow_redirects=True)
    res.raise_for_status()
    return res.json()["properties"]["forecast"]

@async_ttl_cache(ttl=1200, key_fn=lambda forecast_url, client: forecast_url)
async def _forecast_periods(forecast_url: str,client: httpx.AsyncClient) -> list:
    logging.warning(f"Forecast cache miss for {forecast_url}")
    res = await client.get(forecast_url)
    res.raise_for_status()
    return res.json()["properties"]["periods"]

# Helper to strip the NWS fat
def _trim_period(p):
    return {
        "time": p.get("name"), # e.g., "Saturday Night"
        "forecast": p.get("detailedForecast")
    }

@local_tool
async def get_weather_forecast(
        latitude: float | None = None,
        longitude: float | None = None,
        *,
        location: str | None = None,
        timeframe: Literal["today", "tomorrow", "this weekend", "7 day"] = "today",
):
    """Retrieves real-time weather forecasts from the NWS for a specific sector.
Use for current, daily, or weekend forecasts. Provide either latitude and longitude together, or location.
Never provide only one coordinate or combine coordinates with location. Prefer latitude and longitude when known.

The timeframe specifies which forecast period to retrieve.

    PARAMETERS
    ----------
    latitude (float | None): Decimal latitude (e.g., 64.837). Use together with longitude when known.
    longitude (float | None): Decimal longitude (e.g., -147.716). Use together with latitude when known.
    location (str | None): City/State name for non-local scans. Use when coordinates are unavailable.
    timeframe (Literal): The forecast period to retrieve. Defaults to 'today'.

    RETURNS
    -------
    A JSON-encoded weather forecast.
    """
    t0=time.time()
    now = datetime.now() # Mar 19, 2026 (Thursday)

    weather_headers = {"User-Agent": "Starfleet-Command-AVO-Assistant/1.0 (israel@MacStudio)"}
    async with httpx.AsyncClient(timeout=20.0, headers=weather_headers, http2=True) as client:
        if latitude is not None or longitude is not None:
            if latitude is None or longitude is None:
                return json.dumps({
                    'result': "ERROR",
                    'content': "Both latitude and longitude are required when using coordinates."
                })
            # Use direct coordinates (Highest Precision / Lowest Latency)
            lat, lon = latitude, longitude
        elif location:
            # Geocode
            try:
                lat, lon = await _geocode(location,client)
            except (ValueError, httpx.HTTPStatusError) as e:
                return json.dumps({
                    'result': "ERROR",
                    'content': str(e)
                })
        else:
            return json.dumps({
                'result': "ERROR",
                'content': "Provide either both latitude and longitude or a location."
            })

        #  Make sure lat/lon are numbers
        lat = float(lat)
        lon = float(lon)

        try:
            # 2. Get NWS Grid
            forecast_url = await _grid_forecast_url(lat, lon,client)
            # 3. Get Full Forecast (Next 7 days)
            periods = await _forecast_periods(forecast_url,client)
        except httpx.HTTPStatusError as e:
            return json.dumps({'result': "ERROR", 'content': str(e)})

    logging.info(f"Got weather forecast in {time.time()-t0:.2f} seconds")
    if timeframe == "this weekend":
        days_to_friday = (4 - now.weekday()) % 7
        friday_date = (now + timedelta(days=days_to_friday)).date()
        sunday_date = friday_date + timedelta(days=2)
        return json.dumps([_trim_period(p) for p in periods if friday_date <=
                            datetime.fromisoformat(p["startTime"]).date() <= sunday_date])

    if timeframe == "today":
        return json.dumps([_trim_period(p) for p in periods if
                            datetime.fromisoformat(p["startTime"]).date() == now.date()])

    return json.dumps([_trim_period(p) for p in periods[:6]]) # Default to 3 days (day/night pairs)

################ MUSIC PLAYBACK USING apple-music-custom ########################
from .config import ITUNES_URL

@local_tool
async def get_current_playback() -> str:
    """Retrieves the current playback state from Apple Music, including whether something is playing/paused/stopped, the current track details (title, artist, album), playback position, and duration. Use this when the user asks 'what's playing', 'what song is this', or needs context before controlling playback.

    PARAMETERS
    ----------
    None.

    RETURNS
    -------
    The current playback state as a JSON object.
    """
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.get(
                f"{ITUNES_URL.rstrip('/')}/now_playing"
            )
            resp.raise_for_status()
            data = resp.json()

            # Optional: Add a human-friendly summary for the LLM
            if data and isinstance(data, dict):
                state = data.get("state", "stopped")
                track = data.get("track", {})

                if track and state in ("playing", "paused"):
                    summary = f"{state.capitalize()}: {track.get('name')} by {track.get('artist')} " \
                              f"from {track.get('album')}"
                else:
                    summary = "Nothing is currently playing."

                data["friendly_summary"] = summary  # helpful for LLM reasoning

            return json.dumps(data)

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return json.dumps(
                {"state": "stopped", "friendly_summary": "Nothing is currently playing."}
            )
        return json.dumps({"error": f"HTTP error: {e}"})
    except Exception as e:
        return json.dumps({"error": f"Failed to get current playback: {str(e)}"})


async def _set_shuffle(enable: bool):
    """
    Internal helper to set shuffle mode via PUT /shuffle. The body must be 'mode=songs' to enable or 'mode=off' to disable.

    PARAMETERS
    ----------
    enable (bool): Whether shuffle mode should be enabled.

    RETURNS
    -------
    None.
    """
    mode = "songs" if enable else "off"
    async with httpx.AsyncClient() as client:
        await client.put(f"{ITUNES_URL}/shuffle", data={"mode": mode})

@local_tool
async def pause_music():
    """
    Pauses the music. You MUST call this every time a user asks to pause or stop the music

    PARAMETERS
    ----------
    None.

    RETURNS
    -------
    The server response.
    """
    async with httpx.AsyncClient() as client:
        response = await client.put(f"{ITUNES_URL}/pause")
        response.raise_for_status()
        return response.text

@local_tool
async def list_playlists():
    """
    Retrieves all playlists from the library. Call this tool first to find the 'id' of a playlist when the user refers to one by name or to answer questions about available playlists. Returns a list of playlist objects with 'id' and 'name' properties.

    PARAMETERS
    ----------
    None.

    RETURNS
    -------
    A list of playlist objects.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{ITUNES_URL}/playlists")
        response.raise_for_status()
        return response.text

async def list_albums(offset: int = 0, limit: int = 100):
    """
    Retrieves all albums from the library.

    PARAMETERS
    ----------
    offset (int): The offset to use when listing albums.
    limit (int): The number of albums to return in the result set.

    RETURNS
    -------
    A list of album objects.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{ITUNES_URL}/library/albums", params={"offset": offset, 'limit': limit,})
        response.raise_for_status()
        return response.text

@local_tool
async def search_music(query: str, limit: int = 10):
    """Searches the library for albums, artists, or specific tracks. Use this as the
    primary tool for finding music when the exact title is unknown or to verify metadata.
    IMPORTANT:Search using exactly one criterion: a track name, album name, or artist name.
    Do not combine criteria in a single query. If the user provides multiple pieces
    of information, search using only the track name.

    PARAMETERS
    ----------
    query (str): The search term (e.g., 'Dark Side', 'Pink Floyd', or 'Wish You Were Here').
    limit (int): Max results to return.

    RETURNS
    -------
    Matching albums, artists, and tracks.
    """
    from . import cache_utils
    results = {
        'albums': [],
        'artists': [],
        'tracks': [],
    }

    if cache_utils.itunes_cache.get('albums'):
        album_matches = process.extract(
            query,
            cache_utils.itunes_cache['albums'].keys(),
            scorer=fuzz.WRatio,
            processor=utils.default_process,
            limit=limit,
        )
        for title, score, _ in album_matches:
            if score > 55:
                item = dict(cache_utils.itunes_cache['albums'][title])
                item['confidence'] = round(score)
                results['albums'].append(item)

    if cache_utils.itunes_cache.get('artists'):
        artist_matches = process.extract(
            query,
            cache_utils.itunes_cache['artists'].keys(),
            scorer=fuzz.WRatio,
            processor=utils.default_process,
            limit=limit,
        )
        for name, score, _ in artist_matches:
            if score > 55:
                item = dict(cache_utils.itunes_cache['artists'][name])
                item['confidence'] = round(score)
                results['artists'].append(item)

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                "http://localhost:8181/library/search",
                params={"q": query}
            )
            resp.raise_for_status()
            track_data = resp.json()   # Expecting a list of track dicts

            # Add type + confidence so everything is consistent
            for track in track_data['tracks']:  # adjust limit as needed
                item = dict(track)
                item['confidence'] = 80
                results['tracks'].append(item)

    except httpx.HTTPError as e:
        print(f"Track search to server failed: {e}")
        # results['tracks'] stays empty — LLM can still use album/artist matches
    except Exception as e:
        print(f"Unexpected error in track search: {e}")

    return json.dumps(results)


@local_tool
async def play_track(track_id: str):
    """
    Plays a single track by its persistent ID. If the ID is unknown, call search_music first.

    PARAMETERS
    ----------
    track_id (str): The persistent ID of the track.

    RETURNS
    -------
    The server response.
    """
    async with httpx.AsyncClient() as client:
        response = await client.put(f"{ITUNES_URL}/library/tracks/{track_id}/play")
        response.raise_for_status()
        return response.text

@local_tool
async def play_playlist(playlist_id: str, shuffle: bool = False):
    """
    Plays a playlist by ID. Use list_playlists first to resolve a name to an ID. Support optional shuffling.

    PARAMETERS
    ----------
    playlist_id (str): The unique ID of the playlist.
    shuffle (bool): Whether to shuffle the playlist.

    RETURNS
    -------
    The server response.
    """
    await _set_shuffle(shuffle)
    async with httpx.AsyncClient() as client:
        response = await client.put(f"{ITUNES_URL}/playlists/{playlist_id}/play")
        response.raise_for_status()
        return response.text

@local_tool
async def play_album(artist_name: str, album_name: str, shuffle: bool = False):
    """
    Plays a specific album. Requires exact artist and album names, as returned by search_music. Use search_music first to get the exact names.

    PARAMETERS
    ----------
    artist_name (str): The name of the artist.
    album_name (str): The title of the album.
    shuffle (bool): Whether to shuffle the album tracks.

    RETURNS
    -------
    The server response.
    """
    await _set_shuffle(shuffle)
    # quote() ensures spaces/special characters are safe for the URI path
    safe_artist = quote(artist_name, safe='')
    safe_album = quote(album_name, safe='')
    async with httpx.AsyncClient() as client:
        response = await client.put(f"{ITUNES_URL}/library/albums/{safe_artist}/{safe_album}/play")
        response.raise_for_status()
        return response.text

@local_tool
async def play_artist(artist_name: str, shuffle: bool = True):
    """
    Plays all tracks by a specific artist. Defaults to shuffle mode.

    PARAMETERS
    ----------
    artist_name (str): The name of the artist.
    shuffle (bool): Whether to shuffle the album tracks.

    RETURNS
    -------
    The server response.
    """
    await _set_shuffle(shuffle)
    safe_artist = quote(artist_name, safe='')
    async with httpx.AsyncClient() as client:
        response = await client.put(f"{ITUNES_URL}/library/artists/{safe_artist}/play")
        response.raise_for_status()
        return response.text
