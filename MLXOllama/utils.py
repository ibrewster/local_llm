import asyncio
import gc
import hashlib
import json
import logging
import queue
import re
import threading
import uuid

from asyncio import Lock
from concurrent.futures import Future
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Dict, Optional

import mlx.utils as mx_utils
import mlx.core as mx

from cachetools import TTLCache
from huggingface_hub import snapshot_download
from mlx_lm import sample_utils, load

from . import config, mcp_client

##### GLOBAL OBJECTS#######
cache_lock = Lock()
model_ready = {}
loaded_models = {}


class InferenceWorker:
    _STOP = object()
    
    def __init__(self, name="Hermes_Thinker"):
        self.tasks = queue.Queue()
        self.shutdown_flag = False
        
        self.thread = threading.Thread(
            target=self._run,
            name=name,
            daemon=True,
        )
        self.thread.start()

    def submit(self, fn, *args, **kwargs):
        if self.shutdown_flag:
            raise RuntimeError("Worker has been shut down")
        
        future = Future()
        self.tasks.put((future, fn, args, kwargs))
        return future
    
    def shutdown(self, wait=True):
        if not self.shutdown_flag:
            self.shutdown_flag = True
            self.tasks.put(self._STOP)

        if wait:
            self.thread.join()    

    def _run(self):
        while True:
            item = self.tasks.get()
            if item is self._STOP:
                logging.info("Inference worker thread stopping")
                break
            
            future, fn, args, kwargs = item

            if future.set_running_or_notify_cancel():
                try:
                    result = fn(*args, **kwargs)
                except (asyncio.exceptions.CancelledError, Exception) as e:
                    future.set_exception(e)
                else:
                    future.set_result(result)
                # finally:
                    # TODO: figure out if this is overkill/hurting performance
                    # mx.clear_cache()
                    # gc.collect()
                    # mx.clear_cache()                

inference_worker = InferenceWorker(name="Hermes_Thinker")

FUN_SAMPLER = sample_utils.make_sampler(
    temp=1.0, # was 0.95
    top_p=1.0, # was 0.9
    min_p=0.05
#    top_k=20
)

sentence_endings_eager = re.compile(
    r'(?<!\w\.\w)'          # not mid-abbreviation
    r'(?<![A-Z][a-z]\.)'    # not after titles
    r'[.!?]+'               # punctuation
    r'(?!\d)'               # not before a digit
    r'(?=[\s\n]|$)'         # explicitly include newline check
)

sentence_endings_conservative = re.compile(
    r'(?<!\w\.\w)'          # not mid-abbreviation like U.S.A.
    r'(?<![A-Z][a-z]\.)'   # not after titles: Mr. Dr. St. etc.
    r'(?<!\d)'              # not after digit...
    r'[.!?]+'
    r'(?!\d)'               # ...or before digit (together these protect 2.5, -8.2)
    r'(?=\s*(?:\n|$))'      # must be followed by uppercase or end of buffer
)

paragraph_split = re.compile(r'\n{2,}')

conversation_store = TTLCache(maxsize=100, ttl=3600)

MCP_CLIENT_CONFIG = {
    "mcpServers": {
        # "MCPAssist": {
            # "url": "http://10.27.81.207:8090",
        # },
        "homeassistant": {
            "url": "http://10.27.81.207:8123/api/mcp",
            "headers": {
                "Authorization": f"Bearer {config.HA_TOKEN}"
            }
        },
        # "searxng-http": {
            # "url": "http://10.27.81.60:3003/mcp",
            # "env": {
                # "SEARXNG_URL": "http://10.27.81.60:3002/search",
                # "MCP_HTTP_PORT": "3003"
            # }
        # },
        "tavily": {
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={config.TAVILY_TOKEN}",
        },
    }
}

MCP_CLIENT=mcp_client.MCPService(MCP_CLIENT_CONFIG)

mlx_inference_lock = threading.Lock()
###################

def init_logging():
    log_path = config.LOG_PATH

    handler = RotatingFileHandler(
        log_path,
        maxBytes=10_000_000,  # 10MB
        backupCount=5
    )

    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)-5s %(threadName)s %(message)s',
        datefmt='%d/%b/%Y:%H:%M:%S %z'
    )

    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

def get_model_path(repo_id: str) -> Path:
    """Returns the local snapshot path for a HF repo, downloading if needed.
    If already cached, this is instantaneous — no network call."""
    return Path(snapshot_download(repo_id=repo_id, local_files_only=True))

def model_digest(model_id: str) -> str:
    """SHA-256 of the model's config.json — stable across process restarts,
    changes if the model is actually updated."""
    config = get_model_path(model_id) / "config.json"
    return "sha256:" + hashlib.sha256(config.read_bytes()).hexdigest()

def model_details(model_name: str) -> dict:
    """Build details from the actual loaded model config."""

    model_info = loaded_models[model_name]
    repo_id = model_info['repo_id']
    model_config = model_info['config']
    family = family_name(model_info['name'], model_config.get("model_type", "qwen"))
    return {
        "parent_model": "",
        "format": "safetensors",
        "family": family,
        "families": [family],
        "parameter_size": _param_size(model_name, model_config),
        "quantization_level": _quant_from_repo_id(repo_id)
    }

def _param_size(name: str, config: dict):
    if s := _param_size_from_name(name):
        return s

    if n := config.get("num_parameters"):
        return f"{n/1e9:.1f}B"

    return _estimate_from_config(config)


def _param_size_from_name(name: str):
    m = re.search(r'(\d+(?:\.\d+)?)B', name, re.IGNORECASE)
    if m:
        return f"{m.group(1)}B"

def _estimate_from_config(config: dict) -> str:
    """Estimate parameter count from hidden_size/num_layers if not explicit."""
    # Qwen config.json has num_parameters directly in some versions,
    # otherwise derive it or just read it from the directory name / model card
    if n := config.get("num_parameters"):
        return f"{n/1e9:.1f}B"

    h = config.get("hidden_size")
    i = config.get("intermediate_size")
    L = config.get("num_hidden_layers")
    v = config.get("vocab_size")

    if all((h, i, L, v)):
        per_layer = 4 * h * h + 2 * h * i
        total = L * per_layer + v * h
        billions = int(round(total / 1e9))
        return f"{billions}B"

    return "7B"  # fallback

def _quant_from_repo_id(repo_id: str) -> str:
    repo_lower = repo_id.lower()
    for q in ("4bit", "8bit", "3bit", "6bit", "bf16", "fp16"):
        if q in repo_lower:
            return q
    return "unknown"

def base_model_entry(model_info: dict) -> dict:
    repo_id = model_info['repo_id']
    return {
        "name": model_info['name'],
        "model": model_info['name'],
        "modified_at": model_modified_at(repo_id),
        "size": model_size(repo_id),
        "digest": model_digest(repo_id),
        "details": model_details(model_info['name']),
    }

def model_modified_at(repo_id) -> str:
    """Use the mtime of config.json in the snapshot as a proxy for 'last modified'."""
    mtime = (get_model_path(repo_id) / "config.json").stat().st_mtime
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

def model_size(repo_id) -> int:
    """Sum of all blob file sizes — actual disk usage of the model."""
    blobs_dir = get_model_path(repo_id).parent.parent / "blobs"
    return sum(f.stat().st_size for f in blobs_dir.iterdir() if f.is_file())

def family_name(model_name: str, model_type: str):
    # Prioritize obvious families from name
    name_lower = model_name.lower()
    if "qwen3" in name_lower or "qwen2" in name_lower:
        return "qwen"
    if "llama" in name_lower:
        return "llama"
    if "mixtral" in name_lower or "mistral" in name_lower:
        return "mistral"
    # fallback to type
    return model_type

OLLAMA_OPTIONS_MAP = {
    # Ollama option  -> you can map to mlx param names
    "temperature":    "temp",
    "num_predict":    "max_tokens",
    "top_p":          "top_p",
    "top_k":          "top_k",
    "repeat_penalty": "repetition_penalty",
    # "seed":         "seed",  # if mlx_lm supports it
}

def parse_options(raw: dict) -> dict:
    return {OLLAMA_OPTIONS_MAP.get(k, k): v for k, v in (raw or {}).items()}

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
            # smart type coercion (same as before)
            try:
                if val_str.startswith('[') and val_str.endswith(']'):
                    val = json.loads(val_str)
                elif val_str.lower() in ('true', 'false'):
                    val = val_str.lower() == 'true'
                elif val_str.replace('.', '', 1).replace('-', '', 1).isdigit():
                    val = float(val_str) if '.' in val_str else int(val_str)
                else:
                    val = val_str
            except:
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
    # ────────────────────────────────────────────────
    gemma4_pattern = r'<\|tool_call>(.*?)<tool_call\|>'
    gemma4_blocks = re.findall(gemma4_pattern, text, re.DOTALL)

    for block in gemma4_blocks:
        block_full = f'<|tool_call>{block}<tool_call|>'
        cleaned_text = cleaned_text.replace(block_full, '', 1).strip()

        body = block.strip()
        if not body.startswith('call:'):
            continue
        body = body[len('call:'):]

        brace_idx = body.find('{')
        if brace_idx == -1:
            continue
        func_name = body[:brace_idx].strip()
        args_str = body[brace_idx:]

        # Replace <|"|> string delimiters, then quote bare keys
        args_str = args_str.replace('<|"|>', '"')
        args_str = re.sub(r'(?<!["\w])(\w+)\s*:', r'"\1":', args_str)

        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
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

def is_tool_response_request(messages: list) -> bool:
    """Check if this request is just delivering a tool result back to the LLM."""
    if not messages:
        return False
    last = messages[-1]
    return (
        isinstance(last, dict)
        and last.get("role") == "tool"
        and isinstance(last.get("content"), str)
        and "response_type" in last["content"]
    )

def load_model(name, path):
    logging.info(f"Loading {name}")
    model, tokenizer, model_config= load(path, return_config=True)

    model_bytes = sum(x.nbytes for _, x in mx_utils.tree_flatten(model.parameters()))
    
    loaded_models[name] = {
        "model": model,
        "tokenizer": tokenizer,
        "config": model_config,
        "repo_id": path,
        "name": name,
        "size_vram": model_bytes,
    }
    model_ready[name] = threading.Event()
    model_ready[name].set()

def unload_model(name):
    if name not in loaded_models:
        logging.warning(f"Model {name} not found")
        return

    model_ready[name].clear()
    logging.info(f"Unloading {name}")
    del loaded_models[name]  # Remove all references to model, tokenizer, config
    gc.collect()             # Force CPython GC to release the objects
    mx.clear_cache()   # Free MLX's internal Metal buffer cache
