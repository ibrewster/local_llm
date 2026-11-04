import asyncio
import gc
import hashlib
import json
import logging
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
from mlx_vlm import sample_utils, load
from mlx_vlm.utils import load_config
from sentence_transformers import SentenceTransformer

from . import config
from .preemptive_queue import PreemptiveQueue

##### GLOBAL OBJECTS#######
EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

cache_lock = Lock()
# Hugging Face's fast tokenizer is backed by a Rust object whose mutable
# internals cannot be borrowed by two threads at once.  The processor/tokenizer
# is shared by the async request handlers and the inference worker, so protect
# every tokenizer/processor operation that can touch it.
tokenizer_lock = threading.RLock()
model_ready = {}
loaded_models = {}


class InferenceWorker:
    _STOP = object()
    
    def __init__(self, name="Hermes_Thinker"):
        self.tasks = PreemptiveQueue()
        self.shutdown_flag = False
        
        self.thread = threading.Thread(
            target=self._run,
            name=name,
            daemon=True,
        )
        self.thread.start()

    def submit(self, fn, *args, priority=False, **kwargs):
        if self.shutdown_flag:
            raise RuntimeError("Worker has been shut down")
        
        future = Future()
        if priority:
            self.tasks.put_front((future, fn, args, kwargs))
        else:
            self.tasks.put((future, fn, args, kwargs))
        return future
    
    def shutdown(self, wait=True):
        if not self.shutdown_flag:
            self.shutdown_flag = True
            self.tasks.put(self._STOP)

        if wait:
            self.thread.join()

    def pending_count(self):
        """Queued tasks that haven't been canceled (excludes the running task)."""
        return sum(
            1 for item in self.tasks.snapshot()
            if item is not self._STOP and not item[0].cancelled()
        )

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
    """Returns the local snapshot path for a huggingface repo, downloading if needed.
    If already cached, this is instantaneous — no network call."""
    return Path(snapshot_download(repo_id=repo_id, local_files_only=True))

def model_digest(model_id: str) -> str:
    """SHA-256 of the model's config.json — stable across process restarts,
    changes if the model is actually updated."""
    _config = get_model_path(model_id) / "config.json"
    return "sha256:" + hashlib.sha256(_config.read_bytes()).hexdigest()

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

def _param_size(name: str, model_config: dict):
    if s := _param_size_from_name(name):
        return s

    if n := model_config.get("num_parameters"):
        return f"{n/1e9:.1f}B"

    return _estimate_from_config(model_config)


def _param_size_from_name(name: str):
    m = re.search(r'(\d+(?:\.\d+)?)B', name, re.IGNORECASE)
    if m:
        return f"{m.group(1)}B"
    return None


def _estimate_from_config(model_config: dict) -> str:
    """Estimate parameter count from hidden_size/num_layers if not explicit."""
    # Qwen config.json has num_parameters directly in some versions,
    # otherwise derive it or just read it from the directory name / model card
    if n := model_config.get("num_parameters"):
        return f"{n/1e9:.1f}B"

    h = model_config.get("hidden_size")
    i = model_config.get("intermediate_size")
    L = model_config.get("num_hidden_layers")
    v = model_config.get("vocab_size")

    if (
        isinstance(h, (int, float)) and h
        and isinstance(i, (int, float)) and i
        and isinstance(L, (int, float)) and L
        and isinstance(v, (int, float)) and v
    ):
        per_layer = 4 * h * h + 2 * h * i
        total = L * per_layer + v * h
        billions = round(total / 1e9)
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
        "capabilities": ["completion", "vision", "tools"],
    }

def embed_model_entry() -> dict:
    return {
        "name": EMBED_MODEL_NAME,
        "model": EMBED_MODEL_NAME,
        "modified_at": "2024-01-01T00:00:00.000Z",
        "size": 90900000,
        "digest": "sha256:all-minilm-l6-v2",
        "details": {
            "parent_model": "",
            "format": "safetensors",
            "family": "bert",
            "families": ["bert"],
            "parameter_size": "22.7M",
            "quantization_level": "fp32",
        },
        "capabilities": ["embedding"],
    }

def extract_embed_inputs(data: dict) -> list[str]:
    """Extract and normalize inputs from OpenAI or Ollama embedding request payloads."""
    raw = data.get("input") if "input" in data else data.get("prompt", [])
    if isinstance(raw, str):
        return [raw] if raw else []
    if isinstance(raw, list):
        return [str(x) for x in raw if x is not None]
    return []

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
    # Ollama option -> you can map to mlx param names
    "temperature":    "temp",
    "num_predict":    "max_tokens",
    "top_p":          "top_p",
    "top_k":          "top_k",
    "repeat_penalty": "repetition_penalty",
    # "seed":         "seed",  # if mlx-vlm supports it
}

def parse_options(raw: dict) -> dict:
    return {OLLAMA_OPTIONS_MAP.get(k, k): v for k, v in (raw or {}).items()}


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
    model, processor = load(path)
    model_config = load_config(path)

    model_bytes = sum(
        x.nbytes
        for _, x in mx_utils.tree_flatten(model.parameters())
        if isinstance(x, mx.array)
    )
    
    loaded_models[name] = {
        "model": model,
        "processor": processor,
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
