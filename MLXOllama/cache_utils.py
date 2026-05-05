import asyncio
import datetime
import gc
import httpx
import json
import logging
import re
import threading
import time

from dataclasses import dataclass, field
from pathlib import Path

import xxhash

import mlx.core as mx

from aiocache import cached
from cachetools import TTLCache
from mlx_lm.models.cache import make_prompt_cache, save_prompt_cache, load_prompt_cache

from . import utils, config
from .utils import MCP_CLIENT, HA_MARKER

static_caches = TTLCache(maxsize=128, ttl=172800) # Cache for two days, if no use in that time, ditch.
dynamic_caches = TTLCache(maxsize=8, ttl=3600)
state_cache = {}
itunes_cache = {}

logging.getLogger("httpx").setLevel(logging.WARNING)

_background_tasks = set()

async def get_cache(model_info, messages, tools, is_static=False):
    if len(messages) == 1:
        return None, False # No system prompt to cache
    
    first_prompt = messages[0]['content']

    if HA_MARKER in first_prompt:
        # HA request. Deal with the cache
        cache = static_caches.get('HA')
        if cache is not None:
            return cache, True
        
        # No HA cache. Create one.
        cache = await create_ha_cache(model_info, messages, tools)
        static_caches['HA'] = cache
        
        return cache, True
    elif "#####MORNING#####" in first_prompt:
        cache = static_caches.get('MORNING')
        if cache is None:
            cache = await create_morning_cache("") # No news to cache
            static_caches['MORNING'] = cache
        return cache, False
    
    cache_store = static_caches if is_static else dynamic_caches
    cache_key = str(first_prompt) + json.dumps(tools, sort_keys=True)
    cache_hash = xxhash.xxh64(cache_key.encode()).hexdigest()
    
    cache = cache_store.get(cache_hash)
    if cache is None:
        logging.info("Cache Miss")
        cache = await _build_cache(model_info, first_prompt, tools)
    else:
        logging.info("Cache Hit")

    # Bump the TTL
    cache_store[cache_hash] = cache
    if is_static:
        task = asyncio.create_task(_save_static_caches())
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
        
    return cache, False
    
async def _save_static_caches():
    path: Path = Path(__file__).parent / "Caches"
    path.mkdir(exist_ok=True)
    
    # Clean up any existing Caches
    for f in path.glob('*.safetensors'):
        f.unlink()
        await asyncio.sleep(0)
        
    try:
        for key, cache in static_caches.items():
            file = path / f"{key}.safetensors"
            with utils.mlx_inference_lock:
                save_prompt_cache(str(file), cache)
            await asyncio.sleep(0)
    except Exception as e:
        logging.warning(f"Cache save failed (non-critical): {e}")
        
async def _load_static_caches():
    path = Path(__file__).parent / "Caches"
    if path.exists():
        for f in path.glob("*.safetensors"):
            key = f.stem 
            try:
                static_caches[key] = load_prompt_cache(f)
                logging.info(f"Loaded static cache: {key[:16]}...")  # truncate hash for log readability
            except Exception as e:
                logging.warning(f"Failed to load cache {key[:16]}...: {e}")
            await asyncio.sleep(0)
        
@dataclass
class LiveEntity:
    names: list[str]
    domain: str
    state: str
    areas: list[str] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)
 
async def get_rest_data() ->list[dict]:
    headers = {
        "Authorization": f"Bearer {config.HA_TOKEN}",
        "Content-Type": "application/json",
    }    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{config.HA_URL}/api/states",
                headers=headers,
                timeout=10.0,
            )
            response.raise_for_status()
    except Exception as e:
        print(e)
        return
    
    return response.json()

async def state_refresh_loop():
    """Runs forever, refreshing the entity state cache."""
    while True:
        try:
            await refresh_entity_states()
        except Exception as e:  # Never let the task die silently
            logging.error(f"HA state refresh failed: {e}", exc_info=True)
        
        await asyncio.sleep(20)
        
async def refresh_entity_states():
    entity_data = await get_rest_data()
    if entity_data is None:
        logging.warning("Unable to update entity data. No new data recieved.")
        return;
    
    for entity in entity_data:
        entity_id = entity['entity_id']
        attrs = entity.get("attributes", {})
                
        entry = {
            "state": entity["state"],
            "friendly_name": attrs.get(
                "friendly_name",
                entity_id.split(".")[-1].replace("_", " ").title()
            ),
        }
        
        if "unit_of_measurement" in attrs:
            entry["unit_of_measurement"] = attrs["unit_of_measurement"]
        if "device_class" in attrs:
            entry["device_class"] = attrs["device_class"]
        
        # Optional: last_changed
        if "last_changed" in entity and entity["last_changed"] != "unknown":
            entry["last_changed"] = entity["last_changed"]
        
        state_cache[entity_id] = entry
    
@cached(ttl=21600) # six hours
async def get_entity_ids() ->dict[str:str]:
    # Get more information on the entities from the REST api
    response = await get_rest_data()
    
    friendly_to_id: dict[str, str] = {
        s["attributes"].get("friendly_name", "").lower(): s["entity_id"]
        for s in response
    }
    return friendly_to_id

@cached(ttl=86400) # 24 hours
async def get_live_context() -> list[LiveEntity]:
    ha_exposed_entities = await MCP_CLIENT.call_tool('homeassistant_GetLiveContext')
        
    exposed_entities: str = json.loads(ha_exposed_entities.content[0].text)['result']
    exposed_entities = re.sub(r"^Live Context:.*?\n", "", exposed_entities, flags=re.DOTALL).strip()
    
    entities: list[LiveEntity] = []
    current: dict | None = None    
    
    for line in exposed_entities.splitlines():
        # Top-level entity block (starts with "- names:")
        if line.startswith("- names:"):
            if current:
                entities.append(_build_entity(current))
            names_raw = line.removeprefix("- names:").strip()
            current = {"names": names_raw, "attributes": {}}

        elif current is None:
            continue

        elif m := re.match(r"^\s{2}(\w+):\s*(.*)", line):
            key, val = m.group(1), m.group(2).strip().strip("'")
            if key != "attributes":
                current[key] = val

        elif m := re.match(r"^\s{4}(\w+):\s*(.*)", line):
            attr_key, attr_val = m.group(1), m.group(2).strip().strip("'")
            current["attributes"][attr_key] = attr_val or None

    if current:
        entities.append(_build_entity(current))

    return entities    
    
def _build_entity(raw: dict) -> LiveEntity:
    names = [n.strip() for n in raw["names"].split(",")]
    areas = [a.strip() for a in raw.get("areas", "").split(",")] if raw.get("areas") else []
    return LiveEntity(
        names=names,
        domain=raw.get("domain", ""),
        state=raw.get("state", ""),
        areas=areas,
        attributes=raw.get("attributes", {}),
    )


async def cache_refresh_loop():
    from . import speak_queue
    # When starting this loop, first load any disk saved caches
    logging.info("Loading static caches from disk")
    await _load_static_caches()
    if not 'HA' in static_caches:
        ha_cache = await create_ha_cache()
        static_caches['HA'] = ha_cache
        await _save_static_caches()
    else:
        speak_queue.put_nowait("UPDATE")
        
        
    TARGET_TIME = datetime.time(0, 30)  # 12:30 AM
    while True:
        now = datetime.datetime.now()
        target = datetime.datetime.combine(now.date(), TARGET_TIME)
        
        if target <= now:
            target += datetime.timedelta(days=1) # roll to tomorrow
        
        seconds_until = (target - now).total_seconds()
        
        await asyncio.sleep(seconds_until)
        try:
            async with utils.cache_lock:
                # Delete the old first to free memory
                if 'HA' in static_caches:
                    del static_caches['HA']
                mx.clear_cache()
                gc.collect()
                
                cache = await create_ha_cache()
                static_caches['HA'] = cache
            logging.info("HA prompt cache successfully updated")
            await _save_static_caches()

        except Exception as e:  # Never let the task die silently
            logging.error(f"HA cache refresh failed: {e}", exc_info=True)
 
 
async def itunes_cache_refresh():
    while True:
        logging.info("Refreshing iTunes album/artist cache")
        await create_iTunes_cache()
        logging.info("iTunes cache refreshed")
        await asyncio.sleep(60 * 30)
        
async def refresh_cache_background():
    cache = await create_ha_cache()
    static_caches['HA'] = cache
    
@cached(ttl=86400)
async def get_tools():
    mcp_tools = await MCP_CLIENT.list_tools()
    return mcp_tools

async def create_morning_cache(news_prompt):
    logging.info("Creating morning cache")
    t0 = time.time()
    system_prompt = (Path(__file__).parent / "morning_system.txt").read_text()
    model_info = utils.loaded_models[config.MAIN_MODEL]
    cache = await _build_cache(model_info, system_prompt, None, news_prompt)
    static_caches['MORNING'] = cache
    await _save_static_caches()
    logging.info(f"Created morning cache in {time.time() - t0}")
    return cache
    
async def create_ha_cache():
    from . import speak_queue
    from .local_tools import LOCAL_TOOLS
    
    logging.info(f"Creating new Home Assistant cache")
    t0 = time.time()
    model_info = utils.loaded_models[config.MAIN_MODEL]
    
    # Fetch the tools list.
    mcp_tools = await get_tools()
    skipped_tools = {
        'homeassistant_GetLiveContext',
        'MCPAssist_perform_action',
        'homeassistant_HassMediaSearchAndPlay',
        'homeassistant_HassMediaPause'
    }
    tools = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            },
        }
        for tool in mcp_tools
        if tool.name not in skipped_tools
    ]
    
    # Add local tools
    tools += LOCAL_TOOLS.values()
    
    live_context = await get_live_context()
    entity_ids = await get_entity_ids()
    
    lines: list[str] = []
    
    for entity in live_context:
        for name in entity.names:
            if entity_id := entity_ids.get(name.lower()):
                # Using YAML-style structure for maximum Qwen compatibility
                entry = [
                    f"- names: {', '.join(entity.names)}",
                    f"  domain: {entity.domain}",
                    f"  entity_id: {entity_id}",
                ]
                if entity.areas:
                    entry.append(f"  areas: {', '.join(entity.areas)}")
                lines.append("\n".join(entry))
                break
    
    entities = "\n".join(lines)
    system_prompt = (Path(__file__).parent / "HA_system.txt").read_text()
    system_prompt += f"\n{entities}"
    date_string = f"The current date is\
    {time.strftime('%A, %B %-d, %Y')}\n"
    system_prompt = date_string + system_prompt
    
    cache = await _build_cache(model_info, system_prompt, tools)
    
    logging.info(f"Created and populated HA cache in {time.time() - t0}s")
    speak_queue.put_nowait("UPDATE")
    
    return cache    
    
async def _build_cache(model_info, system_prompt, tools, user_prompt=""):
    model = model_info['model']
    tokenizer = model_info['tokenizer']
    
    prefix_messages = [
        {'role': 'system', 'content': system_prompt},  # system
        {"role": "user", "content": user_prompt}
    ]    
    
    prefix_str = tokenizer.apply_chat_template(
        prefix_messages,
        tools=tools,              
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    
    prefix_tokens = tokenizer.encode(prefix_str)
    
    def _populate_cache(model, prefix_tokens):
        cache = make_prompt_cache(model)
        with utils.mlx_inference_lock:
            logits = model(mx.array([prefix_tokens]), cache=cache)
            mx.eval(logits)
        return cache
    
    cache = await asyncio.to_thread(_populate_cache, model, prefix_tokens)
    logging.info(f"Prefix tokens: {len(prefix_tokens)}")
    
    return cache

async def _get_library_items(path):
    from .local_tools import ITUNES_URL
    offset = 0
    limit = 2000    
    all_items = []
    async with httpx.AsyncClient() as client:
        while True:
            response = await client.get(f"{ITUNES_URL}/library/{path}", params={"offset": offset, 'limit': limit,})
            response.raise_for_status()
            data = response.json()
            all_items.append(data)
            if (offset + limit) >= data['total']:
                break

            offset += limit
            
    return all_items

async def create_iTunes_cache():
    # Walk through available albums
    all_albums = await _get_library_items('albums')
    all_albums = all_albums[0]['albums']
    itunes_cache['albums']={a['name']:a for a in all_albums}

    all_artists = await _get_library_items('artists')
    all_artists = all_artists[0]['artists']
    itunes_cache['artists'] = {a['name']: a for a in all_artists}
    
    