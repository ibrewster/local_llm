import asyncio
import base64
from io import BytesIO
import json
import multiprocessing
import logging

from pathlib import Path

import quart
from PIL import Image
import sounddevice as sd
import soundfile as sf

from . import app, speak_queue, config, utils, worker, cache_utils


async def _read_chat_request():
    """Read a chat request from JSON or multipart/form-data.

    Multipart requests may put the Ollama/OpenAI JSON payload in ``data`` and
    one or more image files in fields named ``image`` or ``images``.
    """
    if quart.request.is_json:
        return await quart.request.get_json(), []

    form = await quart.request.form
    raw_data = form.get("data") or form.get("request") or form.get("json")
    if not raw_data:
        raise ValueError("multipart chat requests must include a data field")
    data = json.loads(raw_data)

    images = []
    files = await quart.request.files
    for upload in files.getlist("images") + files.getlist("image"):
        image_bytes = await upload.read()
        images.append(Image.open(BytesIO(image_bytes)).convert("RGB"))
    return data, images


def _images_from_messages(messages):
    """Decode data-URI/base64 images supplied in a JSON message."""
    images = []
    for message in messages:
        # Ollama's native chat format puts base64 image data directly on the
        # message, while OpenAI-compatible clients use image_url content.
        message_images = message.get("images", []) if isinstance(message, dict) else []
        if isinstance(message_images, str):
            message_images = [message_images]
        for encoded in message_images:
            try:
                if encoded.startswith("data:image/"):
                    encoded = encoded.split(",", 1)[1]
                images.append(Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB"))
            except Exception as exc:
                raise ValueError(f"invalid image data: {exc}") from exc

        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            image_url = item.get("image_url", item.get("input_image"))
            url = image_url.get("url") if isinstance(image_url, dict) else image_url
            if not isinstance(url, str) or not url.startswith("data:image/"):
                continue
            try:
                _, encoded = url.split(",", 1)
                images.append(Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB"))
            except Exception as exc:
                raise ValueError(f"invalid image data: {exc}") from exc
    return images


def _message_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            item.get("text", "") for item in content
            if isinstance(item, dict) and item.get("type") in ("text", "input_text")
        )
    return ""


@app.route("/playGoodnight", methods=["POST"])
async def goodnight():
    app.add_background_task(play_goodnight)
    return quart.jsonify({"status": "playing", "message": "Sweet dreams..."}), 202

def play_goodnight():
    app.logger.info("Playback Started")
    goodnight_file = Path(__file__).parent / "GenOut/goodnight/goodnight.wav"
    device = sd.query_devices('BlackHole 2ch')
    
    data, samplerate = sf.read(str(goodnight_file), dtype='float32')
    app.logger.info(f"🎵 Playing on device: {device['name'] if device is not None else 'system default'}")
    
    sd.play(data, samplerate, device=device['index'], latency='low')
    sd.wait()
    
    app.logger.info("Playback finished")
    
@app.route('/generate', methods = ['POST'])
async def generate_response():
    payload = await quart.request.json
    prompt = payload['prompt']
    system = payload.get("system", "")
    speak_queue.put((system, prompt))
    return "Accepted", 202

@app.route('/generate/cache_news', methods=['POST'])
async def cache_news():
    payload = await quart.request.json
    news = payload['news']
    app.add_background_task(cache_utils.create_morning_cache, news)
    return "Accepted", 202

@app.route("/api/version")
async def api_version():
    return quart.jsonify({"version": config.OLLAMA_VERSION})

@app.route("/api/tags")
async def api_tags():
    models = [utils.base_model_entry(info) for info in utils.loaded_models.values()]
    return quart.jsonify({"models": models})

@app.get("/v1/models")
async def list_models():
    models = []
    for model_info in utils.loaded_models.values():
        model = {
            "id": model_info['name'],
            'object': "model",
            "created": 1690000000,
            "owned_by": "mlx-vlm",
        }
        models.append(model)
        
    return {
        "object": "list",
        "data": models
    }

@app.route("/api/show", methods=["POST"])
async def api_show():
    data = await quart.request.get_json()
    name = data.get("name", config.QUICK_MODEL)
    if name not in utils.loaded_models:
        return quart.jsonify({"error": f"model '{name}' not found"}), 404
    
    model_info = utils.loaded_models[name]
    model_config = model_info['config']
    
    return quart.jsonify({
        "modelfile": f"FROM {model_info['repo_id']}",
        "parameters": f"num_predict {model_config.get('max_position_embeddings', 2048)}\ntemperature 0.7",
        "template": model_info['processor'].tokenizer.chat_template,
        "details": utils.model_details(model_info['name']),
        "model_info": {
            "general.architecture": model_config.get("model_type"),
            "general.parameter_count": model_config.get("num_parameters"),
            "qwen2.context_length": model_config.get("max_position_embeddings"),
            "qwen2.embedding_length": model_config.get("hidden_size"),
            "qwen2.block_count": model_config.get("num_hidden_layers"),
            "qwen2.attention.head_count": model_config.get("num_attention_heads"),
            "qwen2.attention.head_count_kv": model_config.get("num_key_value_heads"),
        },
    })

@app.route("/api/ps")
async def api_ps():
    """Report the loaded model as 'running'."""
    entries = []
    for model_info in utils.loaded_models.values():
        entry = utils.base_model_entry(model_info)
        entry.update({
            "expires_at": "2099-12-31T00:00:00Z", 
            "size_vram": model_info['size_vram'],
        })
        entries.append(entry)
        
    return quart.jsonify({"models": entries})

@app.route("/api/blobs/<digest>", methods=["HEAD"])
async def blob_check(digest):
    return quart.Response(status=200)

@app.route("/api/generate", methods=["POST"])
async def api_generate():
    data = await quart.request.get_json()
    model_name = data.get("model", config.QUICK_MODEL)
    prompt = data.get("prompt", "")
    options = utils.parse_options(data.get("options", {}))
    stream = data.get("stream", True)
    context = data.get("context", [])
    
    conv_key = hash(tuple(context)) if context else -1
    prompt = {"role": "user", "content": prompt}
    msg_history = utils.conversation_store.pop(conv_key, [])
    if not context:
        if system := data.get("system"):
            msg_history.append({"role": "system", "content": system})
    
    msg_history.append(prompt)    
 
    model_info = utils.loaded_models[model_name]

    if stream:
        response = quart.Response(
            worker.generate_stream(
                stream,
                model_info,
                msg_history,
                options,
                is_gen=True
                ),
            mimetype="application/x-ndjson"
        )
        response.timeout = None
        return response
    
    else:
        # just collect the last yielded completion object
        async for chunk_json in worker.generate_stream(
            stream,
            model_info,
            msg_history,
            options,
            is_gen=True
        ):
            final_chunk = json.loads(chunk_json)
        
        return quart.jsonify(final_chunk)
    
@app.route("/api/chat", methods=["POST"])
async def api_chat():
    try:
        data, uploaded_images = await _read_chat_request()
        messages = data.get("messages", [])
        images = uploaded_images + _images_from_messages(messages)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        return quart.jsonify({"error": str(exc)}), 400

    model_name = data.get("model", config.QUICK_MODEL)
    options = utils.parse_options(data.get("options", {}))
    stream = data.get("stream", True)
    tools = data.get("tools") or None
    think = data.get("think", False)
    model_info = utils.loaded_models[model_name]
    
    last_message = _message_text(messages[-1].get('content', '')).strip() if messages and messages[-1].get('role') == 'user' else ''

    if last_message == 'refresh cache':
        asyncio.create_task(cache_utils.refresh_cache_background())
        final_payload =  {
            "model": model_info['name'],
            "created_at": utils.now_iso(), 
            "message": {
                "role": "assistant",
                "content": "Aye captian, cache refresh initiated."
            },
            "done": True,
            "done_reason": "stop",
            "finish_reason": "stop",
            "total_duration": 500000000,
            "load_duration": 100000000,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 200000000,
            "eval_count": 25,
            "eval_duration": 200000000,
            "usage": {  # Add this dict
                "prompt_tokens": 10,
                "completion_tokens": 25,
                "total_tokens": 35
            }                    
        }
        
        if not stream:
            return quart.Response(
                json.dumps(final_payload),
                content_type='application/json'
            )
        
        # Streaming mode: emit newline-delimited JSON
        async def generate():
            # Optional: emit intermediate chunk(s) with done=False
            chunk = {
                "model": model_name,
                "created_at": utils.now_iso(),
                "message": {"role": "assistant", "content": "Aye captain, cache refresh initiated."},
                "done": False,
            }
            yield json.dumps(chunk) + "\n"
        
            final_payload['message']['content'] = ''
            # Final chunk with done=True and stats
            yield json.dumps(final_payload) + "\n"
            
        return quart.Response(generate(), content_type='application/x-ndjson')
        
    if stream:
        response = quart.Response(
            worker.generate_stream(
                stream,
                model_info,
                messages,
                options,
                tools=tools,
                think=think,
                images=images,
            ), 
            mimetype="application/x-ndjson",
        )
        response.timeout = None
        return response
    else:
        # just collect the last yielded completion object
        async for chunk_json in worker.generate_stream(
            stream,
            model_info,
            messages,
            options,
            tools=tools,
            think=think,
            images=images,
        ):
            final_chunk = json.loads(chunk_json)
        
        return quart.jsonify(final_chunk)
        
        
@app.route("/api/push", methods=["POST"])
async def api_push():
    return quart.jsonify({"error": "push not supported"}), 501


@app.route("/api/create", methods=["POST"])
async def api_create():
    return quart.jsonify({"error": "create not supported"}), 501


@app.route("/api/copy", methods=["POST"])
async def api_copy():
    return quart.jsonify({"error": "copy not supported"}), 501


@app.route("/api/delete", methods=["DELETE"])
async def api_delete():
    return quart.jsonify({"error": "delete not supported"}), 501

@app.route("/api/pull", methods=["POST"])
async def api_pull():
    """HA won't use this but other clients might. Return a fake success."""
    # data = await quart.request.get_json()
    async def pull_stream():
        yield json.dumps({"status": "pulling manifest"}) + "\n"
        yield json.dumps({"status": "success"}) + "\n"
    return quart.Response(pull_stream(), mimetype="application/x-ndjson")

@app.route("/api/embed", methods=["POST"])
@app.route("/api/embeddings", methods=["POST"])
async def api_embeddings():
    """
    Embeddings not supported — return 503 so clients fail loudly
    rather than silently consuming zero-vectors.
    If you ever need this, consider mlx-embeddings with a dedicated
    model like mlx-community/nomic-embed-text-v1.5
    """
    data = await quart.request.get_json()
    model = data.get("model", config.QUICK_MODEL)
    return quart.jsonify({
        "error": f"Embedding model not available. '{model}' is a generation model only."
    }), 503
        
_whisper_process: multiprocessing.Process | None = None

# Run wyoming STT as a service
@app.before_serving
async def startup():
    global _whisper_process
    
    await utils.MCP_CLIENT.get_client() # ensures it is connected.
    
    asyncio.create_task(
        cache_utils.cache_refresh_loop(),
        name="HA_Cache_Refresh"
    )
    
    asyncio.create_task(
        cache_utils.state_refresh_loop(),
        name="HA_State_refresh"
    )
    
    asyncio.create_task(
        cache_utils.itunes_cache_refresh(),
        name="iTunes cache refresh"
    )

    logging.info("Started HA cache background refresh task")

@app.after_serving
async def shutdown():
    """Triggered once when the server stops."""
    await utils.MCP_CLIENT.close()
