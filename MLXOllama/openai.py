import asyncio
import base64
import json
import time
import uuid
from typing import Any, Mapping, cast

import quart

from . import app, config, utils, worker
from .common import images_from_messages, read_chat_request

assert app is not None # because the app wouldn't be running if it was
JsonObject = dict[str, Any]


def _error(
    message: str,
    status: int = 400,
    error_type: str = "invalid_request_error",
) -> tuple[Any, int]:
    return quart.jsonify({"error": {"message": message, "type": error_type}}), status


def _options(data: Mapping[str, Any]) -> JsonObject:
    options: JsonObject = {}
    if "temperature" in data:
        options["temp"] = data["temperature"]
    for key in ("top_p", "top_k", "seed"):
        if key in data:
            options[key] = data[key]
    max_tokens = data.get("max_completion_tokens", data.get("max_tokens"))
    if max_tokens is not None:
        options["max_tokens"] = max_tokens
    return options


def _format_tool_calls(
    tool_calls: list[dict[str, Any]],
    for_stream: bool = False,
) -> list[dict[str, Any]]:
    formatted = []
    for idx, tc in enumerate(tool_calls):
        fn = tc.get("function", {})
        args = fn.get("arguments", {})
        args_str = json.dumps(args) if isinstance(args, dict) else str(args or "{}")
        item: dict[str, Any] = {
            "id": tc.get("id", f"call_{uuid.uuid4().hex[:10]}"),
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


def _chunk(
    chunk: Mapping[str, Any],
    model_name: str,
    request_id: str,
    created: int,
    include_role: bool = False,
) -> JsonObject:
    message = chunk.get("message") or {}
    message = cast(Mapping[str, Any], message)
    delta: JsonObject = {}
    if include_role:
        delta["role"] = "assistant"
    if message.get("content"):
        delta["content"] = message["content"]
    if message.get("tool_calls"):
        delta["tool_calls"] = _format_tool_calls(message["tool_calls"], for_stream=True)

    choice: JsonObject = {"index": 0, "delta": delta}
    if chunk.get("done"):
        done_reason = chunk.get("done_reason")
        choice["finish_reason"] = "tool_calls" if message.get("tool_calls") or done_reason == "tool_call" else "stop"
    else:
        choice["finish_reason"] = None
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [choice],
    }


@app.route("/v1/chat/completions", methods=["POST"])
async def chat_completions() -> Any:
    """OpenAI-compatible chat completions endpoint."""
    try:
        data, uploaded_images = await read_chat_request()
        raw_messages = data.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            return _error("'messages' must be a non-empty array")
        messages = raw_messages
        images = uploaded_images + images_from_messages(messages)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        return _error(str(exc))

    model_name = data.get("model", config.QUICK_MODEL)
    raw_model_info = utils.loaded_models.get(model_name)
    if raw_model_info is None:
        return _error(f"model '{model_name}' not found", 404, "model_not_found")
    model_info = cast(JsonObject, raw_model_info)

    stream = data.get("stream", False)
    request_id = f"chatcmpl-{base64.urlsafe_b64encode(str(time.time_ns()).encode()).decode().rstrip('=')}"
    created = int(time.time())
    options = _options(data)
    tools = data.get("tools") or None
    think = data.get("think", False)

    if not stream:
        final_chunk: JsonObject | None = None
        async for chunk_json in worker.generate_stream(
            False, model_info, messages, options, tools=tools, think=think, images=images
        ):
            final_chunk = cast(JsonObject, json.loads(chunk_json))
        if final_chunk is None:
            return _error("model returned no response", 500, "server_error")

        message = cast(
            JsonObject,
            final_chunk.get("message", {"role": "assistant", "content": ""}),
        )

        if message.get("tool_calls"):
            message["tool_calls"] = _format_tool_calls(message["tool_calls"], for_stream=False)
            if not message.get("content"):
                message["content"] = None

        finish_reason = "tool_calls" if message.get("tool_calls") or final_chunk.get("done_reason") == "tool_call" else "stop"
        completion: JsonObject = {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": model_info["name"],
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
        }
        prompt_tokens = final_chunk.get("prompt_eval_count")
        completion_tokens = final_chunk.get("eval_count")
        if prompt_tokens is not None or completion_tokens is not None:
            prompt_tokens = prompt_tokens or 0
            completion_tokens = completion_tokens or 0
            completion["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        return quart.jsonify(completion)

    async def completion_stream():
        first = True
        async for chunk_json in worker.generate_stream(
            True, model_info, messages, options, tools=tools, think=think, images=images
        ):
            chunk = cast(JsonObject, json.loads(chunk_json))
            yield f"data: {json.dumps(_chunk(chunk, model_info['name'], request_id, created, first))}\n\n"
            first = False
        yield "data: [DONE]\n\n"

    response = quart.Response(completion_stream(), mimetype="text/event-stream")
    response.timeout = None
    response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/v1/models")
async def list_models() -> Any:
    models = []
    for model_info in utils.loaded_models.values():
        model = {
            "id": model_info['name'],
            'object': "model",
            "created": 1690000000,
            "owned_by": "mlx-vlm",
        }
        models.append(model)

    models.append({
        "id": utils.EMBED_MODEL_NAME,
        "object": "model",
        "created": 1690000000,
        "owned_by": "sentence-transformers",
    })

    return {
        "object": "list",
        "data": models
    }


@app.route("/v1/embeddings", methods=["POST"])
async def embeddings() -> Any:
    data = await quart.request.get_json() or {}
    inputs = utils.extract_embed_inputs(data)
    if not inputs:
        return _error("'input' is required and must not be empty", 400, "invalid_request_error")

    model_name = data.get("model", utils.EMBED_MODEL_NAME)
    vectors = await asyncio.to_thread(utils.embed_model.encode, inputs)
    vectors_list = vectors.tolist()

    response_data = [
        {
            "object": "embedding",
            "index": i,
            "embedding": vector,
        }
        for i, vector in enumerate(vectors_list)
    ]

    return quart.jsonify({
        "object": "list",
        "data": response_data,
        "model": model_name,
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0,
        },
    }), 200
