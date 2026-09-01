import base64
import json
from io import BytesIO
from typing import Any

import quart
from PIL import Image


async def read_chat_request() -> tuple[dict[str, Any], list[Any]]:
    """Read a chat request from JSON or multipart/form-data."""
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


def images_from_messages(messages: list[Any]) -> list[Any]:
    """Decode data-URI/base64 images supplied in JSON messages."""
    images = []
    for message in messages:
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


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            item.get("text", "") for item in content
            if isinstance(item, dict) and item.get("type") in ("text", "input_text")
        )
    return ""
