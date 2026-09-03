import asyncio
import functools

from pathlib import Path

from diskcache import Cache

CACHE_DIR = Path(__file__).parent / "Cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def async_ttl_cache(
    path: str | Path | None = None,
    ttl: float | None = None,
    key_fn=None,
):
    def decorator(fn):
        cache_path = Path(path) if path is not None else Path(fn.__name__)
        if not cache_path.is_absolute():
            cache_path = CACHE_DIR / cache_path
        cache = Cache(cache_path)

        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            key = key_fn(*args, **kwargs) if key_fn else f"{fn.__name__}:{args}:{kwargs}"
            sentinel = object()
            val = await asyncio.to_thread(cache.get, key, sentinel)
            if val is not sentinel:
                return val
            result = await fn(*args, **kwargs)
            await asyncio.to_thread(cache.set, key, result, expire=ttl)
            return result
        return wrapper
    return decorator