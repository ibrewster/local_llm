import asyncio
import functools
from diskcache import Cache

def async_ttl_cache(cache: Cache, ttl: float | None = None, key_fn=None):
    def decorator(fn):
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