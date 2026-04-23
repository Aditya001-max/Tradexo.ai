"""
In-Memory Cache
================
Thread-safe TTL cache replacing Redis for local development.
Swap for redis-py in production.
"""

import threading
from typing import Any, Optional

from cachetools import TTLCache

from backend.config import get_settings

settings = get_settings()

# --- Global cache instance ---
_cache = TTLCache(
    maxsize=settings.CACHE_MAX_SIZE,
    ttl=settings.CACHE_TTL_SECONDS,
)
_lock = threading.Lock()


def cache_get(key: str) -> Optional[Any]:
    """Get a value from the cache. Returns None if not found or expired."""
    with _lock:
        return _cache.get(key)


def cache_set(key: str, value: Any) -> None:
    """Set a value in the cache with TTL."""
    with _lock:
        _cache[key] = value


def cache_delete(key: str) -> None:
    """Delete a key from the cache."""
    with _lock:
        _cache.pop(key, None)


def cache_clear() -> None:
    """Clear all cached values."""
    with _lock:
        _cache.clear()


def cache_has(key: str) -> bool:
    """Check if a key exists in the cache."""
    with _lock:
        return key in _cache
