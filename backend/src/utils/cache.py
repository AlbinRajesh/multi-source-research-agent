"""
Simple file-based research cache.

Keys are an MD5 hash of the normalized topic (+ sources_available),
so re-running the same or near-identical topic during dev/testing
skips the entire graph — zero Groq calls, zero Tavily calls, zero
429s. TTL defaults to 7 days per the original project plan.
"""
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days
CACHE_DIR = Path(".cache/research")


def _cache_key(topic: str, sources_available: Optional[list] = None) -> str:
    """Normalize topic (lowercase, collapse whitespace) before hashing
    so trivial variations ('Elon Musk' vs 'elon musk ') still hit."""
    normalized_topic = " ".join(topic.strip().lower().split())
    normalized_sources = sorted(sources_available or ["web"])
    raw = normalized_topic + "|" + ",".join(normalized_sources)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.json"


def get_cached_result(
    topic: str,
    sources_available: Optional[list] = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> Optional[dict]:
    """Return the cached final_state dict if a fresh entry exists, else None."""
    key = _cache_key(topic, sources_available)
    path = _cache_path(key)

    if not path.exists():
        logger.info(f"[cache] MISS (no entry): {topic!r}")
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            entry = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"[cache] corrupt cache file for {key}, ignoring: {e}")
        return None

    age = time.time() - entry.get("cached_at", 0)
    if age > ttl_seconds:
        logger.info(f"[cache] MISS (expired, age={age/3600:.1f}h): {topic!r}")
        return None

    logger.info(f"[cache] HIT (age={age/3600:.1f}h): {topic!r} — skipping Groq/Tavily entirely")
    return entry.get("result")


def set_cached_result(
    topic: str,
    result: dict,
    sources_available: Optional[list] = None,
) -> None:
    """Store final_state dict, stripped of non-JSON-serializable fields."""
    key = _cache_key(topic, sources_available)
    path = _cache_path(key)

    serializable = _make_json_safe(result)

    entry = {
        "topic": topic,
        "cached_at": time.time(),
        "result": serializable,
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2, default=str)
        logger.info(f"[cache] stored: {topic!r} -> {path.name}")
    except OSError as e:
        logger.warning(f"[cache] failed to write cache for {topic!r}: {e}")


def _make_json_safe(obj: Any) -> Any:
    """Recursively convert pydantic models / non-serializable objects
    to plain dict/list/str so json.dump doesn't choke."""
    if hasattr(obj, "model_dump"):
        return _make_json_safe(obj.model_dump())
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def clear_cache() -> int:
    """Delete all cached entries. Returns count removed. Useful during dev
    when prompts/logic change and stale cache hits would hide new behavior."""
    if not CACHE_DIR.exists():
        return 0
    count = 0
    for f in CACHE_DIR.glob("*.json"):
        f.unlink()
        count += 1
    logger.info(f"[cache] cleared {count} entries")
    return count