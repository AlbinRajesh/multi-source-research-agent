"""
Stage-level latency logging — parsing/chunking/embedding/retrieval/generation.
Appends one JSON line per stage call to logs/stage_timing.jsonl.
"""
import json
import time
import logging
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

STAGE_LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "stage_timing.jsonl"
STAGE_LOG_PATH.parent.mkdir(exist_ok=True)


@contextmanager
def time_stage(stage: str, extra: dict | None = None):
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        row = {"stage": stage, "elapsed_sec": round(elapsed, 4)}
        if extra:
            row.update(extra)
        with open(STAGE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        logger.info(f"[timing] {stage} took {elapsed:.3f}s")