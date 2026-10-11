"""Accountability (G8) audit logging as JSON Lines."""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class AuditLogger:
    """Append-only audit log.

    With ``path=None`` records are kept in memory (``self.records``).
    Raw text is stored only when ``include_text=True``; otherwise only hashes.
    """

    def __init__(self, path: str | Path | None = None, include_text: bool = False):
        self.path = Path(path) if path else None
        self.include_text = include_text
        self.records: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, original: str, final: str, **fields: Any) -> dict[str, Any]:
        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "original_sha256": sha256(original),
            "final_sha256": sha256(final),
            **fields,
        }
        if self.include_text:
            record["original"] = original
            record["final"] = final
        with self._lock:
            self.records.append(record)
            if self.path:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record
