"""Durable F09 advice callback, independent of the F08 result callback."""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.error import HTTPError, URLError

from .f09_adapter import map_f09_advice

LOG = logging.getLogger("gutcare.f09")


class F09Outbox:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS f09_jobs (
                task_id TEXT PRIMARY KEY, state TEXT NOT NULL,
                callback_json TEXT, reason TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.db_path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def prepare(self, run, frozen: dict, f08_status: str) -> None:
        task_id = str(run.task_id)
        mapping = map_f09_advice(run, frozen, f08_status=f08_status)
        state = "pending" if mapping.ready else "skipped"
        encoded = json.dumps(mapping.payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self._lock, self._connect() as db:
            row = db.execute("SELECT state,callback_json FROM f09_jobs WHERE task_id=?", (task_id,)).fetchone()
            if row:
                if row[1] != encoded:
                    raise ValueError("F09 callback changed for an existing task")
                return
            db.execute("INSERT INTO f09_jobs(task_id,state,callback_json,reason) VALUES(?,?,?,?)",
                       (task_id, state, encoded, "no_valid_advice" if not mapping.ready else None))

    def get(self, task_id: str) -> tuple[str, dict] | None:
        with self._connect() as db:
            row = db.execute("SELECT state,callback_json FROM f09_jobs WHERE task_id=?", (task_id,)).fetchone()
        return (row[0], json.loads(row[1])) if row and row[1] else None

    def pending(self) -> list[str]:
        with self._connect() as db:
            return [row[0] for row in db.execute("SELECT task_id FROM f09_jobs WHERE state='pending'")]

    def set_state(self, task_id: str, state: str, reason: str = "") -> None:
        with self._lock, self._connect() as db:
            db.execute("UPDATE f09_jobs SET state=?,reason=?,updated_at=CURRENT_TIMESTAMP WHERE task_id=?",
                       (state, reason[:64], task_id))


def deliver_f09(client, outbox: F09Outbox, task_id: str, *, retry_delay: float = 0.25) -> bool:
    saved = outbox.get(task_id)
    if not saved or saved[0] != "pending":
        return False
    payload = saved[1]
    for attempt in range(3):
        try:
            client.request("POST", f"/internal/v1/agent/advice/{task_id}/callback", payload)
        except HTTPError as exc:
            if exc.code in {400, 404, 409, 422}:
                outbox.set_state(task_id, "stopped", f"http_{exc.code}")
                LOG.warning("F09 callback rejected task_id=%s http_status=%s", task_id, exc.code)
                return False
            if exc.code < 500:
                outbox.set_state(task_id, "stopped", f"http_{exc.code}")
                return False
            if attempt == 2:
                raise
        except (URLError, TimeoutError, OSError):
            if attempt == 2:
                raise
        else:
            outbox.set_state(task_id, "sent")
            LOG.info("F09 callback accepted task_id=%s", task_id)
            return True
        time.sleep(retry_delay * (2 ** attempt))
    return False
