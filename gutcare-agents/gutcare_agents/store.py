"""任务状态与结果存储（SQLite，标准库）。

为什么不用 Postgres/Redis：智能体侧只需要保存开发期任务状态、执行记录、
缓存与用量；正式业务数据权威仍在业务后端。单文件 SQLite 便于演示、
评审复现与打包，同时满足“任务状态可追踪、结果可追溯”的要求。
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .contract import AgentResult, Finding, TaskRun, now_iso

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id       TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,
    patient_id    TEXT NOT NULL,
    status        TEXT NOT NULL,
    snapshot_hash TEXT,
    request_json  TEXT,
    run_json      TEXT,
    error_code    TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_results (
    task_id    TEXT NOT NULL,
    agent      TEXT NOT NULL,
    status     TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (task_id, agent)
);
CREATE TABLE IF NOT EXISTS cache (
    cache_key      TEXT PRIMARY KEY,
    agent          TEXT NOT NULL,
    model_id       TEXT,
    prompt_version TEXT,
    payload_json   TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    last_hit_at    TEXT,
    hits           INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS usage_log (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id           TEXT,
    agent             TEXT,
    model_id          TEXT,
    prompt_tokens     INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    cached            INTEGER DEFAULT 0,
    latency_ms        INTEGER DEFAULT 0,
    created_at        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS findings (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT,
    agent      TEXT,
    code       TEXT,
    severity   TEXT,
    message    TEXT,
    target     TEXT,
    action     TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS safety_alerts (
    alert_id           TEXT PRIMARY KEY,
    patient_id         TEXT NOT NULL,
    dedupe_key         TEXT NOT NULL,
    severity_level     INTEGER NOT NULL,
    rule_id            TEXT NOT NULL,
    rule_version       TEXT NOT NULL,
    message            TEXT NOT NULL,
    recommended_action TEXT,
    trigger_record_ids TEXT,
    status             TEXT NOT NULL DEFAULT 'unread',
    created_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alerts_patient ON safety_alerts(patient_id, created_at);
CREATE INDEX IF NOT EXISTS idx_alerts_dedupe ON safety_alerts(dedupe_key, created_at);
CREATE INDEX IF NOT EXISTS idx_tasks_patient ON tasks(patient_id, created_at);
"""


class Store:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            # WAL：分析任务写库时，控制台/接口仍可并发读（演示现场会同时轮询与写库）
            conn.execute("PRAGMA journal_mode=WAL")
            # 忙等超时：并发提交时避免直接抛 "database is locked"
            conn.execute("PRAGMA busy_timeout=5000")
        self._ensure_cancel_column()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # 任务
    # ------------------------------------------------------------------
    def save_task(self, run: TaskRun, request_payload: dict[str, Any] | None = None) -> None:
        payload = json.dumps(run.to_dict(), ensure_ascii=False)
        with self._lock, self._connect() as conn:
            existing = conn.execute("SELECT request_json FROM tasks WHERE task_id=?", (run.task_id,)).fetchone()
            request_json = (
                json.dumps(request_payload, ensure_ascii=False)
                if request_payload is not None
                else (existing["request_json"] if existing else None)
            )
            conn.execute(
                """
                INSERT INTO tasks (task_id,kind,patient_id,status,snapshot_hash,request_json,run_json,error_code,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(task_id) DO UPDATE SET
                  status=excluded.status, snapshot_hash=excluded.snapshot_hash,
                  request_json=COALESCE(excluded.request_json, tasks.request_json),
                  run_json=excluded.run_json, error_code=excluded.error_code, updated_at=excluded.updated_at
                """,
                (
                    run.task_id,
                    run.kind,
                    run.patient_id,
                    run.status,
                    run.input_snapshot_hash,
                    request_json,
                    payload,
                    (run.error or {}).get("code") if run.error else None,
                    run.created_at,
                    now_iso(),
                ),
            )

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT run_json FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        return json.loads(row["run_json"]) if row else None

    def get_task_request(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT request_json FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        return json.loads(row["request_json"]) if row and row["request_json"] else None

    def find_by_idempotency(self, kind: str, patient_id: str, snapshot_hash: str) -> dict[str, Any] | None:
        """同一患者、同一任务类型、同一输入快照 → 复用旧任务，避免重复报告。

        只认“已结束”的任务：异步提交时会先落一条 queued 占位记录，
        如果把它也当作可复用结果，任务会被自己挡掉而永远停在 queued。
        """
        with self._connect() as conn:
            row = conn.execute(
                """SELECT run_json FROM tasks
                   WHERE kind=? AND patient_id=? AND snapshot_hash=?
                     AND status NOT IN ('queued','running')
                   ORDER BY created_at DESC LIMIT 1""",
                (kind, patient_id, snapshot_hash),
            ).fetchone()
        return json.loads(row["run_json"]) if row else None

    def list_tasks(self, limit: int = 20, patient_id: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT task_id,kind,patient_id,status,created_at,updated_at FROM tasks"
        params: tuple[Any, ...] = ()
        if patient_id:
            sql += " WHERE patient_id=?"
            params = (patient_id,)
        sql += " ORDER BY created_at DESC LIMIT ?"
        with self._connect() as conn:
            rows = conn.execute(sql, params + (limit,)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # 取消（分析任务生命周期里的“已取消”，见详细功能设计 §9.3）
    # ------------------------------------------------------------------
    CANCELLED = "cancelled"
    ACTIVE_STATES = ("queued", "running")

    def _ensure_cancel_column(self) -> None:
        with self._connect() as conn:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(tasks)")}
            if "cancel_requested" not in cols:
                conn.execute("ALTER TABLE tasks ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0")

    def request_cancel(self, task_id: str) -> dict[str, Any] | None:
        """请求取消：只对排队中/运行中的任务生效，已结束的任务不覆盖结果。

        取消标记单独存一列，避免 orchestrator 每写一次进度就被覆盖
        （阶段落库会写 status/running，如果复用 status 字段，取消会“丢失”）。
        """
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT status, cancel_requested FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if row is None:
                return None
            status = str(row["status"])
            if status not in self.ACTIVE_STATES:
                return {"task_id": task_id, "cancelled": False, "status": status}
            conn.execute(
                "UPDATE tasks SET cancel_requested=1, updated_at=? WHERE task_id=?",
                (now_iso(), task_id),
            )
        return {"task_id": task_id, "cancelled": True, "status": self.CANCELLED}

    def is_cancelled(self, task_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT cancel_requested FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        return bool(row and int(row["cancel_requested"] or 0) == 1)

    # ------------------------------------------------------------------
    # Agent 结果
    # ------------------------------------------------------------------
    def save_agent_result(self, result: AgentResult) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO agent_results (task_id,agent,status,result_json,created_at)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(task_id,agent) DO UPDATE SET
                     status=excluded.status, result_json=excluded.result_json""",
                (
                    result.task_id,
                    result.agent,
                    result.status,
                    json.dumps(result.to_dict(), ensure_ascii=False),
                    now_iso(),
                ),
            )

    def get_agent_results(self, task_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            rows = conn.execute("SELECT agent,result_json FROM agent_results WHERE task_id=?", (task_id,)).fetchall()
        return {r["agent"]: json.loads(r["result_json"]) for r in rows}

    # ------------------------------------------------------------------
    # 缓存（领域结果复用：“改了饮食不必重跑医疗”）
    # ------------------------------------------------------------------
    def cache_get(self, cache_key: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT payload_json FROM cache WHERE cache_key=?", (cache_key,)).fetchone()
            if not row:
                return None
            conn.execute(
                "UPDATE cache SET hits=hits+1, last_hit_at=? WHERE cache_key=?",
                (now_iso(), cache_key),
            )
        return json.loads(row["payload_json"])

    def cache_put(
        self,
        cache_key: str,
        agent: str,
        payload: dict[str, Any],
        *,
        model_id: str = "",
        prompt_version: str = "",
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO cache (cache_key,agent,model_id,prompt_version,payload_json,created_at)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(cache_key) DO UPDATE SET
                     payload_json=excluded.payload_json, model_id=excluded.model_id,
                     prompt_version=excluded.prompt_version""",
                (
                    cache_key,
                    agent,
                    model_id,
                    prompt_version,
                    json.dumps(payload, ensure_ascii=False),
                    now_iso(),
                ),
            )

    def cache_stats(self) -> dict[str, int]:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n, COALESCE(SUM(hits),0) AS h FROM cache").fetchone()
        return {"entries": int(row["n"]), "hits": int(row["h"])}

    # ------------------------------------------------------------------
    # 用量与日志
    # ------------------------------------------------------------------
    def log_usage(
        self,
        *,
        task_id: str,
        agent: str,
        model_id: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cached: bool = False,
        latency_ms: int = 0,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO usage_log
                   (task_id,agent,model_id,prompt_tokens,completion_tokens,cached,latency_ms,created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    task_id,
                    agent,
                    model_id,
                    prompt_tokens,
                    completion_tokens,
                    int(cached),
                    latency_ms,
                    now_iso(),
                ),
            )

    def usage_summary(self) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT COUNT(*) AS calls,
                          COALESCE(SUM(prompt_tokens),0) AS pt,
                          COALESCE(SUM(completion_tokens),0) AS ct,
                          COALESCE(SUM(cached),0) AS cached_calls,
                          COALESCE(AVG(latency_ms),0) AS avg_ms
                   FROM usage_log"""
            ).fetchone()
        return {
            "calls": int(row["calls"]),
            "prompt_tokens": int(row["pt"]),
            "completion_tokens": int(row["ct"]),
            "cached_calls": int(row["cached_calls"]),
            "avg_latency_ms": round(float(row["avg_ms"]), 1),
        }

    # ------------------------------------------------------------------
    # 校验发现
    # ------------------------------------------------------------------
    def save_findings(self, task_id: str, agent: str, findings: list[Finding]) -> None:
        if not findings:
            return
        with self._lock, self._connect() as conn:
            conn.executemany(
                """INSERT INTO findings (task_id,agent,code,severity,message,target,action,created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                [
                    (task_id, agent, f.code, f.severity, f.message, f.target, f.action, now_iso())
                    for f in findings
                ],
            )

    def findings_summary(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute("SELECT severity, COUNT(*) AS n FROM findings GROUP BY severity").fetchall()
        return {r["severity"]: int(r["n"]) for r in rows}

    # ------------------------------------------------------------------
    # 安全提示（独立规则路径的落库，便于演示“已读不等于风险解除”）
    # ------------------------------------------------------------------
    def save_alert(self, alert: dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO safety_alerts
                   (alert_id,patient_id,dedupe_key,severity_level,rule_id,rule_version,message,
                    recommended_action,trigger_record_ids,status,created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    alert["alert_id"],
                    alert["patient_id"],
                    alert["dedupe_key"],
                    alert["severity_level"],
                    alert["rule_id"],
                    alert["rule_version"],
                    alert["message"],
                    alert.get("recommended_action", ""),
                    json.dumps(alert.get("trigger_record_ids", []), ensure_ascii=False),
                    alert.get("status", "unread"),
                    alert.get("created_at", now_iso()),
                ),
            )

    def find_recent_alert(self, dedupe_key: str, within_seconds: int = 3600) -> dict[str, Any] | None:
        """去重：同一事件短时间内不反复轰炸，但新的危险事件必须能产生新提示。"""
        from datetime import datetime, timedelta

        cutoff = (datetime.now().astimezone() - timedelta(seconds=within_seconds)).isoformat(timespec="seconds")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM safety_alerts WHERE dedupe_key=? AND created_at>=? ORDER BY created_at DESC LIMIT 1",
                (dedupe_key, cutoff),
            ).fetchone()
        return dict(row) if row else None

    def list_alerts(self, patient_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM safety_alerts WHERE patient_id=? ORDER BY created_at DESC LIMIT ?",
                (patient_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]
