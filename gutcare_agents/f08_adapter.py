"""F08 frozen-input bridge. Business task ownership stays with the backend Worker."""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .contract import TaskKind, TaskRequest, TaskStatus, TimeRange

LOG = logging.getLogger("gutcare.f08")
ANALYSIS_TYPES = {"comprehensive_health_analysis", "periodic_summary"}
RECORD_KINDS = {
    "confirmed_medical_fact": "medical",
    "diet_record": "diet",
    "stool_record": "stool",
    "symptom_record": "symptom",
    "mood_record": "mood",
    "medication_plan": "medication",
    "medication_checkin": "medication",
    "risk_alert": "risk_alert",
}
MOOD_CONSENTS = {"health_data_processing", "mood_data_processing", "mood_ai_sharing", "third_party_ai"}


class F08ContractError(ValueError):
    pass


class F08Journal:
    """Durable dispatch receipt and callback outbox; never stores raw frozen inputs."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS f08_jobs (
                task_id TEXT PRIMARY KEY, analysis_type TEXT NOT NULL,
                dispatch_json TEXT NOT NULL, state TEXT NOT NULL,
                callback_kind TEXT, callback_json TEXT, last_error TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""")

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def accept(self, dispatch: dict) -> str:
        task_id = validate_dispatch(dispatch)
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT analysis_type, state FROM f08_jobs WHERE task_id=?", (task_id,)).fetchone()
            if row:
                if row["analysis_type"] != dispatch["analysis_type"]:
                    raise F08ContractError("task id reused with different analysis type")
                return str(row["state"])
            conn.execute("INSERT INTO f08_jobs(task_id,analysis_type,dispatch_json,state) VALUES(?,?,?,?)",
                         (task_id, dispatch["analysis_type"], json.dumps(dispatch), "accepted"))
        return "accepted"

    def get(self, task_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM f08_jobs WHERE task_id=?", (task_id,)).fetchone()
        return dict(row) if row else None

    def pending_dispatches(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT dispatch_json FROM f08_jobs WHERE state IN ('accepted','running','callback_pending')").fetchall()
        return [json.loads(row["dispatch_json"]) for row in rows]

    def set_state(self, task_id: str, state: str, error: str = "") -> None:
        with self._lock, self._connect() as conn:
            conn.execute("UPDATE f08_jobs SET state=?,last_error=?,updated_at=CURRENT_TIMESTAMP WHERE task_id=?",
                         (state, error[:64], task_id))

    def save_callback(self, task_id: str, kind: str, payload: dict) -> None:
        if kind not in {"result", "fail"}:
            raise F08ContractError("unsupported callback kind")
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT callback_kind,callback_json FROM f08_jobs WHERE task_id=?",
                               (task_id,)).fetchone()
            if row is None:
                raise F08ContractError("dispatch must be saved before callback")
            if row["callback_json"] is not None:
                if row["callback_kind"] != kind or row["callback_json"] != encoded:
                    raise F08ContractError("saved callback differs from retry")
                return
            conn.execute("""UPDATE f08_jobs SET state='callback_pending',callback_kind=?,callback_json=?,
                           updated_at=CURRENT_TIMESTAMP WHERE task_id=?""", (kind, encoded, task_id))


def _uuid(value: object) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, AttributeError) as exc:
        raise F08ContractError("invalid UUID in frozen input") from exc


def validate_dispatch(payload: dict) -> str:
    if set(payload) - {"task_id", "analysis_type", "schema_version", "deadline_at"}:
        raise F08ContractError("unexpected dispatch field")
    if payload.get("schema_version") != "1.0" or payload.get("analysis_type") not in ANALYSIS_TYPES:
        raise F08ContractError("unsupported dispatch schema or analysis type")
    return _uuid(payload.get("task_id"))


def adapt_inputs(raw: dict, task_id: str, analysis_type: str) -> tuple[TaskRequest, dict]:
    """Return existing engine request and the exact frozen provenance index."""
    if raw.get("schema_version") != "1.0" or raw.get("analysis_type") != analysis_type:
        raise F08ContractError("input schema or analysis type mismatch")
    if _uuid(raw.get("task_id")) != task_id:
        raise F08ContractError("task identity mismatch")
    patient_id = _uuid(raw.get("patient_id"))
    snapshot_hash = raw.get("snapshot_hash")
    if not isinstance(snapshot_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", snapshot_hash):
        raise F08ContractError("invalid snapshot hash")
    flags = raw.get("consent_scope_snapshot")
    scope = raw.get("consent_scope")
    if not isinstance(flags, dict) or not isinstance(scope, list):
        raise F08ContractError("missing consent snapshot")
    if not flags.get("health_data_processing") or not flags.get("third_party_ai"):
        raise F08ContractError("analysis consent absent")
    if set(scope) != {key for key, enabled in flags.items() if enabled}:
        raise F08ContractError("consent scope differs from snapshot")
    ordinary = raw.get("records")
    moods = raw.get("mood_records")
    if not isinstance(ordinary, list) or not isinstance(moods, list):
        raise F08ContractError("records must be lists")
    if moods and not all(flags.get(name) is True for name in MOOD_CONSENTS):
        raise F08ContractError("mood records lack consent")
    records: list[dict] = []
    provenance: dict[str, dict] = {}
    seen_ids: set[str] = set()
    for source in ordinary + moods:
        if not isinstance(source, dict):
            raise F08ContractError("record must be object")
        source_type = source.get("record_type")
        if (source in moods) != (source_type == "mood_record"):
            raise F08ContractError("mood segregation invalid")
        source_id = _uuid(source.get("record_id"))
        version = source.get("record_version")
        locator = source.get("locator")
        status = source.get("confirmation_status")
        data = source.get("payload")
        if type(version) is not int or version < 1 or not isinstance(locator, str) or not 1 <= len(locator) <= 160:
            raise F08ContractError("invalid record provenance")
        if status not in {"confirmed", "unresolved"} or not isinstance(data, dict):
            raise F08ContractError("invalid record status or payload")
        if source_id in seen_ids:
            raise F08ContractError("duplicate record identity")
        seen_ids.add(source_id)
        if source_type == "medical_fact_conflict":
            if status != "unresolved":
                raise F08ContractError("medical conflict cannot be confirmed fact")
            continue
        if source_type not in RECORD_KINDS:
            raise F08ContractError("unsupported record type")
        if source_type == "mood_record":
            if status != "confirmed" or set(data) != {"occurred_at", "mood_level", "emotion_tags"}:
                raise F08ContractError("invalid minimal mood payload")
            if type(data["mood_level"]) is not int or not 1 <= data["mood_level"] <= 5:
                raise F08ContractError("invalid mood level")
            tags = data["emotion_tags"]
            allowed_tags = {"calm", "content", "joyful", "sad", "anxious", "irritable",
                            "overwhelmed", "tired", "other"}
            if (not isinstance(tags, list) or len(tags) > 5 or
                    any(not isinstance(tag, str) for tag in tags) or len(set(tags)) != len(tags)
                    or any(tag not in allowed_tags for tag in tags)):
                raise F08ContractError("invalid mood tags")
            data = {"mood_level": data["mood_level"], "mood_tags": data["emotion_tags"]}
        if status == "confirmed":
            provenance[source_id] = {"source_type": source_type, "source_id": source_id,
                                     "source_version": version, "locator": locator}
        records.append({"kind": RECORD_KINDS[source_type], "record_id": source_id,
                        "patient_id": patient_id, "status": status, "version": version,
                        "occurred_at": source.get("occurred_at") or source.get("payload", {}).get("occurred_at"),
                        "data": data})
    time_range = raw.get("time_range") or {}
    if not isinstance(time_range, dict):
        raise F08ContractError("invalid time range")
    request = TaskRequest(task_id=task_id, kind=TaskKind.ANALYSIS, patient_id=patient_id,
                          time_range=TimeRange(time_range.get("from"), time_range.get("to")),
                          records=records, options={"f08_analysis_type": analysis_type,
                                                    "f08_snapshot_hash": snapshot_hash,
                                                    "force_refresh": True})
    return request, provenance


def map_result(run, raw: dict, provenance: dict[str, dict]) -> dict:
    """Map only validated observations and frozen record identities to callback schema."""
    if run.status in {TaskStatus.FAILED, TaskStatus.CANCELLED}:
        raise F08ContractError("analysis failed; use failure callback")
    report = run.report or {}
    observations = []
    evidence = []
    seen = set()
    for obs in report.get("observations") or []:
        statement = obs.get("statement")
        if not isinstance(statement, str) or not statement.strip() or obs.get("needs_confirmation"):
            continue
        ids = obs.get("source_record_ids") or []
        if any(str(source_id) not in provenance for source_id in ids):
            continue  # never transmit a claim whose source is outside the frozen snapshot
        observations.append(statement[:500])
        for source_id in ids:
            if source_id not in seen:
                evidence.append(provenance[source_id])
                seen.add(source_id)
        if len(observations) >= 30:
            break
    missing = [str(item)[:120] for item in (raw.get("missing_data") or [])]
    if any(row.get("record_type") == "medical_fact_conflict" for row in raw.get("records") or []):
        missing.append("存在未解决的医学事实冲突，相关结论需人工确认")
    missing += [str(item)[:120] for item in (report.get("insufficient") or [])]
    missing = list(dict.fromkeys(missing))[:30]
    uncertainties = [str(item)[:500] for item in (report.get("degraded") or [])][:30]
    if run.status == TaskStatus.INSUFFICIENT or not observations:
        status = "needs_input"
        if not missing:
            missing = ["缺少可用于分析的已确认记录"]
    elif run.status == TaskStatus.PARTIAL or missing or uncertainties:
        status = "partial"
    else:
        status = "succeeded"
    summary = str(report.get("narrative_summary") or "").strip()
    if not summary:
        summary = observations[0] if observations else "当前资料不足，暂无法形成可靠分析。"
    callback_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"gutcare:f08:{run.task_id}:{raw['snapshot_hash']}:result"))
    return {"callback_id": callback_id, "status": status, "schema_version": "1.0",
            "summary": summary[:1200], "observations": observations,
            "evidence": evidence[:100], "uncertainties": uncertainties,
            "missing_data": missing}


class BackendClient:
    def __init__(self, base_url: str, service_token: str, timeout: float = 20.0):
        if not base_url or not service_token:
            raise F08ContractError("backend URL and service token required")
        self.base_url, self.service_token, self.timeout = base_url.rstrip("/"), service_token, timeout

    def request(self, method: str, path: str, payload: dict | None = None) -> dict:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        req = Request(self.base_url + path, data=data, method=method,
                      headers={"Authorization": f"Bearer {self.service_token}",
                               "Content-Type": "application/json"})
        with urlopen(req, timeout=self.timeout) as response:
            body = response.read()
            return json.loads(body) if body else {}

    def claim(self) -> dict | None:
        return self.request("POST", "/internal/v1/agent/analysis-tasks/claim").get("task")

    def inputs(self, task_id: str) -> dict:
        return self.request("GET", f"/internal/v1/agent/analysis-tasks/{task_id}/inputs")

    def result(self, task_id: str, payload: dict) -> dict:
        return self.request("POST", f"/internal/v1/agent/analysis-tasks/{task_id}/result", payload)

    def fail(self, task_id: str, code: str) -> dict:
        callback_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"gutcare:f08:{task_id}:fail"))
        return self.request("POST", f"/internal/v1/agent/analysis-tasks/{task_id}/fail",
                            {"callback_id": callback_id, "error_code": code})


def _send_saved_callback(client: BackendClient, journal: F08Journal, task_id: str,
                         *, retry_delay: float) -> dict | None:
    row = journal.get(task_id)
    if not row or row["state"] != "callback_pending" or not row["callback_json"]:
        raise F08ContractError("no pending callback to send")
    payload = json.loads(row["callback_json"])
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            if row["callback_kind"] == "result":
                client.result(task_id, payload)
            else:
                client.fail(task_id, payload["error_code"])
        except HTTPError as exc:
            if exc.code in {404, 409}:
                code = ""
                if exc.code == 409:
                    try:
                        code = str(json.loads(exc.read()).get("code") or "")
                    except (ValueError, AttributeError):
                        pass
                if code == "CALLBACK_ID_CONFLICT":
                    journal.set_state(task_id, "stopped", "CALLBACK_ID_CONFLICT")
                    raise F08ContractError("backend rejected reused callback id") from exc
                # Backend owns cancellation, consent revocation and terminal task state.
                journal.set_state(task_id, "stopped", "BACKEND_TASK_CLOSED")
                LOG.info("F08 backend closed task_id=%s http_status=%s", task_id, exc.code)
                return None
            if exc.code < 500:
                journal.set_state(task_id, "stopped", "CALLBACK_REJECTED")
                raise
            last_error = exc
        except (URLError, TimeoutError, OSError) as exc:
            # The backend may have committed before the response was lost. Retry the
            # *same* callback id and bytes; never replace this with a fail callback.
            last_error = exc
        else:
            journal.set_state(task_id, "completed")
            LOG.info("F08 callback accepted task_id=%s kind=%s", task_id, row["callback_kind"])
            return payload
        if attempt < 2:
            time.sleep(retry_delay * (2 ** attempt))
    assert last_error is not None
    raise last_error


def process_task(client: BackendClient, orchestrator, dispatch: dict,
                 *, journal: F08Journal, retry_delay: float = 0.25,
                 on_result=None) -> dict | None:
    """Run once or replay a saved callback after retry/restart without reanalysis."""
    task_id = validate_dispatch(dispatch)
    state = journal.accept(dispatch)
    if state in {"completed", "stopped"}:
        return None
    if state == "callback_pending":
        return _send_saved_callback(client, journal, task_id, retry_delay=retry_delay)

    frozen = None
    for attempt in range(3):
        try:
            frozen = client.inputs(task_id)
            break
        except HTTPError as exc:
            if exc.code in {404, 409}:
                journal.set_state(task_id, "stopped", "INPUT_UNAVAILABLE")
                LOG.info("F08 input unavailable task_id=%s http_status=%s", task_id, exc.code)
                return None
            if exc.code < 500 or attempt == 2:
                raise
        except (URLError, TimeoutError, OSError):
            if attempt == 2:
                raise
        time.sleep(retry_delay * (2 ** attempt))
    assert frozen is not None
    journal.set_state(task_id, "running")
    try:
        request, provenance = adapt_inputs(frozen, task_id, dispatch["analysis_type"])
        run = orchestrator.run(request)
        result = map_result(run, frozen, provenance)
    except Exception as exc:
        LOG.error("F08 analysis failed task_id=%s error_type=%s", task_id, type(exc).__name__)
        journal.save_callback(task_id, "fail", {"error_code": "AGENT_ANALYSIS_FAILED"})
        return _send_saved_callback(client, journal, task_id, retry_delay=retry_delay)
    if on_result is not None and result["status"] in {"succeeded", "partial"}:
        try:
            on_result(run, frozen, result["status"])
        except Exception as exc:
            # Advice is a separate callback; failure to prepare it must not
            # turn an already-computed F08 result into an analysis failure.
            LOG.error("F09 preparation failed task_id=%s error_type=%s", task_id, type(exc).__name__)
    journal.save_callback(task_id, "result", result)
    return _send_saved_callback(client, journal, task_id, retry_delay=retry_delay)
