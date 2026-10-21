"""F02 medical document bridge; the business backend owns claim and confirmation."""
from __future__ import annotations

import base64
import binascii
import json
import logging
import sqlite3
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from urllib.error import HTTPError, URLError

from .agents.medical import MedicalAgent
from .contract import Status, TaskKind, TaskRequest
from .llm import LLMClient

LOG = logging.getLogger("gutcare.f02")
MIME_SUFFIX = {"application/pdf": ".pdf", "image/jpeg": ".jpg", "image/png": ".png"}
MAX_DOCUMENT_BYTES = 20 * 1024 * 1024
SUPPORTED_TEST_UNITS = {"mg/L", "mg/dL", "g/L", "mmol/L", "umol/L", "10^9/L",
                        "10^12/L", "%", "U/L", "IU/L", "mL", "L", "pg/mL", "ng/mL"}


class F02ContractError(ValueError):
    pass


def _uuid(value: object) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, AttributeError) as exc:
        raise F02ContractError("invalid F02 request UUID") from exc


def validate_dispatch(payload: dict) -> str:
    if not isinstance(payload, dict) or set(payload) != {"request_id", "schema_version"}:
        raise F02ContractError("invalid F02 dispatch fields")
    if payload["schema_version"] != "1.0":
        raise F02ContractError("unsupported F02 dispatch version")
    return _uuid(payload["request_id"])


def validate_input(raw: dict, request_id: str) -> tuple[bytes, str]:
    if not isinstance(raw, dict) or set(raw) != {
        "request_id", "document_id", "document_version", "document_type",
        "document_date", "content_type", "file_name", "content_base64",
    }:
        raise F02ContractError("invalid F02 input fields")
    if _uuid(raw["request_id"]) != request_id:
        raise F02ContractError("F02 input request mismatch")
    _uuid(raw["document_id"])
    if type(raw["document_version"]) is not int or raw["document_version"] < 1:
        raise F02ContractError("invalid document version")
    if not isinstance(raw["document_type"], str) or not isinstance(raw["file_name"], str):
        raise F02ContractError("invalid document metadata")
    if raw["content_type"] not in MIME_SUFFIX:
        raise F02ContractError("unsupported document type")
    encoded = raw["content_base64"]
    if not isinstance(encoded, str) or len(encoded) > ((MAX_DOCUMENT_BYTES + 2) // 3) * 4 + 4:
        raise F02ContractError("document input too large")
    try:
        content = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise F02ContractError("invalid document encoding") from exc
    if not content or len(content) > MAX_DOCUMENT_BYTES:
        raise F02ContractError("invalid document size")
    mime = raw["content_type"]
    if (mime == "application/pdf" and not content.startswith(b"%PDF-")) or (
        mime == "image/png" and not content.startswith(b"\x89PNG\r\n\x1a\n")
    ) or (mime == "image/jpeg" and not (content.startswith(b"\xff\xd8") and content.endswith(b"\xff\xd9"))):
        raise F02ContractError("document content type mismatch")
    return content, MIME_SUFFIX[mime]


class F02Journal:
    """Persist only dispatch and exact callback; never persist document bytes."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS f02_jobs (
                request_id TEXT PRIMARY KEY, dispatch_json TEXT NOT NULL, state TEXT NOT NULL,
                callback_kind TEXT, callback_json TEXT, reason TEXT,
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

    def accept(self, dispatch: dict) -> str:
        request_id = validate_dispatch(dispatch)
        encoded = json.dumps(dispatch, sort_keys=True, separators=(",", ":"))
        with self._lock, self._connect() as db:
            row = db.execute("SELECT dispatch_json,state FROM f02_jobs WHERE request_id=?", (request_id,)).fetchone()
            if row:
                if row[0] != encoded:
                    raise F02ContractError("F02 request id reused with different dispatch")
                return row[1]
            db.execute("INSERT INTO f02_jobs(request_id,dispatch_json,state) VALUES(?,?,?)",
                       (request_id, encoded, "accepted"))
        return "accepted"

    def get(self, request_id: str) -> dict | None:
        with self._connect() as db:
            db.row_factory = sqlite3.Row
            row = db.execute("SELECT * FROM f02_jobs WHERE request_id=?", (request_id,)).fetchone()
        return dict(row) if row else None

    def pending(self) -> list[dict]:
        with self._connect() as db:
            rows = db.execute("SELECT dispatch_json FROM f02_jobs WHERE state IN ('accepted','running','callback_pending')").fetchall()
        return [json.loads(row[0]) for row in rows]

    def state(self, request_id: str, value: str, reason: str = "") -> None:
        with self._lock, self._connect() as db:
            db.execute("UPDATE f02_jobs SET state=?,reason=?,updated_at=CURRENT_TIMESTAMP WHERE request_id=?",
                       (value, reason[:64], request_id))

    def callback(self, request_id: str, kind: str, payload: dict) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self._lock, self._connect() as db:
            row = db.execute("SELECT callback_kind,callback_json FROM f02_jobs WHERE request_id=?",
                             (request_id,)).fetchone()
            if row is None or (row[1] is not None and (row[0] != kind or row[1] != encoded)):
                raise F02ContractError("F02 callback differs from saved retry")
            db.execute("UPDATE f02_jobs SET state='callback_pending',callback_kind=?,callback_json=?,updated_at=CURRENT_TIMESTAMP WHERE request_id=?",
                       (kind, encoded, request_id))


def _page(item: dict, page_count: int | None) -> int | None:
    value = item.get("source_page")
    if type(value) is int and value >= 1 and (page_count is None or value <= page_count):
        return value
    if page_count == 1:
        return 1
    return None


def map_medical_result(request_id: str, result) -> dict:
    """Emit only fields with a source excerpt and an unambiguous page."""
    data = result.data or {}
    pdf = data.get("pdf_processing") or {}
    page_count = pdf.get("processed_pages") if pdf else 1
    if type(page_count) is not int or page_count < 1:
        page_count = None
    candidates: list[dict] = []

    def add(code: str, kind: str, value, excerpt: object, source: dict, unit=None):
        page = _page(source, page_count)
        if page is None or not isinstance(excerpt, str) or not excerpt.strip():
            return
        text = excerpt.strip()[:500]
        if not text or len(candidates) >= 200:
            return
        uncertainty = "模型抽取，待患者逐项确认"
        if pdf.get("truncated"):
            uncertainty += "；PDF 仅处理前几页"
        candidates.append({"field_code": code, "value_type": kind, "value": value, "unit": unit,
                           "confidence": None, "uncertainty": uncertainty,
                           "source_locator": {"page": page}, "excerpt": text})

    for item in data.get("exam_items") or []:
        if not isinstance(item, dict):
            continue
        name, value, excerpt = item.get("name"), item.get("value"), item.get("raw_text")
        if not isinstance(name, str) or not name.strip() or not isinstance(value, str) or not value.strip():
            continue
        unit = item.get("unit") or None
        if unit is not None and (not isinstance(unit, str) or unit not in SUPPORTED_TEST_UNITS):
            continue
        add("test_item", "string", name.strip()[:2000], excerpt, item)
        if len(value.strip()) > 200:
            continue
        reference_range = item.get("reference_range") or None
        if reference_range is not None and not isinstance(reference_range, str):
            continue
        test_result = {"result": value.strip(), "unit": unit,
                       "reference_range": reference_range}
        add("test_result", "object", test_result, excerpt, item, unit)

    if page_count == 1:
        for suggestion in data.get("suggestions") or []:
            if isinstance(suggestion, str) and suggestion.strip():
                add("recommendation", "string", suggestion.strip()[:2000], suggestion, {})

    for item in data.get("diagnosis_candidates") or []:
        if not isinstance(item, dict):
            continue
        certainty = {"confirmed": "explicit", "suspected": "suspected",
                     "history": "history", "ruled_out": "rule_out"}.get(item.get("status"))
        name = item.get("name")
        if certainty and isinstance(name, str) and name.strip():
            add("diagnosis", "diagnosis", {"text": name.strip()[:2000], "certainty": certainty},
                item.get("evidence"), item)

    for item in data.get("medications") or []:
        if not isinstance(item, dict):
            continue
        excerpt = item.get("raw_text")
        for code, key in (("medication_name", "name"), ("medication_frequency", "frequency"),
                          ("medication_duration", "course")):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                add(code, "string", value.strip()[:2000], excerpt, item)

    callback_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"gutcare:f02:{request_id}:candidates"))
    if candidates:
        return {"callback_id": callback_id, "parse_outcome": "extracted",
                "reason_code": None, "candidates": candidates}
    if result.status == Status.INSUFFICIENT and not any(
        data.get(key) for key in ("exam_items", "diagnosis_candidates", "medications", "suggestions", "conclusion")
    ):
        return {"callback_id": callback_id, "parse_outcome": "insufficient_quality",
                "reason_code": "illegible_content", "candidates": []}
    return {"callback_id": callback_id, "parse_outcome": "no_supported_facts",
            "reason_code": "no_supported_fields", "candidates": []}


def _run_medical(settings, request_id: str, raw: dict, content: bytes, suffix: str):
    with tempfile.TemporaryDirectory(prefix="gutcare-f02-") as directory:
        path = Path(directory) / (request_id + suffix)
        path.write_bytes(content)
        request = TaskRequest(task_id=request_id, patient_id=request_id, kind=TaskKind.REPORT_EXTRACT,
                              files=[{"file_id": raw["document_id"], "path": str(path),
                                      "kind": "report", "mime": raw["content_type"]}],
                              options={"force_refresh": True})
        return MedicalAgent(settings, LLMClient(settings)).run(request)


def _send_saved(client, journal: F02Journal, request_id: str, retry_delay: float) -> dict | None:
    row = journal.get(request_id)
    if not row or row["state"] != "callback_pending":
        return None
    payload = json.loads(row["callback_json"])
    endpoint = "candidates" if row["callback_kind"] == "candidates" else "fail"
    path = f"/internal/v1/agent/medical-parse-requests/{request_id}/{endpoint}"
    for attempt in range(3):
        try:
            client.request("POST", path, payload)
        except HTTPError as exc:
            if exc.code in {400, 404, 409, 422}:
                journal.state(request_id, "stopped", f"callback_http_{exc.code}")
                return None
            if exc.code < 500 or attempt == 2:
                raise
        except (URLError, TimeoutError, OSError):
            if attempt == 2:
                raise
        else:
            journal.state(request_id, "completed")
            return payload
        time.sleep(retry_delay * (2 ** attempt))
    return None


def process_medical_task(client, settings, dispatch: dict, *, journal: F02Journal,
                         retry_delay: float = 0.25) -> dict | None:
    request_id = validate_dispatch(dispatch)
    state = journal.accept(dispatch)
    if state in {"completed", "stopped"}:
        return None
    if state == "callback_pending":
        return _send_saved(client, journal, request_id, retry_delay)
    raw = None
    for attempt in range(3):
        try:
            raw = client.request("GET", f"/internal/v1/agent/medical-parse-requests/{request_id}/inputs")
            break
        except HTTPError as exc:
            if exc.code in {404, 409}:
                journal.state(request_id, "stopped", f"input_http_{exc.code}")
                return None
            if exc.code < 500 or attempt == 2:
                raise
        except (URLError, TimeoutError, OSError):
            if attempt == 2:
                raise
        time.sleep(retry_delay * (2 ** attempt))
    assert raw is not None
    journal.state(request_id, "running")
    try:
        content, suffix = validate_input(raw, request_id)
        result = _run_medical(settings, request_id, raw, content, suffix)
        if result.error or result.status == Status.FAILED:
            raise RuntimeError("medical_agent_failed")
        payload = map_medical_result(request_id, result)
        journal.callback(request_id, "candidates", payload)
    except Exception as exc:
        LOG.error("F02 processing failed request_id=%s error_type=%s", request_id, type(exc).__name__)
        error_code = "invalid_document" if isinstance(exc, F02ContractError) else "agent_unavailable"
        journal.callback(request_id, "fail", {"callback_id": str(uuid.uuid5(uuid.NAMESPACE_URL,
                          f"gutcare:f02:{request_id}:fail")), "error_code": error_code})
    return _send_saved(client, journal, request_id, retry_delay)
