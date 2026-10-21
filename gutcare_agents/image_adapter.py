"""F03/F04 image candidate bridge. The backend owns task claim and patient confirmation."""
from __future__ import annotations

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
from urllib.parse import quote
from urllib.request import Request, urlopen

from .agents.diet import DietAgent
from .agents.stool import StoolAgent
from .contract import TaskKind, TaskRequest
from .llm import LLMClient

LOG = logging.getLogger("gutcare.images")
TYPES = {"diet_image_recognition": TaskKind.MEAL_RECOGNIZE,
         "stool_image_recognition": TaskKind.STOOL_IMAGE_RECOGNIZE}
CATEGORIES = {"grain_staple", "vegetable", "fruit", "meat", "seafood", "dairy",
              "egg", "legume", "snack", "beverage", "mixed_dish", "other"}


class ImageContractError(ValueError):
    pass


class ImageProcessingError(RuntimeError):
    def __init__(self, error_code: str):
        super().__init__(error_code)
        self.error_code = error_code


def _uuid(value: object) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, AttributeError) as exc:
        raise ImageContractError("invalid image task UUID") from exc


def validate_dispatch(payload: dict) -> str:
    if not isinstance(payload, dict) or set(payload) != {"request_id", "recognition_type", "attempt",
                                                  "retry_of_request_id", "schema_version"}:
        raise ImageContractError("invalid image dispatch fields")
    if payload["schema_version"] != "1.0" or payload["recognition_type"] not in TYPES:
        raise ImageContractError("unsupported image task")
    if type(payload["attempt"]) is not int or payload["attempt"] < 1:
        raise ImageContractError("invalid attempt")
    if payload["retry_of_request_id"] is not None:
        _uuid(payload["retry_of_request_id"])
    return _uuid(payload["request_id"])


def validate_input(raw: dict, dispatch: dict) -> list[dict]:
    if not isinstance(raw, dict) or raw.get("schema_version") != "1.0":
        raise ImageContractError("invalid image input schema")
    if any(raw.get(key) != dispatch[key] for key in ("request_id", "recognition_type", "attempt",
                                                       "retry_of_request_id")):
        raise ImageContractError("image input differs from dispatch")
    for key in ("record_version", "image_set_version"):
        if type(raw.get(key)) is not int or raw[key] < 1:
            raise ImageContractError("invalid image version")
    _uuid(raw.get("record_id"))
    attachments = raw.get("attachments")
    if not isinstance(attachments, list) or not 1 <= len(attachments) <= 10:
        raise ImageContractError("invalid attachment count")
    seen = set()
    for item in attachments:
        if not isinstance(item, dict):
            raise ImageContractError("invalid attachment")
        aid = _uuid(item.get("attachment_id"))
        if aid in seen or type(item.get("file_version")) is not int or item["file_version"] < 1:
            raise ImageContractError("invalid attachment identity or version")
        if item.get("content_type") not in {"image/jpeg", "image/png"}:
            raise ImageContractError("unsupported image content type")
        capability = item.get("image_capability")
        if not isinstance(capability, str) or not 32 <= len(capability) <= 256:
            raise ImageContractError("invalid image capability")
        seen.add(aid)
    return attachments


class ImageJournal:
    """Store dispatch and exact callback only; never store capabilities or image bytes."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS image_jobs (
                request_id TEXT PRIMARY KEY, recognition_type TEXT NOT NULL,
                dispatch_json TEXT NOT NULL, state TEXT NOT NULL,
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
        with self._lock, self._connect() as db:
            row = db.execute("SELECT recognition_type,dispatch_json,state FROM image_jobs WHERE request_id=?",
                             (request_id,)).fetchone()
            encoded = json.dumps(dispatch, sort_keys=True, separators=(",", ":"))
            if row:
                if row[0] != dispatch["recognition_type"] or row[1] != encoded:
                    raise ImageContractError("image request id reused with different dispatch")
                return row[2]
            db.execute("INSERT INTO image_jobs(request_id,recognition_type,dispatch_json,state) VALUES(?,?,?,?)",
                       (request_id, dispatch["recognition_type"], encoded, "accepted"))
        return "accepted"

    def get(self, request_id: str) -> dict | None:
        with self._connect() as db:
            db.row_factory = sqlite3.Row
            row = db.execute("SELECT * FROM image_jobs WHERE request_id=?", (request_id,)).fetchone()
        return dict(row) if row else None

    def pending(self) -> list[dict]:
        with self._connect() as db:
            rows = db.execute("SELECT dispatch_json FROM image_jobs WHERE state IN ('accepted','running','callback_pending')").fetchall()
        return [json.loads(row[0]) for row in rows]

    def state(self, request_id: str, state: str, reason: str = "") -> None:
        with self._lock, self._connect() as db:
            db.execute("UPDATE image_jobs SET state=?,reason=?,updated_at=CURRENT_TIMESTAMP WHERE request_id=?",
                       (state, reason[:64], request_id))

    def callback(self, request_id: str, kind: str, payload: dict) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self._lock, self._connect() as db:
            row = db.execute("SELECT callback_json FROM image_jobs WHERE request_id=?", (request_id,)).fetchone()
            if row is None or (row[0] is not None and row[0] != encoded):
                raise ImageContractError("image callback differs from saved retry")
            db.execute("UPDATE image_jobs SET state='callback_pending',callback_kind=?,callback_json=?,updated_at=CURRENT_TIMESTAMP WHERE request_id=?",
                       (kind, encoded, request_id))


def _candidate_id(request_id: str, attachment_id: str, index: int, kind: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"gutcare:image:{request_id}:{attachment_id}:{index}:{kind}"))


def map_image_result(request_id: str, recognition_type: str, attachment_id: str, result) -> dict:
    data = result.data or {}
    candidates = []
    if recognition_type == "diet_image_recognition":
        foods = data.get("foods") or []
        for index, food in enumerate(foods[:24]):
            if not isinstance(food, dict) or not isinstance(food.get("name"), str):
                continue
            name = food["name"].strip()[:160]
            if not name:
                continue
            confidence_value = food.get("confidence")
            confidence = confidence_value if isinstance(confidence_value, str) and confidence_value in {"low", "medium", "high"} else "low"
            group = _candidate_id(request_id, attachment_id, index, "group")
            common = {"attachment_id": attachment_id, "candidate_group_key": group,
                      "confidence_band": confidence, "uncertainty_code": "other"}
            candidates.append({**common, "candidate_id": _candidate_id(request_id, attachment_id, index, "name"),
                               "candidate_type": "food_name", "value": name})
            category = food.get("category")
            if isinstance(category, str) and category in CATEGORIES:
                candidates.append({**common, "candidate_id": _candidate_id(request_id, attachment_id, index, "category"),
                                   "candidate_type": "food_category", "value": category})
            if len(candidates) >= 48:
                break
        quality = "usable" if candidates else "uncertain"
        reason = None if candidates else "no_visible_food"
    else:
        quality_info = data.get("image_quality") or {}
        candidate = data.get("candidate_bristol_type")
        if quality_info.get("usable") is False:
            quality, reason = "unusable", "low_image_quality"
        elif type(candidate) is int and 1 <= candidate <= 7:
            confidence_value = data.get("confidence")
            confidence = confidence_value if isinstance(confidence_value, str) and confidence_value in {"low", "medium", "high"} else "low"
            candidates.append({"candidate_id": _candidate_id(request_id, attachment_id, 0, "bristol"),
                               "attachment_id": attachment_id, "candidate_type": "bristol_type", "value": candidate,
                               "confidence_band": confidence, "uncertainty_code": "other"})
            quality, reason = "usable", None
        else:
            quality, reason = "uncertain", "no_reliable_stool_type"
    return {"attachment_id": attachment_id, "image_quality": quality,
            "reason_code": reason, "candidates": candidates}


def _download(client, capability: str, content_type: str, max_bytes: int) -> bytes:
    request = Request(client.base_url + "/internal/v1/agent/record-image-recognition/images/" + quote(capability, safe=""),
                      headers={"Authorization": f"Bearer {client.service_token}"})
    with urlopen(request, timeout=client.timeout) as response:
        actual = response.headers.get_content_type()
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes or actual != content_type:
        raise ImageContractError("image size or type mismatch")
    if content_type == "image/png" and not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ImageContractError("invalid PNG bytes")
    if content_type == "image/jpeg" and not (data.startswith(b"\xff\xd8") and data.endswith(b"\xff\xd9")):
        raise ImageContractError("invalid JPEG bytes")
    return data


def _run_one(settings, request_id: str, recognition_type: str, attachment: dict, image_bytes: bytes) -> dict:
    attachment_id = _uuid(attachment["attachment_id"])
    suffix = ".png" if attachment["content_type"] == "image/png" else ".jpg"
    with tempfile.TemporaryDirectory(prefix="gutcare-image-") as directory:
        path = Path(directory) / (attachment_id + suffix)
        path.write_bytes(image_bytes)
        request = TaskRequest(task_id=request_id, patient_id=request_id, kind=TYPES[recognition_type],
                              files=[{"file_id": attachment_id, "path": str(path),
                                      "kind": "food_image" if recognition_type == "diet_image_recognition" else "stool_image",
                                      "mime": attachment["content_type"]}])
        client = LLMClient(settings)
        agent = DietAgent(settings, client) if recognition_type == "diet_image_recognition" else StoolAgent(settings, client)
        result = agent.run(request)
    if result.error:
        code = str(result.error.get("code") or "") if isinstance(result.error, dict) else ""
        mapped = "timeout" if "timeout" in code.lower() else "agent_unavailable"
        raise ImageProcessingError(mapped)
    return map_image_result(request_id, recognition_type, attachment_id, result)


def _fetch_images(client, settings, dispatch: dict, request_id: str, retry_delay: float) -> tuple[dict, list]:
    """Refresh all one-use capabilities when any transient read fails."""
    for attempt in range(3):
        try:
            raw = client.request("GET", f"/internal/v1/agent/record-image-recognition/requests/{request_id}/input")
            attachments = validate_input(raw, dispatch)
            downloaded = [(item, _download(client, item["image_capability"], item["content_type"],
                                           settings.max_image_bytes)) for item in attachments]
            return raw, downloaded
        except HTTPError as exc:
            if exc.code in {404, 409}:
                raise
            if exc.code < 500 or attempt == 2:
                raise
        except (URLError, TimeoutError, OSError):
            if attempt == 2:
                raise
        time.sleep(retry_delay * (2 ** attempt))
    raise RuntimeError("image input retry exhausted")


def _send_saved(client, journal: ImageJournal, request_id: str, *, retry_delay: float) -> dict | None:
    row = journal.get(request_id)
    if not row or row["state"] != "callback_pending":
        return None
    payload = json.loads(row["callback_json"])
    path = f"/internal/v1/agent/record-image-recognition/requests/{request_id}/"
    path += "candidates" if row["callback_kind"] == "candidates" else "fail"
    for attempt in range(3):
        try:
            client.request("POST", path, payload)
        except HTTPError as exc:
            if exc.code in {400, 404, 409, 422}:
                journal.state(request_id, "stopped", f"http_{exc.code}")
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


def process_image_task(client, settings, dispatch: dict, *, journal: ImageJournal,
                       retry_delay: float = 0.25) -> dict | None:
    request_id = validate_dispatch(dispatch)
    state = journal.accept(dispatch)
    if state in {"completed", "stopped"}:
        return None
    if state == "callback_pending":
        return _send_saved(client, journal, request_id, retry_delay=retry_delay)
    journal.state(request_id, "running")
    try:
        raw, downloaded = _fetch_images(client, settings, dispatch, request_id, retry_delay)
    except HTTPError as exc:
        if exc.code in {404, 409}:
            journal.state(request_id, "stopped", f"input_http_{exc.code}")
            return None
        raise
    except ImageContractError:
        journal.callback(request_id, "fail", {
            "callback_id": _candidate_id(request_id, request_id, 0, "fail"),
            "error_code": "invalid_image",
        })
        return _send_saved(client, journal, request_id, retry_delay=retry_delay)
    try:
        images = []
        for item, image_bytes in downloaded:
            images.append(_run_one(settings, request_id, dispatch["recognition_type"], item, image_bytes))
        payload = {"callback_id": _candidate_id(request_id, request_id, 0, "callback"),
                   "record_version": raw["record_version"], "image_set_version": raw["image_set_version"],
                   "schema_version": "1.0", "images": images}
        journal.callback(request_id, "candidates", payload)
    except Exception as exc:
        LOG.error("Image task failed request_id=%s error_type=%s", request_id, type(exc).__name__)
        error_code = (exc.error_code if isinstance(exc, ImageProcessingError) else
                      "invalid_image" if isinstance(exc, ImageContractError) else "internal_processing_error")
        payload = {"callback_id": _candidate_id(request_id, request_id, 0, "fail"), "error_code": error_code}
        journal.callback(request_id, "fail", payload)
    return _send_saved(client, journal, request_id, retry_delay=retry_delay)
