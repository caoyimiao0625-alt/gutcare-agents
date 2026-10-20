"""测试包适配层。

把 `GutCare测试包_v4` 的样例转成 TaskRequest，使智能体可以**不依赖业务后端**
独立运行与评测。正式联调时，这一层换成后端授权请求的映射，Agent 核心不变
（见《开发边界》§8 与路线文档 §4）。

只读：本模块不修改测试包内任何文件。
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .contract import PatientProfile, TaskKind, TaskRequest, TimeRange, new_id

CATALOG_PATH = "inputs/test_catalog.json"
SEED_PATH = "inputs/seed_database.json"
SQLITE_PATH = "inputs/fixture.sqlite"

#: 样例类别 → 任务类型
CATEGORY_TASK = {
    "food_image": TaskKind.MEAL_RECOGNIZE,
    "chinese_report": TaskKind.REPORT_EXTRACT,
    "ocr": TaskKind.REPORT_OCR,
    "advice_document": TaskKind.ADVICE_DOC_EXTRACT,
    "longitudinal": TaskKind.TREND_ANALYZE,
    "stool": TaskKind.STOOL_RECORD,
    "symptom": TaskKind.SAFETY_CHECK,
    "medication": TaskKind.STOOL_RECORD,
    "activity": TaskKind.STOOL_RECORD,
    "mood": TaskKind.STOOL_RECORD,
    "mood_public": TaskKind.SAFETY_CHECK,
    "workflow": TaskKind.ANALYSIS,
}


class FixtureDatabase:
    """测试包 SQLite 的**只读**适配层（模拟业务后端数据库接入）。

    为什么需要它：真实联调时数据来自业务库/接口，而不是 JSON 文件。
    用测试包自带的 `fixture.sqlite` 做一次“数据库 → TaskRequest”的真实演练，
    可以证明适配层不是只为样例文件写的；同时用 `mode=ro` 打开，
    从代码层面保证我们不会改写后端数据。
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(
                f"未找到测试包数据库：{self.path}（在测试包 inputs/fixture.sqlite）"
            )
        self._conn = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True)
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "FixtureDatabase":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    @staticmethod
    def _payload(row: sqlite3.Row) -> dict[str, Any]:
        return json.loads(row["payload"])

    def patients(self) -> list[dict[str, Any]]:
        return [self._payload(row) for row in self._conn.execute("SELECT payload FROM patients")]

    def patient(self, patient_id: str) -> dict[str, Any]:
        row = self._conn.execute("SELECT payload FROM patients WHERE patient_id=?", (patient_id,)).fetchone()
        if row is None:
            raise KeyError(f"数据库中不存在患者 {patient_id}")
        return self._payload(row)

    def records(self, patient_id: str | None = None, *, status: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT payload FROM records"
        clauses: list[str] = []
        params: list[Any] = []
        if patient_id:
            clauses.append("patient_id=?")
            params.append(patient_id)
        if status:
            clauses.append("status=?")
            params.append(status)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY record_id"
        rows = [self._payload(row) for row in self._conn.execute(sql, params)]
        # occurred_at 在 payload JSON 里，不在列上；这里在内存中按发生时间排序，保证确定性
        rows.sort(key=lambda r: (str(r.get("occurred_at") or ""), str(r.get("record_id") or "")))
        return rows

    def record(self, record_id: str) -> dict[str, Any]:
        row = self._conn.execute("SELECT payload FROM records WHERE record_id=?", (record_id,)).fetchone()
        if row is None:
            raise KeyError(f"数据库中不存在记录 {record_id}")
        return self._payload(row)

    def medication_plans(self, patient_id: str | None = None) -> list[dict[str, Any]]:
        if patient_id:
            rows = self._conn.execute(
                "SELECT payload FROM medication_plans WHERE patient_id=?", (patient_id,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT payload FROM medication_plans").fetchall()
        return [self._payload(row) for row in rows]

    def media(self, asset_id: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT asset_id, owner_fixture_id, path, status, origin FROM media"
        params: tuple[Any, ...] = ()
        if asset_id:
            sql += " WHERE asset_id=?"
            params = (asset_id,)
        return [dict(row) for row in self._conn.execute(sql, params)]

    def stats(self) -> dict[str, int]:
        return {
            table: int(self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("patients", "records", "medication_plans", "media")
        }


class Fixtures:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        if not (self.root / CATALOG_PATH).is_file():
            raise FileNotFoundError(
                f"未找到测试包：{self.root / CATALOG_PATH}（可用 GCA_FIXTURE_ROOT 指定）"
            )
        self._catalog = json.loads((self.root / CATALOG_PATH).read_text(encoding="utf-8"))
        self._by_id = {item["sample_id"]: item for item in self._catalog}
        self._seed: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    def catalog(self, category: str | None = None) -> list[dict[str, Any]]:
        if category is None:
            return self._catalog
        return [item for item in self._catalog if item.get("category") == category]

    def sample(self, sample_id: str) -> dict[str, Any]:
        if sample_id not in self._by_id:
            raise KeyError(f"测试包中没有样例 {sample_id}")
        return self._by_id[sample_id]

    def categories(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self._catalog:
            counts[item["category"]] = counts.get(item["category"], 0) + 1
        return counts

    # ------------------------------------------------------------------
    def load(self, relative_path: str) -> Any:
        return json.loads((self.root / relative_path).read_text(encoding="utf-8"))

    @property
    def seed(self) -> dict[str, Any]:
        if self._seed is None:
            self._seed = self.load(SEED_PATH)
        return self._seed

    def patient(self, patient_id: str) -> dict[str, Any]:
        for item in self.seed["patients"]:
            if item["patient_id"] == patient_id:
                return item
        raise KeyError(patient_id)

    def profile(self, patient_id: str) -> PatientProfile:
        raw = self.patient(patient_id)
        return PatientProfile(
            patient_id=raw["patient_id"],
            display_name=raw.get("display_name", ""),
            age=raw.get("age"),
            diagnosis_status=raw.get("diagnosis_status", "unknown"),
            diagnosis=raw.get("diagnosis"),
            profile_version=int(raw.get("profile_version", 1)),
        )

    def records_of(self, patient_id: str) -> list[dict[str, Any]]:
        return [r for r in self.seed["records"] if r["patient_id"] == patient_id]

    def records_of_case(self, sample_id: str) -> list[dict[str, Any]]:
        item = self.sample(sample_id)
        if item["category"] != "longitudinal":
            return []
        return list(self.load(item["input_path"])["records"])

    # ------------------------------------------------------------------
    def build_request(
        self,
        sample_id: str,
        *,
        task_id: str | None = None,
        options: dict[str, Any] | None = None,
        time_range: TimeRange | None = None,
        extra_records: list[dict[str, Any]] | None = None,
        query: str = "",
        safety_alerts: list[dict[str, Any]] | None = None,
    ) -> TaskRequest:
        item = self.sample(sample_id)
        category = str(item["category"])
        kind = CATEGORY_TASK.get(category, TaskKind.ANALYSIS)
        task_id = task_id or new_id(category)
        options = dict(options or {})

        if category == "longitudinal":
            payload = self.load(item["input_path"])
            profile = _profile_from_dict(payload["profile"])
            return TaskRequest(
                task_id=task_id,
                kind=TaskKind.TREND_ANALYZE,
                patient_id=payload["patient_id"],
                time_range=time_range or TimeRange(payload["window_start"], payload["window_end"]),
                records=list(payload["records"]) + list(extra_records or []),
                profile=profile,
                options=options,
            )

        if category in ("stool", "symptom", "medication", "activity", "mood"):
            payload = self.load(item["input_path"])
            record = payload["record"]
            return TaskRequest(
                task_id=task_id,
                kind=kind,
                patient_id=record["patient_id"],
                records=[record] + list(extra_records or []),
                profile=self.profile(record["patient_id"]),
                options=options,
                query=query,
                safety_alerts=list(safety_alerts or []),
            )

        if category == "mood_public":
            payload = self.load(item["input_path"])
            return TaskRequest(
                task_id=task_id,
                kind=kind,
                patient_id="EXT-openESM0029",
                records=list(extra_records or []),
                options={**options, "external_payload": payload},
                query=query,
                safety_alerts=list(safety_alerts or []),
            )

        if category == "workflow":
            payload = self.load(item["input_path"])
            return TaskRequest(
                task_id=task_id,
                kind=TaskKind.ANALYSIS,
                patient_id=payload["actor_patient_id"],
                records=self.records_of(payload["target_patient_id"]),
                profile=self.profile(payload["actor_patient_id"]),
                options={**options, "workflow": payload},
                safety_alerts=list(safety_alerts or []),
            )

        # 图像类样例
        files = [
            {
                "file_id": item["sample_id"],
                "path": item["input_path"],
                "mime": _guess_mime(item["input_path"]),
                "kind": "report" if category in ("chinese_report", "ocr") else ("advice_doc" if category == "advice_document" else "food_image"),
            }
        ]
        if item.get("linked_report_path"):
            files.append(
                {
                    "file_id": str(item.get("linked_report_id")),
                    "path": item["linked_report_path"],
                    "mime": _guess_mime(item["linked_report_path"]),
                    "kind": "report",
                    "role": "linked_report",
                }
            )
        return TaskRequest(
            task_id=task_id,
            kind=kind,
            patient_id="EXT-assets",
            files=files,
            profile=None,
            options={**options, "source_id": item.get("source_id"), "task_hint": item.get("task")},
            query=query,
            safety_alerts=list(safety_alerts or []),
        )

    # ------------------------------------------------------------------
    def analysis_request(
        self,
        patient_id: str,
        *,
        days: int = 7,
        task_id: str | None = None,
        extra_records: list[dict[str, Any]] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> TaskRequest:
        from datetime import datetime, timedelta

        records = self.records_of(patient_id)
        stamps = sorted(r["occurred_at"] for r in records)
        if start and end:
            window = TimeRange(start, end)
        elif stamps:
            end_dt = datetime.fromisoformat(stamps[-1])
            window = TimeRange((end_dt - timedelta(days=days)).isoformat(), end_dt.isoformat())
        else:
            window = TimeRange()
        return TaskRequest(
            task_id=task_id or new_id("analysis"),
            kind=TaskKind.ANALYSIS,
            patient_id=patient_id,
            time_range=window,
            records=records + list(extra_records or []),
            profile=self.profile(patient_id),
        )

    # ------------------------------------------------------------------
    # 数据库接入：模拟“业务后端提供数据”的真实路径
    # ------------------------------------------------------------------
    @property
    def database(self) -> FixtureDatabase:
        return FixtureDatabase(self.root / SQLITE_PATH)

    def analysis_request_from_db(
        self,
        patient_id: str,
        *,
        days: int = 7,
        task_id: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> TaskRequest:
        """与 `analysis_request` 等价，但记录来自 SQLite（只读）。

        真实联调时这条路径会换成“后端授权接口 → TaskRequest”，
        Agent 核心与编排逻辑不需要改动。
        """
        with self.database as db:
            patient = db.patient(patient_id)
            records = db.records(patient_id)
        profile = _profile_from_dict(patient)
        stamps = sorted(r["occurred_at"] for r in records)
        if start and end:
            window = TimeRange(start, end)
        elif stamps:
            end_dt = datetime.fromisoformat(stamps[-1])
            window = TimeRange((end_dt - timedelta(days=days)).isoformat(), end_dt.isoformat())
        else:
            window = TimeRange()
        return TaskRequest(
            task_id=task_id or new_id("analysis"),
            kind=TaskKind.ANALYSIS,
            patient_id=patient_id,
            time_range=window,
            records=records,
            profile=profile,
        )

    def record_request_from_db(self, record_id: str, *, kind: str = TaskKind.STOOL_RECORD) -> TaskRequest:
        """单条记录任务（例如排便记录校验/安全分流），记录来自数据库。"""
        with self.database as db:
            record = db.record(record_id)
            patient = db.patient(str(record["patient_id"]))
        return TaskRequest(
            task_id=new_id("record"),
            kind=kind,
            patient_id=str(record["patient_id"]),
            records=[record],
            profile=_profile_from_dict(patient),
        )

    def text_request(
        self,
        text: str,
        *,
        patient_id: str = "EXT-assets",
        source_id: str = "report_text",
        task_id: str | None = None,
    ) -> TaskRequest:
        """从 OCR/转写文本构造报告抽取请求（视觉不可用时的降级路径）。"""
        return TaskRequest(
            task_id=task_id or new_id("report_text"),
            kind=TaskKind.REPORT_TEXT_EXTRACT,
            patient_id=patient_id,
            options={"text": text, "source_id": source_id},
        )


def _profile_from_dict(raw: dict[str, Any]) -> PatientProfile:
    """测试包档案比契约多几个字段（origin 等），这里只取契约字段，不做猜补。"""
    return PatientProfile(
        patient_id=str(raw.get("patient_id")),
        display_name=str(raw.get("display_name") or ""),
        age=raw.get("age"),
        diagnosis_status=str(raw.get("diagnosis_status") or "unknown"),
        diagnosis=raw.get("diagnosis"),
        profile_version=int(raw.get("profile_version") or 1),
        restrictions=list(raw.get("restrictions") or []),
    )


def _guess_mime(path: str) -> str:
    lowered = path.lower()
    if lowered.endswith(".png"):
        return "image/png"
    if lowered.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if lowered.endswith(".pdf"):
        return "application/pdf"
    return "application/octet-stream"
