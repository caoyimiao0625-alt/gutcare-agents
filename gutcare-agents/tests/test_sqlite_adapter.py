"""数据库适配层测试：从 SQLite 读数据与从 JSON 读数据必须得到同一份结论。

这条路径对应真实联调（后端给数据），所以两个点必须证明：
1. **等价性**：同一患者、同一时间窗，DB 读与 JSON 读的分析输入和统计口径一致；
2. **只读性**：适配层用 `mode=ro` 打开，写操作必须失败——我们不改后端数据。
"""
from __future__ import annotations

import shutil
import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gutcare_agents.config import load_settings  # noqa: E402
from gutcare_agents.fixtures import FixtureDatabase, Fixtures  # noqa: E402
from gutcare_agents.orchestrator import Orchestrator  # noqa: E402
from gutcare_agents.stats import facts_for_request  # noqa: E402
from gutcare_agents.store import Store  # noqa: E402

FIXTURE_ROOT = ROOT.parent / "testdata_v4"


class _TmpDir:
    def __init__(self) -> None:
        base = ROOT / "data" / "testtmp"
        base.mkdir(parents=True, exist_ok=True)
        self.name = str(base / f"t{uuid.uuid4().hex[:10]}")
        Path(self.name).mkdir(parents=True, exist_ok=True)

    def cleanup(self) -> None:
        shutil.rmtree(self.name, ignore_errors=True)


@unittest.skipUnless((FIXTURE_ROOT / "inputs/fixture.sqlite").is_file(), "缺少测试包数据库")
class TestFixtureDatabase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = _TmpDir()
        self.settings = load_settings(
            mode="mock",
            fixture_root=FIXTURE_ROOT,
            data_dir=Path(self.tmp.name),
            replay_dir=Path(self.tmp.name) / "replays",
            db_path=Path(self.tmp.name) / "t.sqlite3",
            knowledge_dir=ROOT / "knowledge",
            rules_path=ROOT / "knowledge" / "safety_rules.json",
        )
        self.fixtures = Fixtures(FIXTURE_ROOT)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_schema_counts(self) -> None:
        with self.fixtures.database as db:
            stats = db.stats()
        self.assertEqual(stats["patients"], 90)
        self.assertEqual(stats["records"], 3240)
        self.assertEqual(stats["media"], 351)

    def test_database_matches_seed_json(self) -> None:
        """DB 与 seed JSON 必须描述同一批数据（适配层没有偷偷改口径）。"""
        with self.fixtures.database as db:
            db_records = db.records("SYN001")
        json_records = self.fixtures.records_of("SYN001")
        key = lambda r: r["record_id"]  # noqa: E731
        self.assertEqual(
            [(r["record_id"], r["kind"], r["status"], r["data"]) for r in sorted(db_records, key=key)],
            [(r["record_id"], r["kind"], r["status"], r["data"]) for r in sorted(json_records, key=key)],
        )

    def test_database_is_read_only(self) -> None:
        db = self.fixtures.database
        try:
            with self.assertRaises(Exception) as ctx:
                db._conn.execute("DELETE FROM records WHERE record_id='SYN001-diet-1'")
                db._conn.commit()
            self.assertIn("readonly", str(ctx.exception).lower())
        finally:
            db.close()
        # 数据没有被改动
        with self.fixtures.database as db2:
            self.assertEqual(len(db2.records("SYN001")), 41)

    def test_facts_from_db_equal_facts_from_json(self) -> None:
        json_request = self.fixtures.analysis_request("SYN001", days=7, start="2026-09-01T08:00:00+08:00", end="2026-09-08T08:00:00+08:00")
        db_request = self.fixtures.analysis_request_from_db(
            "SYN001", days=7, start="2026-09-01T08:00:00+08:00", end="2026-09-08T08:00:00+08:00"
        )
        json_facts, json_records = facts_for_request(json_request)
        db_facts, db_records = facts_for_request(db_request)
        self.assertEqual(
            sorted(r["record_id"] for r in json_records),
            sorted(r["record_id"] for r in db_records),
        )
        self.assertEqual(json_facts.by_kind, db_facts.by_kind)
        self.assertEqual(json_facts.missing_days, db_facts.missing_days)
        self.assertEqual(json_facts.bristol, db_facts.bristol)
        self.assertEqual(json_facts.symptoms, db_facts.symptoms)

    def test_orchestrator_runs_from_database(self) -> None:
        store = Store(self.settings.db_path)
        orchestrator = Orchestrator(self.settings, store=store)
        request = self.fixtures.analysis_request_from_db("SYN001", days=7)
        run = orchestrator.run(request)
        self.assertIn(run.status, ("succeeded", "partial", "insufficient_data"))
        self.assertIn("trend", [s.stage for s in run.stages])
        self.assertTrue(run.report.get("data_overview", {}).get("by_kind"))
        # 输入记录必须都能回溯到数据库里的真实 id
        eligible = {r["record_id"] for r in request.confirmed_records()}
        self.assertTrue(eligible)
        self.assertTrue(eligible <= {r["record_id"] for r in self.fixtures.records_of("SYN001")})

    def test_record_request_from_db(self) -> None:
        request = self.fixtures.record_request_from_db("SYN001-stool-1")
        self.assertEqual(request.patient_id, "SYN001")
        self.assertEqual(len(request.records), 1)
        self.assertEqual(request.records[0]["record_id"], "SYN001-stool-1")
        self.assertIsNotNone(request.profile)

    def test_unknown_patient_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.fixtures.analysis_request_from_db("SYN999-NOT-EXIST")


if __name__ == "__main__":
    unittest.main(verbosity=2)
