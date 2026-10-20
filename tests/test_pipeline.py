"""流水线行为测试：结果复用、故障注入、上游失败传播、请求校验。

这些用例对应“架构设计与工程质量”里最容易被追问的点：
改了输入是否重跑？模型挂了会怎样？上游失败会不会被伪装成“数据不足”？
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gutcare_agents.api import parse_request  # noqa: E402
from gutcare_agents.config import load_settings  # noqa: E402
from gutcare_agents.contract import AgentResult, Status  # noqa: E402
from gutcare_agents.errors import AgentError  # noqa: E402
from gutcare_agents.fixtures import Fixtures  # noqa: E402
from gutcare_agents.orchestrator import Orchestrator  # noqa: E402
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


def make_settings(tmp: str):
    return load_settings(
        mode="mock",
        fixture_root=FIXTURE_ROOT,
        data_dir=Path(tmp),
        replay_dir=Path(tmp) / "replays",
        db_path=Path(tmp) / "test.sqlite3",
        knowledge_dir=ROOT / "knowledge",
        rules_path=ROOT / "knowledge" / "safety_rules.json",
    )


@unittest.skipUnless((FIXTURE_ROOT / "inputs/test_catalog.json").is_file(), "缺少测试包")
class TestDomainReuseAndFaults(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = _TmpDir()
        self.settings = make_settings(self.tmp.name)
        self.fixtures = Fixtures(self.settings.fixture_root)
        self.store = Store(self.settings.db_path)
        self.orchestrator = Orchestrator(self.settings, store=self.store)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _report_analysis_request(self, task_suffix: str = ""):
        request = self.fixtures.build_request("MED002")
        request.kind = "analysis"
        request.task_id = f"analysis_test{task_suffix}"
        request.records = []
        return request

    def test_domain_result_reused_for_same_file(self) -> None:
        first = self._report_analysis_request("a")
        self.orchestrator.run(first)
        self.assertGreaterEqual(self.store.cache_stats()["entries"], 1)

        second = self._report_analysis_request("b")
        second.options["force_refresh"] = True  # 强制重跑流程，但仍应复用识别结果
        run = self.orchestrator.run(second)
        reused = [s for s in run.stages if s.reused]
        self.assertTrue(reused, "同一文件的领域识别结果应被复用")
        self.assertIn("复用", reused[0].note)

    def test_fault_injection_marks_stage_failed(self) -> None:
        request = self.fixtures.analysis_request("SYN002", days=7)
        request.options["simulate_faults"] = ["trend"]
        run = self.orchestrator.run(request)
        trend_stage = [s for s in run.stages if s.stage == "trend"][0]
        self.assertEqual(trend_stage.status, "failed")
        self.assertIn("注入", trend_stage.note)
        self.assertEqual(run.status, "failed")

    def test_upstream_failure_not_disguised_as_insufficient(self) -> None:
        request = self.fixtures.analysis_request("SYN002", days=7)
        request.options["simulate_faults"] = ["trend"]
        run = self.orchestrator.run(request)
        advice = run.agent_results.get("advice")
        self.assertIsInstance(advice, AgentResult)
        codes = {u.code for u in advice.uncertainties}
        self.assertIn("upstream_trend_failed", codes)
        self.assertIn("上游趋势分析未完成", str(advice.data.get("degraded_reason")))

    def test_records_survive_advice_failure(self) -> None:
        request = self.fixtures.analysis_request("SYN003", days=7)
        request.options["simulate_faults"] = ["advice"]
        run = self.orchestrator.run(request)
        advice = run.agent_results.get("advice")
        self.assertIsInstance(advice, AgentResult)
        self.assertIn(advice.status, (Status.FAILED, Status.PARTIAL))
        self.assertTrue(run.report.get("observations"), "建议失败不应丢掉观察与记录")
        self.assertGreater(len(run.report["observations"]), 0)

    def test_faulted_run_not_reused_by_idempotency(self) -> None:
        request = self.fixtures.analysis_request("SYN004", days=7)
        request.options["simulate_faults"] = ["advice"]
        faulted = self.orchestrator.run(request)
        normal = self.fixtures.analysis_request("SYN004", days=7)
        rerun = self.orchestrator.run(normal)
        self.assertNotEqual(faulted.task_id, rerun.task_id)
        self.assertNotIn("reuse", rerun.report or {})

    def test_cancel_is_cooperative_and_keeps_results(self) -> None:
        import threading
        import time

        request = self.fixtures.analysis_request("SYN006", days=7)
        # 用故障注入制造一段可控的“分析中”窗口，再在窗口里请求取消
        request.options["simulate_faults"] = ["advice"]
        request.options["simulate_delay_ms"] = 1500
        holder: dict = {}

        def worker() -> None:
            holder["run"] = self.orchestrator.run(request)

        thread = threading.Thread(target=worker)
        thread.start()
        # 等任务行真正落库再取消：否则在慢机器上可能“取消”发生在任务创建之前
        deadline = time.time() + 5
        while time.time() < deadline and self.store.get_task(request.task_id) is None:
            time.sleep(0.02)
        cancel = self.store.request_cancel(request.task_id)
        thread.join(timeout=30)

        self.assertIsNotNone(cancel)
        self.assertTrue(cancel["cancelled"])
        run = holder["run"]
        self.assertEqual(run.status, "cancelled")
        self.assertIn("cancelled", run.report)
        # 已完成阶段的结果保留，供用户重新发起分析时复用
        self.assertTrue(run.report["cancelled"]["completed_stages"])

    def test_cancel_after_finish_does_not_overwrite(self) -> None:
        request = self.fixtures.analysis_request("SYN007", days=7)
        run = self.orchestrator.run(request)
        result = self.store.request_cancel(run.task_id)
        self.assertFalse(result["cancelled"])
        self.assertEqual(result["status"], run.status)


    def test_emergency_alert_recorded_once(self) -> None:
        request = self.fixtures.analysis_request("SYN005", days=7)
        request.records.append(
            {
                "record_id": "SYN005-stool-black",
                "patient_id": "SYN005",
                "kind": "stool",
                "occurred_at": "2026-09-05T08:00:00+08:00",
                "status": "confirmed",
                "version": 1,
                "data": {"bristol_type": 6, "note": "黑色柏油样大便"},
            }
        )
        first = self.orchestrator.run(request)
        again = self.fixtures.analysis_request("SYN005", days=7)
        again.records = list(request.records)
        again.options["force_refresh"] = True
        second = self.orchestrator.run(again)
        first_alerts = [a for a in first.report.get("risks", []) if int(a.get("severity_level", 0)) >= 3]
        second_alerts = [a for a in second.report.get("risks", []) if int(a.get("severity_level", 0)) >= 3]
        self.assertTrue(first_alerts)
        self.assertTrue(second_alerts)
        # 同一事件在时间窗内被去重，不应产生第二条新提示
        self.assertTrue(any(a.get("is_new") is False for a in second_alerts))


class TestRequestValidation(unittest.TestCase):
    def test_missing_patient_id_rejected(self) -> None:
        with self.assertRaises(AgentError) as ctx:
            parse_request({"kind": "trend_analyze"})
        self.assertEqual(ctx.exception.code, "invalid_request")

    def test_profile_patient_mismatch_rejected(self) -> None:
        with self.assertRaises(AgentError) as ctx:
            parse_request(
                {
                    "kind": "trend_analyze",
                    "patient_id": "SYN001",
                    "profile": {"patient_id": "SYN002", "diagnosis_status": "confirmed", "diagnosis": "IBS"},
                }
            )
        self.assertEqual(ctx.exception.code, "patient_scope_violation")

    def test_options_and_idempotency_preserved(self) -> None:
        request = parse_request(
            {
                "kind": "analysis",
                "patient_id": "SYN001",
                "options": {"force_refresh": True, "simulate_faults": ["advice"]},
                "idempotency_key": "abc123",
            }
        )
        self.assertTrue(request.options["force_refresh"])
        self.assertEqual(request.idempotency_key, "abc123")


if __name__ == "__main__":
    unittest.main(verbosity=2)
