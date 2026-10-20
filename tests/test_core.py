"""核心回归测试（标准库 unittest，零依赖）。

运行：
    python -m unittest discover -s tests -v

重点验证四类“容易被糊弄”的地方：
1. 记录筛选口径（只有 confirmed 参与、版本/时间窗正确）；
2. 确定性统计与测试包参考答案一致（这是可量化的硬指标）；
3. 安全规则独立可用，且不误报；
4. 输出校验真的会删掉无依据/违规内容，而不是只提醒。
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import uuid
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gutcare_agents.agents.advice import AdviceAgent  # noqa: E402
from gutcare_agents.agents.trend import TrendAgent  # noqa: E402
from gutcare_agents.config import load_settings  # noqa: E402
from gutcare_agents.contract import (  # noqa: E402
    AgentResult,
    ObsKind,
    PatientProfile,
    TaskRequest,
    TimeRange,
)
from gutcare_agents.fixtures import Fixtures  # noqa: E402
from gutcare_agents.knowledge_base import KnowledgeBase  # noqa: E402
from gutcare_agents.llm import LLMClient  # noqa: E402
from gutcare_agents.orchestrator import Orchestrator  # noqa: E402
from gutcare_agents.rules import RuleEngine, build_context  # noqa: E402
from gutcare_agents.stats import facts_for_request  # noqa: E402
from gutcare_agents.store import Store  # noqa: E402
from gutcare_agents.validation import (  # noqa: E402
    check_numeric_grounding,
    scan_text,
    validate_advice_items,
    validate_agent_result,
)

FIXTURE_ROOT = ROOT.parent / "testdata_v4"

class _TmpDir:
    """测试用临时目录：放在工作区内，避免沙箱对系统临时目录的写入限制。"""

    def __init__(self) -> None:
        base = ROOT / "data" / "testtmp"
        base.mkdir(parents=True, exist_ok=True)
        # 不用 mkdtemp：它会创建 0700 目录，在受限令牌下会导致子目录写入被拒
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
        min_days_coverage=3,
        min_records_for_trend=5,
    )


@unittest.skipUnless((FIXTURE_ROOT / "inputs/test_catalog.json").is_file(), "缺少测试包")
class TestFixturesAndStats(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = _TmpDir()
        self.settings = make_settings(self.tmp.name)
        self.fixtures = Fixtures(self.settings.fixture_root)
        self.store = Store(self.settings.db_path)
        self.client = LLMClient(self.settings)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_catalog_counts(self) -> None:
        counts = self.fixtures.categories()
        self.assertEqual(counts["longitudinal"], 90)
        self.assertEqual(counts["chinese_report"], 87)

    def test_trend_eligible_ids_match_reference(self) -> None:
        """与测试包 business_answers 的 eligible_record_ids 逐条比对。"""
        reference = {
            item["sample_id"]: item
            for item in json.loads(
                (self.settings.fixture_root / "reference_answers/business_answers.json").read_text(encoding="utf-8")
            )
        }
        matched = 0
        total = 0
        for sample_id in ("TREND001", "TREND002", "TREND003", "TREND004", "TREND005"):
            request = self.fixtures.build_request(sample_id)
            facts, records = facts_for_request(request)
            expected = reference[sample_id]
            got_ids = {str(r["record_id"]) for r in records}
            total += 1
            if got_ids == set(expected["eligible_record_ids"]) and facts.by_kind == expected["record_counts"]:
                matched += 1
        self.assertEqual(matched, total, "记录筛选/计数与参考答案不一致")

    def test_unconfirmed_records_excluded(self) -> None:
        request = self.fixtures.build_request("TREND001")
        # 人为插入一条 pending_confirmation 与一条 deleted，必须被排除
        request.records.append(
            {
                "record_id": "SYN001-stool-pending",
                "patient_id": "SYN001",
                "kind": "stool",
                "occurred_at": "2026-09-03T08:00:00+08:00",
                "status": "pending_confirmation",
                "version": 1,
                "data": {"bristol_type": 7},
            }
        )
        request.records.append(
            {
                "record_id": "SYN001-stool-deleted",
                "patient_id": "SYN001",
                "kind": "stool",
                "occurred_at": "2026-09-03T08:00:00+08:00",
                "status": "deleted",
                "version": 1,
                "data": {"bristol_type": 1},
            }
        )
        facts, kept = facts_for_request(request)
        self.assertEqual(facts.excluded.get("pending_confirmation"), 1)
        self.assertEqual(facts.excluded.get("deleted"), 1)
        kept_ids = {str(r["record_id"]) for r in kept}
        self.assertNotIn("SYN001-stool-pending", kept_ids)
        self.assertNotIn("SYN001-stool-deleted", kept_ids)
        self.assertEqual(facts.by_kind.get("stool"), 7)

    def test_other_patient_records_ignored(self) -> None:
        request = self.fixtures.build_request("TREND001")
        request.records.append(
            {
                "record_id": "SYN999-diet-1",
                "patient_id": "SYN999",
                "kind": "diet",
                "occurred_at": "2026-09-03T08:00:00+08:00",
                "status": "confirmed",
                "version": 1,
                "data": {"description": "别人的记录"},
            }
        )
        facts, records = facts_for_request(request)
        self.assertTrue(all(r["patient_id"] == "SYN001" for r in records))
        self.assertEqual(facts.total_records, 41)


@unittest.skipUnless((FIXTURE_ROOT / "inputs/test_catalog.json").is_file(), "缺少测试包")
class TestTrendAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = _TmpDir()
        self.settings = make_settings(self.tmp.name)
        self.fixtures = Fixtures(self.settings.fixture_root)
        self.store = Store(self.settings.db_path)
        self.agent = TrendAgent(self.settings, LLMClient(self.settings), self.store)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_deterministic_observations_without_model(self) -> None:
        result = self.agent.run(self.fixtures.build_request("TREND001"))
        self.assertTrue(result.observations)
        self.assertTrue(all(o.source_record_ids for o in result.observations))
        self.assertEqual(result.data["facts"]["total_records"], 41)
        self.assertIn(result.status, ("ok", "partial"))

    def test_empty_window_returns_insufficient(self) -> None:
        request = self.fixtures.build_request("TREND001", time_range=TimeRange("2020-01-01T00:00:00+08:00", "2020-01-02T00:00:00+08:00"))
        result = self.agent.run(request)
        self.assertEqual(result.status, "insufficient_data")
        self.assertFalse([o for o in result.observations if "共 0 条" in o.statement])

    def test_missing_days_not_zero(self) -> None:
        result = self.agent.run(self.fixtures.build_request("TREND001"))
        text = " ".join(o.statement for o in result.observations)
        self.assertIn("没有记录不等于没有发生", text)


class TestSafetyRules(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = _TmpDir()
        self.settings = make_settings(self.tmp.name)
        self.engine = RuleEngine(self.settings.rules_path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _ctx(self, records, profile=None):
        return build_context(records, profile)

    def test_blood_reported_triggers_level2(self) -> None:
        records = [
            {
                "record_id": "r1",
                "patient_id": "P1",
                "kind": "stool",
                "occurred_at": "2026-09-01T08:00:00+08:00",
                "data": {"bristol_type": 3, "blood_reported": True},
            }
        ]
        hits = self.engine.evaluate(self._ctx(records), stage="record_submit")
        ids = {h.rule_id for h in hits}
        self.assertIn("R-BLOOD-STOOL-EXPLICIT", ids)
        self.assertEqual(self.engine.max_level(hits), 2)

    def test_black_tarry_text_escalates_level3(self) -> None:
        records = [
            {
                "record_id": "r2",
                "patient_id": "P1",
                "kind": "symptom",
                "occurred_at": "2026-09-01T08:00:00+08:00",
                "data": {"name": "腹痛", "severity": 4, "note": "这两天是黑色柏油样大便"},
            }
        ]
        hits = self.engine.evaluate(self._ctx(records), stage="record_submit")
        self.assertEqual(self.engine.max_level(hits), 3)
        black = [h for h in hits if h.rule_id == "R-BLACK-TARRY"]
        self.assertTrue(black and black[0].needs_confirmation)

    def test_normal_record_no_alert(self) -> None:
        records = [
            {
                "record_id": "r3",
                "patient_id": "P1",
                "kind": "stool",
                "occurred_at": "2026-09-01T08:00:00+08:00",
                "data": {"bristol_type": 4, "blood_reported": False},
            },
            {
                "record_id": "r4",
                "patient_id": "P1",
                "kind": "symptom",
                "occurred_at": "2026-09-01T09:00:00+08:00",
                "data": {"name": "腹胀", "severity": 3},
            },
        ]
        hits = self.engine.evaluate(self._ctx(records), stage="record_submit")
        self.assertEqual(hits, [])

    def test_profile_rule_only_in_analysis_stage(self) -> None:
        records = [
            {
                "record_id": "r5",
                "patient_id": "SYN009",
                "kind": "stool",
                "occurred_at": "2026-09-01T08:00:00+08:00",
                "data": {"bristol_type": 4},
            }
        ]
        profile = {"diagnosis_status": "unconfirmed", "diagnosis": None}
        submit = self.engine.evaluate(self._ctx(records, profile), stage="record_submit")
        analysis = self.engine.evaluate(self._ctx(records, profile), stage="analysis")
        self.assertNotIn("R-PROFILE-INCOMPLETE", {h.rule_id for h in submit})
        self.assertIn("R-PROFILE-INCOMPLETE", {h.rule_id for h in analysis})


class TestValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = _TmpDir()
        self.settings = make_settings(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _result(self) -> AgentResult:
        return AgentResult(task_id="t1", agent="advice", patient_id="P1")

    def test_causal_and_dosage_wording_blocked(self) -> None:
        codes = {h[0] for h in scan_text("高脂饮食导致了你的腹胀")}
        self.assertIn("causal_claim", codes)
        codes = {h[0] for h in scan_text("建议把药量加倍服用")}
        self.assertIn("dosage_change", codes)
        codes = {h[0] for h in scan_text("这张照片可以确诊为肠炎")}
        self.assertTrue({"photo_diagnosis", "diagnosis_claim"} & codes)

    def test_negated_causal_is_allowed(self) -> None:
        self.assertEqual(scan_text("这只是时间上的邻近，不能据此认定因果，也不能说明是谁导致"), [])

    def test_long_hedged_sentence_is_allowed(self) -> None:
        """长句里的“无法确定……导致……”是合规表述，不能被当成因果断言删掉。"""
        text = "目前无法确定是食物种类还是进食速度等其他因素导致时间邻近，需结合更多细节分析"
        self.assertEqual(scan_text(text), [])

    def test_causal_positive_still_blocked(self) -> None:
        self.assertIn("causal_claim", {h[0] for h in scan_text("高脂饮食导致了你的腹胀")})

    def test_safe_negated_dosage_allowed(self) -> None:
        self.assertEqual(scan_text("请不要自行停药或换药，需咨询医生"), [])

    def test_evidence_missing_observation_dropped(self) -> None:
        result = self._result()
        result.add_obs(ObsKind.MODEL_INFERENCE, "患者最近腹胀明显", record_ids=[])
        result.add_obs(ObsKind.FACT, "记录了 7 次排便", record_ids=["r1"])
        report = validate_agent_result(result, allowed_record_ids=["r1"])
        self.assertEqual(report.dropped_observations, ["advice_obs_001"])
        self.assertEqual(len(result.observations), 1)
        self.assertEqual(result.status, "partial")

    def test_foreign_record_reference_dropped(self) -> None:
        result = self._result()
        result.add_obs(ObsKind.FACT, "来自其他患者的记录", record_ids=["SYN999-diet-1"])
        report = validate_agent_result(result, allowed_record_ids=["r1"])
        self.assertTrue(report.findings)
        self.assertEqual(result.observations, [])

    def test_dates_are_not_treated_as_claims(self) -> None:
        facts = {"total_records": 7, "window_start": "2026-08-31T08:00:00+08:00"}
        for text in ("8月31日无记录", "2026-08-31 当天无记录", "31日缺记录"):
            self.assertEqual(check_numeric_grounding(text, facts), [], f"日期被误判：{text}")
        self.assertTrue(check_numeric_grounding("连续 3 天没有记录", {"total_records": 7}), "真正的数字仍应被检查")

    def test_numeric_grounding(self) -> None:
        facts = {"total_records": 7, "bristol": {"count": 3}}
        self.assertEqual(check_numeric_grounding("本周共 7 条记录", facts), [])
        self.assertTrue(check_numeric_grounding("本周共 12 条记录", facts))

    def test_advice_without_evidence_dropped(self) -> None:
        result = self._result()
        result.add_obs(ObsKind.FACT, "记录了 7 次排便", record_ids=["r1"])
        result.data["advice_items"] = [
            {"advice_id": "a1", "title": "多吃纤维", "body": "每天多吃蔬菜", "evidence_obs_ids": [], "knowledge_ids": []},
            {
                "advice_id": "a2",
                "title": "继续记录",
                "body": "继续按天记录排便与症状",
                "evidence_obs_ids": ["advice_obs_001"],
                "knowledge_ids": [],
            },
        ]
        report = validate_advice_items(result, obs_ids=["advice_obs_001"], knowledge_ids=[], facts={"total_records": 7})
        self.assertEqual([i["advice_id"] for i in result.data["advice_items"]], ["a2"])
        self.assertEqual(report.dropped_advice, ["a1"])

    def test_reference_without_leading_zero_is_accepted(self) -> None:
        """真实模型会把 trend_obs_011 写成 trend_obs_11：应规范化而不是删掉整条建议。"""
        result = self._result()
        result.data["advice_items"] = [
            {
                "advice_id": "a1",
                "title": "继续记录",
                "body": "继续按天记录",
                "evidence_obs_ids": ["trend_obs_11"],
                "knowledge_ids": ["KB-IBS-1"],
            }
        ]
        report = validate_advice_items(
            result, obs_ids=["trend_obs_011"], knowledge_ids=["KB-IBS-001"], facts=None
        )
        self.assertEqual(len(result.data["advice_items"]), 1)
        self.assertEqual(result.data["advice_items"][0]["evidence_obs_ids"], ["trend_obs_011"])
        self.assertEqual(result.data["advice_items"][0]["knowledge_ids"], ["KB-IBS-001"])
        self.assertFalse([f for f in report.findings if f.action == "dropped"])

    def test_observation_reference_normalized(self) -> None:
        result = self._result()
        result.add_obs(ObsKind.MODEL_INFERENCE, "共记录 7 天", record_ids=["SYN001-diet-01"])
        validate_agent_result(result, allowed_record_ids=["SYN001-diet-1"])
        self.assertEqual(len(result.observations), 1)
        self.assertEqual(result.observations[0].source_record_ids, ["SYN001-diet-1"])

    def test_advice_with_unknown_numbers_dropped(self) -> None:
        result = self._result()
        result.data["advice_items"] = [
            {
                "advice_id": "a3",
                "title": "目标",
                "body": "建议每天记录 12 次",
                "evidence_obs_ids": ["o1"],
                "knowledge_ids": [],
            }
        ]
        validate_advice_items(result, obs_ids=["o1"], knowledge_ids=[], facts={"total_records": 7})
        self.assertEqual(result.data["advice_items"], [])


@unittest.skipUnless((FIXTURE_ROOT / "inputs/test_catalog.json").is_file(), "缺少测试包")
class TestOrchestrator(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = _TmpDir()
        self.settings = make_settings(self.tmp.name)
        self.fixtures = Fixtures(self.settings.fixture_root)
        self.store = Store(self.settings.db_path)
        self.orchestrator = Orchestrator(self.settings, store=self.store)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_pipeline_runs_and_persists(self) -> None:
        request = self.fixtures.analysis_request("SYN001", days=7)
        run = self.orchestrator.run(request)
        self.assertEqual(run.status, "partial")  # mock 模式没有模型叙述，属于预期降级
        stage_names = [s.stage for s in run.stages]
        self.assertIn("intake", stage_names)
        self.assertIn("trend", stage_names)
        self.assertIn("advice", stage_names)
        self.assertTrue(self.store.get_task(run.task_id))

    def test_idempotent_reuse(self) -> None:
        request = self.fixtures.analysis_request("SYN001", days=7)
        first = self.orchestrator.run(request)
        second_request = self.fixtures.analysis_request("SYN001", days=7)
        second = self.orchestrator.run(second_request)
        self.assertEqual(second.report.get("reuse", {}).get("reused_task_id"), first.task_id)
        self.assertEqual(second.input_snapshot_hash, first.input_snapshot_hash)

    def test_unconfirmed_diagnosis_degrades_advice(self) -> None:
        request = self.fixtures.analysis_request("SYN009", days=365)
        run = self.orchestrator.run(request)
        advice = run.agent_results.get("advice")
        self.assertIn("advice", run.agent_results)
        from gutcare_agents.contract import AgentResult as AR

        if isinstance(advice, AR):
            self.assertFalse(advice.data.get("advice_items"))
            self.assertIn("已确认", str(advice.data.get("degraded_reason")))

    def test_emergency_suppresses_normal_advice(self) -> None:
        request = self.fixtures.analysis_request("SYN001", days=7)
        request.records.append(
            {
                "record_id": "SYN001-stool-black",
                "patient_id": "SYN001",
                "kind": "stool",
                "occurred_at": "2026-09-05T08:00:00+08:00",
                "status": "confirmed",
                "version": 1,
                "data": {"bristol_type": 6, "note": "黑色柏油样大便"},
            }
        )
        run = self.orchestrator.run(request)
        risks = run.report.get("risks") or []
        self.assertTrue(any(int(r.get("severity_level", 0)) >= 3 for r in risks))
        from gutcare_agents.contract import AgentResult as AR

        advice = run.agent_results.get("advice")
        if isinstance(advice, AR):
            for item in advice.data.get("advice_items") or []:
                self.assertEqual(item.get("category"), "seek_care")


class TestKnowledge(unittest.TestCase):
    def test_retrieval_returns_relevant_doc(self) -> None:
        kb = KnowledgeBase.load(ROOT / "knowledge")
        hits = kb.search("低FODMAP 饮食 限制 症状关联", top_k=2)
        self.assertTrue(hits)
        self.assertEqual(hits[0][0].doc_id, "KB-DIET-002")

    def test_all_docs_marked_unverified(self) -> None:
        kb = KnowledgeBase.load(ROOT / "knowledge")
        self.assertTrue(all(d.review_status == "unverified_demo" for d in kb.docs))


if __name__ == "__main__":
    unittest.main(verbosity=2)
