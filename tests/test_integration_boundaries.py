"""回归此前未覆盖的输出安全、患者隔离和领域协作边界。"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from gutcare_agents.config import load_settings
from gutcare_agents.contract import AgentResult, PatientProfile, Status, TaskRequest, TaskRun
from gutcare_agents.agents.advice import AdviceAgent
from gutcare_agents.agents.diet import DietAgent
from gutcare_agents.agents.trend import TrendAgent
from gutcare_agents.api import parse_request
from gutcare_agents.errors import AgentError
from gutcare_agents.fixtures import Fixtures
from gutcare_agents.llm import LLMResponse
from gutcare_agents.orchestrator import Orchestrator
from gutcare_agents.store import Store
from gutcare_agents.validation import validate_agent_result


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT.parent / "testdata_v4"


class TestGeneratedSummarySafety(unittest.TestCase):
    def test_diet_model_fields_are_sanitized(self) -> None:
        class UnsafeClient:
            def complete_json(self, **kwargs):
                return {
                    "recognizable": True,
                    "foods": [
                        {"name": "米饭", "category": "主食", "confidence": "high"},
                        {"name": "立即停药", "category": "主食", "confidence": "high"},
                    ],
                    "portion_note": "立即停药",
                    "scene_note": "餐盘",
                }, LLMResponse(text="{}", model="test")

        settings = load_settings(mode="mock")
        request = TaskRequest(
            task_id="diet_safety", kind="meal_recognize", patient_id="p",
            files=[{"file_id": "photo_1", "path": "unused.jpg", "kind": "food_image", "mime": "image/jpeg"}],
        )
        result = DietAgent(settings, UnsafeClient()).run(request)
        visible = json.dumps(result.to_dict(), ensure_ascii=False)
        self.assertNotIn("立即停药", visible)
        self.assertEqual([f["name"] for f in result.data["foods"]], ["米饭"])

    def test_unvalidated_advice_is_not_serialized(self) -> None:
        result = AgentResult(task_id="t", agent="advice", patient_id="p")
        result.data["raw_advice_items"] = [{"body": "立即停药"}]
        result.data["advice_items"] = []
        self.assertNotIn("raw_advice_items", result.to_dict()["data"])
        run = TaskRun(task_id="t", kind="analysis", patient_id="p", agent_results={"advice": result})
        self.assertNotIn("raw_advice_items", run.to_dict()["agent_results"]["advice"]["data"])

    def test_unsafe_summary_is_removed_before_return(self) -> None:
        result = AgentResult(task_id="t", agent="trend", patient_id="p")
        result.data["narrative_summary"] = "立即停药，并且 100% 确诊。"

        report = validate_agent_result(result, allowed_record_ids=[], facts={})

        self.assertEqual(result.data["narrative_summary"], "")
        self.assertTrue(report.degraded)
        self.assertTrue(any(f.target == "narrative_summary" and f.action == "dropped" for f in report.findings))

    def test_prior_result_must_belong_to_request_patient(self) -> None:
        with self.assertRaises(AgentError) as error:
            parse_request({
                "kind": "assistant_chat", "patient_id": "patient_a",
                "confirmed_results": [{"patient_id": "patient_b", "agent": "trend"}],
            })
        self.assertEqual(error.exception.code, "patient_scope_violation")

    def test_prior_result_change_invalidates_input_snapshot(self) -> None:
        first = parse_request({
            "kind": "analysis", "patient_id": "patient_a",
            "confirmed_results": [{"patient_id": "patient_a", "agent": "trend", "version": 1}],
        })
        second = parse_request({
            "kind": "analysis", "patient_id": "patient_a",
            "confirmed_results": [{"patient_id": "patient_a", "agent": "trend", "version": 2}],
        })
        self.assertNotEqual(first.snapshot_hash(), second.snapshot_hash())


@unittest.skipUnless((FIXTURE_ROOT / "inputs/test_catalog.json").is_file(), "缺少测试包")
class TestDomainBoundaries(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="boundary_", dir=ROOT / "data"))
        self.settings = load_settings(
            mode="mock", fixture_root=FIXTURE_ROOT, data_dir=self.tmp,
            replay_dir=self.tmp / "replays", db_path=self.tmp / "test.sqlite3",
            knowledge_dir=ROOT / "knowledge", rules_path=ROOT / "knowledge" / "safety_rules.json",
        )
        self.fixtures = Fixtures(FIXTURE_ROOT)
        self.orchestrator = Orchestrator(self.settings, store=Store(self.settings.db_path))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_same_file_is_not_reused_across_patients(self) -> None:
        first = self.fixtures.build_request("MED002")
        first.kind, first.task_id, first.records = "analysis", "patient_a", []
        self.orchestrator.run(first)

        second = self.fixtures.build_request("MED002")
        second.kind, second.task_id, second.records = "analysis", "patient_b", []
        second.patient_id = "OTHER_PATIENT"
        second.profile = PatientProfile(patient_id=second.patient_id)
        run = self.orchestrator.run(second)
        medical = run.agent_results["medical"]

        self.assertEqual(medical.patient_id, second.patient_id)
        self.assertFalse(medical.usage.get("reused", False))

    def test_each_domain_receives_only_its_own_files(self) -> None:
        request = self.fixtures.build_request("MED002")
        request.kind, request.task_id, request.records = "analysis", "mixed_files", []
        report_file = request.files[0]
        food_file = dict(self.fixtures.build_request("FOOD001").files[0])
        food_file["kind"] = "food_image"
        request.files = [report_file, food_file]
        received: dict[str, list[str]] = {}

        class CaptureAgent:
            def __init__(self, name: str) -> None:
                self.name = name

            def run(self, sub_request):
                received[self.name] = [str(f["file_id"]) for f in sub_request.files]
                return AgentResult(
                    task_id=sub_request.task_id, agent=self.name,
                    patient_id=sub_request.patient_id,
                    input_snapshot_hash=sub_request.snapshot_hash(),
                ).finish(Status.OK)

        self.orchestrator.agents["medical"] = CaptureAgent("medical")
        self.orchestrator.agents["diet"] = CaptureAgent("diet")
        run = TaskRun(task_id=request.task_id, kind="analysis", patient_id=request.patient_id)
        self.orchestrator._stage_domain(run, request)

        self.assertEqual(received["medical"], [report_file["file_id"]])
        self.assertEqual(received["diet"], [food_file["file_id"]])

    def test_meal_description_change_invalidates_domain_cache(self) -> None:
        first = self.fixtures.build_request("FOOD001")
        first.kind, first.task_id = "analysis", "meal_a"
        first.options["description"] = "白米饭"
        self.orchestrator.run(first)

        second = self.fixtures.build_request("FOOD001")
        second.kind, second.task_id = "analysis", "meal_b"
        second.options["description"] = "炒饭"
        run = self.orchestrator.run(second)
        self.assertFalse(run.agent_results["diet"].usage.get("reused", False))

    def test_domain_observations_reach_trend(self) -> None:
        run = self.orchestrator.run(self.fixtures.analysis_request("SYN001", days=7))
        stool_ids = {obs.obs_id for obs in run.agent_results["stool"].observations}
        trend_ids = set(run.agent_results["trend"].data["domain_input_obs_ids"])
        self.assertTrue(stool_ids)
        self.assertTrue(stool_ids <= trend_ids)

    def test_unconfirmed_image_candidate_is_visible_to_trend_but_not_advice(self) -> None:
        request = self.fixtures.analysis_request("SYN001", days=7)
        request.domain_results = [{
            "agent": "diet", "status": "ok", "needs_confirmation": True,
            "observations": [{
                "obs_id": "diet_candidate_001", "statement": "候选菜品：米饭",
                "source_record_ids": ["new_food_photo"],
            }],
        }]
        captured: dict[str, str] = {}

        class CandidateClient:
            def complete_json(self, **kwargs):
                captured["prompt"] = kwargs["user"]
                return {
                    "narrative_summary": "本次有一项待确认识别结果。",
                    "observations": [{
                        "statement": "食物图片候选为米饭，仍待患者确认。",
                        "source_record_ids": ["new_food_photo"],
                    }],
                }, LLMResponse(text="{}", model="test")

        trend = TrendAgent(self.settings, CandidateClient()).run(request)
        self.assertIn("diet_candidate_001", captured["prompt"])
        provisional = [o for o in trend.observations if "new_food_photo" in o.source_record_ids]
        self.assertEqual(len(provisional), 1)
        self.assertTrue(provisional[0].value["provisional"])

        advice = AdviceAgent(self.settings, CandidateClient())
        pool = advice._observation_pool(request, trend)
        self.assertNotIn(provisional[0].obs_id, {o["obs_id"] for o in pool})


if __name__ == "__main__":
    unittest.main()
