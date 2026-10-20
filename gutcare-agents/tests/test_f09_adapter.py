"""F09 mapping tests; optionally validate against a checked-out backend model."""
from __future__ import annotations

import os
import sys
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gutcare_agents.f08_adapter import F08ContractError  # noqa: E402
from gutcare_agents.f09_adapter import map_f09_advice  # noqa: E402


def fixture():
    task_id, patient_id, source_id = (str(uuid.uuid4()) for _ in range(3))
    frozen = {
        "task_id": task_id, "patient_id": patient_id,
        "analysis_type": "comprehensive_health_analysis", "schema_version": "1.0",
        "snapshot_hash": "b" * 64, "time_range": None,
        "consent_scope": ["health_data_processing", "third_party_ai"],
        "consent_scope_snapshot": {
            "health_data_processing": True, "mood_data_processing": False,
            "mood_ai_sharing": False, "third_party_ai": True,
        },
        "records": [{"record_type": "stool_record", "record_id": source_id,
                     "record_version": 2, "confirmation_status": "confirmed",
                     "locator": f"stool_record:{source_id}", "payload": {"bristol_type": 4}}],
        "mood_records": [], "missing_data": [],
    }
    item = {"advice_id": "advice_001", "category": "record_management",
            "title": "继续记录", "body": "继续记录排便形态以便核对变化。",
            "uncertainty": "记录较少", "applicability": "仅用于本次记录",
            "evidence_obs_ids": ["obs_001"], "knowledge_ids": []}
    run = SimpleNamespace(task_id=task_id, patient_id=patient_id,
                          report={"observations": [{"obs_id": "obs_001",
                                                    "source_record_ids": [source_id],
                                                    "needs_confirmation": False}],
                                  "advice": [item]})
    return run, frozen


class F09MappingTests(unittest.TestCase):
    def test_patient_evidence_exactly_matches_frozen_source(self):
        run, frozen = fixture()
        mapped = map_f09_advice(run, frozen, f08_status="partial")
        self.assertTrue(mapped.ready)
        self.assertEqual(mapped.dropped, [])
        advice = mapped.payload["advice"][0]
        self.assertEqual(advice["category"], "record_management")
        self.assertEqual(advice["applicability"]["scope"], "general")
        self.assertEqual(advice["uncertainty"]["level"], "high")
        self.assertEqual(advice["evidence"], [{
            "evidence_type": "patient", "source_type": "stool_record",
            "source_id": frozen["records"][0]["record_id"], "source_version": 2,
            "locator": frozen["records"][0]["locator"],
        }])
        self.assertEqual(mapped.payload, map_f09_advice(run, frozen, f08_status="partial").payload)

    def test_unmapped_or_unconfirmed_observation_drops_entire_advice(self):
        run, frozen = fixture()
        run.report["observations"][0]["source_record_ids"] = [str(uuid.uuid4())]
        result = map_f09_advice(run, frozen, f08_status="partial")
        self.assertFalse(result.ready)
        self.assertEqual(result.dropped[0]["reason"], "unmapped_patient_evidence")
        run.report["observations"][0]["source_record_ids"] = [frozen["records"][0]["record_id"]]
        run.report["observations"][0]["needs_confirmation"] = True
        self.assertFalse(map_f09_advice(run, frozen, f08_status="partial").ready)

    def test_demo_knowledge_is_not_presented_as_approved(self):
        run, frozen = fixture()
        run.report["advice"][0]["knowledge_ids"] = ["KB-DEMO"]
        demo = {"KB-DEMO": {"review_status": "unverified_demo", "version": "demo-1",
                            "title": "演示条目", "publisher": "GutCare demo"}}
        result = map_f09_advice(run, frozen, f08_status="partial", approved_knowledge=demo)
        self.assertFalse(result.ready)
        self.assertEqual(result.dropped[0]["reason"], "knowledge_not_approved_or_incomplete")
        demo["KB-DEMO"]["review_status"] = "approved"
        result = map_f09_advice(run, frozen, f08_status="partial", approved_knowledge=demo)
        self.assertTrue(result.ready)
        self.assertEqual(result.payload["advice"][0]["evidence"][1]["knowledge_version"], "demo-1")

    def test_ineligible_f08_status_rejected(self):
        run, frozen = fixture()
        with self.assertRaises(F08ContractError):
            map_f09_advice(run, frozen, f08_status="needs_input")

    def test_latest_backend_callback_model(self):
        source = os.environ.get("GCA_BACKEND_SOURCE")
        if not source:
            self.skipTest("set GCA_BACKEND_SOURCE to check the actual backend Pydantic model")
        sys.path.insert(0, str(Path(source) / "backend" / "src"))
        from gutcare.analysis.advice import AdviceCallbackBody

        run, frozen = fixture()
        mapped = map_f09_advice(run, frozen, f08_status="partial")
        AdviceCallbackBody.model_validate(mapped.payload)


if __name__ == "__main__":
    unittest.main()
