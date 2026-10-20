"""文本抽取降级路径测试（视觉不可用时的备选方案）。

用桩客户端（stub）替代真实模型，验证：
1. 文本确实被送进模型，且走的是文本模型而不是视觉模型；
2. 输出映射成观察时保留原文、不补单位、不把“疑似”写成“确诊”；
3. 没有文本时明确降级，不猜测报告内容；
4. 提示词版本与摘要随结果一起返回。
"""
from __future__ import annotations

import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gutcare_agents.agents.medical import MedicalAgent  # noqa: E402
from gutcare_agents.config import load_settings  # noqa: E402
from gutcare_agents.fixtures import Fixtures  # noqa: E402
from gutcare_agents.llm import LLMResponse  # noqa: E402
from gutcare_agents.store import Store  # noqa: E402

FIXTURE_ROOT = ROOT.parent / "testdata_v4"

STUB_PAYLOAD: dict[str, Any] = {
    "document_type": "检验报告",
    "report_date": "2026-09-02",
    "institution": "示例医院",
    "exam_items": [
        {
            "name": "粪便隐血",
            "value": "阳性",
            "unit": "",
            "reference_range": "阴性",
            "abnormal_flag": "high",
            "raw_text": "粪便隐血 阳性 参考值 阴性",
        },
        {
            "name": "血红蛋白",
            "value": "[遮挡]",
            "unit": "g/L",
            "reference_range": "",
            "abnormal_flag": "unknown",
            "raw_text": "血红蛋白 [遮挡] g/L",
        },
    ],
    "conclusion": "粪便隐血阳性，建议消化内科就诊",
    "suggestions": ["建议消化内科就诊"],
    "diagnosis_candidates": [
        {"name": "肠易激综合征", "status": "suspected", "evidence": "考虑肠易激综合征"}
    ],
    "medications": [],
    "redactions": [{"locator": "第2行", "note": "遮挡"}],
    "uncertainties": [],
}


class _StubClient:
    """记录调用参数并返回固定 JSON 的桩客户端。"""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def complete_json(self, **kwargs: Any) -> tuple[Any, LLMResponse]:
        self.calls.append(kwargs)
        return self.payload, LLMResponse(text=json.dumps(self.payload, ensure_ascii=False), model="stub-model")


class _TmpDir:
    def __init__(self) -> None:
        base = ROOT / "data" / "testtmp"
        base.mkdir(parents=True, exist_ok=True)
        self.name = str(base / f"t{uuid.uuid4().hex[:10]}")
        Path(self.name).mkdir(parents=True, exist_ok=True)

    def cleanup(self) -> None:
        shutil.rmtree(self.name, ignore_errors=True)


class TestTextExtraction(unittest.TestCase):
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
        self.store = Store(self.settings.db_path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _agent(self) -> tuple[MedicalAgent, _StubClient]:
        client = _StubClient(STUB_PAYLOAD)
        return MedicalAgent(self.settings, client, self.store), client  # type: ignore[arg-type]

    def test_text_is_sent_to_text_model(self) -> None:
        agent, client = self._agent()
        fixtures = Fixtures(FIXTURE_ROOT) if (FIXTURE_ROOT / "inputs/test_catalog.json").is_file() else None
        text = "粪便隐血 阳性 参考值 阴性\n血红蛋白 [遮挡] g/L\n考虑肠易激综合征\n建议消化内科就诊"
        request = (
            fixtures.text_request(text, source_id="doc_001")
            if fixtures
            else self._fallback_request(text)
        )
        result = agent.run(request)

        self.assertEqual(len(client.calls), 1, "应只调用一次模型")
        call = client.calls[0]
        self.assertEqual(call["model"], self.settings.text_model, "文本抽取必须走文本模型")
        self.assertFalse(call.get("images"), "文本路径不应传图片")
        self.assertIn("粪便隐血", call["user"])

        self.assertEqual(result.agent, "medical")
        self.assertEqual(result.prompt_version, "v1")
        self.assertEqual(len(result.prompt_sha256), 16)
        self.assertTrue(result.observations)
        self.assertTrue(all(o.source_record_ids == ["doc_001"] for o in result.observations))

    def _fallback_request(self, text: str):
        from gutcare_agents.contract import TaskKind, TaskRequest

        return TaskRequest(
            task_id="t-text",
            kind=TaskKind.REPORT_TEXT_EXTRACT,
            patient_id="EXT-assets",
            options={"text": text, "source_id": "doc_001"},
        )

    def test_units_and_redactions_not_fabricated(self) -> None:
        agent, _client = self._agent()
        text = "粪便隐血 阳性 参考值 阴性"
        request = self._fallback_request(text)
        result = agent.run(request)
        joined = " ".join(o.statement for o in result.observations)
        self.assertIn("粪便隐血=阳性", joined, "单位为空时不应补造 g/L 等单位")
        self.assertIn("[遮挡]", joined, "遮挡内容应原样保留")
        self.assertTrue(result.data.get("redactions"), "遮挡信息应进入结构化结果")
        self.assertTrue(result.needs_confirmation)
        self.assertIn("report.diagnosis", result.confirmation_fields)

    def test_suspected_diagnosis_not_upgraded(self) -> None:
        agent, _client = self._agent()
        result = agent.run(self._fallback_request("考虑肠易激综合征"))
        diagnosis_obs = [o for o in result.observations if "肠道" in o.statement or "肠易激" in o.statement]
        self.assertTrue(diagnosis_obs)
        statement = diagnosis_obs[0].statement
        self.assertIn("疑似", statement)
        self.assertNotIn("原文明确诊断", statement)

    def test_missing_text_degrades_without_calling_model(self) -> None:
        agent, client = self._agent()
        result = agent.run(self._fallback_request("   "))
        self.assertEqual(result.status, "insufficient_data")
        self.assertEqual(client.calls, [], "没有文本时不应调用模型")
        codes = {u.code for u in result.uncertainties}
        self.assertIn("no_text", codes)

    def test_text_file_input(self) -> None:
        agent, client = self._agent()
        path = Path(self.tmp.name) / "ocr.txt"
        path.write_text("粪便隐血 阳性", encoding="utf-8")
        from gutcare_agents.contract import TaskKind, TaskRequest

        request = TaskRequest(
            task_id="t-text-file",
            kind=TaskKind.REPORT_TEXT_EXTRACT,
            patient_id="EXT-assets",
            files=[{"file_id": "ocr_doc", "path": str(path), "mime": "text/plain", "kind": "report"}],
            options={"source_id": "ocr_doc"},
        )
        result = agent.run(request)
        self.assertEqual(len(client.calls), 1)
        self.assertTrue(result.observations)


if __name__ == "__main__":
    unittest.main(verbosity=2)
