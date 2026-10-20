"""新图像候选和 PDF 双路径的边界回归。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from gutcare_agents.agents.medical import MedicalAgent
from gutcare_agents.agents.stool import StoolAgent
from gutcare_agents.config import load_settings
from gutcare_agents.contract import Status, TaskRequest
from gutcare_agents.contract import TaskRun
from gutcare_agents.llm import LLMResponse
from gutcare_agents.orchestrator import Orchestrator
from gutcare_agents.pdf_input import PdfInput, inspect_pdf, render_pdf


class CaptureClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.payload, LLMResponse(text="{}", model="test-vision")


class TestStoolImage(unittest.TestCase):
    def setUp(self):
        self.settings = load_settings(mode="mock")
        self.request = TaskRequest(task_id="stool_photo", kind="stool_image_recognize", patient_id="p", files=[{"file_id": "img1", "kind": "stool_image", "mime": "image/png", "path": "unused.png"}])

    def test_candidate_is_provisional_and_does_not_report_blood(self):
        client = CaptureClient({"quality": {"usable": True, "issue": ""}, "recognizable": True, "bristol_type": 4, "confidence": "medium", "appearance": "光滑条状", "evidence": "画面中央", "blood_reported": True})
        result = StoolAgent(self.settings, client).run(self.request)
        self.assertEqual(result.status, Status.OK)
        self.assertEqual(result.data["candidate_bristol_type"], 4)
        self.assertFalse(result.data["confirmed_by_patient"])
        self.assertTrue(result.needs_confirmation)
        self.assertNotIn("blood_reported", result.data)
        self.assertEqual(result.observations[0].source_refs[0].ref_id, "img1")
        self.assertEqual(client.calls[0]["images"][0].name, "unused.png")

    def test_unusable_or_invalid_image_has_no_candidate(self):
        for payload in (
            {"quality": {"usable": False, "issue": "模糊"}, "recognizable": True, "bristol_type": 4},
            {"quality": {"usable": True}, "recognizable": True, "bristol_type": 8},
        ):
            with self.subTest(payload=payload):
                result = StoolAgent(self.settings, CaptureClient(payload)).run(self.request)
                self.assertEqual(result.status, Status.INSUFFICIENT)
                self.assertIsNone(result.data["candidate_bristol_type"])
                self.assertFalse(result.observations)

    def test_unsafe_model_description_removed(self):
        client = CaptureClient({"quality": {"usable": True}, "recognizable": True, "bristol_type": 4, "appearance": "疑似便血", "evidence": "画面中央"})
        result = StoolAgent(self.settings, client).run(self.request)
        self.assertEqual(result.data["appearance"], "")
        self.assertTrue(any(f.code == "unsafe_content_blocked" for f in result.findings))

    def test_orchestrator_routes_stool_image_to_stool_agent(self):
        class MemoryStore:
            def cache_get(self, key):
                return None
            def cache_put(self, *args, **kwargs):
                pass
            def save_agent_result(self, result):
                pass
            def save_task(self, *args):
                pass
            def save_findings(self, *args):
                pass
            def log_usage(self, **kwargs):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            photo = root / "stool.png"
            photo.write_bytes(b"candidate input")
            settings = load_settings(mode="mock", data_dir=root)
            orchestrator = Orchestrator(settings, store=MemoryStore())
            request = TaskRequest(task_id="route_stool", kind="analysis", patient_id="p", files=[{"file_id": "img1", "kind": "stool_image", "mime": "image/png", "path": str(photo)}])
            run = TaskRun(task_id=request.task_id, kind=request.kind, patient_id=request.patient_id)
            orchestrator._stage_domain(run, request)
            self.assertIn("stool", run.agent_results)
            self.assertEqual(run.agent_results["stool"].status, Status.INSUFFICIENT)
            self.assertEqual(run.agent_results["stool"].data["candidate_bristol_type"], None)


class TestPdfPaths(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "report.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        with self.path.open("wb") as handle:
            writer.write(handle)
        self.settings = load_settings(mode="mock", fixture_root=self.root)
        self.request = TaskRequest(task_id="pdf_1", kind="report_extract", patient_id="p", files=[{"file_id": "pdf1", "kind": "report", "mime": "application/pdf", "path": str(self.path)}])

    def tearDown(self):
        self.temp.cleanup()

    def test_blank_page_uses_visual_path(self):
        parsed = inspect_pdf(self.path)
        self.assertTrue(parsed.needs_visual)
        images = render_pdf(self.path, self.root)
        self.assertEqual(len(images), 1)
        self.assertTrue(images[0].is_file())
        client = CaptureClient({"document_type": "unreadable", "exam_items": [], "conclusion": ""})
        result = MedicalAgent(self.settings, client).run(self.request)
        self.assertEqual(result.data["pdf_processing"]["mode"], "visual")
        self.assertEqual(len(client.calls[0]["images"]), 1)
        self.assertEqual(result.status, Status.INSUFFICIENT)

    def test_embedded_pdf_text_is_extracted_without_visual_fallback(self):
        writer = PdfWriter()
        page = writer.add_blank_page(width=200, height=200)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
        font_ref = writer._add_object(font)
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})})
        content = DecodedStreamObject()
        content.set_data(b"BT /F1 12 Tf 10 100 Td (Hello report, value 123) Tj ET")
        page[NameObject("/Contents")] = writer._add_object(content)
        with self.path.open("wb") as handle:
            writer.write(handle)
        parsed = inspect_pdf(self.path)
        self.assertFalse(parsed.needs_visual)
        self.assertIn("Hello report, value 123", parsed.text)

    def test_embedded_text_uses_text_model_and_original_file_reference(self):
        client = CaptureClient({"document_type": "report", "exam_items": [{"name": "项目", "value": "1", "unit": ""}], "conclusion": ""})
        with patch("gutcare_agents.agents.medical.inspect_pdf", return_value=PdfInput(2, 2, "[第 1 页] 项目 1", False)):
            result = MedicalAgent(self.settings, client).run(self.request)
        self.assertEqual(result.data["pdf_processing"]["mode"], "embedded_text")
        self.assertEqual(client.calls[0]["model"], self.settings.text_model)
        self.assertNotIn("images", client.calls[0])
        self.assertEqual(result.observations[0].source_refs[0].locator, "p1-p2")
        self.assertEqual(result.observations[0].source_refs[0].ref_id, "pdf1")

    def test_ocr_from_embedded_text_does_not_call_model(self):
        self.request.kind = "report_ocr"
        client = CaptureClient({})
        with patch("gutcare_agents.agents.medical.inspect_pdf", return_value=PdfInput(1, 1, "[第 1 页] 原文", False)):
            result = MedicalAgent(self.settings, client).run(self.request)
        self.assertEqual(result.data["transcript"], "[第 1 页] 原文")
        self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
