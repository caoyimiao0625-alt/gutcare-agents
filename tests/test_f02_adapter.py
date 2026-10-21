"""F02 agent-side dispatch, provenance mapping and callback recovery."""
from __future__ import annotations

import base64
import io
import json
import tempfile
import unittest
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from gutcare_agents.api import make_server
from gutcare_agents.config import load_settings
from gutcare_agents.contract import Status
from gutcare_agents.f02_adapter import (
    F02ContractError, F02Journal, map_medical_result, process_medical_task,
    validate_input,
)


class F02AdapterTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.request_id = str(uuid.uuid4())
        self.document_id = str(uuid.uuid4())
        self.dispatch = {"request_id": self.request_id, "schema_version": "1.0"}
        self.raw = {"request_id": self.request_id, "document_id": self.document_id,
                    "document_version": 1, "document_type": "lab_report", "document_date": None,
                    "content_type": "application/pdf", "file_name": "report.pdf",
                    "content_base64": base64.b64encode(b"%PDF-1.4\nsynthetic-test").decode("ascii")}

    def test_input_rejects_mismatched_or_invalid_document(self):
        validate_input(self.raw, self.request_id)
        with self.assertRaises(F02ContractError):
            validate_input({**self.raw, "request_id": str(uuid.uuid4())}, self.request_id)
        with self.assertRaises(F02ContractError):
            validate_input({**self.raw, "content_base64": "!!"}, self.request_id)
        with self.assertRaises(F02ContractError):
            validate_input({**self.raw, "content_base64": base64.b64encode(b"bad").decode()}, self.request_id)

    def test_candidate_mapping_is_backend_shaped_and_skips_unlocated_facts(self):
        result = SimpleNamespace(status=Status.OK, data={"exam_items": [
            {"name": "白细胞", "value": "5.0", "unit": "10^9/L", "raw_text": "白细胞 5.0 10^9/L"}],
            "diagnosis_candidates": [
                {"name": "疑似肠炎", "status": "suspected", "evidence": "考虑肠炎"},
                {"name": "仅提及", "status": "mentioned", "evidence": "仅提及"}],
            "medications": [{"name": "测试药名", "dose_text": "半片", "raw_text": "测试药名 半片"}]})
        payload = map_medical_result(self.request_id, result)
        self.assertEqual(payload["parse_outcome"], "extracted")
        self.assertEqual([c["field_code"] for c in payload["candidates"]],
                         ["test_item", "test_result", "diagnosis", "medication_name"])
        self.assertEqual(payload["candidates"][2]["value"]["certainty"], "suspected")
        self.assertEqual(payload["candidates"][0]["source_locator"], {"page": 1})
        self.assertEqual(payload["candidates"][1]["unit"], "10^9/L")
        self.assertNotIn("medication_dose", [c["field_code"] for c in payload["candidates"]])
        result.data["pdf_processing"] = {"processed_pages": 2}
        self.assertEqual(map_medical_result(self.request_id, result)["parse_outcome"], "no_supported_facts")
        result.data["exam_items"][0]["source_page"] = 2
        self.assertEqual(map_medical_result(self.request_id, result)["candidates"][0]["source_locator"], {"page": 2})

    def test_saved_callback_retries_without_reanalysing(self):
        class Client:
            def __init__(self):
                self.reads = 0
                self.sent = []
                self.fail = True
            def request(self, method, path, payload=None):
                if method == "GET":
                    self.reads += 1
                    return self_raw
                self.sent.append(payload)
                if self.fail:
                    raise URLError("lost response")
                return {}
        self_raw = self.raw
        client = Client()
        journal = F02Journal(self.root / "jobs.sqlite3")
        result = SimpleNamespace(status=Status.OK, error=None, data={"exam_items": [{
            "name": "白细胞", "value": "5.0", "unit": "10^9/L", "raw_text": "白细胞 5.0 10^9/L"}]})
        with patch("gutcare_agents.f02_adapter._run_medical", return_value=result) as run:
            with self.assertRaises(URLError):
                process_medical_task(client, None, self.dispatch, journal=journal, retry_delay=0)
            self.assertEqual(run.call_count, 1)
            client.fail = False
            process_medical_task(client, None, self.dispatch, journal=F02Journal(journal.db_path), retry_delay=0)
            self.assertEqual(run.call_count, 1)
        self.assertEqual(client.reads, 1)
        self.assertEqual(client.sent[0], client.sent[-1])
        self.assertEqual(journal.get(self.request_id)["state"], "completed")
        self.assertNotIn("content_base64", json.dumps(journal.get(self.request_id)))

    def test_stale_input_stops_without_callback(self):
        class Client:
            def request(self, method, path, payload=None):
                raise HTTPError(path, 404, "closed", {}, io.BytesIO(b"{}"))
        journal = F02Journal(self.root / "jobs.sqlite3")
        self.assertIsNone(process_medical_task(Client(), None, self.dispatch, journal=journal))
        self.assertEqual(journal.get(self.request_id)["state"], "stopped")

    def test_bad_input_uses_failure_callback_and_does_not_run_model(self):
        bad = {**self.raw, "content_base64": base64.b64encode(b"not a PDF").decode()}
        sent = []
        class Client:
            def request(self, method, path, payload=None):
                if method == "GET":
                    return bad
                sent.append((path, payload))
                return {}
        with patch("gutcare_agents.f02_adapter._run_medical") as run:
            process_medical_task(Client(), None, self.dispatch,
                                 journal=F02Journal(self.root / "bad.sqlite3"), retry_delay=0)
            run.assert_not_called()
        self.assertEqual(sent[0][1]["error_code"], "invalid_document")
        self.assertTrue(sent[0][0].endswith("/fail"))

    def test_empty_outcome_is_explicit(self):
        empty = SimpleNamespace(status=Status.OK, data={})
        payload = map_medical_result(self.request_id, empty)
        self.assertEqual((payload["parse_outcome"], payload["reason_code"], payload["candidates"]),
                         ("no_supported_facts", "no_supported_fields", []))

    def test_http_dispatch_to_backend_callback(self):
        raw = self.raw
        received = []
        class Business(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass
            def do_GET(self):
                if self.path != f"/internal/v1/agent/medical-parse-requests/{raw['request_id']}/inputs":
                    self.send_error(404)
                    return
                body = json.dumps(raw).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(204)
                self.end_headers()
        backend = ThreadingHTTPServer(("127.0.0.1", 0), Business)
        self.addCleanup(backend.server_close)
        self.addCleanup(backend.shutdown)
        import threading
        threading.Thread(target=backend.serve_forever, daemon=True).start()
        settings = load_settings(mode="mock", http_host="127.0.0.1", http_port=0,
                                 internal_token="agent-test-token", backend_service_token="backend-test-token",
                                 backend_base_url=f"http://127.0.0.1:{backend.server_address[1]}",
                                 db_path=self.root / "agent.sqlite3", data_dir=self.root / "data",
                                 replay_dir=self.root / "replays")
        agent = make_server(settings)
        self.addCleanup(agent.server_close)
        self.addCleanup(agent.shutdown)
        threading.Thread(target=agent.serve_forever, daemon=True).start()
        fake_result = SimpleNamespace(status=Status.OK, error=None, data={"exam_items": [{
            "name": "白细胞", "value": "5.0", "unit": "10^9/L", "raw_text": "白细胞 5.0 10^9/L"}]})
        with patch("gutcare_agents.f02_adapter._run_medical", return_value=fake_result):
            request = Request(f"http://127.0.0.1:{agent.server_address[1]}/internal/v1/medical-parse-requests",
                              data=json.dumps(self.dispatch).encode(), method="POST",
                              headers={"Authorization": "Bearer agent-test-token",
                                       "Idempotency-Key": self.request_id, "Content-Type": "application/json"})
            with urlopen(request, timeout=10) as response:
                self.assertEqual(response.status, 202)
            import time
            for _ in range(100):
                if received:
                    break
                time.sleep(0.01)
        self.assertEqual(received[0]["parse_outcome"], "extracted")
        self.assertEqual(received[0]["candidates"][0]["field_code"], "test_item")


if __name__ == "__main__":
    unittest.main()
