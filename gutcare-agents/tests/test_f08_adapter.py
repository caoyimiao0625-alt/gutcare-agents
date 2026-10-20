"""F08 contract checks and a real localhost HTTP claim/dispatch/callback/query cycle."""
from __future__ import annotations

import json
import io
import shutil
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
import urllib.error
import uuid
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gutcare_agents.api import make_server  # noqa: E402
from gutcare_agents.config import load_settings  # noqa: E402
from gutcare_agents.f08_adapter import (BackendClient, F08ContractError, F08Journal,
                                         adapt_inputs, map_result, process_task)  # noqa: E402

TASK_ID = str(uuid.uuid4())
PATIENT_ID = str(uuid.uuid4())
TOKEN = "f08-test-token"


def frozen_input() -> dict:
    records = []
    for index, kind in enumerate(("diet_record", "symptom_record", "stool_record", "diet_record", "symptom_record")):
        record_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"f08-test-{index}"))
        records.append({"record_type": kind, "record_id": record_id, "record_version": 1,
                        "occurred_at": f"2026-10-{10 + index}T12:00:00+08:00",
                        "confirmation_status": "confirmed", "locator": f"{kind}:{record_id}",
                        "payload": {"symptoms": [], "items": []}})
    return {"task_id": TASK_ID, "patient_id": PATIENT_ID,
            "analysis_type": "comprehensive_health_analysis", "schema_version": "1.0",
            "snapshot_hash": "a" * 64, "time_range": None,
            "consent_scope": ["health_data_processing", "third_party_ai"],
            "consent_scope_snapshot": {"health_data_processing": True,
                                       "mood_data_processing": False, "mood_ai_sharing": False,
                                       "third_party_ai": True},
            "records": records, "mood_records": [], "missing_data": []}


class FakeBusiness(BaseHTTPRequestHandler):
    frozen = frozen_input()
    status = "pending"
    result_payload = None
    callbacks = 0
    error = None

    def log_message(self, *args):
        pass

    def reply(self, status, body):
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def authorized(self):
        return self.headers.get("Authorization") == f"Bearer {TOKEN}"

    def do_GET(self):
        if self.path == f"/api/v1/analysis-tasks/{TASK_ID}":
            if self.headers.get("Authorization") != "Bearer patient-test-token":
                return self.reply(401, {})
            return self.reply(200, {"id": TASK_ID, "status": self.status,
                                    "result": self.result_payload, "error_code": self.error})
        if not self.authorized():
            return self.reply(401, {})
        if self.path == f"/internal/v1/agent/analysis-tasks/{TASK_ID}/inputs":
            return self.reply(200, self.frozen) if self.status == "processing" else self.reply(404, {})
        return self.reply(404, {})

    def do_POST(self):
        if not self.authorized():
            return self.reply(401, {})
        if self.path == "/internal/v1/agent/analysis-tasks/claim":
            if self.status != "pending":
                return self.reply(200, {"task": None})
            type(self).status = "processing"
            return self.reply(200, {"task": {"task_id": TASK_ID,
                                             "analysis_type": "comprehensive_health_analysis", "attempt": 1}})
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path == f"/internal/v1/agent/analysis-tasks/{TASK_ID}/result":
            required = {"callback_id", "status", "schema_version", "summary", "observations",
                        "evidence", "uncertainties", "missing_data"}
            if set(payload) != required or payload["schema_version"] != "1.0":
                return self.reply(422, {"code": "VALIDATION_ERROR"})
            refs = {(row["record_type"], row["record_id"], row["record_version"], row["locator"])
                    for row in self.frozen["records"] + self.frozen["mood_records"]}
            if any((e["source_type"], e["source_id"], e["source_version"], e["locator"]) not in refs
                   for e in payload["evidence"]):
                return self.reply(422, {"code": "INVALID_EVIDENCE"})
            if self.status != "processing":
                return self.reply(409, {"code": "TASK_STATE_CONFLICT"})
            type(self).status = payload["status"]
            type(self).result_payload = payload
            type(self).callbacks += 1
            return self.reply(200, {})
        if self.path == f"/internal/v1/agent/analysis-tasks/{TASK_ID}/fail":
            type(self).status = "failed"
            type(self).error = payload["error_code"]
            return self.reply(200, {})
        return self.reply(404, {})


class F08Tests(unittest.TestCase):
    def _job_fixture(self):
        frozen = frozen_input()
        temp = Path(tempfile.mkdtemp(prefix="f08-journal-"))
        journal = F08Journal(temp / "jobs.sqlite3")
        dispatch = {"task_id": TASK_ID, "analysis_type": frozen["analysis_type"], "schema_version": "1.0"}
        journal.accept(dispatch)

        class Client:
            def __init__(self):
                self.inputs_count = 0
                self.result_payloads = []
                self.fail_payloads = []
                self.result_errors = []
                self.input_errors = []

            def inputs(self, task_id):
                self.inputs_count += 1
                if self.input_errors:
                    raise self.input_errors.pop(0)
                return frozen

            def result(self, task_id, payload):
                self.result_payloads.append(deepcopy(payload))
                if self.result_errors:
                    raise self.result_errors.pop(0)
                return {}

            def fail(self, task_id, code):
                self.fail_payloads.append(code)
                return {}

        class Engine:
            def __init__(self):
                self.count = 0

            def run(self, request):
                self.count += 1
                return SimpleNamespace(task_id=TASK_ID, status="succeeded",
                                       report={"observations": [{"statement": "已确认的观察",
                                                                 "source_record_ids": [frozen["records"][0]["record_id"]]}]})

        return temp, journal, dispatch, Client(), Engine()

    def test_callback_timeout_retries_exact_payload_without_fail(self):
        temp, journal, dispatch, client, engine = self._job_fixture()
        try:
            client.result_errors = [urllib.error.URLError("timeout")]
            process_task(client, engine, dispatch, journal=journal, retry_delay=0)
            self.assertEqual(engine.count, 1)
            self.assertEqual(len(client.result_payloads), 2)
            self.assertEqual(client.result_payloads[0], client.result_payloads[1])
            self.assertEqual(client.fail_payloads, [])
            self.assertEqual(journal.get(TASK_ID)["state"], "completed")
        finally:
            shutil.rmtree(temp)

    def test_restart_replays_saved_callback_without_reanalysis(self):
        temp, journal, dispatch, client, engine = self._job_fixture()
        try:
            client.result_errors = [urllib.error.URLError("timeout")] * 3
            with self.assertRaises(urllib.error.URLError):
                process_task(client, engine, dispatch, journal=journal, retry_delay=0)
            self.assertEqual(journal.get(TASK_ID)["state"], "callback_pending")
            first_payload = deepcopy(client.result_payloads[0])
            resumed_journal = F08Journal(temp / "jobs.sqlite3")
            process_task(client, engine, dispatch, journal=resumed_journal, retry_delay=0)
            self.assertEqual(engine.count, 1)
            self.assertEqual(client.inputs_count, 1)
            self.assertEqual(client.result_payloads[-1], first_payload)
            self.assertEqual(resumed_journal.get(TASK_ID)["state"], "completed")
        finally:
            shutil.rmtree(temp)

    def test_revoked_consent_stops_without_failure_callback(self):
        temp, journal, dispatch, client, engine = self._job_fixture()
        try:
            client.input_errors = [urllib.error.HTTPError("test", 409, "CONSENT_REQUIRED",
                                                          {}, io.BytesIO(b'{"code":"CONSENT_REQUIRED"}'))]
            process_task(client, engine, dispatch, journal=journal, retry_delay=0)
            self.assertEqual(engine.count, 0)
            self.assertEqual(client.fail_payloads, [])
            self.assertEqual(journal.get(TASK_ID)["state"], "stopped")
        finally:
            shutil.rmtree(temp)

    def test_late_result_after_cancel_stops_without_failure_callback(self):
        temp, journal, dispatch, client, engine = self._job_fixture()
        try:
            client.result_errors = [urllib.error.HTTPError("test", 409, "TASK_STATE_CONFLICT",
                                                           {}, io.BytesIO(b'{"code":"TASK_STATE_CONFLICT"}'))]
            process_task(client, engine, dispatch, journal=journal, retry_delay=0)
            self.assertEqual(engine.count, 1)
            self.assertEqual(client.fail_payloads, [])
            self.assertEqual(journal.get(TASK_ID)["state"], "stopped")
        finally:
            shutil.rmtree(temp)

    def test_callback_id_conflict_is_reported_not_hidden(self):
        temp, journal, dispatch, client, engine = self._job_fixture()
        try:
            client.result_errors = [urllib.error.HTTPError("test", 409, "CALLBACK_ID_CONFLICT",
                                                           {}, io.BytesIO(b'{"code":"CALLBACK_ID_CONFLICT"}'))]
            with self.assertRaises(F08ContractError):
                process_task(client, engine, dispatch, journal=journal, retry_delay=0)
            self.assertEqual(client.fail_payloads, [])
            self.assertEqual(journal.get(TASK_ID)["last_error"], "CALLBACK_ID_CONFLICT")
        finally:
            shutil.rmtree(temp)

    def test_input_network_error_retries_without_false_failure(self):
        temp, journal, dispatch, client, engine = self._job_fixture()
        try:
            client.input_errors = [urllib.error.URLError("temporary outage")]
            process_task(client, engine, dispatch, journal=journal, retry_delay=0)
            self.assertEqual(client.inputs_count, 2)
            self.assertEqual(engine.count, 1)
            self.assertEqual(client.fail_payloads, [])
            self.assertEqual(journal.get(TASK_ID)["state"], "completed")
        finally:
            shutil.rmtree(temp)

    def test_analysis_failure_uses_failure_callback_once(self):
        temp, journal, dispatch, client, engine = self._job_fixture()
        try:
            def fail_run(request):
                engine.count += 1
                raise RuntimeError("model unavailable")
            engine.run = fail_run
            process_task(client, engine, dispatch, journal=journal, retry_delay=0)
            self.assertEqual(engine.count, 1)
            self.assertEqual(client.result_payloads, [])
            self.assertEqual(client.fail_payloads, ["AGENT_ANALYSIS_FAILED"])
            process_task(client, engine, dispatch, journal=journal, retry_delay=0)
            self.assertEqual(engine.count, 1)
        finally:
            shutil.rmtree(temp)

    def test_duplicate_dispatch_survives_restart_and_type_conflict(self):
        temp, journal, dispatch, client, engine = self._job_fixture()
        try:
            process_task(client, engine, dispatch, journal=journal, retry_delay=0)
            resumed_journal = F08Journal(temp / "jobs.sqlite3")
            self.assertEqual(resumed_journal.accept(dispatch), "completed")
            process_task(client, engine, dispatch, journal=resumed_journal, retry_delay=0)
            self.assertEqual(engine.count, 1)
            with self.assertRaises(F08ContractError):
                resumed_journal.accept(dict(dispatch, analysis_type="periodic_summary"))
        finally:
            shutil.rmtree(temp)

    def test_service_startup_resends_pending_callback(self):
        FakeBusiness.status = "processing"
        FakeBusiness.result_payload = None
        FakeBusiness.callbacks = 0
        business = ThreadingHTTPServer(("127.0.0.1", 0), FakeBusiness)
        threading.Thread(target=business.serve_forever, daemon=True).start()
        temp = Path(tempfile.mkdtemp(prefix="f08-startup-"))
        agent = None
        try:
            settings = load_settings(mode="mock", data_dir=temp, replay_dir=temp / "replay",
                                     db_path=temp / "tasks.sqlite3", http_host="127.0.0.1", http_port=0,
                                     internal_token=TOKEN,
                                     backend_base_url=f"http://127.0.0.1:{business.server_address[1]}",
                                     backend_service_token=TOKEN)
            journal = F08Journal(settings.db_path)
            dispatch = {"task_id": TASK_ID, "analysis_type": "comprehensive_health_analysis",
                        "schema_version": "1.0"}
            journal.accept(dispatch)
            callback = {"callback_id": str(uuid.uuid4()), "status": "partial", "schema_version": "1.0",
                        "summary": "已保存的回调", "observations": [], "evidence": [],
                        "uncertainties": [], "missing_data": []}
            journal.save_callback(TASK_ID, "result", callback)
            agent = make_server(settings)
            for _ in range(50):
                if FakeBusiness.callbacks:
                    break
                time.sleep(0.05)
            self.assertEqual(FakeBusiness.callbacks, 1)
            self.assertEqual(FakeBusiness.result_payload, callback)
            self.assertEqual(journal.get(TASK_ID)["state"], "completed")
        finally:
            if agent:
                agent.server_close()
            business.shutdown()
            business.server_close()
            shutil.rmtree(temp, ignore_errors=True)

    def test_consent_and_conflict_exclusion(self):
        frozen = frozen_input()
        conflict = deepcopy(frozen["records"][0])
        conflict["record_id"] = str(uuid.uuid4())
        conflict["record_type"] = "medical_fact_conflict"
        conflict["confirmation_status"] = "unresolved"
        frozen["records"].append(conflict)
        request, provenance = adapt_inputs(frozen, TASK_ID, frozen["analysis_type"])
        self.assertEqual(len(request.records), 5)
        self.assertNotIn(conflict["record_id"], provenance)
        frozen["mood_records"] = [dict(conflict, record_type="mood_record")]
        with self.assertRaises(F08ContractError):
            adapt_inputs(frozen, TASK_ID, frozen["analysis_type"])

    def test_unknown_evidence_dropped(self):
        frozen = frozen_input()
        _, provenance = adapt_inputs(frozen, TASK_ID, frozen["analysis_type"])
        class Run:
            task_id = TASK_ID
            status = "succeeded"
            report = {"observations": [{"statement": "outside", "source_record_ids": [str(uuid.uuid4())]},
                                       {"statement": "inside", "source_record_ids": [frozen["records"][0]["record_id"]]}]}
        result = map_result(Run(), frozen, provenance)
        self.assertEqual(result["observations"], ["inside"])
        self.assertEqual(result["evidence"][0]["source_id"], frozen["records"][0]["record_id"])

    def test_claim_analyze_callback_query(self):
        FakeBusiness.status = "pending"
        FakeBusiness.result_payload = None
        FakeBusiness.callbacks = 0
        FakeBusiness.error = None
        business = ThreadingHTTPServer(("127.0.0.1", 0), FakeBusiness)
        business_thread = threading.Thread(target=business.serve_forever, daemon=True)
        business_thread.start()
        temp = Path(tempfile.mkdtemp(prefix="f08-"))
        agent = None
        try:
            business_url = f"http://127.0.0.1:{business.server_address[1]}"
            settings = load_settings(mode="mock", data_dir=temp, replay_dir=temp / "replay",
                                     db_path=temp / "tasks.sqlite3", http_host="127.0.0.1", http_port=0,
                                     internal_token=TOKEN, backend_base_url=business_url,
                                     backend_service_token=TOKEN)
            agent = make_server(settings)
            threading.Thread(target=agent.serve_forever, daemon=True).start()
            client = BackendClient(business_url, TOKEN)
            claimed = client.claim()
            self.assertEqual(FakeBusiness.status, "processing")
            dispatch = {"task_id": claimed["task_id"], "analysis_type": claimed["analysis_type"],
                        "schema_version": "1.0"}
            req = urllib.request.Request(f"http://127.0.0.1:{agent.server_address[1]}/internal/v1/tasks",
                                         data=json.dumps(dispatch).encode(), method="POST",
                                         headers={"Authorization": f"Bearer {TOKEN}",
                                                  "Idempotency-Key": TASK_ID,
                                                  "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=20) as response:
                self.assertEqual(response.status, 202)
                self.assertTrue(json.load(response)["accepted"])
            for _ in range(100):
                if FakeBusiness.status not in {"pending", "processing"}:
                    break
                time.sleep(0.1)
            patient_request = urllib.request.Request(
                f"{business_url}/api/v1/analysis-tasks/{TASK_ID}",
                headers={"Authorization": "Bearer patient-test-token"})
            with urllib.request.urlopen(patient_request, timeout=20) as response:
                query = json.load(response)
            self.assertIn(query["status"], {"succeeded", "partial", "needs_input"})
            self.assertEqual(FakeBusiness.callbacks, 1)
            self.assertEqual(query["result"]["status"], query["status"])
            self.assertGreater(len(query["result"]["evidence"]), 0)
        finally:
            if agent:
                agent.shutdown()
                agent.server_close()
            business.shutdown()
            business.server_close()
            shutil.rmtree(temp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
