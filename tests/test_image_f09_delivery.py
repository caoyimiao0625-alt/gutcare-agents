"""Agent-side F03/F04/F09 contract and durable callback checks."""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from gutcare_agents.api import make_server
from gutcare_agents.config import load_settings
from gutcare_agents.contract import TaskStatus
from gutcare_agents.f08_adapter import F08Journal, process_task
from gutcare_agents.f09_delivery import F09Outbox, deliver_f09
from gutcare_agents.image_adapter import (ImageContractError, ImageJournal, _run_one,
                                          map_image_result, process_image_task, validate_dispatch)


class ImageAndAdviceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / "jobs.sqlite3"
        self.request_id = str(uuid.uuid4())
        self.attachment_id = str(uuid.uuid4())

    @staticmethod
    def _http_error(code):
        return HTTPError("http://local.invalid", code, "test", {}, io.BytesIO(b'{}'))

    def _image_dispatch_and_input(self, recognition_type="diet_image_recognition", count=1):
        dispatch = {"request_id": self.request_id, "recognition_type": recognition_type,
                    "attempt": 1, "retry_of_request_id": None, "schema_version": "1.0"}
        attachments = [{"attachment_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.request_id}:{index}")),
                        "file_version": 1, "content_type": "image/png", "image_capability": "x" * 40}
                       for index in range(count)]
        raw = {**dispatch, "record_id": str(uuid.uuid4()), "record_version": 2,
               "image_set_version": 3, "attachments": attachments}
        return dispatch, raw

    def test_diet_candidates_are_grouped_and_not_facts(self):
        result = SimpleNamespace(data={"foods": [{"name": "米饭", "category": "grain_staple",
                                                    "confidence": "medium"},
                                                   {"name": "不确定菜", "category": "未知", "confidence": "low"}]})
        image = map_image_result(self.request_id, "diet_image_recognition", self.attachment_id, result)
        self.assertEqual(image["image_quality"], "usable")
        self.assertEqual([item["candidate_type"] for item in image["candidates"]],
                         ["food_name", "food_category", "food_name"])
        self.assertEqual(image["candidates"][0]["candidate_group_key"],
                         image["candidates"][1]["candidate_group_key"])
        self.assertNotEqual(image["candidates"][0]["candidate_group_key"],
                            image["candidates"][2]["candidate_group_key"])
        self.assertEqual(image, map_image_result(self.request_id, "diet_image_recognition",
                                                 self.attachment_id, result))

    def test_stool_unusable_never_emits_bristol(self):
        result = SimpleNamespace(data={"image_quality": {"usable": False},
                                       "candidate_bristol_type": 4, "confidence": "high"})
        image = map_image_result(self.request_id, "stool_image_recognition", self.attachment_id, result)
        self.assertEqual(image["candidates"], [])
        self.assertEqual(image["reason_code"], "low_image_quality")

    def test_image_callback_is_replayed_without_model_rerun(self):
        dispatch = {"request_id": self.request_id, "recognition_type": "stool_image_recognition",
                    "attempt": 1, "retry_of_request_id": None, "schema_version": "1.0"}
        raw = {**dispatch, "record_id": str(uuid.uuid4()), "record_version": 2,
               "image_set_version": 3, "attachments": [{"attachment_id": self.attachment_id,
               "file_version": 1, "content_type": "image/png", "image_capability": "x" * 40}]}
        class Client:
            def __init__(self):
                self.sent = []
                self.reads = 0
            def request(self, method, path, payload=None):
                if method == "GET":
                    self.reads += 1
                    return raw
                self.sent.append((path, payload))
                if len(self.sent) == 1:
                    raise URLError("lost response")
                return {}
        client = Client()
        journal = ImageJournal(self.db)
        settings = SimpleNamespace(max_image_bytes=1000)
        image = {"attachment_id": self.attachment_id, "image_quality": "uncertain",
                 "reason_code": "no_reliable_stool_type", "candidates": []}
        with patch("gutcare_agents.image_adapter._download", return_value=b"data") as download, \
             patch("gutcare_agents.image_adapter._run_one", return_value=image) as run:
            payload = process_image_task(client, settings, dispatch, journal=journal, retry_delay=0)
            self.assertEqual(client.reads, 1)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(download.call_count, 1)
            self.assertEqual(client.sent[0][1], client.sent[1][1])
            self.assertEqual(payload["images"], [image])
            self.assertEqual(journal.get(self.request_id)["state"], "completed")
            self.assertIsNone(process_image_task(client, settings, dispatch, journal=journal))
            self.assertEqual(run.call_count, 1)

    def test_restart_replays_saved_image_callback_without_model(self):
        dispatch, raw = self._image_dispatch_and_input()
        class Client:
            def __init__(self, fail):
                self.fail = fail
                self.callbacks = []
            def request(self, method, path, payload=None):
                if method == "GET":
                    return raw
                self.callbacks.append(payload)
                if self.fail:
                    raise URLError("callback response lost")
                return {}
        first = Client(True)
        image = {"attachment_id": raw["attachments"][0]["attachment_id"],
                 "image_quality": "uncertain", "reason_code": "no_visible_food", "candidates": []}
        with patch("gutcare_agents.image_adapter._download", return_value=b"image"), \
             patch("gutcare_agents.image_adapter._run_one", return_value=image) as run:
            with self.assertRaises(URLError):
                process_image_task(first, SimpleNamespace(max_image_bytes=100), dispatch,
                                   journal=ImageJournal(self.db), retry_delay=0)
            self.assertEqual(run.call_count, 1)
            after_restart = Client(False)
            result = process_image_task(after_restart, SimpleNamespace(max_image_bytes=100), dispatch,
                                        journal=ImageJournal(self.db), retry_delay=0)
            self.assertEqual(run.call_count, 1)
        self.assertEqual(result, first.callbacks[0])
        self.assertEqual(ImageJournal(self.db).get(self.request_id)["state"], "completed")

    def test_f09_outbox_retries_same_payload(self):
        outbox = F09Outbox(self.db)
        from tests.test_f09_adapter import fixture
        run, frozen = fixture()
        outbox.prepare(run, frozen, "partial")
        self.request_id = run.task_id
        class Client:
            calls = []
            def request(self, method, path, body):
                self.calls.append((path, body))
                if len(self.calls) == 1:
                    raise URLError("lost response")
                return {}
        client = Client()
        self.assertTrue(deliver_f09(client, outbox, self.request_id, retry_delay=0))
        self.assertEqual(client.calls[0], client.calls[1])
        self.assertEqual(outbox.get(self.request_id)[0], "sent")
        self.assertFalse(deliver_f09(client, outbox, self.request_id, retry_delay=0))

    def test_f08_result_then_independent_f09_callback(self):
        from tests.test_f09_adapter import fixture
        run, frozen = fixture()
        run.status = TaskStatus.PARTIAL
        run.report["observations"][0]["statement"] = "已确认排便记录可供后续观察。"
        run.report["narrative_summary"] = "记录较少，建议继续记录。"
        dispatch = {"task_id": run.task_id, "analysis_type": frozen["analysis_type"],
                    "schema_version": "1.0"}
        outbox = F09Outbox(self.db)
        journal = F08Journal(self.db)
        class Client:
            events = []
            def inputs(self, task_id):
                return frozen
            def result(self, task_id, payload):
                self.events.append(("f08", payload))
                return {}
            def request(self, method, path, body):
                self.events.append(("f09", body))
                return {}
        client = Client()
        orchestrator = SimpleNamespace(run=lambda request: run)
        f08 = process_task(client, orchestrator, dispatch, journal=journal,
                           on_result=outbox.prepare, retry_delay=0)
        self.assertEqual(f08["status"], "partial")
        self.assertTrue(deliver_f09(client, outbox, run.task_id, retry_delay=0))
        self.assertEqual([item[0] for item in client.events], ["f08", "f09"])
        self.assertEqual(client.events[1][1]["advice"][0]["evidence"][0]["source_id"],
                         frozen["records"][0]["record_id"])

    def test_dispatch_requires_backend_schema(self):
        good = {"request_id": self.request_id, "recognition_type": "diet_image_recognition",
                "attempt": 1, "retry_of_request_id": None, "schema_version": "1.0"}
        self.assertEqual(validate_dispatch(good), self.request_id)
        with self.assertRaises(ValueError):
            validate_dispatch({**good, "patient_id": str(uuid.uuid4())})

    def test_duplicate_dispatch_and_changed_payload_conflict(self):
        dispatch, _ = self._image_dispatch_and_input()
        journal = ImageJournal(self.db)
        self.assertEqual(journal.accept(dispatch), "accepted")
        self.assertEqual(journal.accept(dict(dispatch)), "accepted")
        with self.assertRaises(ImageContractError):
            journal.accept({**dispatch, "attempt": 2})

    def test_backend_invalidates_image_task_without_callback(self):
        dispatch, _ = self._image_dispatch_and_input()
        class Client:
            def request(self, method, path, payload=None):
                if method == "GET":
                    raise ImageAndAdviceTests._http_error(409)
                raise AssertionError("cancelled task must not receive a callback")
        journal = ImageJournal(self.db)
        self.assertIsNone(process_image_task(Client(), SimpleNamespace(max_image_bytes=100),
                                             dispatch, journal=journal, retry_delay=0))
        self.assertEqual(journal.get(self.request_id)["state"], "stopped")

    def test_image_callback_conflict_stops_without_reanalysis(self):
        dispatch, raw = self._image_dispatch_and_input()
        class Client:
            calls = 0
            def request(self, method, path, payload=None):
                if method == "GET":
                    return raw
                self.calls += 1
                raise ImageAndAdviceTests._http_error(409)
        client = Client()
        journal = ImageJournal(self.db)
        with patch("gutcare_agents.image_adapter._download", return_value=b"image"), \
             patch("gutcare_agents.image_adapter._run_one", return_value={
                 "attachment_id": raw["attachments"][0]["attachment_id"],
                 "image_quality": "uncertain", "reason_code": "no_visible_food", "candidates": []}) as run:
            self.assertIsNone(process_image_task(client, SimpleNamespace(max_image_bytes=100),
                                                 dispatch, journal=journal, retry_delay=0))
            self.assertIsNone(process_image_task(client, SimpleNamespace(max_image_bytes=100),
                                                 dispatch, journal=journal, retry_delay=0))
            self.assertEqual(run.call_count, 1)
        self.assertEqual(client.calls, 1)
        self.assertEqual(journal.get(self.request_id)["state"], "stopped")

    def test_f09_backend_conflict_stops_retries(self):
        from tests.test_f09_adapter import fixture
        run, frozen = fixture()
        outbox = F09Outbox(self.db)
        outbox.prepare(run, frozen, "partial")
        class Client:
            calls = 0
            def request(self, method, path, body):
                self.calls += 1
                raise ImageAndAdviceTests._http_error(409)
        client = Client()
        self.assertFalse(deliver_f09(client, outbox, run.task_id, retry_delay=0))
        self.assertFalse(deliver_f09(client, outbox, run.task_id, retry_delay=0))
        self.assertEqual(client.calls, 1)
        self.assertEqual(outbox.get(run.task_id)[0], "stopped")

    def test_transient_image_read_refreshes_capabilities(self):
        dispatch, raw = self._image_dispatch_and_input(count=2)
        class Client:
            reads = 0
            callbacks = []
            def request(self, method, path, payload=None):
                if method == "GET":
                    self.reads += 1
                    return raw
                self.callbacks.append(payload)
                return {}
        client = Client()
        journal = ImageJournal(self.db)
        returned = []
        def download(*args):
            if len(returned) == 1:
                returned.append(None)
                raise URLError("temporary image gateway failure")
            returned.append(b"image")
            return b"image"
        def run_one(settings, request_id, recognition_type, item, image):
            return {"attachment_id": item["attachment_id"], "image_quality": "uncertain",
                    "reason_code": "no_visible_food", "candidates": []}
        with patch("gutcare_agents.image_adapter._download", side_effect=download), \
             patch("gutcare_agents.image_adapter._run_one", side_effect=run_one):
            payload = process_image_task(client, SimpleNamespace(max_image_bytes=100),
                                         dispatch, journal=journal, retry_delay=0)
        self.assertEqual(client.reads, 2)
        self.assertEqual(len(payload["images"]), 2)
        self.assertEqual(journal.get(self.request_id)["state"], "completed")

    def test_ten_images_mock_finish_within_lease_and_all_fetched_first(self):
        dispatch, raw = self._image_dispatch_and_input(count=10)
        png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l8sAAAAASUVORK5CYII=")
        class Client:
            def request(self, method, path, payload=None):
                return raw if method == "GET" else {}
        settings = load_settings(mode="mock", db_path=self.db)
        downloaded = 0
        first_run_download_count = []
        def download(*args):
            nonlocal downloaded
            downloaded += 1
            return png
        def run_one(*args):
            first_run_download_count.append(downloaded)
            return _run_one(*args)
        start = time.monotonic()
        with patch("gutcare_agents.image_adapter._download", side_effect=download), \
             patch("gutcare_agents.image_adapter._run_one", side_effect=run_one):
            result = process_image_task(Client(), settings, dispatch, journal=ImageJournal(self.db), retry_delay=0)
        elapsed = time.monotonic() - start
        self.assertEqual(downloaded, 10)
        self.assertEqual(first_run_download_count, [10] * 10)
        self.assertEqual(len(result["images"]), 10)
        self.assertLess(elapsed, 300)

    def test_current_backend_pydantic_callback_models(self):
        source = os.environ.get("GCA_BACKEND_SOURCE")
        if not source:
            self.skipTest("set GCA_BACKEND_SOURCE to validate current backend models")
        sys.path.insert(0, str(Path(source) / "backend" / "src"))
        from gutcare.records.image_worker import CandidateSubmission, RecognitionFailure
        from gutcare.analysis.advice import AdviceCallbackBody
        from tests.test_f09_adapter import fixture

        run, frozen = fixture()
        outbox = F09Outbox(self.db)
        outbox.prepare(run, frozen, "partial")
        AdviceCallbackBody.model_validate(outbox.get(run.task_id)[1])

        for recognition_type, result in (
            ("diet_image_recognition", SimpleNamespace(data={"foods": [
                {"name": "米饭", "category": "grain_staple", "confidence": "medium"}]})),
            ("stool_image_recognition", SimpleNamespace(data={"image_quality": {"usable": True},
                                                         "candidate_bristol_type": 4, "confidence": "high"})),
        ):
            image = map_image_result(self.request_id, recognition_type, self.attachment_id, result)
            body = {"callback_id": str(uuid.uuid4()), "record_version": 1,
                    "image_set_version": 2, "schema_version": "1.0", "images": [image]}
            CandidateSubmission.model_validate(body)
            invalid = deepcopy(body)
            invalid["images"][0]["candidates"][0]["attachment_id"] = "not-a-uuid"
            with self.assertRaises(Exception):
                CandidateSubmission.model_validate(invalid)
        RecognitionFailure.model_validate({"callback_id": str(uuid.uuid4()),
                                           "error_code": "invalid_image"})

    def test_real_local_http_image_dispatch_to_callback(self):
        token = "local-inbound-token"
        service_token = "local-backend-token"
        dispatch = {"request_id": self.request_id, "recognition_type": "diet_image_recognition",
                    "attempt": 1, "retry_of_request_id": None, "schema_version": "1.0"}
        frozen = {**dispatch, "record_id": str(uuid.uuid4()), "record_version": 1,
                  "image_set_version": 1, "attachments": [{"attachment_id": self.attachment_id,
                  "file_version": 1, "content_type": "image/png", "image_capability": "z" * 40}]}
        png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l8sAAAAASUVORK5CYII=")
        class Backend(BaseHTTPRequestHandler):
            callback = None
            def log_message(self, *args):
                pass
            def do_GET(self):
                if self.headers.get("Authorization") != f"Bearer {service_token}":
                    self.send_error(401)
                    return
                if self.path.endswith("/input"):
                    body = json.dumps(frozen).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif "/images/" in self.path:
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(png)))
                    self.end_headers()
                    self.wfile.write(png)
                else:
                    self.send_error(404)
            def do_POST(self):
                if self.headers.get("Authorization") != f"Bearer {service_token}":
                    self.send_error(401)
                    return
                type(self).callback = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.send_response(204)
                self.end_headers()
        backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
        bt = threading.Thread(target=backend.serve_forever, daemon=True)
        bt.start()
        self.addCleanup(backend.server_close)
        self.addCleanup(backend.shutdown)
        settings = load_settings(mode="mock", db_path=self.db, http_port=0,
                                 internal_token=token, backend_service_token=service_token,
                                 backend_base_url=f"http://127.0.0.1:{backend.server_port}")
        agent = make_server(settings)
        at = threading.Thread(target=agent.serve_forever, daemon=True)
        at.start()
        self.addCleanup(agent.server_close)
        self.addCleanup(agent.shutdown)
        self.addCleanup(agent.RequestHandlerClass.state.pool.shutdown)
        request = Request(f"http://127.0.0.1:{agent.server_port}/internal/v1/record-image-recognition/requests",
                          data=json.dumps(dispatch).encode(), method="POST",
                          headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "local-key-123",
                                   "Content-Type": "application/json"})
        without_auth = Request(request.full_url, data=json.dumps(dispatch).encode(), method="POST",
                               headers={"Idempotency-Key": "local-key-123", "Content-Type": "application/json"})
        with self.assertRaises(HTTPError) as denied:
            urlopen(without_auth, timeout=5)
        self.assertEqual(denied.exception.code, 401)
        with urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 202)
        with urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 202)
        changed = Request(request.full_url, data=json.dumps({**dispatch, "attempt": 2}).encode(), method="POST",
                          headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "local-key-123",
                                   "Content-Type": "application/json"})
        with self.assertRaises(HTTPError) as conflict:
            urlopen(changed, timeout=5)
        self.assertEqual(conflict.exception.code, 409)
        for _ in range(100):
            if Backend.callback is not None:
                break
            time.sleep(0.05)
        self.assertIsNotNone(Backend.callback)
        self.assertEqual(Backend.callback["schema_version"], "1.0")
        self.assertEqual(Backend.callback["images"][0]["attachment_id"], self.attachment_id)
        self.assertEqual(agent.RequestHandlerClass.state.image_journal.get(self.request_id)["state"], "completed")


if __name__ == "__main__":
    unittest.main()
