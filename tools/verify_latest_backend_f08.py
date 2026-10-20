"""Exercise the Agent against the actual F08 FastAPI routes from a backend checkout.

The backend service is replaced with an in-memory task because this check does not
have PostgreSQL. Request validation, service bearer, route paths, status codes and
patient response serialization are provided by the real backend source.

Run with a Python environment containing the backend's dependencies:
    python tools/verify_latest_backend_f08.py --backend-checkout PATH
"""
from __future__ import annotations

import argparse
import json
import shutil
import socket
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gutcare_agents.api import make_server  # noqa: E402
from gutcare_agents.config import load_settings  # noqa: E402
from gutcare_agents.f08_adapter import BackendClient  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-checkout", type=Path, required=True)
    args = parser.parse_args()
    backend_root = args.backend_checkout.resolve()
    source = backend_root / "backend" / "src"
    if not (source / "gutcare" / "analysis" / "api.py").is_file():
        parser.error("backend checkout must contain backend/src/gutcare/analysis/api.py")
    sys.path.insert(0, str(source))

    try:
        import uvicorn
        from fastapi import FastAPI
        from gutcare.analysis.api import agent_router, router
        from gutcare.auth.service import AuthContext
        from gutcare.shared.errors import Problem, install_exception_handlers
    except ImportError as exc:
        parser.error(f"install the backend Python dependencies first: {exc}")

    task_id = str(uuid.uuid4())
    patient_id = str(uuid.uuid4())
    source_id = str(uuid.uuid4())
    service_token = f"local-{uuid.uuid4()}"
    patient_token = f"patient-{uuid.uuid4()}"
    now = datetime.now(UTC).isoformat()
    frozen = {
        "task_id": task_id, "patient_id": patient_id,
        "analysis_type": "comprehensive_health_analysis", "schema_version": "1.0",
        "snapshot_hash": "a" * 64, "time_range": None,
        "consent_scope": ["health_data_processing", "third_party_ai"],
        "consent_scope_snapshot": {
            "health_data_processing": True, "mood_data_processing": False,
            "mood_ai_sharing": False, "third_party_ai": True,
        },
        "records": [{
            "record_type": "stool_record", "record_id": source_id,
            "record_version": 1, "occurred_at": now,
            "confirmation_status": "confirmed", "locator": f"stool_record:{source_id}",
            "payload": {"bristol_type": 4},
        }],
        "mood_records": [], "missing_data": [],
    }

    class TaskService:
        status = "pending"
        result_payload = None
        callback_count = 0

        def claim(self):
            if self.status != "pending":
                return None
            self.status = "processing"
            return {"task_id": task_id, "analysis_type": frozen["analysis_type"], "attempt": 1}

        def inputs(self, requested_id):
            if str(requested_id) != task_id or self.status != "processing":
                raise Problem(404, "RESOURCE_NOT_FOUND", "Task input not available")
            return frozen

        def result(self, requested_id, callback_id, payload):
            if str(requested_id) != task_id or self.status != "processing":
                raise Problem(409, "TASK_STATE_CONFLICT", "Task is no longer accepting results")
            refs = {(r["record_type"], r["record_id"], r["record_version"], r["locator"])
                    for r in frozen["records"]}
            for item in payload["evidence"]:
                if (item["source_type"], item["source_id"], item["source_version"], item["locator"]) not in refs:
                    raise Problem(422, "INVALID_EVIDENCE", "Evidence must reference this task snapshot")
            self.status = payload["status"]
            self.result_payload = payload
            self.callback_count += 1

        def fail(self, requested_id, callback_id, error_code):
            self.status = "failed"

        def get(self, context, requested_id):
            if str(requested_id) != task_id:
                raise Problem(404, "RESOURCE_NOT_FOUND", "Task not found")
            payload = self.result_payload
            result = None
            if payload is not None:
                result = {k: v for k, v in payload.items() if k not in {"callback_id", "status"}}
                result["evidence"] = [dict(e, source_deleted=False) for e in result["evidence"]]
            return {
                "id": task_id, "analysis_type": frozen["analysis_type"], "status": self.status,
                "input_summary": {"included_mood_record_count": 0},
                "created_at": now, "updated_at": datetime.now(UTC).isoformat(),
                "attempt": 1, "retry_of": None, "result": result, "error_code": None,
            }

    class PatientAuth:
        def authenticate_access_token(self, token):
            if token != patient_token:
                raise Problem(401, "AUTHENTICATION_REQUIRED", "Authentication required")
            return AuthContext(user_id=uuid.UUID(patient_id), session_id=uuid.uuid4())

    task_service = TaskService()
    backend = FastAPI()
    backend.state.settings = SimpleNamespace(agent_service_token=service_token)
    backend.state.analysis_worker_service = task_service
    backend.state.analysis_service = task_service
    backend.state.auth_service = PatientAuth()
    install_exception_handlers(backend)
    backend.include_router(agent_router)
    backend.include_router(router)

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    backend_port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(backend, host="127.0.0.1", port=backend_port,
                                           log_level="error", access_log=False))
    server_thread = threading.Thread(target=lambda: server.run(sockets=[listener]), daemon=True)
    server_thread.start()
    temp = Path(tempfile.mkdtemp(prefix="gutcare-backend-f08-"))
    agent = None
    try:
        until = time.monotonic() + 10
        while not server.started and time.monotonic() < until:
            time.sleep(0.05)
        if not server.started:
            raise RuntimeError("backend route server did not start")
        backend_url = f"http://127.0.0.1:{backend_port}"
        unauthenticated_claim = urllib.request.Request(
            f"{backend_url}/internal/v1/agent/analysis-tasks/claim", data=b"{}", method="POST")
        try:
            urllib.request.urlopen(unauthenticated_claim, timeout=10)
            raise AssertionError("backend accepted claim without service bearer")
        except urllib.error.HTTPError as exc:
            if exc.code != 401:
                raise
        settings = load_settings(
            mode="mock", data_dir=temp, replay_dir=temp / "replay",
            db_path=temp / "agent.sqlite3", http_host="127.0.0.1", http_port=0,
            internal_token=service_token, backend_base_url=backend_url,
            backend_service_token=service_token,
        )
        agent = make_server(settings)
        threading.Thread(target=agent.serve_forever, daemon=True).start()
        claim = BackendClient(backend_url, service_token).claim()
        if claim is None or claim["task_id"] != task_id:
            raise AssertionError("actual backend claim route did not return the task")
        dispatch = {"task_id": task_id, "analysis_type": claim["analysis_type"],
                    "schema_version": "1.0"}
        request = urllib.request.Request(
            f"http://127.0.0.1:{agent.server_address[1]}/internal/v1/tasks",
            data=json.dumps(dispatch).encode(), method="POST",
            headers={"Authorization": f"Bearer {service_token}",
                     "Idempotency-Key": task_id, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            if response.status != 202 or not json.load(response).get("accepted"):
                raise AssertionError("Agent did not accept the backend dispatch")
        until = time.monotonic() + 20
        while task_service.status == "processing" and time.monotonic() < until:
            time.sleep(0.1)
        patient_query = urllib.request.Request(
            f"{backend_url}/api/v1/analysis-tasks/{task_id}",
            headers={"Authorization": f"Bearer {patient_token}"},
        )
        with urllib.request.urlopen(patient_query, timeout=10) as response:
            patient_result = json.load(response)
        no_patient_token = urllib.request.Request(f"{backend_url}/api/v1/analysis-tasks/{task_id}")
        try:
            urllib.request.urlopen(no_patient_token, timeout=10)
            raise AssertionError("backend accepted patient query without bearer")
        except urllib.error.HTTPError as exc:
            if exc.code != 401:
                raise
        if patient_result["status"] not in {"succeeded", "partial", "needs_input"}:
            raise AssertionError(f"unexpected task status: {patient_result['status']}")
        if task_service.callback_count != 1 or patient_result["result"] is None:
            raise AssertionError("result callback or patient projection missing")
        if not patient_result["result"]["evidence"]:
            raise AssertionError("Agent produced no frozen-source evidence")
        print(json.dumps({"status": "passed", "backend_routes": "actual FastAPI source",
                          "task_status": patient_result["status"],
                          "callback_count": task_service.callback_count,
                          "evidence_count": len(patient_result["result"]["evidence"])}))
        return 0
    finally:
        if agent is not None:
            agent.shutdown()
            agent.server_close()
        server.should_exit = True
        server_thread.join(timeout=5)
        listener.close()
        shutil.rmtree(temp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
