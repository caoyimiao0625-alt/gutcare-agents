"""Development bridge for a backend checkout without its F08 Worker deployment."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gutcare_agents.f08_adapter import BackendClient  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Claim one F08 task and dispatch it to the Agent service")
    parser.add_argument("--agent-url", default="http://127.0.0.1:8100")
    parser.add_argument("--wait-seconds", type=int, default=0,
                        help="With GCA_PATIENT_TOKEN, poll the patient query until terminal state")
    args = parser.parse_args()
    backend_url = os.environ.get("GCA_BACKEND_BASE_URL", "")
    backend_token = os.environ.get("GCA_BACKEND_SERVICE_TOKEN", "")
    agent_token = os.environ.get("GCA_INTERNAL_TOKEN", "")
    if not all((backend_url, backend_token, agent_token)):
        parser.error("Set GCA_BACKEND_BASE_URL, GCA_BACKEND_SERVICE_TOKEN and GCA_INTERNAL_TOKEN")
    patient_token = os.environ.get("GCA_PATIENT_TOKEN", "")
    if args.wait_seconds and not patient_token:
        parser.error("Set GCA_PATIENT_TOKEN to query the task as a patient")
    with urllib.request.urlopen(args.agent_url.rstrip("/") + "/healthz", timeout=5) as response:
        if not json.load(response).get("ok"):
            parser.error("Agent health check failed")
    task = BackendClient(backend_url, backend_token).claim()
    if task is None:
        print(json.dumps({"claimed": False}))
        return 0
    payload = {"task_id": task["task_id"], "analysis_type": task["analysis_type"],
               "schema_version": "1.0"}
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(args.agent_url.rstrip("/") + "/internal/v1/tasks",
                                     data=data, method="POST",
                                     headers={"Authorization": f"Bearer {agent_token}",
                                              "Idempotency-Key": task["task_id"],
                                              "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=20) as response:
        accepted = json.load(response)
    print(json.dumps({"claimed": True, "task_id": task["task_id"], "accepted": accepted["accepted"]}))
    if args.wait_seconds:
        until = time.monotonic() + args.wait_seconds
        while time.monotonic() < until:
            query = urllib.request.Request(
                backend_url.rstrip("/") + f"/api/v1/analysis-tasks/{task['task_id']}",
                headers={"Authorization": f"Bearer {patient_token}"})
            with urllib.request.urlopen(query, timeout=20) as response:
                result = json.load(response)
            if result.get("status") in {"succeeded", "partial", "needs_input", "failed", "cancelled"}:
                print(json.dumps({"task_id": task["task_id"], "status": result["status"],
                                  "result_present": result.get("result") is not None}))
                return 0
            time.sleep(1)
        print(json.dumps({"task_id": task["task_id"], "status": "processing", "timeout": True}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
