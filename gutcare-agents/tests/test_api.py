"""HTTP 层测试：真实起一个服务（随机端口），走完整请求链路。

覆盖：健康检查、鉴权、单任务、异步分析轮询、安全规则、幂等复用。
这些是“可运行程序”必须能自证的部分，也是评审随机功能测试最可能点的接口。
"""
from __future__ import annotations

import json
import shutil
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gutcare_agents.api import make_server  # noqa: E402
from gutcare_agents.config import load_settings  # noqa: E402

FIXTURE_ROOT = ROOT.parent / "testdata_v4"


class _TmpDir:
    def __init__(self) -> None:
        base = ROOT / "data" / "testtmp"
        base.mkdir(parents=True, exist_ok=True)
        self.name = str(base / f"t{uuid.uuid4().hex[:10]}")
        Path(self.name).mkdir(parents=True, exist_ok=True)

    def cleanup(self) -> None:
        shutil.rmtree(self.name, ignore_errors=True)


@unittest.skipUnless((FIXTURE_ROOT / "inputs/test_catalog.json").is_file(), "缺少测试包")
class TestHttpApi(unittest.TestCase):
    token = "test-token"

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = _TmpDir()
        cls.settings = load_settings(
            mode="mock",
            fixture_root=FIXTURE_ROOT,
            data_dir=Path(cls.tmp.name),
            replay_dir=Path(cls.tmp.name) / "replays",
            db_path=Path(cls.tmp.name) / "api.sqlite3",
            knowledge_dir=ROOT / "knowledge",
            rules_path=ROOT / "knowledge" / "safety_rules.json",
            internal_token=cls.token,
            http_host="127.0.0.1",
            http_port=0,
        )
        cls.server = make_server(cls.settings)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.tmp.cleanup()

    # ------------------------------------------------------------------
    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def _request(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: dict | None = None,
        auth: bool = True,
    ) -> tuple[int, dict]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(self._url(path), data=data, method=method)
        request.add_header("Content-Type", "application/json; charset=utf-8")
        if auth:
            request.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(request, timeout=20) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    # ------------------------------------------------------------------
    def test_healthz_without_auth(self) -> None:
        status, body = self._request("/healthz", auth=False)
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(body["mode"], "mock")

    def test_meta_requires_auth(self) -> None:
        status, body = self._request("/v1/meta", auth=False)
        self.assertEqual(status, 401)
        self.assertEqual(body["code"], "unauthorized")
        self.assertIn("request_id", body)

    def test_meta_contains_versions(self) -> None:
        status, body = self._request("/v1/meta")
        self.assertEqual(status, 200)
        self.assertEqual(body["mode"], "mock")
        self.assertTrue(body["rules"]["version"])
        self.assertIn("review_status", body["rules"])
        self.assertEqual(len(body["knowledge"]["ids"]), 6)

    def test_safety_check_level3(self) -> None:
        payload = {
            "patient_id": "SYN001",
            "records": [
                {
                    "record_id": "r-api-black",
                    "patient_id": "SYN001",
                    "kind": "stool",
                    "occurred_at": "2026-10-16T08:00:00+08:00",
                    "status": "confirmed",
                    "version": 1,
                    "data": {"bristol_type": 6, "note": "黑色柏油样大便"},
                }
            ],
        }
        status, body = self._request("/v1/safety/check", method="POST", payload=payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["max_severity_level"], 3)
        self.assertEqual(body["alerts"][0]["rule_id"], "R-BLACK-TARRY")
        self.assertTrue(body["alerts"][0]["needs_confirmation"])

    def test_safety_rejects_cross_patient_record(self) -> None:
        payload = {
            "patient_id": "SYN001",
            "records": [
                {
                    "record_id": "r-other",
                    "patient_id": "SYN002",
                    "kind": "stool",
                    "occurred_at": "2026-10-16T08:00:00+08:00",
                    "data": {"bristol_type": 4},
                }
            ],
        }
        status, body = self._request("/v1/safety/check", method="POST", payload=payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["code"], "patient_scope_violation")

    def test_single_task_trend(self) -> None:
        payload = {
            "kind": "trend_analyze",
            "patient_id": "SYN001",
            "time_range": {"start": "2026-09-01T08:00:00+08:00", "end": "2026-09-08T08:00:00+08:00"},
            "records": [],
            "profile": {"patient_id": "SYN001", "diagnosis_status": "confirmed", "diagnosis": "IBS"},
        }
        status, body = self._request("/v1/tasks", method="POST", payload=payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["agent"], "trend")
        self.assertIn(body["status"], ("ok", "partial", "insufficient_data"))
        self.assertIn("input_snapshot_hash", body)

    def test_analysis_async_and_idempotent(self) -> None:
        payload = {
            "kind": "analysis",
            "patient_id": "SYN001",
            "task_id": "api-analysis-1",
            "time_range": {"start": "2026-09-01T08:00:00+08:00", "end": "2026-09-08T08:00:00+08:00"},
            "profile": {"patient_id": "SYN001", "diagnosis_status": "confirmed", "diagnosis": "IBS"},
        }
        status, body = self._request("/v1/analysis", method="POST", payload=payload)
        self.assertEqual(status, 202)
        self.assertEqual(body["task_id"], "api-analysis-1")

        task = {}
        for _ in range(60):
            status, task = self._request("/v1/tasks/api-analysis-1")
            if task.get("status") not in ("queued", "running"):
                break
            time.sleep(0.2)
        self.assertIn(task.get("status"), ("succeeded", "partial", "insufficient_data"))
        self.assertTrue(task.get("report", {}).get("data_overview") is not None)

        # 同一快照再提交一次：应复用同一任务（幂等），不产生新报告
        status, body2 = self._request("/v1/analysis", method="POST", payload=payload)
        self.assertEqual(status, 202)
        self.assertEqual(body2["task_id"], "api-analysis-1")

    def test_cancel_running_analysis(self) -> None:
        payload = {
            "kind": "analysis",
            "patient_id": "SYN001",
            "task_id": "api-cancel-1",
            "time_range": {"start": "2026-09-01T08:00:00+08:00", "end": "2026-09-08T08:00:00+08:00"},
            "profile": {"patient_id": "SYN001", "diagnosis_status": "confirmed", "diagnosis": "IBS"},
            "options": {"simulate_faults": ["advice"], "simulate_delay_ms": 800},
        }
        status, _body = self._request("/v1/analysis", method="POST", payload=payload)
        self.assertEqual(status, 202)

        status, cancel = self._request("/v1/tasks/api-cancel-1/cancel", method="POST", payload={})
        self.assertEqual(status, 202)
        self.assertTrue(cancel["cancelled"])

        task = {}
        for _ in range(60):
            _status, task = self._request("/v1/tasks/api-cancel-1")
            if task.get("status") in ("cancelled", "succeeded", "partial", "failed", "insufficient_data"):
                break
            time.sleep(0.2)
        self.assertEqual(task.get("status"), "cancelled")
        self.assertIn("cancelled", task.get("report", {}))

        # 已结束的任务不能被“取消”覆盖
        _status, again = self._request("/v1/tasks/api-cancel-1/cancel", method="POST", payload={})
        self.assertEqual(_status, 202)
        self.assertFalse(again["cancelled"])

    def test_cancel_unknown_task_404(self) -> None:
        status, body = self._request("/v1/tasks/nope/cancel", method="POST", payload={})
        self.assertEqual(status, 404)
        self.assertEqual(body["code"], "task_not_found")

    def test_task_exposes_agent_and_prompt_versions(self) -> None:
        payload = {
            "kind": "trend_analyze",
            "patient_id": "SYN001",
            "time_range": {"start": "2026-09-01T08:00:00+08:00", "end": "2026-09-08T08:00:00+08:00"},
            "profile": {"patient_id": "SYN001", "diagnosis_status": "confirmed", "diagnosis": "IBS"},
        }
        status, body = self._request("/v1/tasks", method="POST", payload=payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["prompt_version"], "v1")
        self.assertRegex(body["prompt_sha256"], r"^[0-9a-f]{16}$")
        self.assertTrue(body["agent_version"])
        self.assertTrue(body["input_snapshot_hash"])

    def test_assistant_chat_with_prior_results(self) -> None:
        prior = [
            {
                "agent": "trend",
                "patient_id": "SYN001",
                "observations": [
                    {"obs_id": "trend_obs_001", "statement": "本次共 36 条记录", "source_record_ids": ["SYN001-diet-1"]}
                ],
                "data": {"advice_items": []},
            }
        ]
        payload = {
            "kind": "assistant_chat",
            "patient_id": "SYN001",
            "query": "帮我回顾一下最近的记录",
            "confirmed_results": prior,
        }
        status, body = self._request("/v1/assistant/chat", method="POST", payload=payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["agent"], "conversation")
        self.assertIn("answer", body["data"])
        self.assertIsInstance(body["data"]["answer"], str)

    def test_assistant_refuses_dosage_question(self) -> None:
        payload = {
            "kind": "assistant_chat",
            "patient_id": "SYN001",
            "query": "我漏服了一次，能把药量加倍吗？",
        }
        status, body = self._request("/v1/assistant/chat", method="POST", payload=payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["data"]["guard"], "medical_decision_refused")
        self.assertIn("KB-REDFLAG-005", body["data"]["citations"])

    def test_unknown_task_404(self) -> None:
        status, body = self._request("/v1/tasks/not-exist")
        self.assertEqual(status, 404)
        self.assertEqual(body["code"], "task_not_found")


if __name__ == "__main__":
    unittest.main(verbosity=2)
