"""HTTP 接口（标准库 http.server，零依赖）。

接口边界严格按《开发边界》§6.2：
- 入参带 `task_id`、内部患者标识、分析类型、时间范围、授权数据与幂等标识；
- 返参带 `agent_name`、`status`、错误码、模型/流程版本、输入记录与版本、
  observations / source_references / uncertainties / needs_confirmation；
- 不以“只有一段自然语言”作为唯一返回值。

错误统一 `application/problem+json`，带稳定 `code` 与 `request_id`。
"""
from __future__ import annotations

import json
import logging
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .agents import AGENT_CLASSES
from .config import Settings
from .contract import (
    AgentName,
    PatientProfile,
    TaskKind,
    TaskRequest,
    TaskStatus,
    TimeRange,
    new_id,
    now_iso,
)
from .errors import AgentError, ErrorCode
from .fixtures import Fixtures
from .f08_adapter import BackendClient, F08ContractError, F08Journal, process_task, validate_dispatch
from .orchestrator import Orchestrator
from .rules import RuleEngine, build_context, make_alert
from .storage_bootstrap import get_store

LOG = logging.getLogger("gutcare.api")
CONSOLE_DIR = Path(__file__).resolve().parents[1] / "console"


class AppState:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = get_store(settings)
        self.orchestrator = Orchestrator(settings, store=self.store)
        self.rules = RuleEngine(settings.rules_path)
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.f08_active: set[str] = set()
        self.f08_lock = threading.Lock()
        self.f08_journal = F08Journal(settings.db_path)
        self.fixtures = Fixtures(settings.fixture_root) if (settings.fixture_root / "inputs/test_catalog.json").is_file() else None

    def schedule_f08(self, dispatch: dict[str, Any]) -> None:
        task_id = validate_dispatch(dispatch)
        with self.f08_lock:
            if task_id in self.f08_active:
                return
            self.f08_active.add(task_id)
        self.pool.submit(self._run_f08, dict(dispatch), task_id)

    def _run_f08(self, payload: dict[str, Any], task_id: str) -> None:
        try:
            client = BackendClient(self.settings.backend_base_url, self.settings.backend_service_token,
                                   timeout=self.settings.timeout_s)
            process_task(client, self.orchestrator, payload, journal=self.f08_journal)
        except Exception as exc:
            LOG.error("F08 processing paused task_id=%s error_type=%s", task_id, type(exc).__name__)
        finally:
            with self.f08_lock:
                self.f08_active.discard(task_id)


def _problem(handler: "Handler", status: int, code: str, message: str, request_id: str, detail: Any = None) -> None:
    payload = {
        "type": f"https://gutcare.local/problems/{code}",
        "title": code,
        "status": status,
        "code": code,
        "detail": message,
        "request_id": request_id,
    }
    if detail:
        payload["detail_data"] = detail
    handler._json(status, payload, content_type="application/problem+json")


def parse_request(payload: dict[str, Any]) -> TaskRequest:
    """把外部 JSON 映射为 TaskRequest；缺字段就明确报错，不做静默猜测。"""
    kind = str(payload.get("kind") or "")
    if kind not in vars(TaskKind).values() and not kind:
        raise AgentError(ErrorCode.INVALID_REQUEST, "缺少 kind")
    patient_id = str(payload.get("patient_id") or "")
    if not patient_id:
        raise AgentError(ErrorCode.INVALID_REQUEST, "缺少 patient_id")
    profile_raw = payload.get("profile")
    profile = None
    if profile_raw:
        profile = PatientProfile(
            patient_id=str(profile_raw.get("patient_id") or patient_id),
            display_name=str(profile_raw.get("display_name") or ""),
            age=profile_raw.get("age"),
            diagnosis_status=str(profile_raw.get("diagnosis_status") or "unknown"),
            diagnosis=profile_raw.get("diagnosis"),
            profile_version=int(profile_raw.get("profile_version") or 1),
            restrictions=list(profile_raw.get("restrictions") or []),
        )
        if profile.patient_id != patient_id:
            raise AgentError(ErrorCode.PATIENT_SCOPE_VIOLATION, "profile.patient_id 与 patient_id 不一致")
    confirmed_results = list(payload.get("confirmed_results") or [])
    if any(not isinstance(item, dict) for item in confirmed_results):
        raise AgentError(ErrorCode.INVALID_REQUEST, "confirmed_results 必须是对象列表")
    if any(str(item.get("patient_id") or "") != patient_id for item in confirmed_results):
        raise AgentError(ErrorCode.PATIENT_SCOPE_VIOLATION, "存在不属于该患者的既有分析结果")
    time_range_raw = payload.get("time_range") or {}
    return TaskRequest(
        task_id=str(payload.get("task_id") or new_id("task")),
        kind=kind,
        patient_id=patient_id,
        time_range=TimeRange(time_range_raw.get("start"), time_range_raw.get("end")),
        records=list(payload.get("records") or []),
        profile=profile,
        files=list(payload.get("files") or []),
        confirmed_results=confirmed_results,
        safety_alerts=list(payload.get("safety_alerts") or []),
        query=str(payload.get("query") or ""),
        options=dict(payload.get("options") or {}),
        idempotency_key=str(payload.get("idempotency_key") or ""),
    )


class Handler(BaseHTTPRequestHandler):
    server_version = "GutCareAgents/0.2"
    state: AppState  # 由 make_server 注入

    # ------------------------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:  # 安静一点，结构化日志另记
        LOG.info("%s - %s", self.address_string(), fmt % args)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query)
        request_id = uuid.uuid4().hex[:12]
        try:
            if path in ("/", "/index.html"):
                return self._serve_console()
            if path == "/healthz":
                return self._json(200, {"ok": True, "mode": self.state.settings.mode, "time": now_iso()})
            if not self._authorized():
                return _problem(self, 401, ErrorCode.UNAUTHORIZED, "内部鉴权失败", request_id)
            if path == "/v1/meta":
                return self._json(200, self._meta())
            if path == "/v1/tasks":
                return self._json(
                    200,
                    {
                        "items": self.state.store.list_tasks(
                            limit=int(query.get("limit", ["20"])[0]),
                            patient_id=(query.get("patient_id") or [None])[0],
                        )
                    },
                )
            if path.startswith("/v1/tasks/"):
                task_id = path.split("/")[-1]
                task = self.state.store.get_task(task_id)
                if task is None:
                    return _problem(self, 404, "task_not_found", f"未找到任务 {task_id}", request_id)
                return self._json(200, task)
            if path.startswith("/v1/safety/alerts/"):
                patient_id = path.split("/")[-1]
                return self._json(200, {"items": self.state.store.list_alerts(patient_id)})
            if path == "/v1/demo/cases":
                if not self.state.fixtures:
                    return _problem(self, 404, "fixtures_missing", "未找到测试包", request_id)
                fixtures = self.state.fixtures
                return self._json(
                    200,
                    {
                        "categories": fixtures.categories(),
                        "patients": [
                            {"patient_id": p["patient_id"], "diagnosis": p.get("diagnosis"),
                             "diagnosis_status": p.get("diagnosis_status"), "age": p.get("age")}
                            for p in fixtures.seed["patients"][:12]
                        ],
                    },
                )
            return _problem(self, 404, "not_found", f"未知路径 {path}", request_id)
        except AgentError as exc:
            return _problem(self, 400, exc.code, exc.message, request_id)
        except Exception as exc:  # 兜底，绝不把堆栈返回给调用方
            LOG.error("GET %s failed: %s\n%s", path, exc, traceback.format_exc())
            return _problem(self, 500, ErrorCode.INTERNAL_ERROR, "服务器内部错误", request_id)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        request_id = uuid.uuid4().hex[:12]
        try:
            payload = self._read_json()
            if not self._authorized():
                return _problem(self, 401, ErrorCode.UNAUTHORIZED, "内部鉴权失败", request_id)
            if path == "/v1/safety/check":
                return self._safety(payload, request_id)
            if path == "/internal/v1/tasks":
                return self._f08_dispatch(payload, request_id)
            if path == "/v1/tasks":
                return self._submit(payload, request_id)
            if path == "/v1/analysis":
                return self._analysis(payload, request_id)
            if path == "/v1/assistant/chat":
                return self._chat(payload, request_id)
            if path == "/v1/demo/analysis":
                return self._demo_analysis(payload, request_id)
            if path.startswith("/v1/tasks/") and path.endswith("/cancel"):
                task_id = path.split("/")[-2]
                result = self.state.store.request_cancel(task_id)
                if result is None:
                    return _problem(self, 404, "task_not_found", f"未找到任务 {task_id}", request_id)
                return self._json(202, result | {"request_id": request_id})
            if path.startswith("/v1/safety/alerts/") and path.endswith("/ack"):
                patient_id = path.split("/")[-2]
                return self._json(200, {"patient_id": patient_id, "acknowledged": True,
                                        "note": "已读只更新查看状态，不代表症状消失或风险解除"})
            return _problem(self, 404, "not_found", f"未知路径 {path}", request_id)
        except AgentError as exc:
            return _problem(self, 400, exc.code, exc.message, request_id, exc.detail)
        except json.JSONDecodeError:
            return _problem(self, 400, ErrorCode.INVALID_REQUEST, "请求体不是合法 JSON", request_id)
        except Exception as exc:
            LOG.error("POST %s failed: %s\n%s", path, exc, traceback.format_exc())
            return _problem(self, 500, ErrorCode.INTERNAL_ERROR, "服务器内部错误", request_id)

    # ------------------------------------------------------------------
    def _f08_dispatch(self, payload: dict[str, Any], request_id: str) -> None:
        if not self.state.settings.internal_token or not self.state.settings.backend_service_token:
            return _problem(self, 503, "f08_not_configured", "F08 内部服务未配置", request_id)
        key = self.headers.get("Idempotency-Key", "")
        if not key or len(key) > 128:
            return _problem(self, 400, "idempotency_key_required", "缺少幂等键", request_id)
        try:
            task_id = validate_dispatch(payload)
        except F08ContractError as exc:
            return _problem(self, 400, "invalid_f08_dispatch", str(exc), request_id)
        try:
            state = self.state.f08_journal.accept(payload)
        except F08ContractError as exc:
            return _problem(self, 409, "f08_task_conflict", str(exc), request_id)
        if state not in {"completed", "stopped"}:
            self.state.schedule_f08(payload)
        return self._json(202, {"task_id": task_id, "accepted": True})

    # ------------------------------------------------------------------
    def _submit(self, payload: dict[str, Any], request_id: str) -> None:
        from .orchestrator import run_single

        request = parse_request(payload)
        result = run_single(self.state.settings, request)
        body = result.to_dict()
        body["request_id"] = request_id
        return self._json(200, body)

    def _analysis(self, payload: dict[str, Any], request_id: str) -> None:
        request = parse_request(payload)
        request.kind = TaskKind.ANALYSIS
        # 先落一条 queued，便于调用方立即轮询
        self.state.store.save_task(
            self._placeholder_run(request), request.to_dict()
        )
        self.state.pool.submit(self._run_analysis, request)
        return self._json(202, {"task_id": request.task_id, "status": TaskStatus.QUEUED, "request_id": request_id})

    def _run_analysis(self, request: TaskRequest) -> None:
        try:
            self.state.orchestrator.run(request)
        except Exception as exc:
            LOG.error("analysis %s failed: %s\n%s", request.task_id, exc, traceback.format_exc())

    def _demo_analysis(self, payload: dict[str, Any], request_id: str) -> None:
        if not self.state.fixtures:
            raise AgentError(ErrorCode.INVALID_REQUEST, "未找到测试包，无法使用演示接口")
        patient_id = str(payload.get("patient_id") or "SYN001")
        days = int(payload.get("days") or 7)
        request = self.state.fixtures.analysis_request(patient_id, days=days)
        request.options["force_refresh"] = bool(payload.get("force_refresh"))
        fault = str(payload.get("fault") or "").strip()
        if fault:
            request.options["simulate_faults"] = [fault]
        return self._json(200, self.state.orchestrator.run(request).to_dict() | {"request_id": request_id})

    def _chat(self, payload: dict[str, Any], request_id: str) -> None:
        from .orchestrator import run_single

        request = parse_request(payload)
        request.kind = TaskKind.ASSISTANT_CHAT
        result = run_single(self.state.settings, request)
        return self._json(200, result.to_dict() | {"request_id": request_id})

    def _safety(self, payload: dict[str, Any], request_id: str) -> None:
        records = list(payload.get("records") or [])
        profile = payload.get("profile")
        patient_id = str(payload.get("patient_id") or "")
        if not patient_id:
            raise AgentError(ErrorCode.INVALID_REQUEST, "缺少 patient_id")
        if profile and str(profile.get("patient_id") or patient_id) != patient_id:
            raise AgentError(ErrorCode.PATIENT_SCOPE_VIOLATION, "记录与患者不匹配")
        for record in records:
            if str(record.get("patient_id")) != patient_id:
                raise AgentError(ErrorCode.PATIENT_SCOPE_VIOLATION, "存在不属于该患者的记录")
        context = build_context(records, profile, extra_text=[payload.get("text") or ""])
        hits = self.state.rules.evaluate(context, stage="record_submit")
        alerts = []
        for hit in hits:
            alert = make_alert(patient_id=patient_id, hit=hit)
            recent = self.state.store.find_recent_alert(alert["dedupe_key"], within_seconds=3600)
            if recent is None:
                self.state.store.save_alert(alert)
                alert["is_new"] = True
            else:
                alert["is_new"] = False
                alert["deduplicated_against"] = recent["alert_id"]
            alerts.append(alert)
        return self._json(
            200,
            {
                "request_id": request_id,
                "patient_id": patient_id,
                "rule_set_version": self.state.rules.rule_set_version,
                "review_status": self.state.rules.review_status,
                "max_severity_level": self.state.rules.max_level(hits),
                "alerts": alerts,
                "disclaimer": self.state.rules.disclaimer,
                "latency_note": "本路径不调用任何模型，模型不可用时仍可工作",
            },
        )

    # ------------------------------------------------------------------
    def _placeholder_run(self, request: TaskRequest):
        from .contract import TaskRun

        return TaskRun(
            task_id=request.task_id,
            kind=request.kind,
            patient_id=request.patient_id,
            status=TaskStatus.QUEUED,
            input_snapshot_hash=request.snapshot_hash(),
        )

    def _meta(self) -> dict[str, Any]:
        settings = self.state.settings
        return {
            "service": "gutcare-agents",
            "version": "0.2.0",
            "mode": settings.mode,
            "config": settings.describe(),
            "agents": [
                {"name": name, "requires_model": getattr(AGENT_CLASSES[name], "requires_model", True),
                 "prompt_version": getattr(AGENT_CLASSES[name], "prompt_version", "")}
                for name in AGENT_CLASSES
            ],
            "orchestrator": {"stages": ["intake", "domain", "trend", "advice", "validate"]},
            "rules": {"version": self.state.rules.rule_set_version, "review_status": self.state.rules.review_status,
                      "count": len(self.state.rules.rules)},
            "knowledge": self.state.orchestrator.knowledge.describe(),
            "store": {"cache": self.state.store.cache_stats(), "usage": self.state.store.usage_summary(),
                      "findings": self.state.store.findings_summary()},
            "task_kinds": [v for k, v in vars(TaskKind).items() if not k.startswith("_") and isinstance(v, str)],
        }

    # ------------------------------------------------------------------
    def _authorized(self) -> bool:
        token = self.state.settings.internal_token
        if not token:
            return True
        header = self.headers.get("Authorization", "")
        return header == f"Bearer {token}"

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _json(self, status: int, payload: Any, *, content_type: str = "application/json") -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_console(self) -> None:
        index = CONSOLE_DIR / "index.html"
        if not index.is_file():
            return self._json(404, {"error": "console/index.html 缺失"})
        body = index.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def make_server(settings: Settings) -> ThreadingHTTPServer:
    state = AppState(settings)
    handler = type("BoundHandler", (Handler,), {"state": state})
    server = ThreadingHTTPServer((settings.http_host, settings.http_port), handler)
    server.daemon_threads = True
    if settings.backend_base_url and settings.backend_service_token:
        for dispatch in state.f08_journal.pending_dispatches():
            state.schedule_f08(dispatch)
    return server


def serve(settings: Settings) -> int:
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    server = make_server(settings)
    host, port = server.server_address[:2]
    print(f"GutCare Agents 已启动：http://{host}:{port}  (mode={settings.mode})")
    print(f"演示控制台：http://{host}:{port}/")
    print("按 Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        server.server_close()
    return 0
