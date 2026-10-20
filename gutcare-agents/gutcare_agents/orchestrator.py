"""Orchestrator：把多个 Agent 编成一条可解释、可降级、可复用的工作流。

阶段（与《协作机制说明》§4.2 对应）：
  intake   → 输入校验 + 即时安全规则前置检查（不依赖模型）
  domain   → Medical / Diet / Stool 领域观察（互相独立，可并行；失败不阻塞）
  trend    → 跨来源时间对齐与趋势（确定性统计 + 模型叙述）
  advice   → 建议生成（依赖 Trend 观察 + 检索知识）
  validate → 汇总校验发现，形成对外报告

设计取舍：
- 不做成微服务，也不引入外部编排框架：单人 5 天周期里，透明可控 > 形式先进；
- 每个阶段落库，`GET /tasks/{id}` 轮询即可看到真实进度（不伪造百分比）；
- 同一输入快照复用旧任务，避免重复报告；领域 Agent 结果按文件缓存复用。
"""
from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any, Callable

from .agents import AGENT_CLASSES, BaseAgent
from .agents.advice import AdviceAgent
from .agents.trend import TrendAgent
from .config import Settings
from .contract import (
    AgentName,
    AgentResult,
    Observation,
    PatientProfile,
    SourceRef,
    StageRecord,
    Status,
    TaskKind,
    TaskRequest,
    TaskRun,
    TaskStatus,
    TimeRange,
    agent_result_from_dict,
    new_id,
    now_iso,
    sha256_of,
)
from .errors import AgentError, ErrorCode
from .knowledge_base import KnowledgeBase
from .llm import LLMClient
from .prompts import prompt_meta
from .rules import RuleEngine, build_context, make_alert
from .stats import facts_for_request
from .store import Store


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        *,
        client: LLMClient | None = None,
        store: Store | None = None,
        knowledge: KnowledgeBase | None = None,
    ) -> None:
        self.settings = settings
        self.client = client or LLMClient(settings)
        self.store = store or Store(settings.db_path)
        self.knowledge = knowledge or KnowledgeBase.load(settings.knowledge_dir)
        self.rules = RuleEngine(settings.rules_path)
        self.agents: dict[str, BaseAgent] = {
            name: cls(settings, self.client, self.store, self.knowledge)
            if name in (AgentName.ADVICE, AgentName.CONVERSATION)
            else cls(settings, self.client, self.store)
            for name, cls in AGENT_CLASSES.items()
        }

    # ------------------------------------------------------------------
    def run(self, request: TaskRequest) -> TaskRun:
        faults = bool(request.options.get("simulate_faults"))
        force = bool(request.options.get("force_refresh")) or faults
        # 注入过故障的任务不进入幂等复用（否则一次演练会污染后续正常分析）
        snapshot = sha256_of({
            "input": request.snapshot_hash(),
            "mode": self.settings.mode,
            "text_model": self.settings.text_model,
            "vision_model": self.settings.vision_model,
            "prompts": self._directory_fingerprint(self.settings.prompts_dir),
            "knowledge": self._directory_fingerprint(self.settings.knowledge_dir),
            "rules": hashlib.sha256(self.settings.rules_path.read_bytes()).hexdigest(),
        }) + ("-fault" if faults else "")

        if not force:
            previous = self.store.find_by_idempotency(request.kind, request.patient_id, snapshot)
            if previous:
                run = self._dict_to_run(previous)
                run.status = run.status  # 保持原状态
                for stage in run.stages:
                    stage.note = ""
                run.report.setdefault("reuse", {})
                run.report["reuse"] = {
                    "reused_task_id": run.task_id,
                    "reason": "相同患者与相同输入快照已存在任务，直接复用（幂等）",
                }
                return run

        run = TaskRun(
            task_id=request.task_id,
            kind=request.kind,
            patient_id=request.patient_id,
            status=TaskStatus.RUNNING,
            input_snapshot_hash=snapshot,
        )
        self._persist(run, request)

        try:
            if self._check_cancel(run, request):
                return run
            self._stage_intake(run, request)
            if self._check_cancel(run, request):
                return run
            self._stage_domain(run, request)
            if self._check_cancel(run, request):
                return run
            self._stage_trend(run, request)
            if self._check_cancel(run, request):
                return run
            self._stage_advice(run, request)
            if self._check_cancel(run, request):
                return run
            self._stage_finalize(run, request)
        except Exception as exc:  # 兜底：任何未预期异常都必须变成可读状态
            run.status = TaskStatus.FAILED
            run.error = AgentError(ErrorCode.INTERNAL_ERROR, str(exc)).to_dict()
            run.finished_at = now_iso()
            self._persist(run, request)
        return run

    # ------------------------------------------------------------------
    # 取消（协作式：只在阶段之间检查，已完成的阶段结果保留）
    # ------------------------------------------------------------------
    def _check_cancel(self, run: TaskRun, request: TaskRequest) -> bool:
        if not self.store.is_cancelled(run.task_id):
            return False
        run.status = TaskStatus.CANCELLED
        run.finished_at = now_iso()
        run.report["cancelled"] = {
            "note": "任务已按用户请求取消；已完成阶段的结果与原始记录保留，可重新发起分析",
            "completed_stages": [s.stage for s in run.stages if s.status in (Status.OK, Status.PARTIAL)],
        }
        self._persist(run, request)
        return True

    # ------------------------------------------------------------------
    # 阶段 1：输入与安全前置检查
    # ------------------------------------------------------------------
    def _stage_intake(self, run: TaskRun, request: TaskRequest) -> None:
        started = now_iso()
        records = request.confirmed_records()
        context = build_context(
            records,
            request.profile.to_dict() if request.profile else None,
            extra_text=[request.query] if request.query else None,
        )
        hits = self.rules.evaluate(context, stage="analysis")
        alerts = list(request.safety_alerts)
        for hit in hits:
            alert = make_alert(patient_id=request.patient_id, hit=hit)
            recent = self.store.find_recent_alert(alert["dedupe_key"], within_seconds=3600)
            if recent is None:
                self.store.save_alert(alert)
                alert["is_new"] = True
            else:
                alert["is_new"] = False
                alert["deduplicated_against"] = recent["alert_id"]
            alerts.append(alert)

        request.safety_alerts = alerts
        run.report["risks"] = alerts

        excluded = [r for r in request.records if r.get("status", "confirmed") != "confirmed"]
        if excluded:
            run.report["excluded_records"] = len(excluded)

        status = Status.OK
        note = f"有效记录 {len(records)} 条"
        if any(int(a.get("severity_level", 0)) >= 3 for a in alerts):
            note += "；存在紧急级提示，已优先输出"
        if not records and not request.files:
            status = Status.INSUFFICIENT
            note += "；没有可分析的输入"
        self._record(run, "intake", AgentName.ORCHESTRATOR, status, started, note=note)

    # ------------------------------------------------------------------
    # 阶段 2：领域观察（并行，失败不阻塞）
    # ------------------------------------------------------------------
    def _stage_domain(self, run: TaskRun, request: TaskRequest) -> None:
        planned: dict[str, list[dict[str, Any]]] = {}
        file_kinds = {str(f.get("kind", "")) for f in request.files}
        if any(k in ("report", "document", "advice_doc") for k in file_kinds):
            planned[AgentName.MEDICAL] = [
                f for f in request.files if str(f.get("kind")) in ("report", "document", "advice_doc")
            ]
        if "food_image" in file_kinds:
            planned[AgentName.DIET] = [f for f in request.files if str(f.get("kind")) == "food_image"]
        if "stool_image" in file_kinds:
            planned[AgentName.STOOL] = [f for f in request.files if str(f.get("kind")) == "stool_image"]
        elif any(r.get("kind") == "stool" for r in request.confirmed_records()):
            planned[AgentName.STOOL] = []

        jobs: list[tuple[str, Callable[[], AgentResult]]] = []
        sub_kinds = {
            AgentName.MEDICAL: "report_extract",
            AgentName.DIET: "meal_recognize",
            AgentName.STOOL: "stool_image_recognize" if "stool_image" in file_kinds else "stool_record",
        }
        for name, files in planned.items():
            jobs.append((name, self._make_domain_job(name, request, files, sub_kinds[name])))

        if not jobs:
            self._record(run, "domain", "-", Status.SKIPPED, now_iso(), note="本次输入无需领域识别，直接进入跨来源分析")
            return

        started = now_iso()
        results: dict[str, AgentResult] = {}
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures = {name: pool.submit(fn) for name, fn in jobs}
            for name, future in futures.items():
                try:
                    results[name] = future.result(timeout=self.settings.timeout_s)
                except FutureTimeout:
                    results[name] = self._failed_result(
                        request, name, AgentError(ErrorCode.AGENT_TIMEOUT, f"{name} 阶段超时")
                    )
                except AgentError as exc:
                    results[name] = self._failed_result(request, name, exc)
                except Exception as exc:  # 未预期异常也要降级，不中断整条链
                    results[name] = self._failed_result(
                        request, name, AgentError(ErrorCode.AGENT_FAILED, str(exc), agent=name)
                    )

        for name, result in results.items():
            run.agent_results[name] = result
            run.model_versions[name] = result.model_id
            self.store.save_agent_result(result)
            self.store.save_findings(result.task_id, name, result.findings)
            status = Status.OK if result.status == Status.OK else Status.PARTIAL
            self._record(
                run,
                f"domain.{name}",
                name,
                status,
                result.started_at,
                note=(result.error or {}).get("message", "") if result.error else "",
                ended=result.ended_at,
            )
            if result.usage.get("reused"):
                run.stages[-1].reused = True
                run.stages[-1].note = (run.stages[-1].note or "") + "（复用已确认的识别结果）"
        failed = [n for n, r in results.items() if r.status == Status.FAILED]
        if failed:
            run.report.setdefault("degraded", []).append(
                f"领域识别未完成：{', '.join(failed)}；相关结论已标注不可用，不会用空白数据伪装完整结果"
            )

    # ------------------------------------------------------------------
    # 领域 Agent 作业封装：故障注入 → 结果复用 → 执行
    # ------------------------------------------------------------------
    def _make_domain_job(
        self, name: str, request: TaskRequest, files: list[dict[str, Any]], sub_kind: str
    ) -> Callable[[], AgentResult]:
        def job() -> AgentResult:
            self._raise_if_fault(request, name)
            record_kind = {AgentName.DIET: "diet", AgentName.STOOL: "stool"}.get(name)
            domain_request = self._sub_request(
                request,
                sub_kind,
                files=files,
                records=[r for r in request.records if r.get("kind") == record_kind] if record_kind else [],
                confirmed_results=[],
            )
            cache_key = self._domain_cache_key(name, domain_request, sub_kind)
            # force_refresh 只影响“是否重跑分析”，不影响同一患者、同一输入的识别候选复用。
            if cache_key and not request.options.get("no_reuse"):
                cached = self.store.cache_get(cache_key)
                if (cached and cached.get("patient_id") == request.patient_id
                        and cached.get("input_snapshot_hash") == domain_request.snapshot_hash()):
                    result = agent_result_from_dict(cached)
                    result.task_id = request.task_id
                    result.usage = {**result.usage, "cached": True, "reused": True}
                    return result
            result = self.agents[name].run(domain_request)
            if cache_key and result.status != Status.FAILED and not result.error:
                self.store.cache_put(
                    cache_key,
                    name,
                    result.to_dict(),
                    model_id=result.model_id,
                    prompt_version=result.prompt_version,
                )
            return result

        return job

    def _raise_if_fault(self, request: TaskRequest, name: str) -> None:
        """演示/演练用故障注入：让“异常场景”可以现场复现，而不是只写在文档里。"""
        faults = [str(x) for x in (request.options.get("simulate_faults") or [])]
        if name not in faults and "all" not in faults:
            return
        delay_ms = float(request.options.get("simulate_delay_ms") or 0)
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)
        raise AgentError(
            ErrorCode.AGENT_TIMEOUT,
            f"[演示] 已按请求注入 {name} 故障（模拟模型超时或不可用）",
            agent=name,
        )

    def _domain_cache_key(self, name: str, request: TaskRequest, sub_kind: str = "") -> str:
        """同一患者、同一输入与模型版本 → 复用对应的领域识别候选。

        “改了饮食不必重跑报告解析”就是靠这个键实现的；换模型或改提示词会自动失效。
        `sub_kind` 必须参与：同一个 Medical Agent 承担报告抽取与文字转写，结果不能混用。
        """
        if not request.files:
            return ""
        digests = [self._file_digest(entry) for entry in request.files]
        if not any(digests):
            return ""
        prompt_version = str(getattr(AGENT_CLASSES[name], "prompt_version", "v1"))
        pdf_input = name == AgentName.MEDICAL and any(str(f.get("mime", "")).lower() == "application/pdf" or str(f.get("path", "")).lower().endswith(".pdf") for f in request.files)
        prompt_key = {
            AgentName.MEDICAL: "medical.report",
            AgentName.DIET: "diet.meal",
            AgentName.STOOL: "stool.image",
        }.get(name)
        if pdf_input:
            # 同一 PDF 可能因扫描页走视觉路径；两个提示词都参与失效判断。
            prompt_key = "medical.report_text"
        prompt_hash = (
            str(prompt_meta(prompt_key, prompt_version, root=self.settings.prompts_dir)["sha256"])
            if prompt_key else ""
        )
        model = "|".join((self.settings.text_model, self.settings.vision_model)) if pdf_input else (self.settings.vision_model if name in (AgentName.MEDICAL, AgentName.DIET, AgentName.STOOL) else "deterministic")
        if pdf_input:
            prompt_hash += str(prompt_meta("medical.report", prompt_version, root=self.settings.prompts_dir)["sha256"])
        material = "|".join([
            name, sub_kind, self.settings.mode, request.snapshot_hash(), prompt_version, prompt_hash, model,
            str(request.options.get("description") or ""), *sorted(d for d in digests if d),
        ])
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _file_digest(self, entry: dict[str, Any]) -> str:
        path = Path(str(entry.get("path") or ""))
        if not path.is_absolute():
            path = self.settings.fixture_root / path
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return ""

    @staticmethod
    def _directory_fingerprint(root: Path) -> dict[str, str]:
        return {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.glob("*.md"))
        }

    # ------------------------------------------------------------------
    # 阶段 3：趋势与关联
    # ------------------------------------------------------------------
    def _stage_trend(self, run: TaskRun, request: TaskRequest) -> None:
        started = now_iso()
        domain_results = [
            result.to_dict() for name, result in run.agent_results.items()
            if name in (AgentName.MEDICAL, AgentName.DIET, AgentName.STOOL)
            and isinstance(result, AgentResult)
        ]
        try:
            self._raise_if_fault(request, AgentName.TREND)
            result = self.agents[AgentName.TREND].run(
                self._sub_request(request, TaskKind.TREND_ANALYZE, domain_results=domain_results)
            )
        except AgentError as exc:
            result = self._failed_result(request, AgentName.TREND, exc)
        run.agent_results[AgentName.TREND] = result
        run.model_versions[AgentName.TREND] = result.model_id
        self.store.save_agent_result(result)
        self.store.save_findings(result.task_id, AgentName.TREND, result.findings)
        status = {
            Status.OK: Status.OK,
            Status.INSUFFICIENT: Status.INSUFFICIENT,
            Status.FAILED: Status.FAILED,
        }.get(result.status, Status.PARTIAL)
        self._record(run, "trend", AgentName.TREND, status, started, note=(result.error or {}).get("message", ""),
                     ended=result.ended_at)
        if result.status == Status.INSUFFICIENT:
            run.report.setdefault("insufficient", []).extend(
                u.message for u in result.uncertainties if u.code in ("no_records", "missing_days")
            )

    # ------------------------------------------------------------------
    # 阶段 4：建议
    # ------------------------------------------------------------------
    def _stage_advice(self, run: TaskRun, request: TaskRequest) -> None:
        trend = run.agent_results.get(AgentName.TREND)
        started = now_iso()
        advice_agent: AdviceAgent = self.agents[AgentName.ADVICE]  # type: ignore[assignment]
        try:
            self._raise_if_fault(request, AgentName.ADVICE)
            result = advice_agent.run_with_context(
                self._sub_request(request, TaskKind.ADVICE_GENERATE),
                trend_result=trend if isinstance(trend, AgentResult) else None,
            )
        except AgentError as exc:
            result = self._failed_result(request, AgentName.ADVICE, exc)
        run.agent_results[AgentName.ADVICE] = result
        run.model_versions[AgentName.ADVICE] = result.model_id
        self.store.save_agent_result(result)
        self.store.save_findings(result.task_id, AgentName.ADVICE, result.findings)
        status = Status.OK if result.status == Status.OK else (
            Status.INSUFFICIENT if result.status == Status.INSUFFICIENT else Status.PARTIAL
        )
        self._record(run, "advice", AgentName.ADVICE, status, started,
                     note=str((result.data or {}).get("degraded_reason") or ""), ended=result.ended_at)

    # ------------------------------------------------------------------
    # 阶段 5：汇总
    # ------------------------------------------------------------------
    def _stage_finalize(self, run: TaskRun, request: TaskRequest) -> None:
        started = now_iso()
        trend = run.agent_results.get(AgentName.TREND)
        advice = run.agent_results.get(AgentName.ADVICE)
        facts, records = facts_for_request(request)

        observations: list[dict[str, Any]] = []
        for name, result in run.agent_results.items():
            if not isinstance(result, AgentResult):
                continue
            for obs in result.observations:
                item = obs.to_dict()
                item["agent"] = name
                item["needs_confirmation"] = bool(
                    (name in (AgentName.MEDICAL, AgentName.DIET) and result.needs_confirmation)
                    or (isinstance(obs.value, dict) and obs.value.get("provisional"))
                )
                observations.append(item)

        advice_items = list((advice.data or {}).get("advice_items") or []) if isinstance(advice, AgentResult) else []
        advice_degraded = (
            str((advice.data or {}).get("degraded_reason") or "") if isinstance(advice, AgentResult) else ""
        ) or (str((advice.error or {}).get("message") or "") if isinstance(advice, AgentResult) else "")
        advice_status = {
            "status": advice.status if isinstance(advice, AgentResult) else "missing",
            "degraded_reason": advice_degraded,
            "safety_override": bool((advice.data or {}).get("safety_override")) if isinstance(advice, AgentResult) else False,
            "items_with_evidence": sum(
                1
                for item in advice_items
                if (item.get("evidence_obs_ids") or item.get("knowledge_ids"))
            ),
        }
        findings: list[dict[str, Any]] = []
        for name, result in run.agent_results.items():
            if isinstance(result, AgentResult):
                for finding in result.findings:
                    entry = finding.to_dict()
                    entry["agent"] = name
                    findings.append(entry)
        run.findings = [f for f in (finding for finding in run.findings)]

        run.report.update(
            {
                "analysis_id": run.task_id,
                "patient_id": request.patient_id,
                "window": {"start": facts.window_start, "end": facts.window_end, "days": facts.days},
                "data_overview": {
                    "by_kind": facts.by_kind,
                    "days_with_records": facts.days_with_records,
                    "missing_days": len(facts.missing_days),
                    "excluded_records": run.report.get("excluded_records", 0),
                },
                "narrative_summary": (trend.data or {}).get("narrative_summary") if isinstance(trend, AgentResult) else "",
                "observations": observations,
                "advice": advice_items,
                "advice_status": advice_status,
                "risks": run.report.get("risks", []),
                "insufficient": sorted(set(run.report.get("insufficient", [])))
                + list((facts.sufficiency or {}).get("reasons") or [])
                + ([advice_degraded] if advice_degraded else []),
                "degraded": list(dict.fromkeys(run.report.get("degraded", []))),
                "validation": {
                    "findings": findings,
                    "dropped_observations": sum(
                        1 for f in findings if f.get("action") == "dropped" and f.get("target", "").startswith("obs")
                    ),
                },
                "model_versions": run.model_versions,
                "rule_set_version": self.rules.rule_set_version,
                "rule_review_status": self.rules.review_status,
            }
        )

        status = self._derive_status(run)
        run.status = status
        run.finished_at = now_iso()
        run.usage = self.store.usage_summary() if self.store else {}
        self._record(run, "validate", AgentName.VALIDATOR, Status.OK, started,
                     note=f"校验发现 {len(findings)} 条，删除不合规条目 {run.report['validation']['dropped_observations']} 条")
        self._persist(run, request)

    # ------------------------------------------------------------------
    def _derive_status(self, run: TaskRun) -> str:
        results = [r for r in run.agent_results.values() if isinstance(r, AgentResult)]
        if not results:
            return TaskStatus.FAILED
        trend = run.agent_results.get(AgentName.TREND)
        advice = run.agent_results.get(AgentName.ADVICE)
        if isinstance(trend, AgentResult) and trend.status == Status.FAILED and not (
            isinstance(advice, AgentResult) and advice.data.get("advice_items")
        ):
            return TaskStatus.FAILED
        if any(r.status in (Status.FAILED, Status.PARTIAL) for r in results):
            return TaskStatus.PARTIAL
        if all(r.status == Status.INSUFFICIENT for r in results):
            return TaskStatus.INSUFFICIENT
        return TaskStatus.SUCCEEDED

    # ------------------------------------------------------------------
    def _sub_request(
        self,
        request: TaskRequest,
        kind: str,
        *,
        files: list[dict[str, Any]] | None = None,
        records: list[dict[str, Any]] | None = None,
        confirmed_results: list[dict[str, Any]] | None = None,
        domain_results: list[dict[str, Any]] | None = None,
    ) -> TaskRequest:
        return TaskRequest(
            task_id=request.task_id,
            kind=kind,
            patient_id=request.patient_id,
            time_range=request.time_range,
            records=request.records if records is None else records,
            profile=request.profile,
            files=request.files if files is None else files,
            confirmed_results=request.confirmed_results if confirmed_results is None else confirmed_results,
            domain_results=request.domain_results if domain_results is None else domain_results,
            safety_alerts=request.safety_alerts,
            query=request.query,
            options=request.options,
            idempotency_key=request.idempotency_key,
        )

    def _failed_result(self, request: TaskRequest, agent: str, error: AgentError) -> AgentResult:
        result = AgentResult(
            task_id=request.task_id,
            agent=agent,
            patient_id=request.patient_id,
            status=Status.FAILED,
            agent_version="0.2.0",
            input_snapshot_hash=request.snapshot_hash(),
        )
        result.error = error.to_dict()
        result.add_uncertainty(error.code, error.message)
        return result.finish(Status.FAILED)

    def _record(
        self,
        run: TaskRun,
        stage: str,
        agent: str,
        status: str,
        started: str,
        *,
        note: str = "",
        ended: str = "",
    ) -> None:
        run.stages.append(
            StageRecord(
                stage=stage,
                agent=agent,
                status=status,
                attempt=1,
                started_at=started,
                ended_at=ended or now_iso(),
                note=note,
            )
        )
        run.updated_at = now_iso()
        self.store.save_task(run)

    def _persist(self, run: TaskRun, request: TaskRequest) -> None:
        self.store.save_task(run, request.to_dict())

    @staticmethod
    def _dict_to_run(payload: dict[str, Any]) -> TaskRun:
        run = TaskRun(
            task_id=payload["task_id"],
            kind=payload["kind"],
            patient_id=payload["patient_id"],
            status=payload.get("status", TaskStatus.SUCCEEDED),
            created_at=payload.get("created_at", now_iso()),
            updated_at=payload.get("updated_at", now_iso()),
            finished_at=payload.get("finished_at", ""),
            input_snapshot_hash=payload.get("input_snapshot_hash", ""),
            report=payload.get("report", {}),
            error=payload.get("error"),
            usage=payload.get("usage", {}),
            model_versions=payload.get("model_versions", {}),
        )
        run.stages = [StageRecord(**{k: v for k, v in s.items() if k in StageRecord.__dataclass_fields__}) for s in payload.get("stages", [])]
        return run


# ----------------------------------------------------------------------
# 便捷入口：单任务类型（供 CLI / API 直接调用）
# ----------------------------------------------------------------------
def run_single(settings: Settings, request: TaskRequest) -> AgentResult:
    client = LLMClient(settings)
    store = Store(settings.db_path)
    knowledge = KnowledgeBase.load(settings.knowledge_dir)
    agent_cls = {
        TaskKind.REPORT_EXTRACT: "medical",
        TaskKind.REPORT_TEXT_EXTRACT: "medical",
        TaskKind.REPORT_OCR: "medical",
        TaskKind.ADVICE_DOC_EXTRACT: "medical",
        TaskKind.MEAL_RECOGNIZE: "diet",
        TaskKind.STOOL_RECORD: "stool",
        TaskKind.STOOL_IMAGE_RECOGNIZE: "stool",
        TaskKind.TREND_ANALYZE: "trend",
        TaskKind.ADVICE_GENERATE: "advice",
        TaskKind.ASSISTANT_CHAT: "conversation",
    }.get(request.kind)
    if agent_cls is None:
        raise AgentError(ErrorCode.NOT_IMPLEMENTED, f"不支持的任务类型: {request.kind}")
    agent = AGENT_CLASSES[agent_cls](settings, client, store, knowledge) if agent_cls in ("advice", "conversation") else AGENT_CLASSES[agent_cls](settings, client, store)
    result = agent.run(request)
    store.save_agent_result(result)
    store.save_findings(result.task_id, result.agent, result.findings)
    return result
