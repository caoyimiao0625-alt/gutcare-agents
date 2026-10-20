"""Trend Agent：跨来源趋势与时间关联。

分工要点：
- 计数、分布、缺失日、时间对齐全部由 `stats.compute_facts` 确定性完成；
- 模型只负责把这些数字解释成患者能读懂的观察，且不允许引入新数字；
- 数字落地由 `validation.check_numeric_grounding` 复检，模型说错就标记。

这也是“多智能体不是同一段提示词”的最直接证据：本 Agent 的主要工作在程序里，
模型只是叙述层。
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from ..contract import BJT, AgentName, AgentResult, ObsKind, Status, TaskRequest
from ..errors import AgentError, ErrorCode
from ..stats import facts_for_request, parse_ts
from ..validation import validate_agent_result
from .base import BaseAgent



class TrendAgent(BaseAgent):
    prompt_key = "trend.narrative"
    name = AgentName.TREND
    prompt_version = "v1"
    requires_model = True

    def run(self, request: TaskRequest) -> AgentResult:
        result = self.new_result(request)
        facts, records = facts_for_request(
            request,
            min_days_coverage=self.settings.min_days_coverage,
            min_records_for_trend=self.settings.min_records_for_trend,
        )
        window_start = parse_ts(facts.window_start) or datetime.now(BJT)
        window_end = parse_ts(facts.window_end) or window_start
        result.data["facts"] = facts.to_dict()
        result.data["window"] = {"start": window_start.isoformat(), "end": window_end.isoformat()}
        result.data["eligible_record_ids"] = [str(r.get("record_id")) for r in records]
        result.data["domain_input_obs_ids"] = [
            str(obs.get("obs_id"))
            for domain in request.domain_results
            for obs in (domain.get("observations") or [])
            if obs.get("obs_id")
        ]

        if records:
            self._add_deterministic_observations(result, facts, records)
        else:
            # 空窗口：不生成“0 条记录”这类看似结论的观察，只说明信息不足
            result.add_uncertainty("no_records", "时间范围内没有已确认记录，无法生成趋势")
            report = validate_agent_result(result, allowed_record_ids=[], facts=facts.to_dict())
            if self.store:
                self.store.save_findings(result.task_id, self.name, report.findings)
            return result.finish(Status.INSUFFICIENT)

        # ---- 叙述层（可失败，不影响事实表）----
        try:
            payload, response = self.client.complete_json(
                agent=self.name,
                system=self.system_prompt,
                user=self._user_prompt(request, facts, records),
                context={
                    "facts": facts.to_dict(),
                    "record_ids": [str(r.get("record_id")) for r in records],
                },
            )
            result.model_id = response.model
            result.usage = response.usage()
            self.log_usage(result, response.usage(), cached=response.cached)
            record_ids = [str(r.get("record_id")) for r in records]
            domain_ids = [
                str(record_id)
                for domain in request.domain_results
                for obs in (domain.get("observations") or [])
                for record_id in (obs.get("source_record_ids") or [])
            ]
            self._merge_narrative(result, payload, record_ids + domain_ids, set(record_ids))
        except AgentError as exc:
            result.add_uncertainty(
                exc.code,
                "趋势说明未能由模型生成，已保留程序计算的结果与统计（不编造结论）",
            )
            result.error = exc.to_dict()
            result.status = Status.PARTIAL

        report = validate_agent_result(
            result,
            allowed_record_ids=[str(r.get("record_id")) for r in records] + [
                str(record_id)
                for domain in request.domain_results
                for obs in (domain.get("observations") or [])
                for record_id in (obs.get("source_record_ids") or [])
            ],
            facts=facts.to_dict(),
        )
        if report.degraded and result.status == Status.OK:
            result.status = Status.PARTIAL
        if self.store:
            self.store.save_findings(result.task_id, self.name, report.findings)
        return result.finish()

    # ------------------------------------------------------------------
    def _add_deterministic_observations(self, result: AgentResult, facts: Any, records: list[dict[str, Any]]) -> None:
        ids = [str(r.get("record_id")) for r in records]
        sample_ids = ids[:8]
        by_kind = facts.by_kind
        result.add_obs(
            ObsKind.DERIVED_STAT,
            (
                f"时间范围内共 {len(ids)} 条已确认记录，覆盖 {facts.days_with_records} 天；"
                f"饮食 {by_kind.get('diet', 0)} 条、排便 {by_kind.get('stool', 0)} 条、"
                f"症状 {by_kind.get('symptom', 0)} 条、服药 {by_kind.get('medication', 0)} 条、"
                f"活动 {by_kind.get('activity', 0)} 条、心情 {by_kind.get('mood', 0)} 条。"
            ),
            record_ids=sample_ids,
            confidence="high",
            value={"by_kind": by_kind, "total": len(ids)},
        )

        if facts.missing_days:
            result.add_obs(
                ObsKind.DERIVED_STAT,
                f"有 {len(facts.missing_days)} 天没有任何记录（首次缺记日期 {facts.missing_days[0]}）；"
                "没有记录不等于没有发生，也不等于数值为零。",
                record_ids=sample_ids,
                confidence="high",
            )

        if facts.bristol["count"]:
            dist = "、".join(f"{k}型 {v} 次" for k, v in facts.bristol["distribution"].items())
            result.add_obs(
                ObsKind.DERIVED_STAT,
                f"排便记录 {facts.bristol['count']} 条，Bristol 类型分布为：{dist}；"
                f"未确认或无法判断 {facts.bristol['unknown_or_unconfirmed']} 条，"
                f"患者主动报告便血 {facts.bristol['blood_reported_count']} 条。",
                record_ids=[str(r.get("record_id")) for r in records if r.get("kind") == "stool"][:8],
                confidence="high",
                value=facts.bristol,
            )

        if facts.symptoms["count"]:
            parts = []
            for name, info in facts.symptoms["by_name"].items():
                parts.append(
                    f"{name} {info['count']} 次（自评均值 {info['mean_severity']}，最高 {info['max_severity']}，"
                    f"≥{facts.symptoms['severe_threshold']} 分 {info['severe_count']} 次）"
                )
            result.add_obs(
                ObsKind.DERIVED_STAT,
                "症状记录：" + "；".join(parts) + "。分值为患者自评，不等同于疾病严重程度。",
                record_ids=[str(r.get("record_id")) for r in records if r.get("kind") == "symptom"][:8],
                confidence="high",
                value=facts.symptoms,
            )

        if facts.medication["count"]:
            med = facts.medication
            result.add_obs(
                ObsKind.DERIVED_STAT,
                f"服药打卡 {med['count']} 条：已确认服用 {med['taken']} 条、明确未服用 {med['not_taken']} 条、"
                f"未确认 {med['unconfirmed']} 条；按可判定事件计算的完成比例为 "
                f"{med['adherence_ratio_of_confirmed']}。未确认不等于未服用。",
                record_ids=[str(r.get("record_id")) for r in records if r.get("kind") == "medication"][:8],
                confidence="high",
                value=med,
            )

        if facts.temporal["symptoms_with_preceding_meal"]:
            result.add_obs(
                ObsKind.DERIVED_STAT,
                f"{facts.temporal['symptom_total']} 次症状记录中，有 "
                f"{facts.temporal['symptoms_with_preceding_meal']} 次在症状出现前 "
                f"{facts.temporal['window_hours']} 小时内有餐次记录，平均间隔 "
                f"{facts.temporal['mean_delta_hours']} 小时。这只是时间上的邻近，不能据此认定因果关系，"
                "也不代表已排除其他原因。",
                record_ids=[
                    e["symptom_record_id"] for e in facts.temporal["examples"]
                ][:5] or sample_ids,
                confidence="medium",
                value=facts.temporal,
            )

        if facts.mood["count"]:
            mood = facts.mood
            result.add_obs(
                ObsKind.DERIVED_STAT,
                f"心情记录 {mood['count']} 条（自报 1—5 分，非临床量表）：心情均值 {mood['mean_mood']}、"
                f"压力均值 {mood['mean_stress']}；最低心情 {mood['min_mood']}，最高压力 {mood['max_stress']}。"
                "只能描述同一时间段的共同变化与时间关联，不能据此判断先后或因果。",
                record_ids=[s["record_id"] for s in mood["series"]][:8],
                confidence="high",
                value={k: mood[k] for k in ("count", "mean_mood", "mean_stress", "min_mood", "max_stress", "top_tags")},
            )

    # ------------------------------------------------------------------
    def _merge_narrative(
        self, result: AgentResult, payload: dict[str, Any], allowed_ids: list[str], confirmed_ids: set[str]
    ) -> None:
        summary = str(payload.get("narrative_summary") or "").strip()
        if summary:
            result.data["narrative_summary"] = summary
        for item in payload.get("observations") or []:
            statement = str(item.get("statement") or "").strip()
            if not statement:
                continue
            ids = [str(x) for x in (item.get("source_record_ids") or []) if str(x) in allowed_ids]
            if not ids:
                # 没有可解析来源的叙述不进入结果（交给校验记录）
                result.add_finding(
                    "evidence_missing",
                    "error",
                    f"模型叙述缺少可解析来源，已丢弃：{statement[:40]}",
                    action="dropped",
                )
                continue
            provisional = any(i not in confirmed_ids for i in ids)
            if provisional:
                statement = "待确认识别候选：" + statement
            result.add_obs(
                ObsKind.MODEL_INFERENCE,
                statement,
                record_ids=ids[:5],
                value={"provisional": True} if provisional else None,
                confidence=str(item.get("confidence") or "medium")
                if item.get("confidence") in ("high", "medium", "low")
                else "medium",
            )
        for item in payload.get("uncertainties") or []:
            message = str(item.get("message") or "").strip()
            if message:
                result.add_uncertainty(str(item.get("code") or "model_uncertainty"), message)
        if not (payload.get("observations") or []):
            result.add_uncertainty("no_model_observations", "模型未返回可用的观察，仅保留程序统计结果")
            result.status = Status.PARTIAL

    def _user_prompt(self, request: TaskRequest, facts: Any, records: list[dict[str, Any]]) -> str:
        import json

        profile = request.profile.to_dict() if request.profile else {}
        ids = [str(r.get("record_id")) for r in records]
        domain_context = [
            {
                "agent": domain.get("agent"),
                "status": domain.get("status"),
                "needs_confirmation": domain.get("needs_confirmation", False),
                "observations": [
                    {k: obs.get(k) for k in ("obs_id", "statement", "source_record_ids")}
                    for obs in (domain.get("observations") or [])[:8]
                ],
            }
            for domain in request.domain_results
        ]
        return (
            "## 患者档案（业务后端已确认，只读）\n"
            f"{json.dumps(profile, ensure_ascii=False)}\n\n"
            "## 事实表（由程序计算，禁止改动其中的数字）\n"
            f"{json.dumps(facts.to_dict(), ensure_ascii=False)}\n\n"
            "## 允许引用的 record_id 列表\n"
            f"{json.dumps(ids, ensure_ascii=False)}\n\n"
            "## 本次领域识别候选（尚未经患者确认，不计入事实表）\n"
            f"{json.dumps(domain_context, ensure_ascii=False)}\n\n"
            "## 任务\n"
            f"为患者写一段不超过 150 字的近期状态说明，并给出最多 5 条可核对的观察。"
            f"分析区间：{facts.window_start} 至 {facts.window_end}。"
            "领域候选只可明确标为待确认信息；不能把它写成已确认事实或用于推断因果。"
            "若事实表显示信息不足，请在 uncertainties 中说明，不要补造数据。"
        )
