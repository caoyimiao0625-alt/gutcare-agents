"""Advice Agent：把“已确认背景 + 趋势观察 + 检索到的知识”变成可解释的建议。

三条硬约束：
1. 每条建议必须能指向本次的事实观察（obs_id）或检索到的知识条目（KB id），
   引用解析不了的建议会被 `validate_advice_items` 直接删除；
2. 数据不足时不生成个性化建议（宁可说“信息不足”，也不编）；
3. 已有紧急风险提示时，普通饮食/生活建议不得覆盖或削弱就医提示。
"""
from __future__ import annotations

import json
from typing import Any

from ..contract import AgentName, AgentResult, ObsKind, SourceRef, Status, TaskRequest
from ..errors import AgentError
from ..knowledge_base import KnowledgeBase
from ..validation import scan_text, validate_advice_items
from .base import BaseAgent


CATEGORY_ZH = {
    "diet": "饮食",
    "activity": "日常活动",
    "record_management": "记录管理",
    "medication": "服药打卡",
    "seek_care": "就医咨询",
}


class AdviceAgent(BaseAgent):
    prompt_key = "advice.generate"
    name = AgentName.ADVICE
    prompt_version = "v2"
    requires_model = True

    def __init__(self, settings, client, store=None, knowledge: KnowledgeBase | None = None) -> None:
        super().__init__(settings, client, store)
        self.knowledge = knowledge or KnowledgeBase.load(settings.knowledge_dir)

    # ------------------------------------------------------------------
    def run(self, request: TaskRequest) -> AgentResult:
        return self.run_with_context(request, trend_result=None)

    def run_with_context(self, request: TaskRequest, *, trend_result: AgentResult | None) -> AgentResult:
        from ..stats import facts_for_request

        result = self.new_result(request)
        facts, records = facts_for_request(
            request,
            min_days_coverage=self.settings.min_days_coverage,
            min_records_for_trend=self.settings.min_records_for_trend,
        )
        result.data["facts"] = facts.to_dict()

        obs_pool = self._observation_pool(request, trend_result)
        result.data["observation_pool"] = [{"obs_id": o["obs_id"], "statement": o["statement"]} for o in obs_pool]

        query = self._retrieval_query(request, facts)
        retrieved = self.knowledge.search(query, top_k=3)
        retrieved_ids = [doc.doc_id for doc, _ in retrieved]
        result.data["retrieved_knowledge"] = [
            {"doc_id": doc.doc_id, "title": doc.title, "score": score, "source": doc.source} for doc, score in retrieved
        ]

        # ---- 紧急提示优先：不得用普通建议覆盖就医提示 ----
        emergency = [a for a in request.safety_alerts if int(a.get("severity_level", 0)) >= 3]
        if emergency:
            result.data["safety_override"] = True
            result.add_obs(
                ObsKind.KNOWLEDGE,
                "本次输入已触发紧急危险征象提示；按安全优先原则，不提供可能延误就医的普通生活建议。",
                refs=[SourceRef(ref_type="knowledge", ref_id="KB-REDFLAG-005", locator="第 1 节")],
                confidence="high",
            )
            result.add_uncertainty(
                "emergency_override",
                "存在紧急提示，已抑制普通饮食与活动建议，仅保留就医相关提示。",
            )

        sufficiency = facts.sufficiency
        upstream_failed = trend_result is not None and trend_result.status == Status.FAILED
        if upstream_failed:
            # 上游失败必须显式传给下游，不能把它伪装成“数据不足”
            result.add_uncertainty(
                "upstream_trend_failed",
                "趋势分析阶段失败（模型或流程故障），本次不生成建议；记录与安全提示不受影响，可稍后重试。",
            )
        if (not sufficiency["enough_for_advice"]) or not obs_pool:
            reasons = list(sufficiency["reasons"])
            if upstream_failed:
                reasons.insert(0, "上游趋势分析未完成")
            if not obs_pool:
                reasons.append("缺少可引用的观察")
            result.data["advice_items"] = []
            result.data["degraded_reason"] = "本次不生成个性化建议：" + "；".join(reasons or ["信息不足"])
            result.add_uncertainty("insufficient_for_advice", result.data["degraded_reason"])
            report = validate_agent_result_safe(result, obs_pool, retrieved_ids, facts)
            return result.finish(Status.INSUFFICIENT)

        try:
            payload, response = self.client.complete_json(
                agent=self.name,
                system=self.system_prompt,
                user=self._user_prompt(request, facts, obs_pool, retrieved, emergency),
                context={
                    "facts": facts.to_dict(),
                    "obs_ids": [o["obs_id"] for o in obs_pool],
                    "knowledge_ids": retrieved_ids,
                    "emergency": bool(emergency),
                },
            )
        except AgentError as exc:
            result.error = exc.to_dict()
            result.data["advice_items"] = []
            result.data["degraded_reason"] = "建议生成未完成，可稍后重试；已保存的记录与统计不受影响"
            result.add_uncertainty(exc.code, result.data["degraded_reason"])
            return result.finish(Status.PARTIAL)

        result.model_id = response.model
        result.usage = response.usage()
        self.log_usage(result, response.usage(), cached=response.cached)

        items = self._normalize_items(payload.get("advice_items") or [])
        if emergency:
            items = [i for i in items if i["category"] == "seek_care"]
        # 保留校验前的原始条目：消融实验要能量出“多智能体+校验”到底过滤掉了什么
        result.data["raw_advice_items"] = list(items)
        result.data["advice_items"] = items
        degraded_reason = str(payload.get("degraded_reason") or "")
        if scan_text(degraded_reason):
            result.add_finding(
                "unsafe_content_blocked", "error", "建议状态说明含禁用表述，已移除",
                target="degraded_reason", action="dropped",
            )
            degraded_reason = ""
            result.status = Status.PARTIAL
        result.data["degraded_reason"] = degraded_reason

        report = validate_advice_items(
            result,
            obs_ids=[o["obs_id"] for o in obs_pool],
            knowledge_ids=retrieved_ids,
            facts=facts.to_dict(),
        )
        if self.store:
            self.store.save_findings(result.task_id, self.name, report.findings)
        if not result.data["advice_items"]:
            result.add_uncertainty(
                "no_valid_advice",
                "本次没有通过证据校验的建议条目；已保留观察与知识检索结果供核对。",
            )
            return result.finish(Status.PARTIAL)
        return result.finish()

    # ------------------------------------------------------------------
    def _observation_pool(self, request: TaskRequest, trend_result: AgentResult | None) -> list[dict[str, Any]]:
        pool: list[dict[str, Any]] = []
        if trend_result is not None:
            for obs in trend_result.observations:
                if isinstance(obs.value, dict) and obs.value.get("provisional"):
                    continue
                pool.append({"obs_id": obs.obs_id, "statement": obs.statement, "record_ids": obs.source_record_ids})
        for item in request.confirmed_results:
            if str(item.get("patient_id") or "") != request.patient_id:
                continue
            if str(item.get("agent")) != AgentName.TREND:
                continue
            for obs in item.get("observations") or []:
                if obs.get("obs_id") and obs.get("statement"):
                    pool.append(
                        {
                            "obs_id": str(obs["obs_id"]),
                            "statement": str(obs["statement"]),
                            "record_ids": obs.get("source_record_ids") or [],
                        }
                    )
        # 去重并保持顺序
        seen: set[str] = set()
        unique = []
        for item in pool:
            if item["obs_id"] in seen:
                continue
            seen.add(item["obs_id"])
            unique.append(item)
        return unique

    def _retrieval_query(self, request: TaskRequest, facts: Any) -> str:
        parts = [request.query or ""]
        if request.profile and request.profile.diagnosis:
            parts.append(str(request.profile.diagnosis))
        kinds = facts.by_kind
        if kinds.get("stool"):
            parts.append("排便 形态 记录")
        if facts.symptoms.get("count"):
            parts.append("腹痛 腹胀 症状 记录")
        if kinds.get("diet"):
            parts.append("饮食 症状 关联 观察")
        if kinds.get("medication"):
            parts.append("服药 打卡 依从性")
        if kinds.get("mood"):
            parts.append("压力 情绪 症状 共同变化")
        return " ".join(p for p in parts if p)

    def _user_prompt(
        self,
        request: TaskRequest,
        facts: Any,
        obs_pool: list[dict[str, Any]],
        retrieved: list[tuple[Any, float]],
        emergency: list[dict[str, Any]],
    ) -> str:
        profile = request.profile.to_dict() if request.profile else {}
        knowledge_block = "\n\n".join(
            f"[{doc.doc_id}] {doc.title}（来源：{doc.source or '未标注'}；审核状态：{doc.review_status}）\n{doc.text}"
            for doc, _score in retrieved
        ) or "（本次没有检索到可引用的知识条目）"
        return (
            "## 患者已确认档案（只读，不得改动）\n"
            f"{json.dumps(profile, ensure_ascii=False)}\n\n"
            "## 事实表（程序计算，数字只能引用这里）\n"
            f"{json.dumps(facts.to_dict(), ensure_ascii=False)}\n\n"
            "## 可用观察（只能引用下列 obs_id）\n"
            f"{json.dumps(obs_pool, ensure_ascii=False)}\n\n"
            "## 检索到的知识条目（只能引用下列 knowledge_ids）\n"
            f"{knowledge_block}\n\n"
            + (
                "## 紧急提示（必须遵守）\n"
                f"{json.dumps(emergency, ensure_ascii=False)}\n"
                "本次只输出 category=seek_care 的建议，不得输出普通饮食或活动建议。\n\n"
                if emergency
                else ""
            )
            + "## 任务\n生成 1—3 条可执行建议（不足则更少）。数据不足时把 degraded_reason 写清楚。"
        )

    def _normalize_items(self, raw: list[Any]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for idx, item in enumerate(raw, start=1):
            if not isinstance(item, dict):
                continue
            category = str(item.get("category") or "record_management")
            if category not in CATEGORY_ZH:
                category = "record_management"
            items.append(
                {
                    "advice_id": str(item.get("advice_id") or f"advice_{idx:03d}"),
                    "category": category,
                    "category_label": CATEGORY_ZH[category],
                    "title": str(item.get("title") or "").strip(),
                    "body": str(item.get("body") or "").strip(),
                    "applicability": str(item.get("applicability") or "").strip(),
                    "uncertainty": str(item.get("uncertainty") or "").strip(),
                    "evidence_obs_ids": [str(x) for x in (item.get("evidence_obs_ids") or [])],
                    "knowledge_ids": [str(x) for x in (item.get("knowledge_ids") or [])],
                }
            )
        return items


def validate_agent_result_safe(result: AgentResult, obs_pool, knowledge_ids, facts):
    """数据不足分支也需要走一遍证据校验（保证空建议时观察仍然合规）。"""
    from ..validation import validate_agent_result

    allowed = {str(x) for item in obs_pool for x in (item.get("record_ids") or [])}
    return validate_agent_result(
        result,
        allowed_record_ids=allowed,
        knowledge_ids=knowledge_ids,
        facts=facts.to_dict(),
    )
