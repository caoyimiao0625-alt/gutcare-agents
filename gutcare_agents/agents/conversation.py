"""Conversation Agent：对话助手。

定位（见《协作机制说明》§5）：解释已有结果、回答使用问题、整理随访问题；
需要新的专业分析时，由 Orchestrator 调用对应 Agent，助手本身不改业务数据。

两条防线：
1. 涉及药物调整、诊断结论、急救替代的请求，**不调用模型**，直接给固定安全回应；
2. 回答中的引用必须能解析回已存在的观察、记录或知识条目，否则降级。
"""
from __future__ import annotations

import json
import re
from typing import Any

from ..contract import AgentName, AgentResult, ObsKind, SourceRef, Status, TaskRequest
from ..errors import AgentError
from ..knowledge_base import KnowledgeBase
from ..validation import scan_text
from .base import BaseAgent


MEDICAL_DECISION_PATTERN = re.compile(
    r"(加量|加倍|减量|停药|换药|换一种药|剂量|多吃一片|补服|能不能吃.*药|"
    r"确诊|是不是癌|是不是肿瘤|严重吗|还能活)"
)


class ConversationAgent(BaseAgent):
    prompt_key = "conversation.assistant"
    name = AgentName.CONVERSATION
    prompt_version = "v1"
    requires_model = True

    def __init__(self, settings, client, store=None, knowledge: KnowledgeBase | None = None) -> None:
        super().__init__(settings, client, store)
        self.knowledge = knowledge or KnowledgeBase.load(settings.knowledge_dir)

    def run(self, request: TaskRequest) -> AgentResult:
        result = self.new_result(request)
        query = (request.query or "").strip()
        if not query:
            result.add_uncertainty("empty_query", "问题为空")
            return result.finish(Status.INSUFFICIENT)

        if self._is_medical_decision(query) or scan_text(query):
            result.data["guard"] = "medical_decision_refused"
            result.data["answer"] = (
                "这类问题（药物剂量、停药换药、是否患有某种疾病、检查结果是否严重）需要由医生或药师判断，"
                "我不能给出结论，也不会替医生解释检查结果。\n\n"
                "可以做的：把最近的症状、用药记录和报告整理成一份清单，在复诊时直接给医生看；"
                "如果出现黑便、呕血、持续剧烈腹痛等情况，请立即就医。"
            )
            result.data["citations"] = ["KB-REDFLAG-005"]
            result.add_obs(
                ObsKind.KNOWLEDGE,
                "该问题属于药物或诊断决策类请求，按安全边界不调用模型作答，改为引导就医/咨询药师。",
                refs=[SourceRef(ref_type="knowledge", ref_id="KB-REDFLAG-005", locator="第 1 节")],
                confidence="high",
            )
            result.needs_confirmation = False
            return result.finish(Status.OK)

        context = self._build_context(request, query)
        try:
            payload, response = self.client.complete_json(
                agent=self.name,
                system=self.system_prompt,
                user=self._user_prompt(query, context),
                context={"query": query, "obs_ids": context["allowed_ids"], "knowledge_ids": context["knowledge_ids"]},
            )
        except AgentError as exc:
            result.error = exc.to_dict()
            result.data["answer"] = (
                "暂时无法生成回答（分析服务不可用）。你仍可以在“智能分析”里查看已保存的记录与报告；"
                "如需重新分析，可稍后重试。"
            )
            result.add_uncertainty(exc.code, "对话服务未完成，已保留安全兜底文案")
            return result.finish(Status.PARTIAL)

        result.model_id = response.model
        result.usage = response.usage()
        self.log_usage(result, response.usage(), cached=response.cached)

        answer = str(payload.get("answer") or "").strip()
        citations = [str(x) for x in (payload.get("citations") or [])]
        answer, dropped = self._sanitize(answer, citations, context)
        result.data["answer"] = answer
        result.data["citations"] = citations
        result.data["needs_more_info"] = bool(payload.get("needs_more_info"))
        if dropped:
            result.add_finding(
                "evidence_unresolved",
                "warn",
                f"回答中无法解析的引用已移除：{dropped[:3]}",
                action="downgraded",
            )
        for cid in citations:
            if cid.startswith("KB-"):
                result.add_obs(
                    ObsKind.KNOWLEDGE,
                    f"回答引用了知识条目 {cid}",
                    refs=[SourceRef(ref_type="knowledge", ref_id=cid, locator="全文")],
                    confidence="high",
                )
        return result.finish()

    # ------------------------------------------------------------------
    @staticmethod
    def _is_medical_decision(query: str) -> bool:
        return bool(MEDICAL_DECISION_PATTERN.search(query))

    def _build_context(self, request: TaskRequest, query: str) -> dict[str, Any]:
        obs: list[dict[str, Any]] = []
        for item in request.confirmed_results:
            if str(item.get("patient_id") or "") != request.patient_id:
                continue
            for o in item.get("observations") or []:
                if o.get("obs_id") and o.get("statement"):
                    obs.append(
                        {
                            "obs_id": str(o["obs_id"]),
                            "agent": str(item.get("agent", "")),
                            "statement": str(o["statement"]),
                        }
                    )
        retrieved = self.knowledge.search(query, top_k=2, min_score=0.04)
        knowledge_ids = [doc.doc_id for doc, _ in retrieved]
        advice: list[dict[str, Any]] = []
        for item in request.confirmed_results:
            if str(item.get("patient_id") or "") != request.patient_id:
                continue
            for adv in (item.get("data") or {}).get("advice_items") or []:
                advice.append(
                    {
                        "advice_id": adv.get("advice_id"),
                        "title": adv.get("title"),
                        "category": adv.get("category"),
                    }
                )
        return {
            "observations": obs[:20],
            "allowed_ids": [o["obs_id"] for o in obs[:20]],
            "knowledge": [
                {"doc_id": doc.doc_id, "title": doc.title, "text": doc.text, "source": doc.source}
                for doc, _ in retrieved
            ],
            "knowledge_ids": knowledge_ids,
            "advice": advice,
        }

    def _user_prompt(self, query: str, context: dict[str, Any]) -> str:
        return (
            "## 患者问题\n"
            f"{query}\n\n"
            "## 已有分析观察（只能引用这些 obs_id）\n"
            f"{json.dumps(context['observations'], ensure_ascii=False)}\n\n"
            "## 已有建议条目\n"
            f"{json.dumps(context['advice'], ensure_ascii=False)}\n\n"
            "## 检索到的知识条目（只能引用这些 doc_id）\n"
            f"{json.dumps(context['knowledge'], ensure_ascii=False)}\n\n"
            "## 任务\n依据以上材料回答；没有依据时明确说明缺什么，并引导到对应页面。"
        )

    def _sanitize(self, answer: str, citations: list[str], context: dict[str, Any]) -> tuple[str, list[str]]:
        allowed = set(context["allowed_ids"]) | set(context["knowledge_ids"])
        dropped = [c for c in citations if c not in allowed]
        citations[:] = [c for c in citations if c in allowed]

        hits = scan_text(answer)
        if hits:
            answer = (
                "这个问题涉及药物或诊断判断，我不能给出结论。"
                "建议把近期记录与报告带给医生或药师核对；如出现黑便、呕血或持续剧烈腹痛，请立即就医。"
            )
            citations[:] = [c for c in citations if c.startswith("KB-")]
        # 去掉正文里无法解析的编号引用
        for token in re.findall(r"\[([A-Za-z0-9_\-]+)\]", answer):
            if token not in allowed:
                answer = answer.replace(f"[{token}]", "")
                dropped.append(token)
        return answer.strip(), dropped
