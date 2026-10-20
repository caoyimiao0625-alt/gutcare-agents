"""Diet Agent：单餐饮食识别与结构化。

分工要点：
- 只处理“这一张图/这一条记录”，不做膳食干预建议（那是 Advice 的职责）；
- 允许部分识别，模糊或多菜混合时明确返回不确定项；
- 不估算精确热量与营养素，不做“某食物对你有害”的判断。

单条识别与综合分析的边界见《协作机制说明》§3.2。
"""
from __future__ import annotations

import json
from typing import Any

from ..contract import AgentName, AgentResult, ObsKind, SourceRef, Status, TaskRequest
from ..errors import AgentError
from ..validation import scan_text, validate_agent_result
from .base import BaseAgent, file_evidence_id, resolve_input_path



class DietAgent(BaseAgent):
    prompt_key = "diet.meal"
    name = AgentName.DIET
    prompt_version = "v1"
    requires_model = True

    def run(self, request: TaskRequest) -> AgentResult:
        result = self.new_result(request)
        image_files = [
            f
            for f in request.files
            if str(f.get("kind", "food_image")) in ("food_image", "meal", "image") or "image" in str(f.get("mime", ""))
        ]
        text_description = self._text_description(request)

        if not image_files:
            return self._text_only(result, text_description, request)

        images = [resolve_input_path(str(f.get("path")), self.settings.fixture_root) for f in image_files]
        evidence_ids = [file_evidence_id(f) for f in image_files if file_evidence_id(f)]
        allowed = evidence_ids + [str(r.get("record_id")) for r in request.confirmed_records()]

        try:
            payload, response = self.client.complete_json(
                agent=self.name,
                system=self.system_prompt,
                user=self._user_prompt(text_description),
                images=images,
                model=self.settings.vision_model,
                context={"image_names": [p.name for p in images], "description": text_description},
            )
        except AgentError as exc:
            result.add_uncertainty(
                exc.code,
                "图像识别未完成；患者仍可通过文字方式保存这条饮食记录（不允许因此阻断记录）",
            )
            result.error = exc.to_dict()
            result.needs_confirmation = True
            result.confirmation_fields = ["diet.foods"]
            return result.finish(Status.PARTIAL if exc.code != "unsupported_media_type" else Status.PARTIAL)

        result.model_id = response.model
        result.usage = response.usage()
        self.log_usage(result, response.usage(), cached=response.cached)

        recognizable = bool(payload.get("recognizable", False))
        foods = []
        for food in payload.get("foods") or []:
            if not isinstance(food, dict) or not food.get("name"):
                continue
            if any(scan_text(str(food.get(key) or "")) for key in ("name", "category", "evidence")):
                result.add_finding("unsafe_content_blocked", "error", "识别候选含禁用表述，已移除", action="dropped")
                result.status = Status.PARTIAL
                continue
            foods.append(food)
        ref = SourceRef(ref_type="image_region", ref_id=evidence_ids[0] if evidence_ids else "image", locator="整图")

        if not recognizable or not foods:
            result.data["foods"] = []
            result.data["portion_note"] = ""
            result.add_uncertainty(
                "image_not_recognizable",
                "无法从这张图片可靠识别食物，请自行填写食物名称；记录仍可正常保存。",
            )
            for item in payload.get("uncertainties") or []:
                if item.get("message"):
                    message = self._safe_model_text(result, item["message"], "uncertainty")
                    if message:
                        result.add_uncertainty(str(item.get("code") or "image"), message)
            result.status = Status.INSUFFICIENT
        else:
            result.data["foods"] = foods
            result.data["portion_note"] = self._safe_model_text(result, payload.get("portion_note"), "portion_note")
            result.data["scene_note"] = self._safe_model_text(result, payload.get("scene_note"), "scene_note")
            names = "、".join(f"{f['name']}（{f.get('category', '未分类')}）" for f in foods)
            result.add_obs(
                ObsKind.MODEL_INFERENCE,
                f"图像识别到的可见食物：{names}。份量判断：{result.data['portion_note'] or '无法判断'}。"
                "以上为 AI 识别候选，需患者确认后有效；画面外的配料与调味料无法得知。",
                record_ids=evidence_ids[:1],
                refs=[ref],
                confidence=self._overall_confidence(foods),
                value={"foods": foods},
            )
            for item in foods:
                result.add_obs(
                    ObsKind.MODEL_INFERENCE,
                    f"「{item['name']}」：{item.get('evidence') or '可见'}（置信度 {item.get('confidence', 'medium')}）",
                    record_ids=evidence_ids[:1],
                    refs=[ref],
                    confidence=str(item.get("confidence") or "medium")
                    if item.get("confidence") in ("high", "medium", "low")
                    else "medium",
                )
            for item in payload.get("uncertainties") or []:
                if item.get("message"):
                    message = self._safe_model_text(result, item["message"], "uncertainty")
                    if message:
                        result.add_uncertainty(str(item.get("code") or "image"), message)
            result.needs_confirmation = True
            result.confirmation_fields = ["diet.foods", "diet.portion"]

        report = validate_agent_result(result, allowed_record_ids=allowed)
        if self.store:
            self.store.save_findings(result.task_id, self.name, report.findings)
        return result.finish()

    @staticmethod
    def _safe_model_text(result: AgentResult, value: Any, field: str) -> str:
        text = str(value or "")
        if scan_text(text):
            result.add_finding("unsafe_content_blocked", "error", f"{field} 含禁用表述，已移除", action="dropped")
            result.status = Status.PARTIAL
            return ""
        return text

    # ------------------------------------------------------------------
    def _text_only(self, result: AgentResult, description: str, request: TaskRequest) -> AgentResult:
        record_ids = [
            str(r.get("record_id"))
            for r in request.confirmed_records()
            if r.get("kind") == "diet"
        ]
        if not description:
            result.add_uncertainty(
                "no_input",
                "没有图片也没有文字描述；需要至少一种有效内容才能形成饮食记录",
            )
            return result.finish(Status.INSUFFICIENT)
        result.add_obs(
            ObsKind.PATIENT_REPORT,
            f"患者手动记录的饮食内容：{description}。该内容由患者提供，不需要模型确认；"
            "烹饪方式与调味料未记录，不做推测。",
            record_ids=record_ids[:1],
            confidence="high",
            value={"description": description},
        )
        result.data["foods"] = [{"name": description, "category": "未分类", "confidence": "patient_confirmed"}]
        result.data["portion_note"] = ""
        report = validate_agent_result(result, allowed_record_ids=record_ids)
        if self.store:
            self.store.save_findings(result.task_id, self.name, report.findings)
        return result.finish()

    @staticmethod
    def _overall_confidence(foods: list[dict[str, Any]]) -> str:
        order = {"low": 0, "medium": 1, "high": 2}
        values = [order.get(str(f.get("confidence", "medium")), 1) for f in foods]
        return {0: "low", 1: "medium", 2: "high"}[min(values)] if values else "low"

    def _text_description(self, request: TaskRequest) -> str:
        description = str(request.options.get("description") or "").strip()
        if description:
            return description
        for record in request.confirmed_records():
            if record.get("kind") == "diet":
                value = (record.get("data") or {}).get("description")
                if value:
                    return str(value)
        return ""

    def _user_prompt(self, description: str) -> str:
        extra = f"\n患者补充的文字说明（可能为空）：{description}" if description else ""
        return (
            "识别这张饮食图片中可见的食物，按 schema 输出 JSON。"
            "不要估算热量或营养素，不要判断该食物对患者是否有害。" + extra
        )
