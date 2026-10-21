"""Stool Agent：手动记录校验，或图片质量评估与 Bristol 类型候选。"""
from __future__ import annotations

from typing import Any
import re

from ..contract import AgentName, AgentResult, ObsKind, SourceRef, Status, TaskKind, TaskRequest
from ..errors import AgentError
from ..validation import scan_text, validate_agent_result
from .base import BaseAgent, file_evidence_id, resolve_input_path

BRISTOL_VALID = {1, 2, 3, 4, 5, 6, 7}
BRISTOL_DESCRIPTION = {
    1: "分离的硬球状（难以排出）",
    2: "香肠状但表面凹凸",
    3: "香肠状，表面有裂纹",
    4: "光滑柔软香肠状",
    5: "柔软团块，边缘清晰",
    6: "糊状，边缘不清",
    7: "水样，无固体成分",
}
#: 仅作“值得记录”的提示，不作为疾病判断依据
ATTENTION_TYPES = {1, 2, 6, 7}


class StoolAgent(BaseAgent):
    name = AgentName.STOOL
    prompt_key = "stool.image"
    prompt_version = "v1"
    requires_model = True

    def run(self, request: TaskRequest) -> AgentResult:
        if request.kind == TaskKind.STOOL_IMAGE_RECOGNIZE:
            return self._run_image(request)
        result = self.new_result(request, model="deterministic")
        stools = [r for r in request.confirmed_records() if r.get("kind") == "stool"]
        if not stools:
            result.add_uncertainty("no_stool_records", "本次没有提交排便记录")
            return result.finish(Status.INSUFFICIENT)

        structured: list[dict[str, Any]] = []
        route_to_safety = False
        missing_bristol: list[str] = []
        invalid_bristol: list[str] = []

        for record in stools:
            record_id = str(record.get("record_id"))
            data = record.get("data") or {}
            bristol = data.get("bristol_type")
            blood = data.get("blood_reported")
            item: dict[str, Any] = {
                "record_id": record_id,
                "occurred_at": record.get("occurred_at"),
                "bristol_type": bristol,
                "bristol_confirmed_by_patient": isinstance(bristol, int) and bristol in BRISTOL_VALID,
            }
            if isinstance(bristol, int) and bristol in BRISTOL_VALID:
                item["description"] = BRISTOL_DESCRIPTION[bristol]
                item["attention_flag"] = bristol in ATTENTION_TYPES
                result.add_obs(
                    ObsKind.FACT,
                    f"排便记录（{record.get('occurred_at')}）记录为 Bristol {bristol} 型：{BRISTOL_DESCRIPTION[bristol]}；"
                    "该类型已写入并确认；若来自排便图片，则由排便识别智能体先生成候选，患者确认后系统写入，无需手动填写。",
                    record_ids=[record_id],
                    confidence="high",
                    value={"bristol_type": bristol},
                )
                if bristol in ATTENTION_TYPES:
                    result.add_uncertainty(
                        "attention_type",
                        f"Bristol {bristol} 型属于值得继续观察的类型，建议保持按次记录；"
                        "类型本身不能说明原因，也不是疾病诊断。",
                        affects=[record_id],
                    )
            elif bristol in (None, "", 0):
                missing_bristol.append(record_id)
                item["description"] = "未确认"
            else:
                invalid_bristol.append(record_id)
                item["description"] = f"无效值({bristol!r})"

            if blood is True:
                route_to_safety = True
                result.add_obs(
                    ObsKind.PATIENT_REPORT,
                    "患者主动记录排便时出现血液；系统不依据图像或文本判断原因，"
                    "该主诉已交给独立安全规则路径处理。",
                    record_ids=[record_id],
                    confidence="high",
                    value={"blood_reported": True},
                )
            item["blood_reported"] = blood is True
            structured.append(item)

        if missing_bristol:
            result.needs_confirmation = True
            result.confirmation_fields = ["stool.bristol_type"]
            result.add_uncertainty(
                "bristol_missing",
                "部分排便记录没有选择 Bristol 类型；可以稍后补选，补选前不参与类型分布统计。",
                affects=missing_bristol,
            )
        if invalid_bristol:
            result.add_uncertainty(
                "bristol_invalid",
                "部分记录的 Bristol 类型不在 1—7 范围内，已标记为无效，不会被强行改写。",
                affects=invalid_bristol,
            )
            result.needs_confirmation = True
            result.confirmation_fields = sorted(set(result.confirmation_fields + ["stool.bristol_type"]))

        result.data["structured"] = structured
        result.data["route_to_safety"] = route_to_safety
        result.data["invalid_bristol_record_ids"] = invalid_bristol
        result.data["missing_bristol_record_ids"] = missing_bristol

        report = validate_agent_result(
            result,
            allowed_record_ids=[str(r.get("record_id")) for r in stools],
        )
        if self.store:
            self.store.save_findings(result.task_id, self.name, report.findings)
        status = Status.PARTIAL if (missing_bristol or invalid_bristol) else Status.OK
        return result.finish(status)

    def _run_image(self, request: TaskRequest) -> AgentResult:
        result = self.new_result(request, model=self.settings.vision_model)
        files = [f for f in request.files if str(f.get("kind")) in ("stool_image", "image")]
        if len(files) != 1:
            result.add_uncertainty("one_image_required", "请提交一张排便图片；多张图片需逐张识别。")
            return result.finish(Status.INSUFFICIENT)
        entry = files[0]
        evidence_id = file_evidence_id(entry) or "stool_image"
        image = resolve_input_path(str(entry.get("path") or ""), self.settings.fixture_root)
        result.needs_confirmation = True
        result.confirmation_fields = ["stool.bristol_type"]
        try:
            payload, response = self.client.complete_json(
                agent=self.name,
                system=self.system_prompt,
                user="先判断图片是否清晰且可辨认，再只按可见形态给出 Bristol 1—7 型候选；无法辨认就留空。只输出 JSON。",
                images=[image],
                model=self.settings.vision_model,
                context={"mode": "image", "image_names": [image.name]},
            )
        except AgentError as exc:
            result.error = exc.to_dict()
            result.add_uncertainty(exc.code, "图片识别未完成；请手动选择 Bristol 类型，原始图片保持不变。")
            return result.finish(Status.PARTIAL)

        result.model_id = response.model
        result.usage = response.usage()
        self.log_usage(result, response.usage(), cached=response.cached)
        quality = payload.get("quality") if isinstance(payload.get("quality"), dict) else {}
        usable = quality.get("usable") is True
        recognizable = payload.get("recognizable") is True
        candidate = payload.get("bristol_type")
        valid_candidate = type(candidate) is int and candidate in BRISTOL_VALID
        result.data.update({
            "image_quality": {"usable": usable, "issue": self._safe_text(result, quality.get("issue"), "quality.issue")},
            "recognizable": recognizable,
            "candidate_bristol_type": candidate if usable and recognizable and valid_candidate else None,
            "appearance": self._safe_text(result, payload.get("appearance"), "appearance"),
            "visual_evidence": self._safe_text(result, payload.get("evidence"), "evidence"),
            "confidence": payload.get("confidence") if payload.get("confidence") in ("high", "medium", "low") else "low",
            "confirmed_by_patient": False,
        })
        if result.data["candidate_bristol_type"] is None:
            result.add_uncertainty("image_not_recognizable", "图片质量不足或形态不可辨认，未给出类型；请重新拍摄或手动选择。")
            result.status = Status.INSUFFICIENT
        else:
            result.add_obs(
                ObsKind.MODEL_INFERENCE,
                f"按图片可见形态，AI 给出的 Bristol 候选为 {candidate} 型；需患者确认，不能据此判断疾病或便血。",
                record_ids=[evidence_id],
                refs=[SourceRef(ref_type="image_region", ref_id=evidence_id, locator="整图")],
                confidence=result.data["confidence"],
                value={"candidate_bristol_type": candidate},
            )
        report = validate_agent_result(result, allowed_record_ids=[evidence_id])
        if self.store:
            self.store.save_findings(result.task_id, self.name, report.findings)
        return result.finish()

    @staticmethod
    def _safe_text(result: AgentResult, value: Any, field: str) -> str:
        text = str(value or "")[:200]
        if scan_text(text) or re.search(r"便血|出血|黑便|炎症|肿瘤|感染|癌症|诊断|确诊|治疗|用药", text):
            result.add_finding("unsafe_content_blocked", "error", f"{field} 含禁用表述，已移除", action="dropped")
            return ""
        return text
