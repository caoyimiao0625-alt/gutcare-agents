"""Medical Agent：医疗报告理解、可见文字转写、模拟医嘱抽取。

三条任务共用“图像 → 结构化”能力，但输出契约不同，分开处理：
- report_extract：检查项/结论/建议/诊断候选/药物候选，全部保留原文片段；
- report_ocr：只转写可见文字，遮挡写 [遮挡]，无法辨认写 [不清]；
- advice_doc_extract：抽取医嘱文书字段并保留“模拟/测试”标记。

安全边界：不把“疑似/考虑”升级为确诊；原文没有的单位与剂量不补造；
报告里的异常标记只作为抽取核对，不作为临床诊断标准。
"""
from __future__ import annotations

import json
from dataclasses import replace
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import Any

from ..contract import AgentName, AgentResult, ObsKind, SourceRef, Status, TaskKind, TaskRequest
from ..errors import AgentError, ErrorCode
from ..validation import validate_agent_result
from ..prompts import load_prompt, prompt_meta
from ..pdf_input import inspect_pdf, render_pdf
from .base import BaseAgent, file_evidence_id, resolve_input_path





class MedicalAgent(BaseAgent):
    name = AgentName.MEDICAL
    prompt_version = "v1"
    requires_model = True

    def run(self, request: TaskRequest) -> AgentResult:
        kind = request.kind
        if kind == TaskKind.REPORT_TEXT_EXTRACT:
            return self._run_text(request)
        pdf_files = [f for f in request.files if str(f.get("mime", "")).lower() == "application/pdf" or str(f.get("path", "")).lower().endswith(".pdf")]
        if pdf_files:
            return self._run_pdf(request, pdf_files)
        result = self.new_result(request)
        image_files = [f for f in request.files if "image" in str(f.get("mime", "")) or str(f.get("kind")) in ("report", "document", "advice_doc", "image")]
        if not image_files:
            result.add_uncertainty(
                "no_file",
                "本次请求没有可解析的文件；如果视觉模型不可用，可改走 report_text_extract 从转写文本抽取",
            )
            return result.finish(Status.INSUFFICIENT)

        images = [resolve_input_path(str(f.get("path")), self.settings.fixture_root) for f in image_files]
        evidence_ids = [file_evidence_id(f) for f in image_files if file_evidence_id(f)]
        return self._run_images(request, images, evidence_ids, result)

    def _run_images(self, request: TaskRequest, images: list[Path], evidence_ids: list[str], result: AgentResult, *, prompt_extra: str = "", locator: str = "p1") -> AgentResult:
        kind = request.kind
        prompt_name = {
            TaskKind.REPORT_OCR: "medical.ocr",
            TaskKind.ADVICE_DOC_EXTRACT: "medical.advice_doc",
        }.get(kind, "medical.report")
        # 按任务加载提示词；用局部变量而不是实例属性，避免并发请求互相覆盖
        system_prompt = load_prompt(prompt_name, self.prompt_version, root=self.settings.prompts_dir)
        prompt_info = prompt_meta(prompt_name, self.prompt_version, root=self.settings.prompts_dir)
        result.prompt_sha256 = str(prompt_info.get("sha256", ""))

        try:
            payload, response = self.client.complete_json(
                agent=self.name,
                system=system_prompt,
                user=self._user_prompt(kind) + prompt_extra,
                images=images,
                model=self.settings.vision_model,
                context={"mode": kind, "image_names": [p.name for p in images]},
            )
        except AgentError as exc:
            result.error = exc.to_dict()
            result.add_uncertainty(
                exc.code,
                "报告解析未完成；原始文件保持不变，可稍后重试或改用手动录入关键字段",
            )
            result.needs_confirmation = True
            result.confirmation_fields = ["report.fields"]
            return result.finish(Status.FAILED)

        result.model_id = response.model
        result.usage = response.usage()
        self.log_usage(result, response.usage(), cached=response.cached)

        if kind == TaskKind.REPORT_OCR:
            self._handle_ocr(result, payload, evidence_ids, locator=locator)
        elif kind == TaskKind.ADVICE_DOC_EXTRACT:
            self._handle_advice_doc(result, payload, evidence_ids, locator=locator)
        else:
            self._handle_report(result, payload, evidence_ids, locator=locator)

        report = validate_agent_result(result, allowed_record_ids=evidence_ids)
        if self.store:
            self.store.save_findings(result.task_id, self.name, report.findings)
        return result.finish()

    def _run_pdf(self, request: TaskRequest, pdf_files: list[dict[str, Any]]) -> AgentResult:
        result = self.new_result(request)
        result.needs_confirmation = True
        result.confirmation_fields = ["report.fields"]
        if len(pdf_files) != 1 or len(request.files) != 1:
            result.add_uncertainty("one_pdf_required", "PDF 报告请一次提交一份；多份文件需分别处理。")
            return result.finish(Status.INSUFFICIENT)
        entry = pdf_files[0]
        path = resolve_input_path(str(entry.get("path") or ""), self.settings.fixture_root)
        evidence_id = file_evidence_id(entry) or "report_pdf"
        try:
            parsed = inspect_pdf(path)
        except AgentError as exc:
            result.error = exc.to_dict()
            result.add_uncertainty(exc.code, "PDF 解析失败；请上传可读文件或手动录入关键字段。")
            return result.finish(Status.PARTIAL)
        locator = f"p1-p{parsed.processed_pages}"
        metadata = {"mode": "visual" if parsed.needs_visual else "embedded_text", "page_count": parsed.page_count, "processed_pages": parsed.processed_pages, "truncated": parsed.page_count > parsed.processed_pages}
        if not parsed.needs_visual:
            if request.kind == TaskKind.REPORT_OCR:
                result.model_id = "pdf_text_extractor"
                result.data.update({"transcript": parsed.text, "page_note": "PDF 内嵌文本", "pdf_processing": metadata})
                result.add_obs(ObsKind.FACT, "已提取 PDF 页面内嵌文本；请核对文本与原报告是否一致。", record_ids=[evidence_id], refs=[SourceRef(ref_type="report_page", ref_id=evidence_id, locator=locator)], confidence="high")
                if metadata["truncated"]:
                    result.add_uncertainty("pdf_page_limit", "PDF 仅处理前 5 页，后续页面未解析。")
                return result.finish()
            if request.kind == TaskKind.REPORT_EXTRACT:
                text_request = replace(request, kind=TaskKind.REPORT_TEXT_EXTRACT, files=[], options={**request.options, "text": parsed.text, "source_id": evidence_id, "source_locator": locator})
                extracted = self._run_text(text_request)
                extracted.input_snapshot_hash = request.snapshot_hash()
                extracted.data["pdf_processing"] = metadata
                if metadata["truncated"]:
                    extracted.add_uncertainty("pdf_page_limit", "PDF 仅处理前 5 页，后续页面未解析。")
                return extracted
        try:
            with TemporaryDirectory(prefix="gutcare_pdf_") as tmp:
                images = render_pdf(path, Path(tmp), max_pages=parsed.processed_pages, max_side=self.settings.max_image_side)
                extra = f"\n图片按 PDF 页面顺序排列，共处理前 {parsed.processed_pages} 页。"
                if parsed.text:
                    extra += f"\n以下是 PDF 内嵌文字，仅作辅助，需以页面可见内容核对：\n{parsed.text[:6000]}"
                result = self._run_images(request, images, [evidence_id], result, prompt_extra=extra, locator=locator)
        except AgentError as exc:
            result.error = exc.to_dict()
            result.add_uncertainty(exc.code, "PDF 扫描页无法解析；请上传清晰图片或手动录入关键字段。")
            return result.finish(Status.PARTIAL)
        result.data["pdf_processing"] = metadata
        if metadata["truncated"]:
            result.add_uncertainty("pdf_page_limit", "PDF 仅处理前 5 页，后续页面未解析。")
        return result

    # ------------------------------------------------------------------
    def _run_text(self, request: TaskRequest) -> AgentResult:
        """视觉不可用时的降级路径：从 OCR/转写文本抽取，契约与报告抽取一致。

        典型用法：后端自带 OCR（或患者手工粘贴文本），智能体只做字段抽取与校验。
        """
        result = self.new_result(request)
        prompt_name = "medical.report_text"
        system_prompt = load_prompt(prompt_name, self.prompt_version, root=self.settings.prompts_dir)
        prompt_info = prompt_meta(prompt_name, self.prompt_version, root=self.settings.prompts_dir)
        result.prompt_sha256 = str(prompt_info.get("sha256", ""))

        text = str(request.options.get("text") or "").strip()
        if not text:
            for entry in request.files:
                if "text" in str(entry.get("mime", "")):
                    try:
                        text = resolve_input_path(str(entry.get("path")), self.settings.fixture_root).read_text(
                            encoding="utf-8"
                        )
                    except OSError:
                        text = ""
                    break
        evidence_id = str(request.options.get("source_id") or "report_text")
        if not text:
            result.add_uncertainty(
                "no_text",
                "本任务需要 OCR/转写后的文本；未提供文本时无法抽取（不猜测报告内容）",
            )
            result.needs_confirmation = True
            result.confirmation_fields = ["report.source_text"]
            return result.finish(Status.INSUFFICIENT)

        result.data["source_text_chars"] = len(text)
        try:
            payload, response = self.client.complete_json(
                agent=self.name,
                system=system_prompt,
                user=self._text_prompt(text),
                model=self.settings.text_model,
                context={"mode": "report_text", "text_chars": len(text)},
            )
        except AgentError as exc:
            result.error = exc.to_dict()
            result.add_uncertainty(exc.code, "文本抽取未完成；原文不变，可稍后重试或手动录入关键字段")
            result.needs_confirmation = True
            result.confirmation_fields = ["report.fields"]
            return result.finish(Status.FAILED)

        result.model_id = response.model
        result.usage = response.usage()
        self.log_usage(result, response.usage(), cached=response.cached)
        self._handle_report(result, payload, [evidence_id], locator=str(request.options.get("source_locator") or "p1"))
        report = validate_agent_result(result, allowed_record_ids=[evidence_id])
        if self.store:
            self.store.save_findings(result.task_id, self.name, report.findings)
        return result.finish()

    def _text_prompt(self, text: str) -> str:
        clipped = text[:12000]
        return (
            "## 待抽取文本（来自报告 OCR/转写，可能含 [遮挡]/[不清] 标记）\n"
            f"{clipped}\n\n"
            "## 任务\n按 schema 输出 JSON。只使用文本中出现的字段："
            "文本没有的项目与单位不要补造，遮挡处留空，疑似不得写成确诊。"
        )

    # ------------------------------------------------------------------
    def _handle_report(self, result: AgentResult, payload: dict[str, Any], evidence_ids: list[str], *, locator: str = "p1") -> None:
        data = {
            "document_type": payload.get("document_type") or "无法判断",
            "report_date": payload.get("report_date"),
            "institution": payload.get("institution"),
            "exam_items": payload.get("exam_items") or [],
            "conclusion": payload.get("conclusion") or "",
            "suggestions": payload.get("suggestions") or [],
            "diagnosis_candidates": payload.get("diagnosis_candidates") or [],
            "medications": payload.get("medications") or [],
            "redactions": payload.get("redactions") or [],
        }
        result.data.update(data)
        ref = SourceRef(ref_type="report_page", ref_id=evidence_ids[0] if evidence_ids else "document", locator=locator)

        items = data["exam_items"]
        if items:
            rendered = "；".join(
                f"{i.get('name', '')}={i.get('value', '')}{i.get('unit', '')}"
                + (f"（参考 {i.get('reference_range')}）" if i.get("reference_range") else "")
                for i in items[:12]
            )
            result.add_obs(
                ObsKind.FACT,
                f"从报告中抽取到 {len(items)} 项检查内容：{rendered}。以上均按原文抽取，未换算单位。",
                record_ids=evidence_ids[:1],
                refs=[ref],
                confidence="high",
                value={"exam_items": items},
            )
        else:
            result.add_uncertainty("no_exam_items", "未抽取到明确的检查项，可能是遮挡严重或非结构化文书")

        if data["conclusion"]:
            result.add_obs(
                ObsKind.FACT,
                f"原文结论：{data['conclusion']}",
                record_ids=evidence_ids[:1],
                refs=[ref],
                confidence="high",
            )
        if data["suggestions"]:
            result.add_obs(
                ObsKind.FACT,
                "原文建议：" + "；".join(str(s) for s in data["suggestions"][:6]),
                record_ids=evidence_ids[:1],
                refs=[ref],
                confidence="high",
            )

        for candidate in data["diagnosis_candidates"]:
            status = str(candidate.get("status") or "mentioned")
            name = str(candidate.get("name") or "")
            if not name:
                continue
            label = {
                "confirmed": "原文明确诊断",
                "suspected": "原文为疑似/考虑/待排，不能当作确诊",
                "history": "原文为既往史",
                "mentioned": "原文仅提及",
                "ruled_out": "原文为已排除",
            }.get(status, "原文提及")
            result.add_obs(
                ObsKind.MODEL_INFERENCE,
                f"诊断候选「{name}」：{label}。证据片段：{candidate.get('evidence') or '[遮挡]'}",
                record_ids=evidence_ids[:1],
                refs=[ref],
                confidence="medium",
                value={"name": name, "status": status},
            )

        if data["medications"]:
            names = "、".join(
                f"{m.get('name')}（剂量原文：{m.get('dose_text') or '未写出'}）" for m in data["medications"][:6]
            )
            result.add_obs(
                ObsKind.FACT,
                f"原文药物信息：{names}。剂量与频次严格按原文，未做任何推断或换算。",
                record_ids=evidence_ids[:1],
                refs=[ref],
                confidence="high",
            )

        if data["redactions"]:
            result.add_uncertainty(
                "redacted_regions",
                f"报告中有 {len(data['redactions'])} 处遮挡或裁切，相关字段已留空，不能据此推断内容。",
            )
        for item in payload.get("uncertainties") or []:
            if item.get("message"):
                result.add_uncertainty(str(item.get("code") or "report"), str(item["message"]))

        result.needs_confirmation = True
        result.confirmation_fields = ["report.diagnosis", "report.suggestions", "report.medications"]
        if not items and not data["conclusion"]:
            # 什么都没抽到就不要显示“解析完成”，明确降级为信息不足
            result.add_uncertainty(
                "nothing_extracted",
                "未从该文件抽取到可用的检查项或结论；请更换更清晰的文件，或手动录入关键字段。",
            )
            result.status = Status.INSUFFICIENT

    # ------------------------------------------------------------------
    def _handle_ocr(self, result: AgentResult, payload: dict[str, Any], evidence_ids: list[str], *, locator: str = "p1") -> None:
        transcript = str(payload.get("transcript") or "").strip()
        result.data["transcript"] = transcript
        result.data["page_note"] = payload.get("page_note") or ""
        if not transcript:
            result.add_uncertainty("ocr_empty", "未转写出可见文字，可能是图像质量不足")
            result.status = Status.INSUFFICIENT
        else:
            masked = transcript.count("[遮挡]") + transcript.count("[不清]")
            result.add_obs(
                ObsKind.FACT,
                f"转写完成，共 {len(transcript.splitlines())} 行可见文字"
                + (f"，其中 {masked} 处标注为遮挡或不清" if masked else "")
                + "。转写严格按可见内容，未复原被遮挡信息。",
                record_ids=evidence_ids[:1],
                refs=[SourceRef(ref_type="report_page", ref_id=evidence_ids[0] if evidence_ids else "document", locator=locator)],
                confidence="high",
            )
        for item in payload.get("uncertainties") or []:
            if item.get("message"):
                result.add_uncertainty(str(item.get("code") or "ocr"), str(item["message"]))
        result.needs_confirmation = True
        result.confirmation_fields = ["ocr.transcript"]

    # ------------------------------------------------------------------
    def _handle_advice_doc(self, result: AgentResult, payload: dict[str, Any], evidence_ids: list[str], *, locator: str = "p1") -> None:
        items = [str(x) for x in (payload.get("items") or []) if str(x).strip()]
        marker = bool(payload.get("warning_marker_present"))
        result.data.update(
            {
                "document_kind": payload.get("document_kind") or "无法判断",
                "title": payload.get("title") or "",
                "warning_marker_present": marker,
                "warning_text": payload.get("warning_text") or "",
                "institution": payload.get("institution"),
                "document_date": payload.get("document_date"),
                "linked_report_id": payload.get("linked_report_id"),
                "report_type_metadata": payload.get("report_type_metadata"),
                "items": items,
                "clinical_use_allowed": False,
            }
        )
        ref = SourceRef(ref_type="report_page", ref_id=evidence_ids[0] if evidence_ids else "document", locator=locator)
        if items:
            result.add_obs(
                ObsKind.FACT,
                f"从医嘱文书中抽取到 {len(items)} 条随访说明："
                + "；".join(items[:6])
                + "。内容为软件测试用模拟文书，不构成诊断、处方或真实医疗建议。",
                record_ids=evidence_ids[:1],
                refs=[ref],
                confidence="high",
                value={"items": items},
            )
        if not marker:
            result.add_uncertainty(
                "synthetic_marker_missing",
                "未识别到“仅用于测试/非真实医疗建议”标记；该文书不得进入真实建议依据，需要人工核对来源。",
            )
            result.needs_confirmation = True
            result.confirmation_fields = ["advice_doc.marker"]
        for item in payload.get("uncertainties") or []:
            if item.get("message"):
                result.add_uncertainty(str(item.get("code") or "advice_doc"), str(item["message"]))

    def _user_prompt(self, kind: str) -> str:
        if kind == TaskKind.REPORT_OCR:
            return "逐行转写这张文档图片上可见的中文文字，按 schema 输出 JSON。遮挡处写 [遮挡]，无法辨认处写 [不清]。"
        if kind == TaskKind.ADVICE_DOC_EXTRACT:
            return "抽取这张医嘱文书图片的字段，按 schema 输出 JSON；注意保留其模拟/测试标记，不得推断诊断、药物或剂量。"
        return (
            "抽取这张医疗报告图片的结构化信息，按 schema 输出 JSON。"
            "只抽取可见内容，遮挡处留空或写 [遮挡]，不补造单位与剂量，不把疑似写成确诊。"
        )
