"""生成联调样例（data/generated_samples/，不纳入文档目录）。

背景：联调条件（《开发边界》§8）要求提供“报告抽取、饮食识别、排便识别、综合分析和
建议生成”的样例请求/响应，以及“成功、部分成功、无法识别、数据不足、超时、服务异常”
的完整状态样例。

本脚本直接跑真实代码路径生成样例，不手写假数据：
- 默认 mock 模式 → 样例用于**结构联调**，响应里带 `_sample_note` 说明“未执行真实识别”；
- `--mode live` → 用真实模型生成，样例同时可用于质量对照（会消耗额度）。

用法：
    # 需要单独提供 testdata_v4；仓库不含该测试包
    # PowerShell: $env:GCA_FIXTURE_ROOT = "<testdata_v4 所在目录>"
    python tools/make_samples.py                 # 结构样例
    python tools/make_samples.py --mode live     # 真实输出样例
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gutcare_agents.config import load_settings
from gutcare_agents.fixtures import Fixtures
from gutcare_agents.orchestrator import Orchestrator, run_single
from gutcare_agents.rules import RuleEngine, build_context, make_alert
from gutcare_agents.store import Store

OUT = ROOT / "data" / "generated_samples"

SAMPLE_NOTE = {
    "mock": "本样例由 mock 模式生成，用于接口结构联调；识图类任务明确未执行真实识别，"
    "不得把本样例内容当作模型效果证据。",
    "live": "本样例由 live 模式生成，模型与 prompt 版本见 model_id/prompt_version 字段。",
    "replay": "本样例由 replay 模式生成（读取已录制的真实响应）。",
}


def dump(name: str, payload: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mock", "live", "replay"], default="mock")
    args = parser.parse_args()

    settings = load_settings(mode=args.mode)
    fixtures = Fixtures(settings.fixture_root)
    store = Store(settings.db_path)
    note = SAMPLE_NOTE[args.mode]
    manifest: list[dict[str, Any]] = []

    print(f"生成联调样例（mode={args.mode}）→ {OUT}")

    # ---- 单任务：报告字段抽取 ----
    for sample_id, prefix in (
        ("MED002", "report_extract"),
        ("ADVICE001", "advice_doc_extract"),
        ("FOOD001", "meal_recognize"),
        ("STOOL001", "stool_record"),
        ("TREND001", "trend_analyze"),
    ):
        request = fixtures.build_request(sample_id)
        result = run_single(settings, request)
        payload = result.to_dict() | {"_sample_note": note, "_sample_task": prefix,
                                      "_sample_source": sample_id, "_mode": args.mode}
        dump(f"{prefix}.request.json", request.to_dict())
        dump(f"{prefix}.response.json", payload)
        manifest.append(
            {
                "kind": prefix,
                "source_sample": sample_id,
                "request": f"{prefix}.request.json",
                "response": f"{prefix}.response.json",
                "expected_status": payload["status"],
                "has_images": bool(request.files),
                "note": "mock 模式下识图类任务会返回 insufficient_data 并说明未执行真实识别"
                if args.mode == "mock"
                else "",
            }
        )

    # ---- 安全规则：正常 / 便血 / 黑色柏油样便 ----
    engine = RuleEngine(settings.rules_path)
    for label, record in (
        (
            "normal",
            {
                "record_id": "sample-stool-normal",
                "patient_id": "SYN001",
                "kind": "stool",
                "occurred_at": "2026-10-16T08:10:00+08:00",
                "created_at": "2026-10-16T08:12:00+08:00",
                "status": "confirmed",
                "version": 1,
                "data": {"bristol_type": 4, "blood_reported": False},
            },
        ),
        (
            "blood",
            {
                "record_id": "sample-stool-blood",
                "patient_id": "SYN001",
                "kind": "stool",
                "occurred_at": "2026-10-16T08:10:00+08:00",
                "status": "confirmed",
                "version": 1,
                "data": {"bristol_type": 6, "blood_reported": True},
            },
        ),
        (
            "black_tarry",
            {
                "record_id": "sample-stool-black",
                "patient_id": "SYN001",
                "kind": "stool",
                "occurred_at": "2026-10-16T08:10:00+08:00",
                "status": "confirmed",
                "version": 1,
                "data": {"bristol_type": 6, "note": "黑色柏油样大便"},
            },
        ),
    ):
        profile = fixtures.profile("SYN001").to_dict()
        request_payload = {"patient_id": "SYN001", "profile": profile, "records": [record]}
        hits = engine.evaluate(build_context([record], profile), stage="record_submit")
        alerts = [make_alert(patient_id="SYN001", hit=hit) for hit in hits]
        dump(f"safety_check.{label}.request.json", request_payload)
        dump(
            f"safety_check.{label}.response.json",
            {
                "rule_set_version": engine.rule_set_version,
                "review_status": engine.review_status,
                "max_severity_level": engine.max_level(hits),
                "alerts": alerts,
                "disclaimer": engine.disclaimer,
                "latency_note": "本路径不调用任何模型，模型不可用时仍可工作",
                "_sample_note": note,
            },
        )
        manifest.append(
            {
                "kind": f"safety_check.{label}",
                "request": f"safety_check.{label}.request.json",
                "response": f"safety_check.{label}.response.json",
                "expected_max_severity": engine.max_level(hits),
            }
        )

    # ---- 综合分析：完整 / 降级 / 数据不足 ----
    orchestrator = Orchestrator(settings, store=store)

    full = fixtures.analysis_request("SYN001", days=7)
    full.options["force_refresh"] = True
    run_full = orchestrator.run(full)
    dump("analysis_full.request.json", full.to_dict())
    dump("analysis_full.response.json", run_full.to_dict() | {"_sample_note": note, "_mode": args.mode})
    manifest.append(
        {
            "kind": "analysis_full",
            "request": "analysis_full.request.json",
            "response": "analysis_full.response.json",
            "expected_status": run_full.status,
            "note": "mock 模式下 trend 的模型叙述缺失，因此任务状态为 partial（这是真实的降级结果）",
        }
    )

    degraded = fixtures.analysis_request("SYN002", days=7)
    degraded.options["force_refresh"] = True
    degraded.options["simulate_faults"] = ["advice"]
    run_degraded = orchestrator.run(degraded)
    dump("analysis_degraded.response.json", run_degraded.to_dict() | {"_sample_note": note, "_mode": args.mode})
    manifest.append(
        {
            "kind": "analysis_degraded",
            "request": "analysis_full.request.json",
            "response": "analysis_degraded.response.json",
            "expected_status": run_degraded.status,
            "note": "通过 options.simulate_faults 注入 advice 故障，用于演示部分成功",
        }
    )

    insufficient = fixtures.analysis_request("SYN009", days=365)
    insufficient.options["force_refresh"] = True
    run_insufficient = orchestrator.run(insufficient)
    dump(
        "analysis_insufficient.response.json",
        run_insufficient.to_dict() | {"_sample_note": note, "_mode": args.mode},
    )
    manifest.append(
        {
            "kind": "analysis_insufficient",
            "request": "analysis_full.request.json",
            "response": "analysis_insufficient.response.json",
            "expected_status": run_insufficient.status,
            "note": "SYN009 没有已确认诊断，用于演示“不得生成疾病专属建议”",
        }
    )

    # ---- 错误响应（与 api.py 的 problem+json 保持一致）----
    dump(
        "error_patient_scope_violation.response.json",
        {
            "type": "https://gutcare.local/problems/patient_scope_violation",
            "title": "patient_scope_violation",
            "status": 400,
            "code": "patient_scope_violation",
            "detail": "records[2] 的 patient_id 与请求的 patient_id 不一致",
            "request_id": "3f9c1a7d0b2e",
        },
    )
    dump(
        "error_model_unavailable.response.json",
        {
            "type": "https://gutcare.local/problems/model_unavailable",
            "title": "model_unavailable",
            "status": 400,
            "code": "model_unavailable",
            "detail": "未配置 GCA_API_KEY，无法使用 live 模式；可改用 GCA_MODE=mock 或 replay",
            "request_id": "8b21c4e0f7aa",
        },
    )
    dump(
        "error_agent_timeout.response.json",
        {
            "type": "https://gutcare.local/problems/agent_timeout",
            "title": "agent_timeout",
            "status": 400,
            "code": "agent_timeout",
            "detail": "[演示] 已按请求注入 advice 故障（模拟模型超时或不可用）",
            "request_id": "c07d2e51b9f4",
            "detail_data": {
                "agent": "advice",
                "retryable": True,
                "hint": "业务后端可重试；记录、任务状态与安全提示不受影响",
            },
        },
    )

    dump(
        "manifest.json",
        {
            "generated_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"),
            "mode": args.mode,
            "model": settings.vision_model,
            "fixture_root": str(settings.fixture_root),
            "samples": manifest,
            "usage": {
                "mock": "用于接口结构联调；识图类结果明确标注未执行真实识别",
                "live": "用于质量对照；注意模型与 prompt 版本字段",
            },
            "not_included": [
                "真实患者数据（本仓库只使用测试包合成/公开脱敏数据）",
                "临床审核过的规则与知识内容",
            ],
        },
    )
    print(f"完成，共 {len(manifest)} 组任务样例。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
