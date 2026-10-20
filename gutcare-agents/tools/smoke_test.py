"""部署自检：一条命令验证“这个包到底能不能跑”。

场景：
1. 队友拿到压缩包后自检；
2. 评审现场开始演示前跑一遍（30 秒内出结论）；
3. CI / 交付前回归。

用法：
    python tools/smoke_test.py                          # 默认连 http://127.0.0.1:8100
    python tools/smoke_test.py --base-url http://host:8100 --token xxx
    python tools/smoke_test.py --no-fixtures            # 服务端没挂测试包时跳过演示用例

退出码：0 全部通过（可含 SKIP）；1 有 FAIL。
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

RESULTS: list[tuple[str, str, str]] = []  # (状态, 名称, 说明)


def http(
    base: str, path: str, *, method: str = "GET", payload: dict | None = None, token: str = "", timeout: float = 120.0
) -> tuple[int, dict]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(f"{base}{path}", data=data, method=method)
    request.add_header("Content-Type", "application/json; charset=utf-8")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "ignore")
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, {"raw": body[:200]}
    except urllib.error.URLError as exc:
        return 0, {"error": str(exc.reason)}


def check(name: str, ok: bool, detail: str = "", *, skip: bool = False) -> None:
    if skip:
        RESULTS.append(("SKIP", name, detail))
        print(f"  [SKIP] {name}  {detail}")
    elif ok:
        RESULTS.append(("PASS", name, detail))
        print(f"  [PASS] {name}  {detail}")
    else:
        RESULTS.append(("FAIL", name, detail))
        print(f"  [FAIL] {name}  {detail}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8100")
    parser.add_argument("--token", default="")
    parser.add_argument("--no-fixtures", action="store_true", help="服务端未挂测试包时跳过演示用例")
    parser.add_argument("--patient", default="SYN001")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    print(f"GutCare Agents 部署自检 · {base}\n")

    # 1) 健康检查
    status, body = http(base, "/healthz", token=args.token, timeout=10)
    ok = status == 200 and bool(body.get("ok"))
    check("健康检查 /healthz", ok, f"HTTP {status} mode={body.get('mode')}")
    if not ok:
        print("\n服务不可达，后续检查跳过。")
        return 1
    mode = body.get("mode")

    # 2) 能力清单
    status, meta = http(base, "/v1/meta", token=args.token, timeout=15)
    agents = [a.get("name") for a in meta.get("agents", [])] if status == 200 else []
    check(
        "能力清单 /v1/meta",
        status == 200 and len(agents) >= 6,
        f"HTTP {status} agents={','.join(agents) or '无'}",
    )
    rules = meta.get("rules", {}) if isinstance(meta, dict) else {}
    check(
        "规则版本与审核状态",
        bool(rules.get("version")) and bool(rules.get("review_status")),
        f"{rules.get('version')} / {rules.get('review_status')}",
    )
    kb = (meta.get("knowledge") or {}) if isinstance(meta, dict) else {}
    check("知识库可加载", int(kb.get("documents") or 0) > 0, f"{kb.get('documents')} 条")

    # 3) 鉴权：未带令牌应被拒（仅在服务端配置了令牌时才可验证）
    status, _unauth = http(base, "/v1/meta", token="definitely-wrong-token", timeout=10)
    if args.token:
        check("内部鉴权生效", status == 401, f"错误令牌 → HTTP {status}")
    else:
        check("内部鉴权", True, "未配置令牌（仅建议本地开发）", skip=True)

    # 4) 安全规则：正常记录不误报
    normal = {
        "patient_id": args.patient,
        "records": [
            {
                "record_id": "smoke-normal",
                "patient_id": args.patient,
                "kind": "stool",
                "occurred_at": "2026-10-16T08:00:00+08:00",
                "status": "confirmed",
                "version": 1,
                "data": {"bristol_type": 4, "blood_reported": False},
            }
        ],
    }
    status, body = http(base, "/v1/safety/check", method="POST", payload=normal, token=args.token, timeout=15)
    check("安全规则不误报", status == 200 and body.get("max_severity_level") == 0, f"max_level={body.get('max_severity_level')}")

    # 5) 安全规则：黑色柏油样便 → L3
    black = {
        "patient_id": args.patient,
        "records": [
            {
                "record_id": "smoke-black",
                "patient_id": args.patient,
                "kind": "stool",
                "occurred_at": "2026-10-16T08:00:00+08:00",
                "status": "confirmed",
                "version": 1,
                "data": {"bristol_type": 6, "note": "黑色柏油样大便"},
            }
        ],
    }
    status, body = http(base, "/v1/safety/check", method="POST", payload=black, token=args.token, timeout=15)
    alerts = body.get("alerts") or []
    check(
        "危险征象触发 L3",
        status == 200 and body.get("max_severity_level") == 3 and alerts,
        f"rule={(alerts[0].get('rule_id') if alerts else '无')} needs_confirmation={(alerts[0].get('needs_confirmation') if alerts else None)}",
    )

    # 6) 越权记录必须被拒
    cross = dict(black)
    cross["records"] = [dict(black["records"][0], patient_id="SYN999", record_id="smoke-cross")]
    status, body = http(base, "/v1/safety/check", method="POST", payload=cross, token=args.token, timeout=15)
    check("跨患者记录被拒", status == 400 and body.get("code") == "patient_scope_violation", f"code={body.get('code')}")

    # 7) 综合分析（需要服务端挂测试包）
    if args.no_fixtures:
        check("综合分析链路", True, "按要求跳过（--no-fixtures）", skip=True)
        check("数据不足降级", True, "按要求跳过", skip=True)
    else:
        status, run = http(
            base,
            "/v1/demo/analysis",
            method="POST",
            payload={"patient_id": args.patient, "days": 7, "force_refresh": True},
            token=args.token,
            timeout=180,
        )
        stages = [s.get("stage") for s in (run.get("stages") or [])] if status == 200 else []
        report = run.get("report") or {}
        check(
            "综合分析链路",
            status == 200 and "trend" in stages and "advice" in stages,
            f"status={run.get('status')} stages={','.join(str(s) for s in stages)}",
        )
        advice = report.get("advice") or []
        with_evidence = [a for a in advice if (a.get("evidence_obs_ids") or a.get("knowledge_ids"))]
        check(
            "建议带可解析依据",
            len(with_evidence) == len(advice),
            f"建议 {len(advice)} 条，带依据 {len(with_evidence)} 条",
        )
        check(
            "报告含缺口/降级说明字段",
            "insufficient" in report and "advice_status" in report,
            f"advice_status={((report.get('advice_status') or {}).get('status'))}",
        )

        status, run2 = http(
            base,
            "/v1/demo/analysis",
            method="POST",
            payload={"patient_id": "SYN009", "days": 365, "force_refresh": True},
            token=args.token,
            timeout=180,
        )
        advice2 = ((run2.get("report") or {}).get("advice") or []) if status == 200 else []
        check(
            "无确诊不给疾病专属建议",
            status == 200 and len(advice2) == 0,
            f"advice={len(advice2)} 条",
        )

    # 汇总
    failed = [r for r in RESULTS if r[0] == "FAIL"]
    passed = [r for r in RESULTS if r[0] == "PASS"]
    skipped = [r for r in RESULTS if r[0] == "SKIP"]
    print(f"\n结果：PASS {len(passed)} · FAIL {len(failed)} · SKIP {len(skipped)}（模式 {mode}）")
    if failed:
        print("未通过项：")
        for state, name, detail in failed:
            print(f"  - {name}: {detail}")
        return 1
    print("自检通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
