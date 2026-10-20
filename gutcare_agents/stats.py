"""确定性统计。

原则：凡是“计数、分布、缺失、时间对齐”都必须由程序算出，
模型只负责解释这些数字（`Trend` 的 fact 表）。这样做的直接收益是：
建议里的每个数字都能追溯到一条记录，评审也能当场核对。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

BRISTOL_LABELS = {
    1: "1型：分离硬球状",
    2: "2型：香肠状但表面凹凸",
    3: "3型：香肠状表面有裂纹",
    4: "4型：光滑柔软香肠状",
    5: "5型：柔软团块边缘清晰",
    6: "6型：糊状边缘不清",
    7: "7型：水样无固体成分",
}

SYMPTOM_SEVERE_THRESHOLD = 7
MEAL_WINDOW_HOURS = 6


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _kind_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(r.get("kind")) for r in records))


def _days_with_records(records: list[dict[str, Any]]) -> set[str]:
    days = set()
    for r in records:
        ts = parse_ts(r.get("occurred_at"))
        if ts:
            days.add(ts.date().isoformat())
    return days


def _date_range(start: datetime, end: datetime) -> list[str]:
    days: list[str] = []
    cursor = start.date()
    while cursor <= end.date():
        days.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return days


@dataclass
class Facts:
    """跨来源事实表。字段命名尽量贴近业务术语，便于把 LLM 的输出对齐。"""

    patient_id: str
    window_start: str
    window_end: str
    days: int
    days_with_records: int
    total_records: int
    missing_days: list[str]
    excluded: dict[str, int]
    by_kind: dict[str, int]
    per_kind_days: dict[str, int]
    bristol: dict[str, Any]
    symptoms: dict[str, Any]
    medication: dict[str, Any]
    diet: dict[str, Any]
    activity: dict[str, Any]
    mood: dict[str, Any]
    temporal: dict[str, Any]
    sufficiency: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def compute_facts(
    *,
    patient_id: str,
    profile: dict[str, Any] | None,
    records: list[dict[str, Any]],
    all_records: list[dict[str, Any]],
    window_start: datetime,
    window_end: datetime,
    min_days_coverage: int = 3,
    min_records_for_trend: int = 5,
) -> Facts:
    """records 已由调用方按 confirmed + 归属 + 时间窗过滤。"""
    kinds = _kind_counts(records)
    days_in_window = _date_range(window_start, window_end)
    days_present = _days_with_records(records)
    missing = [d for d in days_in_window if d not in days_present]

    excluded = dict(
        Counter(
            str(r.get("status"))
            for r in all_records
            if str(r.get("status", "confirmed")) != "confirmed"
        )
    )

    per_kind_days: dict[str, int] = {}
    for kind in ("diet", "stool", "symptom", "medication", "activity", "mood"):
        per_kind_days[kind] = len(_days_with_records([r for r in records if r.get("kind") == kind]))

    # ---- 排便 ----
    stool = [r for r in records if r.get("kind") == "stool"]
    bristol_counter: Counter[int] = Counter()
    bristol_unknown = 0
    blood_reported = 0
    for r in stool:
        data = r.get("data") or {}
        value = data.get("bristol_type")
        if isinstance(value, int) and 1 <= value <= 7:
            bristol_counter[value] += 1
        elif value in (None, "", 0):
            bristol_unknown += 1
        if data.get("blood_reported") is True:
            blood_reported += 1
    bristol = {
        "count": len(stool),
        "distribution": {str(k): bristol_counter[k] for k in sorted(bristol_counter)},
        "distribution_labels": {BRISTOL_LABELS[k]: bristol_counter[k] for k in sorted(bristol_counter)},
        "unknown_or_unconfirmed": bristol_unknown,
        "blood_reported_count": blood_reported,
        "abnormal_ratio": round(
            sum(v for k, v in bristol_counter.items() if k in (1, 2, 6, 7)) / max(1, sum(bristol_counter.values())),
            3,
        ),
    }

    # ---- 症状 ----
    symptoms = [r for r in records if r.get("kind") == "symptom"]
    sev_by_name: dict[str, list[int]] = defaultdict(list)
    relation = Counter()
    for r in symptoms:
        data = r.get("data") or {}
        name = str(data.get("name") or "未命名症状")
        sev = data.get("severity")
        if isinstance(sev, (int, float)):
            sev_by_name[name].append(int(sev))
        relation[str(data.get("event_relation") or "unknown")] += 1
    symptom_summary = {
        "count": len(symptoms),
        "by_name": {
            name: {
                "count": len(vals),
                "mean_severity": round(sum(vals) / len(vals), 1) if vals else None,
                "max_severity": max(vals) if vals else None,
                "severe_count": sum(1 for v in vals if v >= SYMPTOM_SEVERE_THRESHOLD),
            }
            for name, vals in sev_by_name.items()
        },
        "event_relation": dict(relation),
        "severe_threshold": SYMPTOM_SEVERE_THRESHOLD,
    }

    # ---- 服药 ----
    meds = [r for r in records if r.get("kind") == "medication"]
    taken = sum(1 for r in meds if (r.get("data") or {}).get("taken") is True)
    not_taken = sum(1 for r in meds if (r.get("data") or {}).get("taken") is False)
    unconfirmed = sum(1 for r in meds if (r.get("data") or {}).get("taken") not in (True, False))
    plans = {
        str((r.get("data") or {}).get("plan_id") or ""): {
            "medicine_name": (r.get("data") or {}).get("medicine_name", ""),
            "records": 1,
        }
        for r in meds
    }
    medication = {
        "count": len(meds),
        "taken": taken,
        "not_taken": not_taken,
        "unconfirmed": unconfirmed,
        "adherence_ratio_of_confirmed": round(taken / max(1, taken + not_taken), 3) if (taken + not_taken) else None,
        "plans": list(plans.keys()),
        "note": "「未确认」不等于「未服用」；依从性分母只含可判定事件",
    }

    # ---- 饮食 ----
    diets = [r for r in records if r.get("kind") == "diet"]
    meal_counter: Counter[str] = Counter()
    food_counter: Counter[str] = Counter()
    for r in diets:
        data = r.get("data") or {}
        meal_counter[str(data.get("meal_type") or "unknown")] += 1
        desc = str(data.get("description") or "")
        for token in desc.replace("，", "、").replace(",", "、").replace(" ", "、").split("、"):
            token = token.strip()
            if token:
                food_counter[token] += 1
    diet = {
        "count": len(diets),
        "by_meal_type": dict(meal_counter),
        "top_food_tokens": food_counter.most_common(8),
        "distinct_food_tokens": len(food_counter),
    }

    # ---- 活动 ----
    acts = [r for r in records if r.get("kind") == "activity"]
    total_minutes = 0
    for r in acts:
        value = (r.get("data") or {}).get("duration_minutes")
        if isinstance(value, (int, float)):
            total_minutes += int(value)
    activity = {
        "count": len(acts),
        "total_minutes": total_minutes,
        "by_type": dict(Counter(str((r.get("data") or {}).get("activity_type") or "unknown") for r in acts)),
    }

    # ---- 心情 ----
    moods = [r for r in records if r.get("kind") == "mood"]
    mood_vals: list[int] = []
    stress_vals: list[int] = []
    tag_counter: Counter[str] = Counter()
    series: list[dict[str, Any]] = []
    for r in moods:
        data = r.get("data") or {}
        mv, sv = data.get("mood_level"), data.get("stress_level")
        if isinstance(mv, (int, float)):
            mood_vals.append(int(mv))
        if isinstance(sv, (int, float)):
            stress_vals.append(int(sv))
        for tag in data.get("mood_tags") or []:
            tag_counter[str(tag)] += 1
        ts = parse_ts(r.get("occurred_at"))
        series.append(
            {
                "record_id": r.get("record_id"),
                "date": ts.date().isoformat() if ts else "",
                "mood_level": mv,
                "stress_level": sv,
            }
        )
    mood = {
        "count": len(moods),
        "scale": "project_1_to_5（患者自报，非临床量表）",
        "mean_mood": round(sum(mood_vals) / len(mood_vals), 2) if mood_vals else None,
        "mean_stress": round(sum(stress_vals) / len(stress_vals), 2) if stress_vals else None,
        "min_mood": min(mood_vals) if mood_vals else None,
        "max_stress": max(stress_vals) if stress_vals else None,
        "top_tags": tag_counter.most_common(5),
        "series": series,
        "note": "只能描述同一时间段的共同变化或关联，不能声称情绪导致消化症状",
    }

    # ---- 时间邻近（餐次之后 0-6 小时内出现的症状）----
    meal_times = [
        (r.get("record_id"), parse_ts(r.get("occurred_at")))
        for r in diets
        if parse_ts(r.get("occurred_at"))
    ]
    adjacencies: list[dict[str, Any]] = []
    for r in symptoms:
        ts = parse_ts(r.get("occurred_at"))
        if not ts:
            continue
        candidates = [
            (rid, meal_ts)
            for rid, meal_ts in meal_times
            if meal_ts and timedelta(0) <= (ts - meal_ts) <= timedelta(hours=MEAL_WINDOW_HOURS)
        ]
        if not candidates:
            continue
        rid, meal_ts = min(candidates, key=lambda x: ts - x[1])
        adjacencies.append(
            {
                "symptom_record_id": r.get("record_id"),
                "meal_record_id": rid,
                "delta_hours": round((ts - meal_ts).total_seconds() / 3600, 2),
                "symptom": (r.get("data") or {}).get("name"),
                "severity": (r.get("data") or {}).get("severity"),
                "meal": (r.get("data") or {}).get("description"),
            }
        )
    temporal = {
        "window_hours": MEAL_WINDOW_HOURS,
        "symptom_total": len(symptoms),
        "symptoms_with_preceding_meal": len(adjacencies),
        "ratio": round(len(adjacencies) / max(1, len(symptoms)), 3),
        "mean_delta_hours": round(
            sum(a["delta_hours"] for a in adjacencies) / len(adjacencies), 2
        )
        if adjacencies
        else None,
        "examples": adjacencies[:5],
        "interpretation_limit": "只能表述为时间邻近，不能据此认定因果关系；也不代表已排除其他原因",
    }

    # ---- 数据充足性 ----
    reasons: list[str] = []
    profile = profile or {}
    if not (profile.get("diagnosis_status") == "confirmed" and profile.get("diagnosis")):
        reasons.append("缺少已确认的诊断信息，只能给出一般性记录建议")
    if not records:
        reasons.append("时间范围内没有有效记录")
    if len(days_present) < min_days_coverage:
        reasons.append(f"有记录的天数为 {len(days_present)} 天，少于 {min_days_coverage} 天")
    if len(records) < min_records_for_trend:
        reasons.append(f"有效记录 {len(records)} 条，少于 {min_records_for_trend} 条")
    if kinds.get("diet", 0) == 0 or kinds.get("symptom", 0) == 0:
        reasons.append("饮食或症状记录缺失，无法做跨来源时间关联")
    sufficiency = {
        "enough_for_advice": not reasons,
        "reasons": reasons,
        "min_days_coverage": min_days_coverage,
        "min_records_for_trend": min_records_for_trend,
    }

    return Facts(
        patient_id=patient_id,
        window_start=window_start.isoformat(),
        window_end=window_end.isoformat(),
        days=len(days_in_window),
        days_with_records=len(days_present),
        total_records=len(records),
        missing_days=missing,
        excluded=excluded,
        by_kind=kinds,
        per_kind_days=per_kind_days,
        bristol=bristol,
        symptoms=symptom_summary,
        medication=medication,
        diet=diet,
        activity=activity,
        mood=mood,
        temporal=temporal,
        sufficiency=sufficiency,
    )


# ----------------------------------------------------------------------
# 从 TaskRequest 直接构建事实表（Trend 与 Advice 共用，避免两套筛选口径）
# ----------------------------------------------------------------------
def request_records(request: Any) -> list[dict[str, Any]]:
    """已确认 + 属于本患者 + 落在时间窗内的记录。"""
    start = parse_ts(getattr(request.time_range, "start", None))
    end = parse_ts(getattr(request.time_range, "end", None))
    out = []
    for record in request.confirmed_records():
        ts = parse_ts(record.get("occurred_at"))
        if ts is None:
            continue
        if start and ts < start:
            continue
        if end and ts > end:
            continue
        out.append(record)
    return out


def window_of(records: list[dict[str, Any]], request: Any) -> tuple[datetime, datetime]:
    start = parse_ts(getattr(request.time_range, "start", None))
    end = parse_ts(getattr(request.time_range, "end", None))
    if start and end:
        return start, end
    stamps = [s for s in (parse_ts(r.get("occurred_at")) for r in records) if s]
    if stamps:
        return (start or min(stamps)), (end or max(stamps))
    from datetime import datetime as _dt, timedelta as _td

    now = _dt.now().astimezone()
    return (start or now - _td(days=7)), (end or now)


def facts_for_request(
    request: Any,
    *,
    min_days_coverage: int = 3,
    min_records_for_trend: int = 5,
) -> tuple[Facts, list[dict[str, Any]]]:
    records = request_records(request)
    window_start, window_end = window_of(records, request)
    facts = compute_facts(
        patient_id=request.patient_id,
        profile=request.profile.to_dict() if request.profile else None,
        records=records,
        all_records=request.records,
        window_start=window_start,
        window_end=window_end,
        min_days_coverage=min_days_coverage,
        min_records_for_trend=min_records_for_trend,
    )
    return facts, records
