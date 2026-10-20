"""Convert validated internal Advice items to the backend's strict F09 callback shape.

This module only prepares a payload. It does not submit a callback or claim that
the current demonstration knowledge articles have been professionally reviewed.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlsplit

from .f08_adapter import F08ContractError, adapt_inputs
from .validation import scan_text


CATEGORY_MAP = {
    "diet": "diet",
    "activity": "lifestyle",
    "record_management": "record_management",
    "medication": "medication_adherence",
    "seek_care": "care_consultation",
}
KNOWLEDGE_REF = re.compile(r"^[A-Za-z0-9._:/-]{1,160}$")


@dataclass(frozen=True)
class F09Mapping:
    payload: dict[str, Any]
    dropped: list[dict[str, str]]

    @property
    def ready(self) -> bool:
        return bool(self.payload["advice"])


def _approved_knowledge(ref: str, catalog: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    meta = catalog.get(ref)
    if not isinstance(meta, Mapping) or meta.get("review_status") != "approved":
        return None
    version = meta.get("version")
    title = meta.get("title")
    publisher = meta.get("publisher")
    if not KNOWLEDGE_REF.fullmatch(ref):
        return None
    if not all(isinstance(value, str) and value.strip() for value in (version, title, publisher)):
        return None
    if len(version) > 64 or len(title) > 240 or len(publisher) > 160:
        return None
    evidence = {
        "evidence_type": "knowledge", "knowledge_ref": ref,
        "knowledge_version": version, "knowledge_title": title,
        "knowledge_publisher": publisher,
    }
    url = meta.get("citation_url")
    if url is not None:
        if not isinstance(url, str):
            return None
        try:
            parsed = urlsplit(url)
            safe = parsed.scheme.lower() == "https" and bool(parsed.hostname)
            safe = safe and parsed.username is None and parsed.password is None and parsed.port != 0
        except ValueError:
            return None
        if not safe:
            return None
        evidence["citation_url"] = url
    return evidence


def map_f09_advice(
    run: Any,
    frozen_input: dict[str, Any],
    *,
    f08_status: str,
    approved_knowledge: Mapping[str, Mapping[str, Any]] | None = None,
) -> F09Mapping:
    """Map advice after F08 is accepted as succeeded/partial by the backend.

    Patient evidence is reconstructed only from the validated frozen input. Any
    unresolved observation or unapproved knowledge citation drops its entire
    advice item, because retaining the text with partial evidence could mislead.
    """
    if f08_status not in {"succeeded", "partial"}:
        raise F08ContractError("F09 requires an accepted succeeded/partial F08 task")
    task_id = str(uuid.UUID(str(run.task_id)))
    request, provenance = adapt_inputs(frozen_input, task_id, frozen_input.get("analysis_type"))
    if str(run.patient_id) != request.patient_id:
        raise F08ContractError("F09 patient identity differs from frozen input")
    report = run.report or {}
    if not isinstance(report, dict):
        raise F08ContractError("F09 report must be object")
    observations = {obs.get("obs_id"): obs for obs in report.get("observations") or []
                    if isinstance(obs, dict) and isinstance(obs.get("obs_id"), str)}
    catalog = approved_knowledge or {}
    mapped: list[dict[str, Any]] = []
    dropped: list[dict[str, str]] = []

    for index, item in enumerate(report.get("advice") or [], start=1):
        item_id = str(item.get("advice_id") or f"advice_{index:03d}") if isinstance(item, dict) else f"advice_{index:03d}"

        def reject(reason: str) -> None:
            dropped.append({"advice_id": item_id, "reason": reason})

        if not isinstance(item, dict):
            reject("invalid_item")
            continue
        raw_category = item.get("category")
        category = CATEGORY_MAP.get(raw_category) if isinstance(raw_category, str) else None
        title, body = item.get("title"), item.get("body")
        if not category or not isinstance(title, str) or not isinstance(body, str):
            reject("invalid_core_fields")
            continue
        title, body = title.strip(), body.strip()
        if not 1 <= len(title) <= 160 or not 1 <= len(body) <= 3000 or scan_text(title + " " + body):
            reject("invalid_or_unsafe_text")
            continue
        uncertainty, applicability = item.get("uncertainty") or "", item.get("applicability") or ""
        if not isinstance(uncertainty, str) or not isinstance(applicability, str):
            reject("invalid_qualifiers")
            continue
        if len(uncertainty) > 240 or len(applicability) > 240:
            reject("qualifier_too_long")
            continue
        observation_ids = item.get("evidence_obs_ids") or []
        knowledge_ids = item.get("knowledge_ids") or []
        if not isinstance(observation_ids, list) or not isinstance(knowledge_ids, list):
            reject("invalid_references")
            continue
        evidence: list[dict[str, Any]] = []
        keys: set[tuple[Any, ...]] = set()
        invalid = False
        for obs_id in observation_ids:
            obs = observations.get(obs_id)
            if not obs or obs.get("needs_confirmation") or not obs.get("source_record_ids"):
                invalid = True
                break
            for source_id in obs["source_record_ids"]:
                source = provenance.get(str(source_id))
                if not source:
                    invalid = True
                    break
                key = (source["source_type"], source["source_id"], source["source_version"], source["locator"])
                if key not in keys:
                    keys.add(key)
                    evidence.append({"evidence_type": "patient", **source})
            if invalid:
                break
        if invalid:
            reject("unmapped_patient_evidence")
            continue
        for ref in knowledge_ids:
            knowledge = _approved_knowledge(ref, catalog) if isinstance(ref, str) else None
            if knowledge is None:
                invalid = True
                break
            if ("knowledge", ref) not in keys:
                keys.add(("knowledge", ref))
                evidence.append(knowledge)
        if invalid:
            reject("knowledge_not_approved_or_incomplete")
            continue
        if not 1 <= len(evidence) <= 30:
            reject("evidence_count_invalid")
            continue
        mapped.append({
            "category": category,
            "priority": "important" if item["category"] == "seek_care" else "routine",
            "title": title, "body": body,
            "uncertainty": {"level": "high", "reasons": [uncertainty] if uncertainty else []},
            "applicability": {"scope": "general", "limitations": [applicability] if applicability else []},
            "safety_boundary": {"not_diagnosis": True, "not_prescription": True,
                                "risk_alert_priority": True},
            "version": "1.0",
            "evidence": evidence,
        })
        if len(mapped) >= 20:
            break
    callback_id = str(uuid.uuid5(uuid.NAMESPACE_URL,
                                 f"gutcare:f09:{task_id}:{frozen_input['snapshot_hash']}:advice:v1"))
    return F09Mapping(
        payload={"callback_id": callback_id, "schema_version": "1.0", "advice": mapped},
        dropped=dropped,
    )
