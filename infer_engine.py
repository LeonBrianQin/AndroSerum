# infer_engine_incremental.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from .llm_client import OpenAIClient, LLMError
except Exception:
    from llm_client import OpenAIClient, LLMError  # type: ignore

try:
    from .params import LLMParams
except Exception:
    from params import LLMParams  # type: ignore

try:
    from .proto_db import PrototypeDB, ClusterMatcher, MatchParams
except Exception:
    from proto_db import PrototypeDB, ClusterMatcher, MatchParams  # type: ignore

try:
    from .utils_io import safe_read_json, safe_write_json, resolve_under_out
except Exception:
    from utils_io import safe_read_json, safe_write_json, resolve_under_out  # type: ignore


def _parse_years(spec: Any) -> List[str]:
    if spec is None:
        return []
    if isinstance(spec, (list, tuple, set)):
        return [str(x).strip() for x in spec if str(x).strip()]
    s = str(spec).strip()
    if not s:
        return []
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    if "-" in s and s.replace("-", "").isdigit():
        a, b = s.split("-", 1)
        if a.isdigit() and b.isdigit():
            aa, bb = int(a), int(b)
            lo, hi = (aa, bb) if aa <= bb else (bb, aa)
            return [str(y) for y in range(lo, hi + 1)]
    return [s]


def _extract_text(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        choices = raw.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: List[str] = []
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                if parts:
                    return "\n".join(parts)
        if isinstance(raw.get("output_text"), str):
            return raw.get("output_text") or ""
        out = raw.get("output")
        if isinstance(out, list):
            parts: List[str] = []
            for item in out:
                if not isinstance(item, dict):
                    continue
                for c in item.get("content") or []:
                    if isinstance(c, dict) and isinstance(c.get("text"), str):
                        parts.append(c["text"])
            if parts:
                return "\n".join(parts)
        for k in ["text", "content"]:
            if isinstance(raw.get(k), str):
                return raw.get(k) or ""
    return str(raw)


def _safe_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    l = text.find("{")
    r = text.rfind("}")
    if l >= 0 and r > l:
        try:
            return json.loads(text[l:r + 1])
        except Exception:
            return {}
    return {}


def _resolve_single_apk_to_dir(out_dir: Path, single_apk: Optional[str]) -> List[Path]:
    if not single_apk:
        return []
    sp = Path(str(single_apk))
    if sp.is_dir():
        return [sp]
    if sp.suffix.lower() == ".apk":
        cand = out_dir / sp.parent.name / sp.stem.upper()
        return [cand] if cand.is_dir() else []
    sha = sp.stem.upper() if sp.suffix else str(single_apk).upper()
    for yd in sorted(out_dir.iterdir()):
        if not yd.is_dir() or yd.name.startswith("_"):
            continue
        cand = yd / sha
        if cand.is_dir():
            return [cand]
    return []


def _iter_apk_dirs(out_dir: Path, years: List[str], single_apk: Optional[str], limit_apks: int) -> List[Path]:
    if single_apk:
        return _resolve_single_apk_to_dir(out_dir, single_apk)
    items: List[Path] = []
    for y in years:
        yd = out_dir / y
        if not yd.exists():
            continue
        for apk_dir in sorted([p for p in yd.iterdir() if p.is_dir()]):
            items.append(apk_dir)
            if limit_apks > 0 and len(items) >= limit_apks:
                return items
    return items


def _load_units(apk_dir: Path, kb_subdir: str, raw_subdir: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    kb_dir = apk_dir / kb_subdir
    raw_dir = apk_dir / raw_subdir
    if not kb_dir.exists():
        return items
    for uf in sorted(kb_dir.glob("unit_*.json")):
        kb = safe_read_json(uf)
        if not kb:
            continue
        raw = safe_read_json(raw_dir / uf.name) if raw_dir.exists() else None
        items.append({"kb": kb, "raw": raw, "name": uf.name})
    return items


def _chunked(seq: List[Any], batch_size: int) -> Iterable[List[Any]]:
    if batch_size <= 0:
        batch_size = 1
    for i in range(0, len(seq), batch_size):
        yield seq[i:i + batch_size]


def _float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _trim_raw_nodes(raw: Optional[Dict[str, Any]], max_nodes: int) -> List[Dict[str, Any]]:
    if not raw:
        return []
    nodes = list(raw.get("nodes") or [])
    nodes.sort(key=lambda n: _float((n or {}).get("score"), 0.0), reverse=True)
    out: List[Dict[str, Any]] = []
    for n in nodes[:max_nodes]:
        ev = n.get("evidence") or {}
        out.append({
            "node_id": n.get("node_id"),
            "sig": n.get("sig"),
            "score": round(_float(n.get("score"), 0.0), 4),
            "primary_role": n.get("primary_role"),
            "role_flags": list(n.get("role_flags") or [])[:10],
            "perms": list(n.get("perms") or [])[:8],
            "active_symbolic_features": list(n.get("active_symbolic_features") or [])[:12],
            "evidence": {
                "strings": list(ev.get("strings_raw_preview") or [])[:6],
                "domains": list(ev.get("domains_topk") or [])[:6],
                "urls": list(ev.get("urls_topk") or [])[:6],
                "ips": list(ev.get("ips_topk") or [])[:6],
                "headers": list(ev.get("http_headers_topk") or [])[:6],
                "intent_actions": list(ev.get("intent_actions_topk") or [])[:6],
                "file_paths": list(ev.get("file_paths_topk") or [])[:6],
                "spans": list(ev.get("evidence_spans") or [])[:6],
            },
        })
    return out


def _light_exemplar(e: Dict[str, Any], max_items: int) -> Dict[str, Any]:
    ss = e.get("symbolic_summary") or {}
    return {
        "unit_id": str(e.get("unit_id", "")),
        "sim": round(_float(e.get("sim"), 0.0), 4),
        "q": round(_float(e.get("q"), 0.0), 4),
        "symbolic_summary": {
            "sapi": list(ss.get("sapi") or [])[:max_items],
            "perms": list(ss.get("perms") or [])[:max_items],
            "hints": list(ss.get("hints") or [])[:max_items],
            "lexical": list(ss.get("lexical") or [])[:max_items],
        },
    }


def _proto_summary(rank: Dict[str, Any], max_exemplars: int, max_freq_psi: int, max_items: int) -> Dict[str, Any]:
    cluster = rank["cluster"]
    base = cluster.get("base_info") or {}
    sym = cluster.get("symbolic_info") or {}
    audit = cluster.get("audit_info") or {}
    size = int(base.get("size", 0) or 0)
    psi_count = sym.get("psi_count") or {}
    freq = []
    for k, v in psi_count.items():
        try:
            c = int(v)
        except Exception:
            continue
        freq.append({"psi": str(k), "count": c, "support": round(c / (size or 1), 4)})
    freq.sort(key=lambda x: (x["count"], x["psi"]), reverse=True)
    return {
        "cluster_id": str(base.get("cluster_id", "")),
        "size": size,
        "score": round(_float(rank.get("score"), 0.0), 4),
        "semantic_sim": round(_float(rank.get("semantic_sim"), 0.0), 4),
        "symbolic_jaccard": round(_float(rank.get("symbolic_jaccard"), 0.0), 4),
        "family_prior": dict(sym.get("family_count") or {}),
        "behavior_prior": dict(sym.get("behavior_count") or {}),
        "freq_psi": freq[:max_freq_psi],
        "role_count": audit.get("role_count") or {},
        "anchor_sigs": list(audit.get("anchor_sigs") or [])[:12],
        "exemplars": [
            _light_exemplar(e, max_items=max_items)
            for e in list(cluster.get("exemplars") or [])[:max_exemplars]
        ],
    }


def _build_bu_pack(
    kb: Dict[str, Any],
    raw: Optional[Dict[str, Any]],
    ranks: List[Dict[str, Any]],
    max_exemplars: int,
    max_freq_psi: int,
    max_items: int,
    max_nodes: int,
) -> Dict[str, Any]:
    base = kb.get("base_info") or {}
    sym = kb.get("symbolic_info") or {}
    audit = kb.get("audit_info") or {}
    return {
        "unit_id": str(base.get("unit_id", "")),
        "bu_id": str(base.get("unit_id", "")),
        "symbolic": {
            "sapi": list(sym.get("SAPI_set") or [])[:max_items],
            "perms": sorted(list((sym.get("perms_set") or {}).keys()))[:max_items],
            "hints": list(sym.get("hint_set") or [])[:max_items],
            "lexical": list(sym.get("lexical_set") or [])[:max_items],
            "domains": list(sym.get("domains_topk") or [])[:12],
            "urls": list(sym.get("urls_topk") or [])[:12],
            "ips": list(sym.get("ips_topk") or [])[:12],
            "headers": list(sym.get("http_headers_topk") or [])[:12],
            "intent_actions": list(sym.get("intent_actions_topk") or [])[:12],
            "file_paths": list(sym.get("file_paths_topk") or [])[:12],
        },
        "audit": {
            "anchor_sigs": list(audit.get("anchor_sigs") or [])[:16],
            "role_count": audit.get("role_count") or {},
            "source_methods": list(audit.get("source_methods") or [])[:16],
            "sink_methods": list(audit.get("sink_methods") or [])[:16],
            "transform_methods": list(audit.get("transform_methods") or [])[:16],
            "top_nodes": list(audit.get("top_nodes") or [])[:10],
            "behavior_evidence_summary": audit.get("behavior_evidence_summary") or {},
            "evidence_spans_topk": list(audit.get("evidence_spans_topk") or [])[:16],
        },
        "raw_nodes": _trim_raw_nodes(raw, max_nodes=max_nodes),
        "prototype_matches": [
            _proto_summary(
                r,
                max_exemplars=max_exemplars,
                max_freq_psi=max_freq_psi,
                max_items=max_items,
            )
            for r in ranks
        ],
    }


PROMPT_BU_LOCAL = (
    "You are an expert in static analysis of Android malware.\n"
    "You will receive a Behavior Unit (BU) and its most relevant historical prototype.\n"
    "Please output a JSON object based only on the given fields:\n"
    "1) The local behavioral intent of this BU (local_intents)\n"
    "2) The risk tendency (risk_signal: malicious|suspicious|benign|unknown)\n"
    "3) A list of the most relevant prototypes\n"
    "4) Supporting citations that use only bu_id / cluster_id / node_id visible in the input.\n"
    "Do not provide an APK-level final conclusion, and do not fabricate any information."
)

SCHEMA_BU_LOCAL = {
    "bu_id": "str",
    "risk_signal": "malicious|suspicious|benign|unknown",
    "confidence": "float in [0,100]",
    "local_intents": [{
        "label": "str",
        "confidence": "float",
        "support": [{"bu_id": "...", "cluster_id": "optional", "node_id": "optional"}]
    }],
    "best_clusters": [{"cluster_id": "str", "reason": "str"}],
    "summary": "short summary"
}

PROMPT_APK_UPDATE = (
    "You are an expert in static analysis of Android malware.\n"
    "You will receive newly added BU-level local summaries from the same APK across multiple rounds.\n"
    "Please incrementally update the APK-level judgment based on previous_state and new_bu_summaries, and output the complete JSON state.\n"
    "Requirements:\n"
    "1) Update only based on the input evidence, and do not forget conclusions in previous_state that are still supported.\n"
    "2) If a newly added BU changes a previous judgment, you must explain it in changed_fields.\n"
    "3) family must be unknown if the evidence is insufficient.\n"
    "4) support can only cite bu_id / cluster_id / node_id that appear in the input."
)

SCHEMA_APK_STATE = {
    "apk_id": "str",
    "verdict": "malicious|benign|unknown",
    "confidence": "float in [0,100]",
    "family": "str or unknown",
    "family_candidates": [{"family": "str", "confidence": "float", "support_bu_ids": ["..."]}],
    "behaviors": [{"label": "str", "confidence": "float", "support_bu_ids": ["..."]}],
    "mechanisms": [{"claim": "str", "support": [{"bu_id": "...", "cluster_id": "optional", "node_id": "optional"}]}],
    "processed_bu_ids": ["..."],
    "changed_fields": ["verdict", "family", "behaviors", "mechanisms"],
    "note": "short rationale"
}

PROMPT_FINAL_RECONCILE = (
    "You are an expert in static analysis of Android malware.\n"
    "You will receive the final candidate state of an APK and all BU-level local summaries.\n"
    "Please perform a final consistency refinement: remove unsupported conclusions, merge duplicate behaviors/mechanisms, and output the final JSON.\n"
    "Do not add assertions that are not supported by the input."
)


def _empty_state(apk_id: str) -> Dict[str, Any]:
    return {
        "apk_id": apk_id,
        "verdict": "unknown",
        "confidence": 0.0,
        "family": "unknown",
        "family_candidates": [],
        "behaviors": [],
        "mechanisms": [],
        "processed_bu_ids": [],
        "changed_fields": [],
        "note": "",
    }


def _normalize_local_summary(apk_id: str, bu_id: str, obj: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "bu_id": bu_id,
        "risk_signal": str(obj.get("risk_signal", "unknown")),
        "confidence": _float(obj.get("confidence"), 0.0),
        "local_intents": obj.get("local_intents") if isinstance(obj.get("local_intents"), list) else [],
        "best_clusters": obj.get("best_clusters") if isinstance(obj.get("best_clusters"), list) else [],
        "summary": str(obj.get("summary", ""))[:1200],
    }
    return out


def _normalize_state(apk_id: str, obj: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(fallback)
    out["apk_id"] = apk_id

    verdict = str(obj.get("verdict", fallback.get("verdict", "unknown"))).strip().lower()
    if verdict not in {"malicious", "benign", "unknown"}:
        verdict = str(fallback.get("verdict", "unknown")).strip().lower()
    if verdict not in {"malicious", "benign", "unknown"}:
        verdict = "unknown"
    out["verdict"] = verdict

    out["confidence"] = max(
        0.0,
        min(100.0, _float(obj.get("confidence"), _float(fallback.get("confidence"), 0.0)))
    )

    out["family"] = str(obj.get("family", fallback.get("family", "unknown"))).strip() or "unknown"

    out["family_candidates"] = (
        obj.get("family_candidates")
        if isinstance(obj.get("family_candidates"), list)
        else list(fallback.get("family_candidates") or [])
    )
    out["behaviors"] = (
        obj.get("behaviors")
        if isinstance(obj.get("behaviors"), list)
        else list(fallback.get("behaviors") or [])
    )
    out["mechanisms"] = (
        obj.get("mechanisms")
        if isinstance(obj.get("mechanisms"), list)
        else list(fallback.get("mechanisms") or [])
    )
    out["processed_bu_ids"] = (
        obj.get("processed_bu_ids")
        if isinstance(obj.get("processed_bu_ids"), list)
        else list(fallback.get("processed_bu_ids") or [])
    )
    out["changed_fields"] = (
        obj.get("changed_fields")
        if isinstance(obj.get("changed_fields"), list)
        else []
    )
    out["note"] = str(obj.get("note", fallback.get("note", "")))[:2000]

    
    if out["verdict"] != "malicious":
        out["family"] = "unknown"

    return out


def _call_llm_json(client: OpenAIClient, prompt: str, max_output_tokens: int) -> Dict[str, Any]:
    raw = client.chat_completions(prompt, max_output_tokens=max_output_tokens)
    text = _extract_text(raw)
    obj = _safe_json(text)
    return {"raw": raw, "text": text, "obj": obj}


def _call_llm_bu_local(client: OpenAIClient, apk_id: str, pack: Dict[str, Any], max_output_tokens: int) -> Dict[str, Any]:
    prompt = (
        PROMPT_BU_LOCAL
        + "\nJSON schema:\n"
        + json.dumps(SCHEMA_BU_LOCAL, ensure_ascii=False)
        + "\nAPK_ID: " + apk_id
        + "\nBU_PACK:\n"
        + json.dumps(pack, ensure_ascii=False)
    )
    return _call_llm_json(client, prompt, max_output_tokens=max_output_tokens)


def _call_llm_state_update(
    client: OpenAIClient,
    apk_id: str,
    previous_state: Dict[str, Any],
    new_bu_summaries: List[Dict[str, Any]],
    step_index: int,
    total_steps: int,
    max_output_tokens: int,
) -> Dict[str, Any]:
    prompt = (
        PROMPT_APK_UPDATE
        + "\nJSON schema:\n"
        + json.dumps(SCHEMA_APK_STATE, ensure_ascii=False)
        + f"\nAPK_ID: {apk_id}\nSTEP: {step_index}/{total_steps}"
        + "\nPREVIOUS_STATE:\n"
        + json.dumps(previous_state, ensure_ascii=False)
        + "\nNEW_BU_SUMMARIES:\n"
        + json.dumps(new_bu_summaries, ensure_ascii=False)
    )
    return _call_llm_json(client, prompt, max_output_tokens=max_output_tokens)


def _call_llm_final_reconcile(
    client: OpenAIClient,
    apk_id: str,
    current_state: Dict[str, Any],
    all_local_summaries: List[Dict[str, Any]],
    max_output_tokens: int,
) -> Dict[str, Any]:
    prompt = (
        PROMPT_FINAL_RECONCILE
        + "\nJSON schema:\n"
        + json.dumps(SCHEMA_APK_STATE, ensure_ascii=False)
        + f"\nAPK_ID: {apk_id}"
        + "\nCURRENT_STATE:\n"
        + json.dumps(current_state, ensure_ascii=False)
        + "\nALL_BU_LOCAL_SUMMARIES:\n"
        + json.dumps(all_local_summaries, ensure_ascii=False)
    )
    return _call_llm_json(client, prompt, max_output_tokens=max_output_tokens)


def _pack_sort_key(pack: Dict[str, Any]) -> Any:
    proto = list(pack.get("prototype_matches") or [])
    best_proto = _float(proto[0].get("score"), 0.0) if proto else 0.0
    anchors = len(pack.get("audit", {}).get("anchor_sigs", []) or [])
    node_score = sum(_float(n.get("score"), 0.0) for n in list(pack.get("raw_nodes") or [])[:6])
    return (-best_proto, -anchors, -node_score, str(pack.get("bu_id", "")))


def _collect_behavior_labels(behaviors: Any) -> List[str]:
    out: List[str] = []
    if not isinstance(behaviors, list):
        return out
    for item in behaviors:
        if isinstance(item, dict):
            s = str(item.get("label", "")).strip()
        else:
            s = str(item).strip()
        if s:
            out.append(s.lower())
    return out


def _collect_mechanism_texts(mechanisms: Any) -> List[str]:
    out: List[str] = []
    if not isinstance(mechanisms, list):
        return out
    for item in mechanisms:
        if isinstance(item, dict):
            s = str(item.get("claim", "")).strip()
        else:
            s = str(item).strip()
        if s:
            out.append(s.lower())
    return out


def _count_strong_malicious_signals(behavior_labels: List[str], mechanism_texts: List[str]) -> int:
    text = "\n".join(list(behavior_labels) + list(mechanism_texts)).lower()

    strong_keywords = [
        "credential theft",
        "account theft",
        "exfiltration",
        "data exfiltration",
        "spyware",
        "overlay abuse",
        "overlay attack",
        "banker",
        "ransomware",
        "sms fraud",
        "premium sms",
        "botnet",
        "dropper",
        "payload download",
        "accessibility abuse",
        "keylogging",
        "steal",
        "stolen",
    ]

    hits = 0
    for kw in strong_keywords:
        if kw in text:
            hits += 1
    return hits


def _apply_conservative_verdict_gate(state: Dict[str, Any], gate_cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(state or {})
    verdict = str(out.get("verdict", "unknown")).strip().lower()
    if verdict not in {"malicious", "benign", "unknown"}:
        verdict = "unknown"
    out["verdict"] = verdict

    conf = max(0.0, min(100.0, _float(out.get("confidence"), 0.0)))
    out["confidence"] = conf

    if verdict != "malicious":
        out["family"] = "unknown"
        return out

    behavior_labels = _collect_behavior_labels(out.get("behaviors"))
    mechanism_texts = _collect_mechanism_texts(out.get("mechanisms"))
    strong_hits = _count_strong_malicious_signals(behavior_labels, mechanism_texts)

    min_conf = max(0.0, min(100.0, _float(gate_cfg.get("min_confidence"), 70.0)))
    min_strong = max(0, int(gate_cfg.get("min_strong_signals", 1) or 1))
    min_behavior_items = max(0, int(gate_cfg.get("min_behavior_items", 1) or 1))

    weak_to = str(gate_cfg.get("weak_malicious_to", "unknown")).strip().lower()
    if weak_to not in {"unknown", "benign"}:
        weak_to = "unknown"

    weaken = (
        conf < min_conf
        or strong_hits < min_strong
        or len(behavior_labels) < min_behavior_items
    )

    if weaken:
        old_note = str(out.get("note", "")).strip()
        gate_note = (
            f"verdict_gate: weaken malicious->{weak_to} "
            f"(conf={conf:.1f}, strong_hits={strong_hits}, behaviors={len(behavior_labels)})"
        )
        out["verdict"] = weak_to
        out["family"] = "unknown"
        out["note"] = f"{old_note} | {gate_note}" if old_note else gate_note

    return out


def run_with_config(CONFIG: Dict[str, Any]) -> List[Dict[str, Any]]:
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)
    out_dir = (project_root / str(CONFIG.get("out_dir", "out"))).resolve()
    years = _parse_years(CONFIG.get("years"))
    single_apk = CONFIG.get("single_apk")
    limit_apks = int(CONFIG.get("limit_apks", 0) or 0)

    kb_subdir = str(CONFIG.get("kb_out_subdir", "behavior_units_kb"))
    raw_subdir = str(CONFIG.get("units_subdir", "behavior_units"))
    infer_subdir = str(CONFIG.get("infer_out_subdir", "infer"))

    proto_root = resolve_under_out(out_dir, str(CONFIG.get("infer_proto_root", "_prototypes"))).resolve()
    params = MatchParams(
        tau=float(CONFIG.get("infer_tau", 0.55)),
        gamma=float(CONFIG.get("infer_gamma", 0.10)),
        eta=float(CONFIG.get("infer_eta", 0.60)),
        w_unit=float(CONFIG.get("proto_w_unit", 0.45)),
        w_internal=float(CONFIG.get("proto_w_internal", 0.20)),
        w_sensitive=float(CONFIG.get("proto_w_sensitive", 0.20)),
        w_anchor=float(CONFIG.get("proto_w_anchor", 0.10)),
        w_max=float(CONFIG.get("proto_w_max", 0.05)),
    )
    db = PrototypeDB(proto_root, params)
    db.load()
    matcher = ClusterMatcher(db)

    llm_params = LLMParams(
        provider=str(CONFIG.get("llm_provider", "openai")),
        model=str(CONFIG.get("llm_model", "gpt-4o-mini")),
        api_key=str(CONFIG.get("llm_api_key", "")),
        api_key_env=str(CONFIG.get("llm_api_key_env", "OPENAI_API_KEY")),
        base_url=str(CONFIG.get("llm_base_url", "https://api.openai.com/v1")),
        proxy=str(CONFIG.get("llm_proxy", "")),
        connect_timeout_sec=int(CONFIG.get("llm_connect_timeout_sec", CONFIG.get("llm_timeout_sec", 10))),
        read_timeout_sec=int(CONFIG.get("llm_read_timeout_sec", CONFIG.get("llm_timeout_sec", 180))),
        max_retries=int(CONFIG.get("llm_max_retries", 6)),
        retry_base_sec=float(CONFIG.get("llm_retry_base_sec", 1.5)),
        temperature=float(CONFIG.get("llm_temperature", 0.0)),
        max_output_tokens_bu=int(CONFIG.get("llm_max_output_tokens_bu", 500)),
        max_output_tokens_final=int(CONFIG.get("llm_max_output_tokens_final", 900)),
        site_url=str(CONFIG.get("llm_site_url", "")),
        app_name=str(CONFIG.get("llm_app_name", "")),
    )
    client = OpenAIClient(llm_params)

    gate_cfg = {
        "min_confidence": float(CONFIG.get("infer_gate_min_confidence", 70.0)),
        "min_strong_signals": int(CONFIG.get("infer_gate_min_strong_signals", 1) or 1),
        "min_behavior_items": int(CONFIG.get("infer_gate_min_behavior_items", 1) or 1),
        "weak_malicious_to": str(CONFIG.get("infer_gate_weak_malicious_to", "unknown") or "unknown"),
    }

    infer_topk_bu = int(CONFIG.get("infer_topk_bu", 8) or 8)
    infer_topk_proto = int(CONFIG.get("infer_topk_proto_per_bu", CONFIG.get("infer_max_exemplars", 2)) or 2)
    infer_batch_size = int(CONFIG.get("infer_batch_size", 2) or 2)
    max_nodes = int(CONFIG.get("infer_max_ev_nodes", 12) or 12)
    max_items = int(CONFIG.get("infer_max_lex_items", 24) or 24)
    max_freq_psi = int(CONFIG.get("infer_max_cluster_freq_psi", 12) or 12)
    final_reconcile = bool(CONFIG.get("infer_final_reconcile", True))
    max_state_tokens = int(CONFIG.get("llm_max_output_tokens_state", CONFIG.get("llm_max_output_tokens_final", 900)) or 900)
    max_reconcile_tokens = int(CONFIG.get("llm_max_output_tokens_reconcile", CONFIG.get("llm_max_output_tokens_final", 900)) or 900)

    apk_dirs = _iter_apk_dirs(out_dir, years, single_apk, limit_apks)
    print(f"[INFER] out_dir={out_dir} apk_targets={len(apk_dirs)} proto_root={proto_root}")

    results: List[Dict[str, Any]] = []
    csv_path = out_dir / str(CONFIG.get("exp1_csv_name", "_exp1_androserum.csv"))
    is_new = not csv_path.exists()

    with csv_path.open("a", encoding="utf-8", newline="") as fcsv:
        writer = csv.DictWriter(
            fcsv,
            fieldnames=["year", "sha", "apk_id", "verdict", "confidence", "family", "behaviors", "note"],
        )
        if is_new:
            writer.writeheader()

        for i, apk_dir in enumerate(apk_dirs, start=1):
            apk_id = f"{apk_dir.parent.name}/{apk_dir.name}"
            infer_dir = apk_dir / infer_subdir
            infer_dir.mkdir(parents=True, exist_ok=True)

            units = _load_units(apk_dir, kb_subdir=kb_subdir, raw_subdir=raw_subdir)
            if not units:
                obj = {
                    "apk_id": apk_id,
                    "verdict": "unknown",
                    "confidence": 0,
                    "family": "unknown",
                    "behaviors": [],
                    "mechanisms": [],
                    "note": "no_units_loaded",
                }
                safe_write_json(infer_dir / "apk_result.json", obj)
                row = {
                    "year": apk_dir.parent.name,
                    "sha": apk_dir.name,
                    "apk_id": apk_id,
                    "verdict": "unknown",
                    "confidence": 0,
                    "family": "unknown",
                    "behaviors": "[]",
                    "note": "no_units_loaded",
                }
                writer.writerow(row)
                results.append({"apk_id": apk_id, "result": obj})
                print(f"[INFER][{i}/{len(apk_dirs)}] {apk_id} verdict=unknown note=no_units_loaded")
                continue

            packs: List[Dict[str, Any]] = []
            for item in units:
                ranks = matcher.rank_clusters(item["kb"], topk=infer_topk_proto)
                packs.append(
                    _build_bu_pack(
                        item["kb"],
                        item["raw"],
                        ranks,
                        max_exemplars=infer_topk_proto,
                        max_freq_psi=max_freq_psi,
                        max_items=max_items,
                        max_nodes=max_nodes,
                    )
                )

            packs.sort(key=_pack_sort_key)
            packs = packs[:infer_topk_bu]

            if not packs:
                obj = {
                    "apk_id": apk_id,
                    "verdict": "unknown",
                    "confidence": 0,
                    "family": "unknown",
                    "behaviors": [],
                    "mechanisms": [],
                    "note": "no_bu_packs",
                }
                safe_write_json(infer_dir / "apk_result.json", obj)
                row = {
                    "year": apk_dir.parent.name,
                    "sha": apk_dir.name,
                    "apk_id": apk_id,
                    "verdict": "unknown",
                    "confidence": 0,
                    "family": "unknown",
                    "behaviors": "[]",
                    "note": "no_bu_packs",
                }
                writer.writerow(row)
                results.append({"apk_id": apk_id, "result": obj})
                print(f"[INFER][{i}/{len(apk_dirs)}] {apk_id} verdict=unknown note=no_bu_packs")
                continue

            safe_write_json(infer_dir / "apk_aligned_bus.json", {"apk_id": apk_id, "units": packs})

            state = _empty_state(apk_id)
            local_trace: List[Dict[str, Any]] = []
            state_trace: List[Dict[str, Any]] = []

            try:
                # Stage A: BU-local summaries
                local_summaries: List[Dict[str, Any]] = []
                for j, pack in enumerate(packs, start=1):
                    out_local = _call_llm_bu_local(client, apk_id, pack, llm_params.max_output_tokens_bu)
                    bu_id = str(pack.get("bu_id", ""))
                    local_obj = _normalize_local_summary(apk_id, bu_id, out_local.get("obj") or {})
                    local_summaries.append(local_obj)
                    local_trace.append({
                        "step": j,
                        "bu_id": bu_id,
                        "prompt_type": "bu_local",
                        "raw_text": out_local.get("text", ""),
                        "result": local_obj,
                    })

                safe_write_json(infer_dir / "apk_local_bu_summaries.json", {
                    "apk_id": apk_id,
                    "local_summaries": local_summaries,
                })
                safe_write_json(infer_dir / "apk_local_bu_trace.json", {
                    "apk_id": apk_id,
                    "trace": local_trace,
                })

                # Stage B: incremental APK-state update
                batches = list(_chunked(local_summaries, infer_batch_size))
                for step_idx, batch in enumerate(batches, start=1):
                    out_state = _call_llm_state_update(
                        client,
                        apk_id,
                        state,
                        batch,
                        step_index=step_idx,
                        total_steps=len(batches),
                        max_output_tokens=max_state_tokens,
                    )
                    state = _normalize_state(apk_id, out_state.get("obj") or {}, fallback=state)
                    state = _apply_conservative_verdict_gate(state, gate_cfg)

                    state_trace.append({
                        "step": step_idx,
                        "prompt_type": "state_update",
                        "input_bu_ids": [str(x.get("bu_id", "")) for x in batch],
                        "raw_text": out_state.get("text", ""),
                        "state": state,
                    })

                # Stage C: final reconcile
                if final_reconcile:
                    out_final = _call_llm_final_reconcile(
                        client,
                        apk_id,
                        state,
                        local_summaries,
                        max_output_tokens=max_reconcile_tokens,
                    )
                    state = _normalize_state(apk_id, out_final.get("obj") or {}, fallback=state)
                    state = _apply_conservative_verdict_gate(state, gate_cfg)

                    state_trace.append({
                        "step": len(state_trace) + 1,
                        "prompt_type": "final_reconcile",
                        "input_bu_ids": [str(x.get("bu_id", "")) for x in local_summaries],
                        "raw_text": out_final.get("text", ""),
                        "state": state,
                    })

                obj = _apply_conservative_verdict_gate(state, gate_cfg)

                safe_write_json(infer_dir / "apk_state_trace.json", {"apk_id": apk_id, "trace": state_trace})
                safe_write_json(infer_dir / "apk_result.json", obj)
                (infer_dir / "apk_report.txt").write_text(
                    json.dumps(
                        {
                            "apk_id": apk_id,
                            "local_summaries": local_summaries,
                            "final_state": obj,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                    errors="ignore",
                )

            except LLMError as e:
                obj = {
                    "apk_id": apk_id,
                    "verdict": "unknown",
                    "confidence": 0,
                    "family": "unknown",
                    "behaviors": [],
                    "mechanisms": [],
                    "note": f"llm_error: {e}",
                }
                safe_write_json(infer_dir / "apk_result.json", obj)
                (infer_dir / "apk_report.txt").write_text(str(e), encoding="utf-8", errors="ignore")

            row = {
                "year": apk_dir.parent.name,
                "sha": apk_dir.name,
                "apk_id": apk_id,
                "verdict": str(obj.get("verdict", "unknown")).lower(),
                "confidence": obj.get("confidence", 0),
                "family": str(obj.get("family", "unknown")),
                "behaviors": json.dumps(obj.get("behaviors", []), ensure_ascii=False),
                "note": str(obj.get("note", "")),
            }
            writer.writerow(row)
            results.append({"apk_id": apk_id, "result": obj})
            print(
                f"[INFER][{i}/{len(apk_dirs)}] {apk_id} "
                f"verdict={row['verdict']} family={row['family']} note={row['note']}"
            )

    print(f"[INFER][DONE] processed={len(results)} csv={csv_path}")
    return results
