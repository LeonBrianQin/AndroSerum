# proto_builder.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from .utils_io import safe_read_json, safe_write_json, safe_append_jsonl, now_ts, l2_normalize, cos, jaccard, resolve_under_out
except Exception:
    from utils_io import safe_read_json, safe_write_json, safe_append_jsonl, now_ts, l2_normalize, cos, jaccard, resolve_under_out  # type: ignore


@dataclass
class ClusterParams:
    tau: float = 0.55
    gamma: float = 0.10
    eta: float = 0.60

    w_unit: float = 0.45
    w_internal: float = 0.20
    w_sensitive: float = 0.20
    w_anchor: float = 0.10
    w_max: float = 0.05

    top_k_members: int = 20
    top_m_exemplars: int = 8
    top_l_psi: int = 1024

    max_center_step: float = 0.02
    alpha_fallback: float = 0.10
    w_min: float = 0.5
    w_max_center: float = 3.0
    s_exemplar: float = 0.75


def _parse_years(spec: Any) -> Optional[Set[str]]:
    if spec is None:
        return None
    if isinstance(spec, (list, tuple, set)):
        xs = {str(x).strip() for x in spec if str(x).strip()}
        return xs or None
    s = str(spec).strip()
    if not s:
        return None
    if "," in s:
        xs = {x.strip() for x in s.split(",") if x.strip()}
        return xs or None
    if "-" in s and s.replace("-", "").isdigit():
        a, b = s.split("-", 1)
        if a.isdigit() and b.isdigit():
            aa, bb = int(a), int(b)
            lo, hi = (aa, bb) if aa <= bb else (bb, aa)
            return {str(y) for y in range(lo, hi + 1)}
    return {s}


def _normalize_unit_id(unit: Dict[str, Any], fallback_name: str = "") -> str:
    base = unit.get("base_info") or {}
    uid = str(base.get("unit_id", "")).strip()
    return uid or fallback_name


def _extract_vec(unit: Dict[str, Any], key: str) -> Optional[List[float]]:
    sem = unit.get("semantic_info") or {}
    vec = sem.get(key)
    if not isinstance(vec, list) or not vec:
        return None
    out: List[float] = []
    for x in vec:
        if not isinstance(x, (int, float)):
            return None
        out.append(float(x))
    return l2_normalize(out)


def _symbolic_tokens(unit: Dict[str, Any]) -> Set[str]:
    sym = unit.get("symbolic_info") or {}
    toks: Set[str] = set()

    for x in sym.get("SAPI_set") or []:
        s = str(x).strip()
        if s:
            toks.add(f"SAPI:{s}")

    perms_set = sym.get("perms_set") or {}
    for p in perms_set.keys():
        s = str(p).strip()
        if s:
            toks.add(f"PERM:{s}")

    for x in sym.get("hint_set") or []:
        s = str(x).strip()
        if s:
            toks.add(f"HINT:{s}")

    for x in sym.get("lexical_set") or []:
        s = str(x).strip()
        if s:
            toks.add(f"LIT:{s}")

    audit = unit.get("audit_info") or {}
    role_count = audit.get("role_count") or {}
    for r, cnt in role_count.items():
        try:
            c = int(cnt)
        except Exception:
            c = 0
        if c > 0:
            toks.add(f"ROLE:{str(r).strip()}")

    return toks


def _q_score(unit: Dict[str, Any]) -> float:
    sym = unit.get("symbolic_info") or {}
    audit = unit.get("audit_info") or {}
    st = unit.get("structure_info") or {}

    q = 1.0
    q += 0.6 * math.log1p(len(sym.get("SAPI_set") or []))
    q += 0.5 * math.log1p(len((sym.get("perms_set") or {}).keys()))
    q += 0.4 * math.log1p(len(sym.get("hint_set") or []))
    q += 0.3 * math.log1p(len(sym.get("lexical_set") or []))
    q += 0.2 * math.log1p(len(st.get("edges") or []))
    q += 0.2 * math.log1p(len(audit.get("anchor_sigs") or []))
    return float(q)


def _update_weight(q: float, p: ClusterParams) -> float:
    w = math.log1p(max(q, 0.0))
    return float(min(max(w, p.w_min), p.w_max_center))


def _weighted_semantic_sim(cluster: Dict[str, Any], unit: Dict[str, Any], p: ClusterParams) -> float:
    center = (cluster.get("semantic_info") or {}).get("center") or {}
    score = 0.0
    weight_sum = 0.0
    views = [
        ("unit_emb", p.w_unit),
        ("unit_emb_internal", p.w_internal),
        ("unit_emb_sensitive", p.w_sensitive),
        ("unit_emb_anchor", p.w_anchor),
        ("unit_emb_max", p.w_max),
    ]
    for key, w in views:
        cv = center.get(key)
        uv = _extract_vec(unit, key)
        if cv and uv and len(cv) == len(uv):
            score += w * cos(uv, cv)
            weight_sum += w
    if weight_sum <= 0:
        return -1.0
    return float(score / weight_sum)


def _init_center(unit: Dict[str, Any]) -> Dict[str, Any]:
    center: Dict[str, Any] = {"n_eff": 1.0}
    for k in ["unit_emb", "unit_emb_internal", "unit_emb_sensitive", "unit_emb_anchor", "unit_emb_max"]:
        v = _extract_vec(unit, k)
        if v is not None:
            center[k] = v
    return center


def _update_center(center: Dict[str, Any], unit: Dict[str, Any], weight: float, p: ClusterParams) -> None:
    n_eff = float(center.get("n_eff", 0.0) or 0.0)
    if n_eff <= 0:
        alpha = p.alpha_fallback
    else:
        alpha = weight / (n_eff + weight)
    alpha = min(max(alpha, 1e-6), p.max_center_step)

    for k in ["unit_emb", "unit_emb_internal", "unit_emb_sensitive", "unit_emb_anchor", "unit_emb_max"]:
        uv = _extract_vec(unit, k)
        if uv is None:
            continue
        cv = center.get(k)
        if not cv:
            center[k] = uv
            continue
        nv = [(1.0 - alpha) * float(a) + alpha * float(b) for a, b in zip(cv, uv)]
        center[k] = l2_normalize(nv)
    center["n_eff"] = float(n_eff + weight)


def _cluster_frequent_psi(cluster: Dict[str, Any], eta: float, cap: int = 256) -> Set[str]:
    size = int((cluster.get("base_info") or {}).get("size", 0) or 0)
    if size <= 0:
        return set()
    psi_count = (cluster.get("symbolic_info") or {}).get("psi_count") or {}
    thr = eta * size
    items: List[Tuple[int, str]] = []
    for k, v in psi_count.items():
        try:
            c = int(v)
        except Exception:
            continue
        if c >= thr:
            items.append((c, str(k)))
    items.sort(reverse=True)
    return {k for _, k in items[:cap]}

def _compact_raw_exemplar(raw_unit: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not raw_unit:
        return {"anchors_meta": {}, "top_nodes": [], "nodes": [], "edges": []}
    anchors_meta = raw_unit.get("anchors_meta") or {}
    nodes = list(raw_unit.get("nodes") or [])
    edges = list(raw_unit.get("edges") or [])
    out_nodes = []
    for n in nodes[:16]:
        out_nodes.append({
            "node_id": n.get("node_id"),
            "sig": n.get("sig"),
            "primary_role": n.get("primary_role"),
            "score": n.get("score"),
            "perms": list(n.get("perms") or [])[:8],
            "active_symbolic_features": list(n.get("active_symbolic_features") or [])[:16],
            "evidence": {
                "strings": list(((n.get("evidence") or {}).get("strings_raw_preview") or []))[:8],
                "domains": list(((n.get("evidence") or {}).get("domains_topk") or []))[:8],
                "urls": list(((n.get("evidence") or {}).get("urls_topk") or []))[:8],
                "ips": list(((n.get("evidence") or {}).get("ips_topk") or []))[:8],
                "headers": list(((n.get("evidence") or {}).get("http_headers_topk") or []))[:8],
                "spans": list(((n.get("evidence") or {}).get("evidence_spans") or []))[:8],
            },
        })
    return {
        "anchors_meta": {
            "anchors": list(anchors_meta.get("anchors") or [])[:16],
            "top_nodes": list(anchors_meta.get("top_nodes") or [])[:16],
            "score_components": list(anchors_meta.get("score_components") or [])[:16],
        },
        "nodes": out_nodes,
        "edges": edges[:80],
    }


def _new_cluster(cluster_id: str, kb_unit: Dict[str, Any], raw_unit: Optional[Dict[str, Any]], q: float, psi: Set[str]) -> Dict[str, Any]:
    base = kb_unit.get("base_info") or {}
    sym = kb_unit.get("symbolic_info") or {}
    audit = kb_unit.get("audit_info") or {}
    created = now_ts()
    uid = _normalize_unit_id(kb_unit)

    return {
        "base_info": {
            "cluster_id": cluster_id,
            "created_time": created,
            "updated_time": created,
            "size": 1,
        },
        "semantic_info": {
            "center": _init_center(kb_unit),
        },
        "symbolic_info": {
            "psi_count": {x: 1 for x in psi},
            "family_count": {},
            "behavior_count": {},
        },
        "audit_info": {
            "role_count": dict(audit.get("role_count") or {}),
            "anchor_sigs": list(audit.get("anchor_sigs") or [])[:32],
            "source_methods": list(audit.get("source_methods") or [])[:32],
            "sink_methods": list(audit.get("sink_methods") or [])[:32],
            "transform_methods": list(audit.get("transform_methods") or [])[:32],
        },
        "top_members": [{"unit_id": uid, "q": q, "sim": 1.0, "time": created}],
        "exemplars": [{
            "unit_id": uid,
            "q": q,
            "sim": 1.0,
            "symbolic_summary": {
                "SAPI_set": list(sym.get("SAPI_set") or [])[:64],
                "perms_set": dict(list((sym.get("perms_set") or {}).items())[:64]),
                "hint_set": list(sym.get("hint_set") or [])[:64],
                "lexical_set": list(sym.get("lexical_set") or [])[:64],
            },
            "raw_ref": _compact_raw_exemplar(raw_unit),
            "time": created,
        }],
    }


def _merge_label_counts(cluster: Dict[str, Any], label_info: Dict[str, Any]) -> None:
    if not label_info:
        return
    sym = cluster.setdefault("symbolic_info", {})
    fam_count = dict(sym.get("family_count") or {})
    beh_count = dict(sym.get("behavior_count") or {})
    fam = str(label_info.get("family", "")).strip()
    if fam and fam.lower() != "unknown":
        fam_count[fam] = int(fam_count.get(fam, 0) or 0) + 1
    for b in label_info.get("behaviors") or []:
        bb = str(b).strip()
        if bb:
            beh_count[bb] = int(beh_count.get(bb, 0) or 0) + 1
    sym["family_count"] = fam_count
    sym["behavior_count"] = beh_count
    cluster["symbolic_info"] = sym


def _update_cluster(cluster: Dict[str, Any], kb_unit: Dict[str, Any], raw_unit: Optional[Dict[str, Any]], q: float, psi: Set[str], sim: float, params: ClusterParams, label_info: Optional[Dict[str, Any]]) -> None:
    base = cluster.setdefault("base_info", {})
    base["updated_time"] = now_ts()
    base["size"] = int(base.get("size", 0) or 0) + 1

    weight = _update_weight(q, params)
    center = cluster.setdefault("semantic_info", {}).setdefault("center", {})
    _update_center(center, kb_unit, weight, params)

    sym = cluster.setdefault("symbolic_info", {})
    psi_count = dict(sym.get("psi_count") or {})
    for it in psi:
        psi_count[it] = int(psi_count.get(it, 0) or 0) + 1
    if len(psi_count) > params.top_l_psi:
        psi_count = dict(sorted(psi_count.items(), key=lambda kv: kv[1], reverse=True)[:params.top_l_psi])
    sym["psi_count"] = psi_count
    cluster["symbolic_info"] = sym

    audit = cluster.setdefault("audit_info", {})
    role_count = dict(audit.get("role_count") or {})
    src_role_count = (kb_unit.get("audit_info") or {}).get("role_count") or {}
    for k, v in src_role_count.items():
        try:
            c = int(v)
        except Exception:
            c = 0
        role_count[str(k)] = int(role_count.get(str(k), 0) or 0) + c
    audit["role_count"] = role_count

    for key in ["anchor_sigs", "source_methods", "sink_methods", "transform_methods"]:
        old = list(audit.get(key) or [])
        seen = set(old)
        for x in (kb_unit.get("audit_info") or {}).get(key) or []:
            sx = str(x).strip()
            if sx and sx not in seen:
                old.append(sx)
                seen.add(sx)
            if len(old) >= 32:
                break
        audit[key] = old

    members = list(cluster.get("top_members") or [])
    members.append({"unit_id": _normalize_unit_id(kb_unit), "q": q, "sim": sim, "time": base["updated_time"]})
    members.sort(key=lambda x: (float(x.get("sim", -1.0)), float(x.get("q", -1.0))), reverse=True)
    cluster["top_members"] = members[:params.top_k_members]

    if sim >= params.s_exemplar:
        sym_src = kb_unit.get("symbolic_info") or {}
        exemplars = list(cluster.get("exemplars") or [])
        exemplars.append({
            "unit_id": _normalize_unit_id(kb_unit),
            "q": q,
            "sim": sim,
            "symbolic_summary": {
                "SAPI_set": list(sym_src.get("SAPI_set") or [])[:64],
                "perms_set": dict(list((sym_src.get("perms_set") or {}).items())[:64]),
                "hint_set": list(sym_src.get("hint_set") or [])[:64],
                "lexical_set": list(sym_src.get("lexical_set") or [])[:64],
            },
            "raw_ref": _compact_raw_exemplar(raw_unit),
            "time": base["updated_time"],
        })
        dedup = []
        seen_u = set()
        for ex in sorted(exemplars, key=lambda x: (float(x.get("sim", -1.0)), float(x.get("q", -1.0))), reverse=True):
            uid = str(ex.get("unit_id", ""))
            if uid in seen_u:
                continue
            seen_u.add(uid)
            dedup.append(ex)
            if len(dedup) >= params.top_m_exemplars:
                break
        cluster["exemplars"] = dedup

    _merge_label_counts(cluster, label_info or {})

def _load_label_manifest(p: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if p is None or not p.exists():
        return {}
    if p.suffix.lower() == ".jsonl":
        out: Dict[str, Dict[str, Any]] = {}
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            apk_id = str(obj.get("apk_id") or obj.get("sha") or "").strip()
            if apk_id:
                out[apk_id] = {
                    "family": obj.get("family", "unknown"),
                    "behaviors": obj.get("behaviors") or [],
                }
        return out
    raw = safe_read_json(p)
    if not raw:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if not isinstance(v, dict):
                continue
            out[str(k)] = {
                "family": v.get("family", "unknown"),
                "behaviors": v.get("behaviors") or [],
            }
    return out


def _load_apk_targets(out_dir: Path, years_filter: Optional[Set[str]], apk_manifest: Optional[Path], limit_apks: int) -> List[Path]:
    targets: List[Path] = []
    if apk_manifest and apk_manifest.exists():
        lines = [x.strip() for x in apk_manifest.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
        for x in lines:
            p = Path(x)
            if p.is_absolute() and p.is_dir():
                targets.append(p)
                continue
            parts = x.replace("\\", "/").split("/")
            if len(parts) >= 2:
                year, sha = parts[-2], parts[-1]
                cand = out_dir / year / sha.upper().replace('.APK', '')
                if cand.is_dir():
                    targets.append(cand)
    else:
        for yd in sorted(out_dir.iterdir()):
            if not yd.is_dir() or yd.name.startswith("_"):
                continue
            if years_filter is not None and yd.name not in years_filter:
                continue
            for apk_dir in sorted([p for p in yd.iterdir() if p.is_dir()]):
                targets.append(apk_dir)
                if limit_apks > 0 and len(targets) >= limit_apks:
                    return targets
    if limit_apks > 0:
        targets = targets[:limit_apks]
    return targets


def _resolve_single_apk_targets(out_dir: Path, single_apk: Any, years_filter: Optional[Set[str]]) -> List[Path]:
    """
    Resolve CONFIG['single_apk'] into exactly one out/<year>/<sha> directory when possible.

    Supported forms:
      1) out/<year>/<sha> directory path
      2) .../<year>/<sha>.apk file path
      3) "2013/<sha>" or "2013/<sha>.apk"
      4) bare sha256 (searched under years_filter or all years)
    """
    if not single_apk:
        return []

    raw = str(single_apk).strip()
    if not raw:
        return []

    sp = Path(raw)

    # 1) direct out/<year>/<sha> dir
    if sp.is_dir():
        return [sp.resolve()]

    # 2) APK file path: .../<year>/<sha>.apk
    if sp.suffix.lower() == ".apk":
        year = sp.parent.name.strip()
        sha = sp.stem.upper()
        cand = out_dir / year / sha
        if cand.is_dir():
            return [cand]

    # 3) explicit "year/sha" or "year/sha.apk"
    parts = [p for p in raw.replace("\\", "/").split("/") if p]
    if len(parts) >= 2:
        year = parts[-2].strip()
        sha = parts[-1].strip().upper().replace(".APK", "")
        cand = out_dir / year / sha
        if cand.is_dir():
            return [cand]

    # 4) bare sha
    apk_key = (sp.stem if sp.suffix else raw).strip().upper()

    year_dirs: List[Path] = []
    if years_filter:
        for y in sorted(years_filter):
            yd = out_dir / y
            if yd.is_dir():
                year_dirs.append(yd)
    else:
        if out_dir.exists():
            year_dirs = [p for p in sorted(out_dir.iterdir()) if p.is_dir() and not p.name.startswith("_")]

    for yd in year_dirs:
        cand = yd / apk_key
        if cand.is_dir():
            return [cand]

    return []


def _prepare_proto_output(proto_root: Path, reset: bool = True) -> Tuple[Path, Path, Path, Path]:
    """
    Prepare prototype output directory.

    IMPORTANT:
    Current builder rebuilds clusters from scratch in memory on every run.
    Therefore, old assignments / clusters should be cleared by default,
    otherwise repeated single-APK debugging will accumulate misleading history.
    """
    clusters_dir = proto_root / "clusters"
    index_path = proto_root / "cluster_index.json"
    assign_path = proto_root / "assignments.jsonl"
    summary_path = proto_root / "cluster_build_summary.json"

    if reset and proto_root.exists():
        if clusters_dir.exists():
            shutil.rmtree(clusters_dir, ignore_errors=True)
        for p in [index_path, assign_path, summary_path]:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    proto_root.mkdir(parents=True, exist_ok=True)
    clusters_dir.mkdir(parents=True, exist_ok=True)
    return clusters_dir, index_path, assign_path, summary_path


def run_with_config(CONFIG: Dict[str, Any]) -> None:
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    out_dir = (project_root / str(CONFIG.get("out_dir", "out"))).resolve()
    years_filter = _parse_years(CONFIG.get("years"))

    kb_subdir = str(CONFIG.get("proto_in_subdir") or CONFIG.get("kb_out_subdir") or "behavior_units_kb")
    raw_subdir = str(CONFIG.get("units_subdir", "behavior_units"))

    proto_root_cfg = str(CONFIG.get("proto_root_rel", "_prototypes"))
    proto_root = resolve_under_out(out_dir, proto_root_cfg)

    # NEW: reset outputs by default, because builder is a fresh rebuild each run
    proto_reset = bool(CONFIG.get("proto_reset", True))
    clusters_dir, index_path, assign_path, summary_path = _prepare_proto_output(proto_root, reset=proto_reset)

    apk_manifest = CONFIG.get("proto_apk_manifest")
    label_manifest = CONFIG.get("proto_label_manifest")
    single_apk = CONFIG.get("single_apk")

    apk_manifest_path = Path(str(apk_manifest)) if apk_manifest else None
    if apk_manifest_path and not apk_manifest_path.is_absolute():
        apk_manifest_path = (project_root / apk_manifest_path).resolve()

    label_manifest_path = Path(str(label_manifest)) if label_manifest else None
    if label_manifest_path and not label_manifest_path.is_absolute():
        label_manifest_path = (project_root / label_manifest_path).resolve()

    labels = _load_label_manifest(label_manifest_path)

    params = ClusterParams(
        tau=float(CONFIG.get("proto_tau", 0.55)),
        gamma=float(CONFIG.get("proto_gamma", 0.10)),
        eta=float(CONFIG.get("proto_eta", 0.60)),
        w_unit=float(CONFIG.get("proto_w_unit", 0.45)),
        w_internal=float(CONFIG.get("proto_w_internal", 0.20)),
        w_sensitive=float(CONFIG.get("proto_w_sensitive", 0.20)),
        w_anchor=float(CONFIG.get("proto_w_anchor", 0.10)),
        w_max=float(CONFIG.get("proto_w_max", 0.05)),
        top_k_members=int(CONFIG.get("proto_top_k_members", 20) or 20),
        top_m_exemplars=int(CONFIG.get("proto_top_m_exemplars", 8) or 8),
        top_l_psi=int(CONFIG.get("proto_top_l_psi", 1024) or 1024),
        max_center_step=float(CONFIG.get("proto_max_center_step", 0.02)),
        alpha_fallback=float(CONFIG.get("proto_alpha_fallback", 0.10)),
        s_exemplar=float(CONFIG.get("proto_s_exemplar", 0.75)),
    )

    clusters: Dict[str, Dict[str, Any]] = {}
    inv: Dict[str, Set[str]] = {}
    max_id_num = 0

    # NEW: single_apk has highest priority, consistent with kb_builder
    if single_apk:
        targets = _resolve_single_apk_targets(out_dir, single_apk, years_filter)
        source_mode = "single_apk"
    else:
        targets = _load_apk_targets(
            out_dir,
            years_filter,
            apk_manifest_path,
            int(CONFIG.get("limit_apks", 0) or 0),
        )
        source_mode = "manifest" if apk_manifest_path else "years"

    # safer limit: only apply positive int, and do it after single_apk resolve too
    limit_apks = CONFIG.get("limit_apks")
    if limit_apks is not None:
        try:
            k = int(limit_apks)
            if k > 0:
                targets = targets[:k]
        except Exception:
            pass

    print(f"[PROTO] out_dir={out_dir} mode={source_mode} targets={len(targets)} kb_subdir={kb_subdir} reset={proto_reset}")
    if single_apk:
        print(f"[PROTO] single_apk={single_apk}")
    if not targets:
        print("[PROTO][WARN] no targets resolved; nothing to build")
        return

    apk_count = 0
    bu_count = 0
    created_count = 0
    merged_count = 0

    for apk_dir in targets:
        kb_dir = apk_dir / kb_subdir
        raw_dir = apk_dir / raw_subdir
        if not kb_dir.exists():
            print(f"[PROTO][SKIP] kb_dir missing: {kb_dir}")
            continue

        unit_files = sorted(kb_dir.glob("unit_*.json"))
        if not unit_files:
            print(f"[PROTO][SKIP] no kb units: {kb_dir}")
            continue

        apk_count += 1
        apk_id = f"{apk_dir.parent.name}/{apk_dir.name}"
        apk_created = 0
        apk_merged = 0
        apk_used = 0

        for uf in unit_files:
            kb_unit = safe_read_json(uf)
            if not kb_unit:
                continue

            uid = _normalize_unit_id(kb_unit, uf.stem)
            psi = _symbolic_tokens(kb_unit)
            if not psi:
                continue

            apk_used += 1
            bu_count += 1

            raw_unit = safe_read_json(raw_dir / uf.name) if raw_dir.exists() else None
            q = _q_score(kb_unit)

            cand_ids: Set[str] = set()
            for it in psi:
                cand_ids |= inv.get(it, set())

            best_cid: Optional[str] = None
            best_sem = -1.0
            best_j = 0.0
            best_score = -1.0

            for cid in cand_ids:
                c = clusters[cid]
                sem = _weighted_semantic_sim(c, kb_unit, params)
                if sem < params.tau:
                    continue

                freq = _cluster_frequent_psi(c, params.eta, cap=256)
                j = jaccard(psi, freq)
                if j < params.gamma:
                    continue

                score = 0.85 * sem + 0.15 * j
                if score > best_score:
                    best_score = score
                    best_sem = sem
                    best_j = j
                    best_cid = cid

            label_info = labels.get(apk_id) or labels.get(apk_dir.name) or {}

            if best_cid is None:
                max_id_num += 1
                cid = f"p_{max_id_num:06d}"
                clusters[cid] = _new_cluster(cid, kb_unit, raw_unit, q, psi)
                _merge_label_counts(clusters[cid], label_info)

                freq = _cluster_frequent_psi(clusters[cid], params.eta, cap=256)
                for it in freq:
                    inv.setdefault(it, set()).add(cid)

                created_count += 1
                apk_created += 1
                safe_append_jsonl(assign_path, {
                    "time": now_ts(),
                    "apk_id": apk_id,
                    "unit_id": uid,
                    "cluster_id": cid,
                    "action": "create",
                    "semantic_sim": 1.0,
                    "symbolic_jaccard": 1.0
                })
            else:
                cid = best_cid

                old_freq = _cluster_frequent_psi(clusters[cid], params.eta, cap=256)
                for it in old_freq:
                    s = inv.get(it)
                    if s:
                        s.discard(cid)
                        if not s:
                            inv.pop(it, None)

                _update_cluster(clusters[cid], kb_unit, raw_unit, q, psi, best_sem, params, label_info)

                new_freq = _cluster_frequent_psi(clusters[cid], params.eta, cap=256)
                for it in new_freq:
                    inv.setdefault(it, set()).add(cid)

                merged_count += 1
                apk_merged += 1
                safe_append_jsonl(assign_path, {
                    "time": now_ts(),
                    "apk_id": apk_id,
                    "unit_id": uid,
                    "cluster_id": cid,
                    "action": "merge",
                    "semantic_sim": round(best_sem, 6),
                    "symbolic_jaccard": round(best_j, 6)
                })

        print(f"[PROTO][{apk_count}/{len(targets)}] {apk_id} used={apk_used} create={apk_created} merge={apk_merged}")

    cluster_ids = sorted(clusters.keys())
    for cid in cluster_ids:
        safe_write_json(clusters_dir / f"{cid}.json", clusters[cid])

    inv_out = {k: sorted(v) for k, v in inv.items()}
    index = {
        "schema_version": "androserum_prototype_index_v2",
        "time": now_ts(),
        "cluster_ids": cluster_ids,
        "inverted_index": inv_out,
        "params": {
            "tau": params.tau,
            "gamma": params.gamma,
            "eta": params.eta,
            "w_unit": params.w_unit,
            "w_internal": params.w_internal,
            "w_sensitive": params.w_sensitive,
            "w_anchor": params.w_anchor,
            "w_max": params.w_max,
        },
        "source": {
            "out_dir": str(out_dir),
            "kb_subdir": kb_subdir,
            "raw_subdir": raw_subdir,
            "single_apk": str(single_apk or ""),
            "proto_apk_manifest": str(apk_manifest_path) if apk_manifest_path else "",
            "proto_label_manifest": str(label_manifest_path) if label_manifest_path else "",
            "mode": source_mode,
        },
        "summary": {
            "apks_processed": apk_count,
            "bus_processed": bu_count,
            "clusters_total": len(cluster_ids),
            "clusters_created": created_count,
            "clusters_merged": merged_count,
        }
    }

    safe_write_json(index_path, index)
    safe_write_json(summary_path, index)
    print(f"[PROTO][DONE] apks={apk_count} bus={bu_count} clusters={len(cluster_ids)} root={proto_root}")
