# kb_builder.py
# -*- coding: utf-8 -*-


import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# ----------------------------
# Time / IO helpers
# ----------------------------

def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, p: Path) -> None:
    safe_mkdir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def normalize_sig(sig: str) -> str:
    return (sig or "").strip()


def parse_years(years_val: Union[None, int, str, List[Any], Tuple[Any, ...]]) -> List[str]:
    """
    Normalize CONFIG['years'] into a list of year directory names (strings).

    Supports:
      - [2004, 2005] or ["2004","2005"]
      - "2004,2005,2006"
      - "2004-2024"
      - 2004
      - None -> []
    """
    if years_val is None:
        return []

    if isinstance(years_val, (list, tuple)):
        out = []
        for x in years_val:
            if x is None:
                continue
            s = str(x).strip()
            if s:
                out.append(s)
        return out

    if isinstance(years_val, int):
        return [str(years_val)]

    if isinstance(years_val, str):
        s = years_val.strip()
        if not s:
            return []

        if "-" in s:
            a, b = [p.strip() for p in s.split("-", 1)]
            if a.isdigit() and b.isdigit():
                ia, ib = int(a), int(b)
                lo, hi = (ia, ib) if ia <= ib else (ib, ia)
                return [str(y) for y in range(lo, hi + 1)]

        if "," in s:
            return [p.strip() for p in s.split(",") if p.strip()]

        return [s]

    return [str(years_val)]


# ----------------------------
# Smali extraction (robust)
# ----------------------------

def get_smali_text(node: Dict[str, Any]) -> str:
    """
    Robustly extract smali text from different export formats.

    Supported node["smali"] formats:
      - str: full smali text
      - list[str]: smali lines
      - dict: {"content": list[str] or str, ...}
      - None / missing
    """
    smali = node.get("smali")

    if isinstance(smali, str):
        return smali

    if isinstance(smali, list):
        try:
            return "\n".join(str(x) for x in smali)
        except Exception:
            return ""

    if isinstance(smali, dict):
        content = smali.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            try:
                return "\n".join(str(x) for x in content)
            except Exception:
                return ""
        for k in ("text", "smali_text", "body"):
            v = smali.get(k)
            if isinstance(v, str):
                return v

    return ""


# ----------------------------
# Hint extraction
# ----------------------------

_URL_RE = re.compile(r"https?://|ftp://", re.IGNORECASE)
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_HEX_RE = re.compile(r"\b0x[0-9a-fA-F]{6,}\b")
# Intent extra: STREAM (share/attachment URI)
_INTENT_EXTRA_STREAM = "android.intent.extra.STREAM"
_INTENT_GET_PARCELABLE_HINTS = (
    "Landroid/content/Intent;->getParcelableExtra",
    "Landroid/content/Intent;->getParcelableArrayListExtra",
)


_REFLECTION_HINTS = (
    "Ljava/lang/reflect;",
    "Ljava/lang/Class;->forName",
    "Ljava/lang/reflect/Method;->invoke",
    "Ljava/lang/reflect/Constructor;->newInstance",
    "Ljava/lang/reflect/Field;->get",
    "Ljava/lang/reflect/Field;->set",
)
_INTENT_PREFIX = "android.intent."
_PATH_RE = re.compile(r"^/|/sdcard/|/Android/", re.IGNORECASE)
_APK_RE = re.compile(r"\.apk$", re.IGNORECASE)
_ALLCAPS_RE = re.compile(r"^[A-Z0-9_]{3,}$")
_DYNLOAD_HINTS = ("Ldalvik/system/DexClassLoader;", "PathClassLoader", "loadClass")
_CRYPTO_HINTS = ("Ljavax/crypto/", "MessageDigest", "Cipher", "Mac")
_NET_HINTS = ("Ljava/net/", "Lokhttp", "Lorg/apache/http", "HttpURLConnection", "Socket")
_SMS_HINTS = ("Landroid/telephony/SmsManager;", "sendTextMessage")
_LOCATION_HINTS = ("Landroid/location/", "getLastKnownLocation")
_DEVICE_ID_HINTS = ("getDeviceId", "ANDROID_ID", "getSubscriberId")
_ACCESSIBILITY_HINTS = ("Landroid/accessibilityservice/",)
_OVERLAY_HINTS = ("SYSTEM_ALERT_WINDOW", "TYPE_APPLICATION_OVERLAY")


def is_anchor_token(t: str) -> bool:
    tl = t.lower()

    if tl.startswith(_INTENT_PREFIX):
        return True
    if tl.startswith("android.permission."):
        return True
    if _PATH_RE.search(t) or _APK_RE.search(t):
        return True
    if t in {"GET", "POST", "PUT", "DELETE"}:
        return True
    if "/" in t and ("vnd.android" in tl or "application/" in tl):
        return True
    if _ALLCAPS_RE.fullmatch(t) and t not in {"TRUE", "FALSE"}:
        return True

    return False


def clean_lexical(items: List[str], limit: int) -> List[str]:
    stop = {"", ";", ":", ",", ".", "true", "false", "null", "none"}
    out: List[str] = []
    seen = set()

    for s in items:
        if s is None:
            continue
        t = str(s).strip()
        if not t or len(t) < 3:
            continue
        if t.lower() in stop:
            continue
        if t in seen:
            continue

        if not is_anchor_token(t):
            continue

        seen.add(t)
        out.append(t)
        if len(out) >= limit:
            break
    return out


def extract_hints_from_smali_text(smali_text: str) -> Set[str]:
    s = smali_text or ""
    out: Set[str] = set()

    if any(h in s for h in _REFLECTION_HINTS):
        out.add("HINT:REFLECTION")
    if any(h in s for h in _DYNLOAD_HINTS):
        out.add("HINT:DYNLOAD")
    if any(h in s for h in _CRYPTO_HINTS):
        out.add("HINT:CRYPTO")
    if any(h in s for h in _NET_HINTS):
        out.add("HINT:NETWORK")
    if any(h in s for h in _SMS_HINTS):
        out.add("HINT:SMS")
    if any(h in s for h in _LOCATION_HINTS):
        out.add("HINT:LOCATION")
    if any(h in s for h in _DEVICE_ID_HINTS):
        out.add("HINT:DEVICE_ID")
    if any(h in s for h in _ACCESSIBILITY_HINTS):
        out.add("HINT:ACCESSIBILITY")
    if any(h in s for h in _OVERLAY_HINTS):
        out.add("HINT:OVERLAY")

    if _URL_RE.search(s):
        out.add("LIT:URL_PRESENT")
    if _IP_RE.search(s):
        out.add("LIT:IP_PRESENT")
    if _HEX_RE.search(s):
        out.add("LIT:HEX_LIKE")

    if (_INTENT_EXTRA_STREAM in s) and any(h in s for h in _INTENT_GET_PARCELABLE_HINTS):
        out.add("HINT:INTENT_EXTRA_STREAM")

    return out


_CONST_STRING_RE = re.compile(r'^\s*const-string(?:/jumbo)?\s+v\d+,\s+"(.*)"\s*$')


def extract_const_strings(smali_lines: List[str], limit: int) -> List[str]:
    out: List[str] = []
    for line in smali_lines:
        m = _CONST_STRING_RE.match(line)
        if m:
            out.append(m.group(1))
            if len(out) >= limit:
                break
    return out


def looks_like_base64_literal(s: str) -> bool:
    s = (s or "").strip()
    if len(s) < 40:
        return False
    if not re.fullmatch(r"[A-Za-z0-9+/=]+", s):
        return False
    if ("+" not in s) and ("/" not in s) and ("=" not in s):
        return False
    return True


# ----------------------------
# Structure compaction (NEW)
# ----------------------------

def compact_structure(nodes: List[Dict[str, Any]], edges_raw: Any) -> Dict[str, Any]:
    """
    Build compact structure_info:
      - node_id_map: {old_id: new_id}
      - nodes: [{id, sig}]
      - edges: [[src_id, dst_id], ...] (dedup)

    Supports two edge formats:
      A) edges_raw: list[list[int]] or list[tuple[int,int]] (src/dst are node indices)
      B) edges_raw: list[dict] with keys {src,dst,src_sig,dst_sig}
    """
    # --- Case A: list of pairs ---
    if isinstance(edges_raw, list) and edges_raw and isinstance(edges_raw[0], (list, tuple)):
        # prefer node index mapping (0..len-1)
        idx_to_sig: Dict[int, str] = {}
        for i, n in enumerate(nodes):
            sig = normalize_sig(n.get("sig", ""))
            idx_to_sig[i] = sig

        old_ids = sorted(idx_to_sig.keys())
        node_id_map = {old_id: new_id for new_id, old_id in enumerate(old_ids)}
        nodes_compact = [{"id": node_id_map[i], "sig": idx_to_sig[i]} for i in old_ids]

        edges_set: Set[Tuple[int, int]] = set()
        for e in edges_raw:
            if not isinstance(e, (list, tuple)) or len(e) != 2:
                continue
            try:
                s_old = int(e[0])
                t_old = int(e[1])
            except Exception:
                continue
            if s_old not in node_id_map or t_old not in node_id_map:
                continue
            s = node_id_map[s_old]
            t = node_id_map[t_old]
            if s != t:
                edges_set.add((s, t))

        edges_compact = [[s, t] for (s, t) in sorted(edges_set)]
        return {"node_id_map": node_id_map, "nodes": nodes_compact, "edges": edges_compact}

    # --- Case B: list of dict edges ---
    if isinstance(edges_raw, list) and edges_raw and isinstance(edges_raw[0], dict):
        oldid_to_sig: Dict[int, str] = {}

        for e in edges_raw:
            try:
                s_old = int(e.get("src"))
                t_old = int(e.get("dst"))
            except Exception:
                continue
            s_sig = normalize_sig(e.get("src_sig") or "")
            t_sig = normalize_sig(e.get("dst_sig") or "")
            if s_sig:
                oldid_to_sig.setdefault(s_old, s_sig)
            if t_sig:
                oldid_to_sig.setdefault(t_old, t_sig)

        # if nodes carry node_id, use it; otherwise ignore nodes list (because cannot map reliably)
        for n in nodes:
            if "node_id" in n:
                try:
                    nid = int(n["node_id"])
                    sig = normalize_sig(n.get("sig", ""))
                    if sig:
                        oldid_to_sig.setdefault(nid, sig)
                except Exception:
                    pass

        old_ids = sorted(oldid_to_sig.keys())
        node_id_map = {old_id: new_id for new_id, old_id in enumerate(old_ids)}
        nodes_compact = [{"id": node_id_map[oid], "sig": oldid_to_sig[oid]} for oid in old_ids]

        edges_set: Set[Tuple[int, int]] = set()
        for e in edges_raw:
            try:
                s_old = int(e.get("src"))
                t_old = int(e.get("dst"))
            except Exception:
                continue
            if s_old not in node_id_map or t_old not in node_id_map:
                continue
            s = node_id_map[s_old]
            t = node_id_map[t_old]
            if s != t:
                edges_set.add((s, t))

        edges_compact = [[s, t] for (s, t) in sorted(edges_set)]
        return {"node_id_map": node_id_map, "nodes": nodes_compact, "edges": edges_compact}

    # --- empty / unknown ---
    # fallback: still keep sigs if we have nodes
    nodes_compact = [{"id": i, "sig": normalize_sig(n.get("sig", ""))} for i, n in enumerate(nodes)]
    return {"node_id_map": {i: i for i in range(len(nodes_compact))}, "nodes": nodes_compact, "edges": []}


# ----------------------------
# KB schema builder
# ----------------------------

@dataclass
class KBConfig:
    in_subdir: str = "behavior_units"
    out_subdir: str = "behavior_units_kb"

    lexical_limit: int = 128
    hint_limit: int = 256

    pool_mode: str = "mean_norm"  # "mean" | "mean_norm"


def pool_unit_embedding(nodes: List[Dict[str, Any]], mode: str) -> Optional[List[float]]:
    emb_sum: Optional[List[float]] = None
    cnt = 0
    for n in nodes:
        h = n.get("h")
        if isinstance(h, list) and h and all(isinstance(x, (int, float)) for x in h):
            if emb_sum is None:
                emb_sum = [0.0] * len(h)
            if len(h) != len(emb_sum):
                continue
            for i, x in enumerate(h):
                emb_sum[i] += float(x)
            cnt += 1
    if emb_sum is None or cnt == 0:
        return None
    unit_emb = [x / cnt for x in emb_sum]
    if mode == "mean_norm":
        return l2_normalize(unit_emb)
    return unit_emb


def build_kb_unit_from_raw_unit(raw_unit: Dict[str, Any], cfg: KBConfig) -> Dict[str, Any]:
    year = str(raw_unit.get("year", "unknown"))
    sha = str(raw_unit.get("apk_sha256", "unknown"))
    unit_id = str(raw_unit.get("unit_id", "unknown"))
    kb_unit_id = f"{year}/{sha}/unit_{unit_id}"

    nodes: List[Dict[str, Any]] = raw_unit.get("nodes") or []
    edges_raw = raw_unit.get("edges") or []          # may be list[pair] or list[dict]
    anchors_meta = raw_unit.get("anchors_meta") or {}

    sapi_set: Set[str] = set()
    perms_set: Dict[str, List[str]] = {}
    lexical_set_raw: List[str] = []
    hint_set: Set[str] = set()

    # node-level
    for n in nodes:
        sig = normalize_sig(n.get("sig", ""))

        if n.get("is_sensitive") and sig:
            sapi_set.add(sig)
            for p in (n.get("perms") or []):
                perms_set.setdefault(str(p), []).append(sig)

        smali_text = get_smali_text(n)
        if smali_text:
            hint_set |= extract_hints_from_smali_text(smali_text)

            if len(lexical_set_raw) < cfg.lexical_limit:
                lines = smali_text.splitlines()
                lexical_set_raw.extend(
                    extract_const_strings(lines, cfg.lexical_limit - len(lexical_set_raw))
                )

    # anchors_meta: stable symbolic APIs/perms
    for a in anchors_meta.get("anchors") or []:
        api = normalize_sig(a.get("sig") or a.get("api") or "")
        perms = a.get("perms") or []

        if api:
            sapi_set.add(api)
            for p in perms:
                perms_set.setdefault(str(p), []).append(api)
        else:
            for p in perms:
                perms_set.setdefault(str(p), [])

   
    for perm, apis in list(perms_set.items()):
        perms_set[perm] = sorted(set(normalize_sig(x) for x in apis if x))

    
    if any(looks_like_base64_literal(s) for s in lexical_set_raw):
        hint_set.add("LIT:BASE64_LIKE")

    lexical_set = clean_lexical(lexical_set_raw, cfg.lexical_limit)

    hint_list = sorted(hint_set)
    if len(hint_list) > cfg.hint_limit:
        hint_list = hint_list[: cfg.hint_limit]

    symbolic_info = {
        "SAPI_set": sorted(sapi_set),
        "perms_set": perms_set,
        "lexical_set": lexical_set,
        "hint_set": hint_list,
    }

    unit_emb = pool_unit_embedding(nodes, cfg.pool_mode)
    semantic_info = {"unit_emb": unit_emb}

    
    structure_info = compact_structure(nodes, edges_raw)

    # mining_meta = {"anchors_meta": anchors_meta}

    return {
        "base_info": {
            "unit_id": kb_unit_id,
            "created_time": now_ts(),
        },
        "symbolic_info": symbolic_info,
        "semantic_info": semantic_info,
        "structure_info": structure_info,
        # "mining_meta": mining_meta,
    }


# ----------------------------
# Per-APK build / Stage entrypoint
# ----------------------------

def build_kb_for_apk_dir(apk_out_dir: Path, cfg: KBConfig, show: bool = True) -> Dict[str, Any]:
    raw_dir = apk_out_dir / cfg.in_subdir
    out_dir = apk_out_dir / cfg.out_subdir
    safe_mkdir(out_dir)

    if not raw_dir.exists():
        if show:
            print(f"[KB][WARN] raw units dir not found: {raw_dir}")
        return {"skipped": True, "reason": "raw_units_dir_missing", "apk_out_dir": str(apk_out_dir)}

    unit_paths = sorted(raw_dir.glob("unit_*.json"))
    if show:
        print(f"[KB] apk={apk_out_dir.name} in={raw_dir.name} units={len(unit_paths)} out={out_dir.name}")

    written = 0
    unit_ids: List[str] = []

    for up in unit_paths:
        raw = read_json(up)
        kb_unit = build_kb_unit_from_raw_unit(raw, cfg=cfg)
        write_json(kb_unit, out_dir / up.name)
        written += 1
        unit_ids.append(kb_unit["base_info"]["unit_id"])

    index = {
        "raw_units_dir": str(raw_dir),
        "out_units_dir": str(out_dir),
        "num_units": written,
        "unit_ids": unit_ids,
        "kb_config": {
            "in_subdir": cfg.in_subdir,
            "out_subdir": cfg.out_subdir,
            "lexical_limit": cfg.lexical_limit,
            "hint_limit": cfg.hint_limit,
            "pool_mode": cfg.pool_mode,
        },
    }
    write_json(index, out_dir / "index.json")
    return {"units_written": written, "out_dir": str(out_dir)}


def run_with_config(CONFIG: Dict[str, Any]) -> None:
    out_root = Path(CONFIG.get("out_dir", "out"))
    years = parse_years(CONFIG.get("years"))
    single_apk = CONFIG.get("single_apk")
    limit_apks = CONFIG.get("limit_apks")
    show = bool(CONFIG.get("progress_enable", True))

    cfg = KBConfig(
        in_subdir=CONFIG.get("kb_in_subdir", "behavior_units"),
        out_subdir=CONFIG.get("kb_out_subdir", "behavior_units_kb"),
        lexical_limit=int(CONFIG.get("kb_lexical_limit", 128)),
        hint_limit=int(CONFIG.get("kb_hint_limit", 256)),
        pool_mode=str(CONFIG.get("kb_pool_mode", "mean_norm")),
    )

    targets: List[Path] = []

    if single_apk:
        sp = Path(str(single_apk))

       
        if sp.is_dir():
            targets = [sp]

       
        elif sp.suffix.lower() == ".apk":
            year = sp.parent.name
            sha = sp.stem.upper()
            cand = out_root / year / sha
            if cand.is_dir():
                targets = [cand]

        
        else:
            apk_key = sp.stem.upper() if sp.suffix else str(single_apk).upper()
            for y in years:
                cand = out_root / str(y) / apk_key
                if cand.is_dir():
                    targets = [cand]
                    break
    else:
        for y in years:
            y_dir = out_root / str(y)
            if not y_dir.exists():
                continue
            for sha in sorted(os.listdir(y_dir)):
                apk_dir = y_dir / sha
                if apk_dir.is_dir():
                    targets.append(apk_dir)

    # ✅ safer limit: only apply when positive int
    if limit_apks is not None:
        try:
            k = int(limit_apks)
            if k > 0:
                targets = targets[:k]
        except Exception:
            pass

    if show:
        print(f"[KB] out_root={out_root} targets={len(targets)} cfg={cfg}")

    done = 0
    skipped = 0
    for i, apk_out_dir in enumerate(targets, start=1):
        try:
            res = build_kb_for_apk_dir(apk_out_dir, cfg=cfg, show=show)
            if res.get("skipped"):
                skipped += 1
                if show:
                    print(f"[KB][{i}/{len(targets)}] skipped {apk_out_dir.name} reason={res.get('reason')}")
            else:
                done += 1
                if show:
                    print(f"[KB][{i}/{len(targets)}] done {apk_out_dir.name} units_written={res.get('units_written')}")
        except Exception as e:
            skipped += 1
            print(f"[KB][ERR] {apk_out_dir} -> {repr(e)}")

    print(f"[KB][DONE] processed={done} skipped={skipped} out={out_root}")
