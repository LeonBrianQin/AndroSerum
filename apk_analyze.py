#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import shutil
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm

from analyze_tools import tokenization
from analyze_tools.models import DexBERT, Config
from analyze_tools.dataloader import PreprocessEmbedding


# ----------------------------
# Helpers
# ----------------------------
def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def sha_from_apk(apk_path: Path) -> str:
    return apk_path.stem.upper()


def package_to_prefix(pkg: str) -> str:
    return (pkg or "").strip().replace(".", "/").strip("/")


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def append_text(p: Path, s: str) -> None:
    safe_mkdir(p.parent)
    with open(p, "a", encoding="utf-8", errors="ignore") as f:
        f.write(s)


def _p(cfg: dict, key: str, default=False) -> bool:
    # progress helper
    return bool(cfg.get(key, default))


# ----------------------------
# APK package / manifest (fast-ish)
# ----------------------------
def get_manifest_package(apk_path: Path) -> str:
    try:
        from androguard.core.bytecodes.apk import APK
        a = APK(str(apk_path))
        return a.get_package() or ""
    except Exception:
        return ""


def get_manifest_metadata(apk_path: Path) -> Dict[str, object]:
    meta: Dict[str, object] = {
        "package": "",
        "permissions": [],
        "activities": [],
        "services": [],
        "receivers": [],
        "providers": [],
        "components": [],
    }
    try:
        from androguard.core.bytecodes.apk import APK
        a = APK(str(apk_path))
        pkg = a.get_package() or ""
        perms = sorted(set(a.get_permissions() or []))
        acts = sorted(set(a.get_activities() or []))
        svcs = sorted(set(a.get_services() or []))
        rcvs = sorted(set(a.get_receivers() or []))
        provs = sorted(set(a.get_providers() or []))
        meta.update({
            "package": pkg,
            "permissions": perms,
            "activities": acts,
            "services": svcs,
            "receivers": rcvs,
            "providers": provs,
            "components": sorted(set(acts + svcs + rcvs + provs)),
        })
    except Exception:
        pass
    return meta


def build_component_role_map(manifest_meta: Dict[str, object]) -> Dict[str, Set[str]]:
    role_map: Dict[str, Set[str]] = {}
    for cls in manifest_meta.get("activities", []) or []:
        role_map.setdefault(str(cls).replace('.', '/'), set()).add("activity")
    for cls in manifest_meta.get("services", []) or []:
        role_map.setdefault(str(cls).replace('.', '/'), set()).add("service")
    for cls in manifest_meta.get("receivers", []) or []:
        role_map.setdefault(str(cls).replace('.', '/'), set()).add("receiver")
    for cls in manifest_meta.get("providers", []) or []:
        role_map.setdefault(str(cls).replace('.', '/'), set()).add("provider")
    return role_map


# ----------------------------
# Multi-dex disassemble with baksmali
# ----------------------------
DEX_RE = re.compile(r"^classes(\d*)\.dex$")


def list_dex_entries(apk_path: Path) -> List[str]:
    dexes: List[str] = []
    with zipfile.ZipFile(apk_path, "r") as zf:
        for name in zf.namelist():
            base = name.split("/")[-1]
            if DEX_RE.match(base):
                dexes.append(base)

    def dex_key(d: str) -> int:
        m = re.match(r"classes(\d*)\.dex", d)
        if not m:
            return 999999
        g = m.group(1)
        return 1 if g == "" else int(g)

    dexes.sort(key=dex_key)
    return dexes


def run_baksmali_for_dex(
    baksmali_jar: Path,
    apk_path: Path,
    dex_entry: str,
    out_dir: Path,
    log_path: Path,
) -> None:
    safe_mkdir(out_dir)
    safe_mkdir(log_path.parent)

    cmd = [
        "java", "-jar", str(baksmali_jar),
        "disassemble",
        f"{str(apk_path)}/{dex_entry}",
        "-o", str(out_dir),
    ]

    with open(log_path, "w", encoding="utf-8") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    if p.returncode != 0:
        raise RuntimeError(f"baksmali failed for {dex_entry}, see log: {log_path}")


def disassemble_all_dex(
    baksmali_jar: Path,
    apk_path: Path,
    smali_root: Path,
    logs_dir: Path,
) -> List[str]:
    dexes = list_dex_entries(apk_path)
    if not dexes:
        raise RuntimeError("No classes*.dex found in APK")

    safe_mkdir(smali_root)
    safe_mkdir(logs_dir)

    for dex in dexes:
        dex_name = dex.replace(".dex", "")
        out_dir = smali_root / dex_name
        log_path = logs_dir / f"baksmali_{dex_name}.log"
        run_baksmali_for_dex(baksmali_jar, apk_path, dex, out_dir, log_path)

    return dexes


# ----------------------------
# Smali parsing
# ----------------------------
@dataclass
class MethodRecord:
    class_name: str
    method_name: str
    method_sig: str
    instructions: List[str]
    invokes: List[str]
    strings: List[str]
    token_count_raw: int


INVOKE_PREFIX = ("invoke-",)
CONST_STRING_PREFIX = ("const-string", "const-string/jumbo")
STR_LIT_RE = re.compile(r"\"(.*)\"")

URL_EXTRACT_RE = re.compile(r"(?i)https?://[^\s\"'<>]+")
DOMAIN_RE = re.compile(r"(?i)\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}\b")
IP_EXTRACT_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
INTENT_ACTION_RE = re.compile(r"android\.intent\.[A-Z0-9_\.]+")
CONTENT_URI_RE = re.compile(r"content://[^\s\"'<>]+", re.IGNORECASE)
FILE_PATH_RE = re.compile(r"(?:/data/data|/sdcard|/storage|/mnt/|/system/)[^\s\"'<>]+", re.IGNORECASE)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
HEADER_KEY_RE = re.compile(r"(?i)^(host|user-agent|cookie|authorization|referer|x-[a-z0-9-]+|udid|imei|imsi|mac|token)$")
HEX_FULL_RE = re.compile(r"^[0-9a-fA-F]{16,}$")
BASE64_FULL_RE = re.compile(r"^[A-Za-z0-9+/=]{24,}$")
PACKAGE_LIKE_RE = re.compile(r"^[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*){2,}$")

SYMBOLIC_FEATURE_NAMES = [
    "has_sensitive_api", "has_url", "has_ip", "has_domain", "has_content_uri", "has_file_path",
    "has_intent_action", "has_base64_like", "has_hex_like", "api_network", "api_location",
    "api_device_id", "api_sms", "api_crypto", "api_dynload", "api_reflection", "role_activity",
    "role_service", "role_receiver", "role_provider", "cb_lifecycle", "cb_onreceive",
    "cb_accessibility", "has_permission_string",
]


def normalize_string_lit(s: str) -> str:
    ss = s.strip()
    if re.search(r"(?i)\bhttps?://", ss):
        return "<URL>"
    if re.search(r"\b\d{1,3}(\.\d{1,3}){3}\b", ss):
        return "<IP>"
    if len(ss) >= 40 and re.fullmatch(r"[A-Za-z0-9+/=]+", ss):
        return f"<BASE64({len(ss)})>"
    if len(ss) >= 32 and re.fullmatch(r"[0-9a-fA-F]+", ss):
        return f"<HEX({len(ss)})>"
    if len(ss) <= 64:
        return ss
    return ss[:32] + f"...<LEN={len(ss)}>"


def looks_like_base64_literal(s: str) -> bool:
    s = (s or "").strip()
    return len(s) >= 24 and BASE64_FULL_RE.fullmatch(s) is not None and any(ch in s for ch in "+/=")


def looks_like_hex_literal(s: str) -> bool:
    s = (s or "").strip()
    return HEX_FULL_RE.fullmatch(s) is not None


def stable_topk(items: List[str], limit: int) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for x in items:
        sx = str(x).strip()
        if not sx or sx in seen:
            continue
        seen.add(sx)
        out.append(sx)
        if len(out) >= limit:
            break
    return out


def infer_method_family_tags(class_name: str, method_name: str, invokes: List[str]) -> List[str]:
    tags: Set[str] = set()
    joined = "\n".join(invokes)
    if "Landroid/location/" in joined or "LocationManager" in joined:
        tags.add("location")
    if any(k in joined for k in ["getDeviceId", "getSubscriberId", "getLine1Number", "WifiInfo"]):
        tags.add("device_id")
    if any(k in joined for k in ["Ljava/net/", "Lorg/apache/http", "HttpURLConnection", "Lokhttp"]):
        tags.add("network")
    if any(k in joined for k in ["Ljavax/crypto/", "MessageDigest", "Cipher", "Mac"]):
        tags.add("crypto")
    if any(k in joined for k in ["Ldalvik/system/DexClassLoader;", "PathClassLoader", "loadClass"]):
        tags.add("dynload")
    if any(k in joined for k in ["Ljava/lang/reflect/", "Class;->forName", "Method;->invoke"]):
        tags.add("reflection")
    mn = (method_name or "").lower()
    if mn.startswith("oncreate") or mn.startswith("onstart") or mn.startswith("onresume"):
        tags.add("lifecycle")
    if mn.startswith("onreceive"):
        tags.add("receiver_callback")
    if mn.startswith("onaccessibility"):
        tags.add("accessibility_callback")
    return sorted(tags)


def extract_audit_literals(strings_raw: List[str], instructions: List[str], limit: int = 16) -> Dict[str, List[str]]:
    text = "\n".join(list(strings_raw) + list(instructions[:256]))
    raw_strings = stable_topk(strings_raw, limit)
    urls = stable_topk(URL_EXTRACT_RE.findall(text), limit)
    ips = stable_topk(IP_EXTRACT_RE.findall(text), limit)
    domains = stable_topk([d for d in DOMAIN_RE.findall(text) if d not in urls and not d.startswith("android.")], limit)
    intent_actions = stable_topk(INTENT_ACTION_RE.findall(text), limit)
    content_uris = stable_topk(CONTENT_URI_RE.findall(text), limit)
    file_paths = stable_topk(FILE_PATH_RE.findall(text), limit)
    emails = stable_topk(EMAIL_RE.findall(text), limit)
    base64_lits = stable_topk([s for s in strings_raw if looks_like_base64_literal(s)], limit)
    hex_lits = stable_topk([s for s in strings_raw if looks_like_hex_literal(s)], limit)
    package_like = stable_topk([s for s in strings_raw if PACKAGE_LIKE_RE.fullmatch((s or "").strip())], limit)
    headers = stable_topk([s for s in strings_raw if HEADER_KEY_RE.fullmatch((s or "").strip())], limit)
    return {
        "raw_strings_topk": raw_strings,
        "urls_topk": urls,
        "domains_topk": domains,
        "ips_topk": ips,
        "http_headers_topk": headers,
        "intent_actions_topk": intent_actions,
        "content_uris_topk": content_uris,
        "file_paths_topk": file_paths,
        "emails_topk": emails,
        "base64_literals_topk": base64_lits,
        "hex_literals_topk": hex_lits,
        "package_like_names_topk": package_like,
    }


def extract_method_evidence_spans(instructions: List[str], sensitive_invokes: List[Dict[str, object]], limit: int = 16) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    sens_map = {str(x.get("callee")): list(x.get("perms") or []) for x in sensitive_invokes}
    for i, line in enumerate(instructions, start=1):
        s = line.strip()
        if s.startswith(INVOKE_PREFIX):
            callee = s.split()[-1].strip().strip(',')
            role_hint = "other"
            if any(k in callee for k in ["getSubscriberId", "getDeviceId", "getLine1Number", "WifiInfo", "LocationManager"]):
                role_hint = "source"
            elif any(k in callee for k in ["HttpURLConnection", "Lorg/apache/http", "Socket", "execute(", "openConnection"]):
                role_hint = "sink"
            elif any(k in callee for k in ["MessageDigest", "Cipher", "Mac", "Base64"]):
                role_hint = "transform"
            item = {"kind": "API_CALL", "line_no": i, "text": s, "callee": callee, "role_hint": role_hint}
            if callee in sens_map:
                item["perms"] = sens_map[callee]
            out.append(item)
        elif s.startswith(CONST_STRING_PREFIX):
            m = STR_LIT_RE.search(s)
            if m:
                val = m.group(1)
                kind = "STRING_LITERAL"
                role_hint = "other"
                if URL_EXTRACT_RE.search(val):
                    role_hint = "network_indicator"
                elif INTENT_ACTION_RE.search(val):
                    role_hint = "intent_indicator"
                elif FILE_PATH_RE.search(val):
                    role_hint = "file_indicator"
                out.append({"kind": kind, "line_no": i, "text": s, "value": val[:240], "role_hint": role_hint})
        if len(out) >= limit:
            break
    return out


def build_method_symbolic_features(
    class_name: str,
    method_name: str,
    invokes: List[str],
    strings: List[str],
    sensitive_invokes: List[Dict[str, object]],
    role_map: Dict[str, Set[str]],
) -> List[float]:
    joined_inv = "\n".join(invokes)
    joined_str = "\n".join(strings)
    roles = role_map.get(class_name, set())
    mn = (method_name or "").lower()
    feats = {name: 0.0 for name in SYMBOLIC_FEATURE_NAMES}
    feats["has_sensitive_api"] = 1.0 if sensitive_invokes else 0.0
    feats["has_url"] = 1.0 if URL_EXTRACT_RE.search(joined_str) else 0.0
    feats["has_ip"] = 1.0 if IP_EXTRACT_RE.search(joined_str) else 0.0
    feats["has_domain"] = 1.0 if DOMAIN_RE.search(joined_str) else 0.0
    feats["has_content_uri"] = 1.0 if CONTENT_URI_RE.search(joined_str) else 0.0
    feats["has_file_path"] = 1.0 if FILE_PATH_RE.search(joined_str) else 0.0
    feats["has_intent_action"] = 1.0 if INTENT_ACTION_RE.search(joined_str) else 0.0
    feats["has_base64_like"] = 1.0 if any(looks_like_base64_literal(s) for s in strings) else 0.0
    feats["has_hex_like"] = 1.0 if any(looks_like_hex_literal(s) for s in strings) else 0.0
    feats["api_network"] = 1.0 if any(k in joined_inv for k in ["Ljava/net/", "Lorg/apache/http", "HttpURLConnection", "Lokhttp", "Socket"]) else 0.0
    feats["api_location"] = 1.0 if "Landroid/location/" in joined_inv else 0.0
    feats["api_device_id"] = 1.0 if any(k in joined_inv for k in ["getDeviceId", "getSubscriberId", "getLine1Number", "WifiInfo"]) else 0.0
    feats["api_sms"] = 1.0 if "SmsManager" in joined_inv else 0.0
    feats["api_crypto"] = 1.0 if any(k in joined_inv for k in ["Ljavax/crypto/", "MessageDigest", "Cipher", "Mac"]) else 0.0
    feats["api_dynload"] = 1.0 if any(k in joined_inv for k in ["DexClassLoader", "PathClassLoader", "loadClass"]) else 0.0
    feats["api_reflection"] = 1.0 if any(k in joined_inv for k in ["Ljava/lang/reflect/", "Class;->forName", "Method;->invoke"]) else 0.0
    feats["role_activity"] = 1.0 if "activity" in roles else 0.0
    feats["role_service"] = 1.0 if "service" in roles else 0.0
    feats["role_receiver"] = 1.0 if "receiver" in roles else 0.0
    feats["role_provider"] = 1.0 if "provider" in roles else 0.0
    feats["cb_lifecycle"] = 1.0 if mn.startswith(("oncreate", "onstart", "onresume", "onpause", "onstop", "ondestroy")) else 0.0
    feats["cb_onreceive"] = 1.0 if mn.startswith("onreceive") else 0.0
    feats["cb_accessibility"] = 1.0 if mn.startswith("onaccessibility") else 0.0
    feats["has_permission_string"] = 1.0 if "android.permission." in joined_str else 0.0
    return [float(feats[name]) for name in SYMBOLIC_FEATURE_NAMES]


def iter_smali_files(smali_root: Path) -> Iterator[Path]:
    for p in smali_root.rglob("*.smali"):
        yield p


def parse_smali_methods(
    smali_root: Path,
    keep_prefix: Optional[str] = None,
    dedup_instructions: bool = False,
) -> Tuple[List[MethodRecord], List[Tuple[str, str]]]:
    methods: List[MethodRecord] = []
    edges: List[Tuple[str, str]] = []

    for smali_file in iter_smali_files(smali_root):
        try:
            lines = smali_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue

        class_name = ""
        in_method = False
        method_name = ""
        instrs: List[str] = []
        invokes: List[str] = []
        strings: List[str] = []
        seen = set()
        tok_count_raw = 0

        for line in lines:
            if line.startswith(".class"):
                try:
                    raw = line.strip().split(" ")[-1]
                    if raw.startswith("L") and raw.endswith(";"):
                        class_name = raw[1:-1]
                    else:
                        class_name = raw
                except Exception:
                    class_name = ""
                continue

            if line.startswith(".method"):
                in_method = True
                instrs, invokes, strings = [], [], []
                seen = set()
                tok_count_raw = 0
                parts = line.split(" ")
                method_name = parts[-1].strip()
                continue

            if in_method and line.startswith(".end method"):
                in_method = False
                if not class_name or not method_name:
                    method_name = ""
                    continue

                if keep_prefix and not class_name.startswith(keep_prefix):
                    method_name = ""
                    continue

                method_sig = f"L{class_name};->{method_name}"
                methods.append(MethodRecord(
                    class_name=class_name,
                    method_name=method_name,
                    method_sig=method_sig,
                    instructions=instrs,
                    invokes=invokes,
                    strings=strings,
                    token_count_raw=tok_count_raw,
                ))
                method_name = ""
                continue

            if not in_method:
                continue

            s = line.strip()
            if not s or s.startswith("."):
                continue

            if s.startswith(CONST_STRING_PREFIX):
                m = STR_LIT_RE.search(s)
                if m:
                    strings.append(m.group(1))

            if s.startswith(INVOKE_PREFIX):
                callee = s.split()[-1].strip().strip(",")
                if callee.startswith("L") and "->" in callee:
                    invokes.append(callee)

            if dedup_instructions:
                if s in seen:
                    continue
                seen.add(s)

            instrs.append(s)
            tok_count_raw += 1

    for m in methods:
        for callee in m.invokes:
            edges.append((m.method_sig, callee))

    return methods, edges


# ----------------------------
# DexBERT tokenization + embedding
# ----------------------------
def load_model(
    cfg_path: Path,
    vocab_path: Path,
    weights_path: Path,
    device: torch.device,
) -> Tuple[DexBERT, Config, tokenization.FullTokenizer, PreprocessEmbedding]:
    cfg = Config.from_json(str(cfg_path))
    tok = tokenization.FullTokenizer(vocab_file=str(vocab_path), do_lower_case=True)

    model = DexBERT(cfg)
    state = torch.load(str(weights_path), map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    proc = PreprocessEmbedding(tok.convert_tokens_to_ids, max_len=cfg.max_len)
    return model, cfg, tok, proc


def build_method_tokens(
    tok: tokenization.FullTokenizer,
    method_name: str,
    instructions: List[str],
    include_methodname_line: bool = True,
    invoke_only: bool = False,
) -> List[str]:
    tokens: List[str] = []
    if include_methodname_line:
        tokens += tok.tokenize(tok.convert_to_unicode(f"MethodName: {method_name}"))

    if invoke_only:
        for ins in instructions:
            if ins.startswith(INVOKE_PREFIX):
                callee = ins.split()[-1].strip().strip(",")
                tokens += tok.tokenize(tok.convert_to_unicode(callee))
    else:
        for ins in instructions:
            tokens += tok.tokenize(tok.convert_to_unicode(ins))

    return tokens


def build_signature_tokens(tok: tokenization.FullTokenizer, sig: str, perms: Optional[List[str]] = None) -> List[str]:
    base = f"MethodName: {sig}"
    if perms:
        base += " Perms: " + " ".join(perms[:6])
    return tok.tokenize(tok.convert_to_unicode(base))


def chunk_tokens(tokens: List[str], max_len: int) -> List[List[str]]:
    if len(tokens) <= max_len:
        return [tokens]
    return [tokens[i:i + max_len] for i in range(0, len(tokens), max_len)]


def embed_tokens_one(
    model: DexBERT,
    proc: PreprocessEmbedding,
    cfg: Config,
    device: torch.device,
    tokens: List[str],
) -> np.ndarray:
    input_ids, segment_ids, input_mask, _ = proc((list(tokens), 0))
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    segment_ids = torch.tensor([segment_ids], dtype=torch.long, device=device)
    input_mask = torch.tensor([input_mask], dtype=torch.long, device=device)
    with torch.no_grad():
        vec = model(input_ids, segment_ids, input_mask)
    return vec.detach().cpu().numpy()[0]


def pool_vectors(vectors: List[np.ndarray], mode: str) -> np.ndarray:
    arr = np.stack(vectors, axis=0)
    if mode == "mean":
        return arr.mean(axis=0)
    if mode == "max":
        return arr.max(axis=0)
    raise ValueError(f"Unknown pool mode: {mode}")


# ----------------------------
# PScout parse & signature convert (Java -> smali)
# ----------------------------
JAVA_SIG_RE = re.compile(r"^<(?P<cls>[^:]+):\s*(?P<ret>[^ ]+)\s+(?P<meth>[^(]+)\((?P<args>[^)]*)\)>")


PRIM = {
    "boolean": "Z", "byte": "B", "char": "C", "short": "S",
    "int": "I", "long": "J", "float": "F", "double": "D",
    "void": "V"
}


def _strip_generics(t: str) -> str:
    out, depth = [], 0
    for ch in t:
        if ch == "<":
            depth += 1
            continue
        if ch == ">":
            depth = max(depth - 1, 0)
            continue
        if depth == 0:
            out.append(ch)
    return "".join(out)


def java_type_to_smali_desc(t: str) -> str:
    t = (t or "").strip()
    if not t:
        return ""
    t = _strip_generics(t)
    if t.endswith("..."):
        t = t[:-3] + "[]"
    arr_dim = 0
    while t.endswith("[]"):
        arr_dim += 1
        t = t[:-2].strip()
    if t in PRIM:
        base = PRIM[t]
    else:
        base = "L" + t.replace(".", "/") + ";"
    return ("[" * arr_dim) + base


def pscout_java_sig_to_smali(sig_line: str) -> Optional[str]:
    m = JAVA_SIG_RE.search(sig_line.strip())
    if not m:
        return None
    cls = m.group("cls").strip()
    ret = m.group("ret").strip()
    meth = m.group("meth").strip()
    args = m.group("args").strip()

    cls_desc = "L" + cls.replace(".", "/") + ";"
    ret_desc = java_type_to_smali_desc(ret)

    arg_descs = []
    if args:
        for a in args.split(","):
            a = a.strip()
            if a:
                arg_descs.append(java_type_to_smali_desc(a))

    return f"{cls_desc}->{meth}({''.join(arg_descs)}){ret_desc}"


def load_pscout_allmappings(allmappings_path: Path) -> Dict[str, Set[str]]:
    api2perms: Dict[str, Set[str]] = {}
    cur_perm: Optional[str] = None
    with open(allmappings_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("Permission:"):
                cur_perm = line.split("Permission:", 1)[1].strip()
                continue
            if cur_perm and line.startswith("<") and ">" in line:
                sig_part = line.split(">", 1)[0] + ">"
                smali = pscout_java_sig_to_smali(sig_part)
                if smali:
                    api2perms.setdefault(smali, set()).add(cur_perm)
    return api2perms


# ----------------------------
# Build GNN-ready FCG
# ----------------------------
def build_gnn_fcg(
    apk_out: Path,
    year: str,
    sha: str,
    apk_path: Path,
    internal_nodes: List[str],
    edges: List[Tuple[str, str]],
    emb_by_sig: Dict[str, np.ndarray],
    method_sym_by_sig: Optional[Dict[str, np.ndarray]],
    model: DexBERT,
    tok: tokenization.FullTokenizer,
    proc: PreprocessEmbedding,
    cfg: Config,
    device: torch.device,
    pscout_api2perms: Dict[str, Set[str]],
    embed_external_apis: str,
    add_reverse_edges: bool,
    add_self_loops: bool,
    keep_isolated_nodes: bool,
) -> None:
    graph_pt = apk_out / "graph.pt"
    graph_json = apk_out / "graph.json"
    graph_meta = apk_out / "graph_meta.json"

    internal_set = set(internal_nodes)
    external_nodes = sorted({callee for _, callee in edges if callee not in internal_set})

    if not keep_isolated_nodes:
        deg = {n: 0 for n in internal_nodes}
        for u, v in edges:
            if u in deg:
                deg[u] += 1
            if v in deg:
                deg[v] += 1
        internal_nodes = [n for n in internal_nodes if deg.get(n, 0) > 0]
        internal_set = set(internal_nodes)

    node_sigs = internal_nodes + external_nodes
    idx = {sig: i for i, sig in enumerate(node_sigs)}

    dim = next(iter(emb_by_sig.values())).shape[0] if emb_by_sig else 128
    sym_dim = len(SYMBOLIC_FEATURE_NAMES)
    x_base = np.zeros((len(node_sigs), dim), dtype=np.float32)
    x_symbolic = np.zeros((len(node_sigs), sym_dim), dtype=np.float32)

    for sig in internal_nodes:
        i = idx[sig]
        if sig in emb_by_sig:
            x_base[i] = emb_by_sig[sig].astype(np.float32)
        if method_sym_by_sig and sig in method_sym_by_sig:
            x_symbolic[i] = method_sym_by_sig[sig].astype(np.float32)

    node_perms: List[List[str]] = [[] for _ in range(len(node_sigs))]
    is_external = np.zeros((len(node_sigs),), dtype=np.int64)
    is_sensitive = np.zeros((len(node_sigs),), dtype=np.int64)

    ext_cache: Dict[str, np.ndarray] = {}
    for sig in external_nodes:
        i = idx[sig]
        is_external[i] = 1
        perms = sorted(list(pscout_api2perms.get(sig, set())))
        if perms:
            is_sensitive[i] = 1
            node_perms[i] = perms

        if embed_external_apis == "zero":
            continue
        if embed_external_apis == "signature":
            if sig not in ext_cache:
                tokens = build_signature_tokens(tok, sig, perms=perms)
                if len(tokens) > cfg.max_len:
                    tokens = tokens[:cfg.max_len]
                ext_cache[sig] = embed_tokens_one(model, proc, cfg, device, tokens).astype(np.float32)
            x_base[i] = ext_cache[sig]
        else:
            raise ValueError(f"Unknown embed_external_apis: {embed_external_apis}")

    edge_set_fwd = set()
    edge_set = set()
    for u, v in edges:
        if u not in idx or v not in idx:
            continue
        uu, vv = idx[u], idx[v]
        edge_set_fwd.add((uu, vv))
        edge_set.add((uu, vv))
        if add_reverse_edges:
            edge_set.add((vv, uu))

    if add_self_loops:
        for i in range(len(node_sigs)):
            edge_set.add((i, i))

    edge_list_fwd = sorted(edge_set_fwd)
    edge_list = sorted(edge_set)
    edge_index_fwd = (
        torch.tensor(edge_list_fwd, dtype=torch.long).t().contiguous()
        if edge_list_fwd else torch.zeros((2, 0), dtype=torch.long)
    )
    edge_index = (
        torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        if edge_list else torch.zeros((2, 0), dtype=torch.long)
    )

    x = np.concatenate([x_base, x_symbolic], axis=1)
    payload = {
        "year": year,
        "apk_sha256": sha,
        "apk_path": str(apk_path),
        "node_sigs": node_sigs,
        "x": torch.from_numpy(x),
        "x_base": torch.from_numpy(x_base),
        "x_symbolic": torch.from_numpy(x_symbolic),
        "sym_feat_names": SYMBOLIC_FEATURE_NAMES,
        "edge_index": edge_index,
        "edge_index_fwd": edge_index_fwd,
        "is_external": torch.from_numpy(is_external),
        "is_sensitive": torch.from_numpy(is_sensitive),
        "node_perms": node_perms,
    }
    torch.save(payload, graph_pt)

    graph_json.write_text(json.dumps({
        "year": year,
        "apk_sha256": sha,
        "apk_path": str(apk_path),
        "num_nodes": len(node_sigs),
        "num_edges": int(edge_index.shape[1]),
        "node_sigs": node_sigs,
        "edges": [(int(a), int(b)) for (a, b) in edge_list],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    graph_meta.write_text(json.dumps({
        "year": year,
        "apk_sha256": sha,
        "apk_path": str(apk_path),
        "nodes_total": len(node_sigs),
        "edges_total": int(edge_index.shape[1]),
        "edges_total_fwd": int(edge_index_fwd.shape[1]),
        "internal_nodes": len(internal_nodes),
        "external_nodes": len(external_nodes),
        "sensitive_external_nodes": int(is_sensitive.sum()),
        "embed_external_apis": embed_external_apis,
        "created_at": now_ts(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def load_methods_jsonl(methods_out: Path) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    internal_nodes: List[str] = []
    emb_by_sig: Dict[str, np.ndarray] = {}
    sym_by_sig: Dict[str, np.ndarray] = {}
    with open(methods_out, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            sig = obj["method_sig"]
            vec = np.array(obj["embedding"], dtype=np.float32)
            internal_nodes.append(sig)
            emb_by_sig[sig] = vec
            if "symbolic_features" in obj:
                sym_by_sig[sig] = np.array(obj["symbolic_features"], dtype=np.float32)
    return internal_nodes, emb_by_sig, sym_by_sig


def load_edges_jsonl(edges_out: Path) -> List[Tuple[str, str]]:
    edges: List[Tuple[str, str]] = []
    with open(edges_out, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            edges.append((obj["caller"], obj["callee"]))
    return edges


def parse_years(s: str) -> List[str]:
    s = s.strip()
    if "-" in s:
        a, b = s.split("-", 1)
        return [str(y) for y in range(int(a), int(b) + 1)]
    return [x.strip() for x in s.split(",") if x.strip()]


def collect_apks(apks_root: Path, years: List[str]) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for y in years:
        yd = apks_root / y
        if not yd.exists():
            continue
        for apk in sorted(yd.glob("*.apk")):
            out.append((y, apk))
    return out


def process_one_apk(
    apk_path: Path,
    year: str,
    out_dir: Path,
    tmp_root: Path,
    baksmali_jar: Path,
    model: DexBERT,
    cfg: Config,
    tok: tokenization.FullTokenizer,
    proc: PreprocessEmbedding,
    device: torch.device,
    pscout_api2perms: Dict[str, Set[str]],
    keep_only_app_code: bool,
    dedup_instructions: bool,
    include_methodname_line: bool,
    invoke_only_for_embedding: bool,
    long_method_mode: str,
    pool_mode: str,
    max_strings_per_method: int,
    build_fcg: bool,
    embed_external_apis: str,
    fcg_add_reverse_edges: bool,
    fcg_add_self_loops: bool,
    fcg_keep_isolated_nodes: bool,
    force_rebuild_graph: bool,
    record_sensitive_invokes: bool,
    max_sensitive_invokes_per_method: int,
    _progress: bool = True,
    _task_i: int = 0,
    _task_n: int = 0,
) -> None:
    sha = sha_from_apk(apk_path)
    apk_out = out_dir / year / sha
    methods_out = apk_out / "methods.jsonl"
    edges_out = apk_out / "edges.jsonl"
    meta_out = apk_out / "meta.json"
    manifest_out = apk_out / "manifest.json"
    err_out = apk_out / "error.log"
    graph_pt = apk_out / "graph.pt"

    safe_mkdir(apk_out)

    # graph-only rebuild
    if build_fcg and methods_out.exists() and edges_out.exists():
        if force_rebuild_graph or (not graph_pt.exists()) or graph_pt.stat().st_size == 0:
            if _progress:
                print(f"[{now_ts()}][APK][{_task_i}/{_task_n}] {year}/{sha} graph-only rebuild ...")
            internal_nodes, emb_by_sig, method_sym_by_sig = load_methods_jsonl(methods_out)
            edges = load_edges_jsonl(edges_out)
            build_gnn_fcg(
                apk_out=apk_out,
                year=year,
                sha=sha,
                apk_path=apk_path,
                internal_nodes=internal_nodes,
                edges=edges,
                emb_by_sig=emb_by_sig,
                method_sym_by_sig=method_sym_by_sig,
                model=model,
                tok=tok,
                proc=proc,
                cfg=cfg,
                device=device,
                pscout_api2perms=pscout_api2perms,
                embed_external_apis=embed_external_apis,
                add_reverse_edges=fcg_add_reverse_edges,
                add_self_loops=fcg_add_self_loops,
                keep_isolated_nodes=fcg_keep_isolated_nodes,
            )
        else:
            if _progress:
                print(f"[{now_ts()}][APK][{_task_i}/{_task_n}] {year}/{sha} graph exists, skip.")
        return

    if build_fcg and graph_pt.exists() and graph_pt.stat().st_size > 0 and (not force_rebuild_graph):
        if _progress:
            print(f"[{now_ts()}][APK][{_task_i}/{_task_n}] {year}/{sha} graph exists, skip.")
        return

    workdir = tmp_root / f"{year}_{sha}"
    smali_root = workdir / "smali"
    logs_dir = workdir / "logs"

    if workdir.exists():
        shutil.rmtree(workdir, ignore_errors=True)
    safe_mkdir(workdir)

    try:
        t0 = time.time()
        if _progress:
            print(f"[{now_ts()}][APK][{_task_i}/{_task_n}] START {year}/{sha}")
            print(f"  - apk={apk_path}")
            print(f"  - out={apk_out}")
            print(f"  - keep_only_app_code={keep_only_app_code} long_method_mode={long_method_mode} build_fcg={build_fcg}")

        manifest_meta = get_manifest_metadata(apk_path)
        pkg = str(manifest_meta.get("package", "") or "")
        keep_prefix = package_to_prefix(pkg) if (keep_only_app_code and pkg) else None
        role_map = build_component_role_map(manifest_meta)

        if _progress:
            print(f"  - package={pkg or '(empty)'} keep_prefix={keep_prefix or '(none)'}")

        if _progress:
            print(f"  - step: baksmali disassemble multi-dex ...")
        dexes = disassemble_all_dex(baksmali_jar, apk_path, smali_root, logs_dir)

        if _progress:
            print(f"  - step: parse smali -> methods + edges ...")
        methods, edges = parse_smali_methods(
            smali_root=smali_root,
            keep_prefix=keep_prefix,
            dedup_instructions=dedup_instructions,
        )
        if not methods:
            raise RuntimeError("No methods parsed (maybe filtered too hard).")

        if _progress:
            print(f"  - parsed: methods={len(methods)} edges={len(edges)} dex={len(dexes)}")

        emb_by_sig: Dict[str, np.ndarray] = {}
        method_sym_by_sig: Dict[str, np.ndarray] = {}

        show_embed_pbar = True  # default
        # allow external caller to switch tqdm on/off by setting this attribute in CONFIG;
        # we pass via _progress to keep minimal diff. (tqdm itself is already in your code)
        # You can disable pbar by setting progress_show_apk_embed=False, then we fallback to plain loop.
        # (we implement by checking an env var style: not changing logic)
        # We'll read a global from os.environ not needed; instead, caller passes _progress and set show_embed_pbar through it.
        # Keep simplest: if _progress is False, disable tqdm.
        if not _progress:
            show_embed_pbar = False

        with open(methods_out, "w", encoding="utf-8") as mf:
            iterator = methods
            if show_embed_pbar:
                iterator = tqdm(methods, desc=f"[{year}] {sha} embed", unit="method")
            for m in iterator:
                tokens = build_method_tokens(
                    tok=tok,
                    method_name=m.method_name,
                    instructions=m.instructions,
                    include_methodname_line=include_methodname_line,
                    invoke_only=invoke_only_for_embedding,
                )

                truncated = False
                num_chunks = 1

                if long_method_mode == "truncate":
                    if len(tokens) > cfg.max_len:
                        truncated = True
                        tokens = tokens[:cfg.max_len]
                    vec = embed_tokens_one(model, proc, cfg, device, tokens)
                elif long_method_mode == "chunk_pool":
                    chunks = chunk_tokens(tokens, cfg.max_len)
                    num_chunks = len(chunks)
                    vecs = [embed_tokens_one(model, proc, cfg, device, ch) for ch in chunks]
                    vec = pool_vectors(vecs, pool_mode)
                else:
                    raise ValueError(f"Unknown long_method_mode: {long_method_mode}")

                emb_by_sig[m.method_sig] = vec.astype(np.float32)
                strings_raw_preview = stable_topk(m.strings[: max_strings_per_method * 2], max_strings_per_method)
                norm_strings = [normalize_string_lit(s) for s in strings_raw_preview]

                sens_inv = []
                sens_inv_total = 0
                if record_sensitive_invokes:
                    for callee in m.invokes:
                        perms = sorted(list(pscout_api2perms.get(callee, set())))
                        if perms:
                            sens_inv_total += 1
                            if len(sens_inv) < max_sensitive_invokes_per_method:
                                sens_inv.append({"callee": callee, "perms": perms})

                method_stats = {
                    "invoke_count": len(m.invokes),
                    "strings_count": len(m.strings),
                    "sensitive_invoke_count": len(sens_inv),
                    "sensitive_invoke_count_total": sens_inv_total,
                    "token_count_raw_instr_lines": m.token_count_raw,
                }
                audit_literals = extract_audit_literals(strings_raw_preview, m.instructions, limit=max_strings_per_method)
                evidence_spans = extract_method_evidence_spans(m.instructions, sens_inv, limit=16)
                family_tags = infer_method_family_tags(m.class_name, m.method_name, m.invokes)
                sym_feat = np.array(
                    build_method_symbolic_features(
                        class_name=m.class_name,
                        method_name=m.method_name,
                        invokes=m.invokes,
                        strings=strings_raw_preview,
                        sensitive_invokes=sens_inv,
                        role_map=role_map,
                    ),
                    dtype=np.float32,
                )
                method_sym_by_sig[m.method_sig] = sym_feat

                rec = {
                    "year": year,
                    "apk_sha256": sha,
                    "apk_path": str(apk_path),
                    "package": pkg,
                    "keep_only_app_code": keep_only_app_code,
                    "keep_prefix": keep_prefix,
                    "method_sig": m.method_sig,
                    "class_name": m.class_name,
                    "method_name": m.method_name,
                    "invokes": m.invokes,
                    "invokes_preview": stable_topk(m.invokes, 24),
                    "strings": norm_strings,
                    "strings_raw_preview": strings_raw_preview,
                    "sensitive_invokes": sens_inv,
                    "token_len_raw": len(tokens),
                    "token_len_used": min(len(tokens), cfg.max_len) if long_method_mode == "truncate" else len(tokens),
                    "token_len": len(tokens),
                    "long_method_mode": long_method_mode,
                    "truncated": truncated,
                    "num_chunks": num_chunks,
                    "chunk_token_lengths": [len(ch) for ch in chunk_tokens(tokens, cfg.max_len)] if long_method_mode == "chunk_pool" else [len(tokens)],
                    "method_family_tags": family_tags,
                    "class_role": sorted(role_map.get(m.class_name, set())),
                    "callback_flags": [t for t in family_tags if t.endswith("callback") or t == "lifecycle"],
                    "method_stats": method_stats,
                    "audit_strings": audit_literals,
                    "evidence_spans": evidence_spans,
                    "evidence_summary": {
                        "has_source_like": any(x.get("role_hint") == "source" for x in evidence_spans),
                        "has_sink_like": any(x.get("role_hint") == "sink" for x in evidence_spans),
                        "has_transform_like": any(x.get("role_hint") == "transform" for x in evidence_spans),
                    },
                    "emb_dim": int(vec.shape[0]),
                    "symbolic_feature_names": SYMBOLIC_FEATURE_NAMES,
                    "symbolic_features": sym_feat.astype(np.float32).tolist(),
                    "embedding": vec.astype(np.float32).tolist(),
                }
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if _progress:
            print(f"  - step: write edges.jsonl ...")
        with open(edges_out, "w", encoding="utf-8") as ef:
            for caller, callee in edges:
                perms = sorted(list(pscout_api2perms.get(callee, set())))
                ef.write(json.dumps({
                    "year": year,
                    "apk_sha256": sha,
                    "caller": caller,
                    "callee": callee,
                    "kind": "invoke",
                    "callee_is_sensitive": 1 if perms else 0,
                    "callee_perms": perms,
                }, ensure_ascii=False) + "\n")

        if build_fcg:
            if _progress:
                print(f"  - step: build graph.pt (FCG) ...")
            internal_nodes = [m.method_sig for m in methods]
            build_gnn_fcg(
                apk_out=apk_out,
                year=year,
                sha=sha,
                apk_path=apk_path,
                internal_nodes=internal_nodes,
                edges=edges,
                emb_by_sig=emb_by_sig,
                method_sym_by_sig=method_sym_by_sig,
                model=model,
                tok=tok,
                proc=proc,
                cfg=cfg,
                device=device,
                pscout_api2perms=pscout_api2perms,
                embed_external_apis=embed_external_apis,
                add_reverse_edges=fcg_add_reverse_edges,
                add_self_loops=fcg_add_self_loops,
                keep_isolated_nodes=fcg_keep_isolated_nodes,
            )

        manifest_out.write_text(json.dumps(manifest_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        meta_out.write_text(json.dumps({
            "year": year,
            "apk_sha256": sha,
            "apk_path": str(apk_path),
            "package": pkg,
            "keep_only_app_code": keep_only_app_code,
            "keep_prefix": keep_prefix,
            "dex_entries": dexes,
            "methods_total": len(methods),
            "edges_total": len(edges),
            "manifest_permissions": len(manifest_meta.get("permissions", []) or []),
            "symbolic_dim": len(SYMBOLIC_FEATURE_NAMES),
            "created_at": now_ts(),
            "time_sec_total": round(time.time() - t0, 3),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        if _progress:
            print(f"[{now_ts()}][APK][{_task_i}/{_task_n}] DONE  {year}/{sha}  time_sec={round(time.time()-t0,3)}")

    except Exception as e:
        append_text(err_out, f"[{now_ts()}] FAIL apk={apk_path}\n{repr(e)}\n")
        if _progress:
            print(f"[{now_ts()}][APK][{_task_i}/{_task_n}] FAIL  {year}/{sha} -> {repr(e)}  (see {err_out})")
        raise
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run_with_config(CONFIG: dict):
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    def R(p: str) -> Path:
        return (project_root / p).expanduser().resolve()

    progress = bool(CONFIG.get("progress_every_apk", True))

    # weights_path = R(CONFIG.get("analyze_tools/weights"))
    def _pick(cfg: dict, *keys: str):
        for k in keys:
            v = cfg.get(k)
            if v:
                return v
        return None

    weights_rel = (
            _pick(CONFIG, "bert_weights", "analyze_tools/weights")
            or (CONFIG.get("analyze_tools") or {}).get("bert_weights")  # 兼容嵌套写法 analyze_tools: {weights: ...}
    )

    if not weights_rel:
        raise KeyError(
            "Missing weights path in CONFIG. Provide one of: "
            "'weights', 'analyze_tools/weights', or 'analyze_tools':{'weights':...}"
        )

    weights_path = R(weights_rel)

    vocab_path = R(CONFIG.get("bert_vocab"))
    cfg_path = R(CONFIG.get("bert_cfg"))
    baksmali_jar = R(CONFIG.get("baksmali_jar"))

    out_dir = R(CONFIG.get("out_dir", "apk_analysis/out"))
    tmp_dir = R(CONFIG.get("tmp_dir", "apk_analysis/_tmp"))
    safe_mkdir(out_dir)
    safe_mkdir(tmp_dir)

    pscout_path = Path(str(CONFIG.get("pscout_allmappings", ""))).expanduser()
    if not pscout_path.exists():
        raise FileNotFoundError(f"pscout_allmappings not found: {pscout_path}")
    print(f"[PScout] loading: {pscout_path}")
    pscout_api2perms = load_pscout_allmappings(pscout_path)
    print(f"[PScout] loaded APIs: {len(pscout_api2perms)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ENV] device={device}")
    print(f"[ENV] weights={weights_path.name} ({weights_path.stat().st_size/1024/1024/1024:.2f} GB)")

    model, cfg, tok, proc = load_model(cfg_path, vocab_path, weights_path, device)

    single_apk = (CONFIG.get("single_apk") or "").strip()
    if single_apk:
        apk_path = R(single_apk)
        year = apk_path.parent.name
        tasks = [(year, apk_path)]
    else:
        apks_root = R(CONFIG.get("apks_root", "downloaded_samples"))
        years = parse_years(CONFIG.get("years", "2014-2024"))
        tasks = collect_apks(apks_root, years)

    limit = int(CONFIG.get("limit_apks", 0) or 0)
    if limit > 0:
        tasks = tasks[:limit]

    print(f"[APK] tasks={len(tasks)} out_dir={out_dir}")
    fails = 0

    for i, (year, apk_path) in enumerate(tasks, start=1):
        try:
            process_one_apk(
                apk_path=apk_path,
                year=year,
                out_dir=out_dir,
                tmp_root=tmp_dir,
                baksmali_jar=baksmali_jar,
                model=model,
                cfg=cfg,
                tok=tok,
                proc=proc,
                device=device,
                pscout_api2perms=pscout_api2perms,

                keep_only_app_code=bool(CONFIG.get("keep_only_app_code", False)),
                dedup_instructions=bool(CONFIG.get("dedup_instructions", False)),
                include_methodname_line=bool(CONFIG.get("include_methodname_line", True)),
                invoke_only_for_embedding=bool(CONFIG.get("invoke_only_for_embedding", False)),
                long_method_mode=str(CONFIG.get("long_method_mode", "truncate")),
                pool_mode=str(CONFIG.get("pool_mode", "mean")),
                max_strings_per_method=int(CONFIG.get("max_strings_per_method", 8)),

                build_fcg=bool(CONFIG.get("build_fcg", True)),
                embed_external_apis=str(CONFIG.get("embed_external_apis", "signature")),
                fcg_add_reverse_edges=bool(CONFIG.get("fcg_add_reverse_edges", True)),
                fcg_add_self_loops=bool(CONFIG.get("fcg_add_self_loops", False)),
                fcg_keep_isolated_nodes=bool(CONFIG.get("fcg_keep_isolated_nodes", True)),
                force_rebuild_graph=bool(CONFIG.get("force_rebuild_graph", False)),

                record_sensitive_invokes=bool(CONFIG.get("record_sensitive_invokes", True)),
                max_sensitive_invokes_per_method=int(CONFIG.get("max_sensitive_invokes_per_method", 30)),

                _progress=progress,
                _task_i=i,
                _task_n=len(tasks),
            )
        except Exception as e:
            fails += 1
            print(f"[APK][ERR] {apk_path} -> {repr(e)}")

    print(f"[APK][DONE] total={len(tasks)}, fails={fails}, out={out_dir}")
