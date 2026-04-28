#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
from typing import Any, List
import shutil
import subprocess
import zipfile
from collections import deque
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# PyG（没装会报错）
try:
    from torch_geometric.nn import SAGEConv, GATConv
    from torch_geometric.utils import negative_sampling
except Exception:
    SAGEConv = None
    GATConv = None
    negative_sampling = None

DEX_RE = re.compile(r"^classes(\d*)\.dex$")


# -------------------------
# utils
# -------------------------
def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_years(years: Any) -> List[str]:
    """
    Robust year parser.
    Accepts:
      - "2014-2024"
      - "2014,2016,2018"
      - ["2014","2015"] / [2014, 2015]
      - {"2019", 2020}
      - 2017
    Returns:
      - List[str] like ["2014", "2015", ...], sorted ascending, unique.
    """
    if years is None:
        return []

    # 1) Iterable input: list/tuple/set
    if isinstance(years, (list, tuple, set)):
        out: List[str] = []
        for x in years:
            if x is None:
                continue
            sx = str(x).strip()
            if not sx:
                continue
            # allow elements like "2014-2016"
            out.extend(parse_years(sx))
        # de-dup + sort (numeric if possible)
        uniq = sorted(set(out), key=lambda z: int(z) if z.isdigit() else z)
        return uniq

    # 2) Scalar input: int/str/other
    s = str(years).strip()
    if not s:
        return []

    # Support forms like "2014 - 2024"
    if "-" in s:
        a, b = s.split("-", 1)
        a, b = a.strip(), b.strip()
        if a.isdigit() and b.isdigit():
            aa, bb = int(a), int(b)
            lo, hi = (aa, bb) if aa <= bb else (bb, aa)
            return [str(y) for y in range(lo, hi + 1)]
        # fallback: treat as comma-separated if range invalid

    # Comma separated
    parts = [x.strip() for x in s.split(",") if x.strip()]
    # de-dup + sort
    uniq = sorted(set(parts), key=lambda z: int(z) if z.isdigit() else z)
    return uniq


def find_graph_pts(graphs_root: Path, years: List[str]) -> List[Path]:
    pts = []
    for y in years:
        yd = graphs_root / y
        if not yd.exists():
            continue
        for sha_dir in sorted(yd.iterdir()):
            p = sha_dir / "graph.pt"
            if p.exists() and p.stat().st_size > 0:
                pts.append(p)
    return pts


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a, b, dim=0).item())


# -------------------------
# baksmali (for exporting smali snippets)
# -------------------------
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


def run_baksmali_for_dex(baksmali_jar: Path, apk_path: Path, dex_entry: str, out_dir: Path) -> None:
    safe_mkdir(out_dir)
    cmd = [
        "java", "-jar", str(baksmali_jar),
        "disassemble",
        f"{str(apk_path)}/{dex_entry}",
        "-o", str(out_dir),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"baksmali failed for {dex_entry}:\n{p.stdout[:2000]}")


def disassemble_all_dex(baksmali_jar: Path, apk_path: Path, smali_root: Path) -> None:
    dexes = list_dex_entries(apk_path)
    if not dexes:
        raise RuntimeError("No classes*.dex found in APK")
    safe_mkdir(smali_root)
    for dex in dexes:
        dex_name = dex.replace(".dex", "")
        run_baksmali_for_dex(baksmali_jar, apk_path, dex, smali_root / dex_name)


def extract_smali_blocks_for_methods(
    smali_root: Path,
    target_method_sigs: Set[str],
    max_lines_per_method: int,
) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    remaining = set(target_method_sigs)

    for smali_file in smali_root.rglob("*.smali"):
        if not remaining:
            break
        try:
            lines = smali_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue

        class_name = ""
        in_method = False
        method_name = ""
        buf: List[str] = []

        for line in lines:
            if line.startswith(".class"):
                raw = line.strip().split(" ")[-1]
                if raw.startswith("L") and raw.endswith(";"):
                    class_name = raw[1:-1]
                else:
                    class_name = raw
                continue

            if line.startswith(".method"):
                in_method = True
                buf = [line]
                method_name = line.split(" ")[-1].strip()
                continue

            if in_method:
                buf.append(line)
                if line.startswith(".end method"):
                    in_method = False
                    if class_name and method_name:
                        sig = f"L{class_name};->{method_name}"
                        if sig in remaining:
                            trunc = False
                            smali_lines = buf
                            if len(smali_lines) > max_lines_per_method:
                                trunc = True
                                smali_lines = smali_lines[:max_lines_per_method] + [
                                    "    # ... <TRUNCATED> ...",
                                    ".end method",
                                ]
                            out[sig] = {
                                "smali": "\n".join(smali_lines),
                                "file": str(smali_file),
                                "truncated": trunc,
                            }
                            remaining.remove(sig)
                    method_name = ""
                    buf = []
    return out


def load_methods_meta(methods_out: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not methods_out.exists():
        return out
    with methods_out.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            sig = str(obj.get('method_sig', '')).strip()
            if sig:
                out[sig] = obj
    return out


_URL_RE = re.compile(r"(?i)https?://[^\s\"'<>]+")
_DOMAIN_RE = re.compile(r"(?i)\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}\b")
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_INTENT_ACTION_RE = re.compile(r"android\.intent\.[A-Z0-9_\.]+")
_CONTENT_URI_RE = re.compile(r"content://[^\s\"'<>]+", re.IGNORECASE)
_FILE_PATH_RE = re.compile(r"(?:/data/data|/sdcard|/storage|/mnt/|/system/)[^\s\"'<>]+", re.IGNORECASE)
_HEADER_KEY_RE = re.compile(r"(?i)^(host|user-agent|cookie|authorization|referer|x-[a-z0-9-]+|udid|imei|imsi|mac|token)$")
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/=]{24,}$")
_HEX_RE = re.compile(r"^[0-9a-fA-F]{16,}$")


def _stable_uniq(items: List[str], limit: int) -> List[str]:
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


def _extract_structured_literals(strings_raw: List[str], smali_text: str, limit: int = 16) -> Dict[str, List[str]]:
    text = "\n".join(strings_raw) + "\n" + (smali_text or "")
    urls = _stable_uniq(_URL_RE.findall(text), limit)
    ips = _stable_uniq(_IP_RE.findall(text), limit)
    domains = _stable_uniq([d for d in _DOMAIN_RE.findall(text) if not d.startswith('android.')], limit)
    actions = _stable_uniq(_INTENT_ACTION_RE.findall(text), limit)
    content_uris = _stable_uniq(_CONTENT_URI_RE.findall(text), limit)
    file_paths = _stable_uniq(_FILE_PATH_RE.findall(text), limit)
    headers = _stable_uniq([s for s in strings_raw if _HEADER_KEY_RE.fullmatch((s or '').strip())], limit)
    base64s = _stable_uniq([s for s in strings_raw if _BASE64_RE.fullmatch((s or '').strip()) and any(ch in s for ch in '+/=')], limit)
    hexes = _stable_uniq([s for s in strings_raw if _HEX_RE.fullmatch((s or '').strip())], limit)
    return {
        'urls_topk': urls, 'domains_topk': domains, 'ips_topk': ips, 'http_headers_topk': headers,
        'intent_actions_topk': actions, 'content_uris_topk': content_uris, 'file_paths_topk': file_paths,
        'base64_literals_topk': base64s, 'hex_literals_topk': hexes,
    }


def _extract_spans(smali_text: str, sensitive_invokes: List[Dict[str, Any]], limit: int = 24) -> List[Dict[str, Any]]:
    lines = (smali_text or '').splitlines()
    sens_map = {str(x.get('callee')): list(x.get('perms') or []) for x in (sensitive_invokes or [])}
    out: List[Dict[str, Any]] = []
    for i, line in enumerate(lines, start=1):
        s = line.strip()
        if s.startswith('invoke-'):
            callee = s.split()[-1].strip().strip(',')
            role = 'other'
            if any(k in callee for k in ['getSubscriberId', 'getDeviceId', 'getLine1Number', 'WifiInfo', 'LocationManager']):
                role = 'source'
            elif any(k in callee for k in ['HttpURLConnection', 'Lorg/apache/http', 'Socket', 'execute(', 'openConnection']):
                role = 'sink'
            elif any(k in callee for k in ['MessageDigest', 'Cipher', 'Mac', 'Base64']):
                role = 'transform'
            item = {'kind': 'API_CALL', 'line_no': i, 'text': s[:280], 'callee': callee, 'role_hint': role}
            if callee in sens_map:
                item['perms'] = sens_map[callee]
            out.append(item)
        elif s.startswith('const-string') or s.startswith('const-string/jumbo'):
            m = re.search(r'"(.*)"', s)
            if m:
                val = m.group(1)
                role = 'other'
                if _URL_RE.search(val) or _IP_RE.search(val):
                    role = 'network_indicator'
                elif _INTENT_ACTION_RE.search(val):
                    role = 'intent_indicator'
                elif _CONTENT_URI_RE.search(val):
                    role = 'content_uri_indicator'
                elif _FILE_PATH_RE.search(val):
                    role = 'file_indicator'
                out.append({'kind': 'STRING_LITERAL', 'line_no': i, 'text': s[:280], 'value': val[:240], 'role_hint': role})
        if len(out) >= limit:
            break
    return out


def _infer_primary_role(node: Dict[str, Any]) -> Tuple[str, List[str]]:
    feats = set(node.get('active_symbolic_features') or [])
    evidence = node.get('evidence') or {}
    role_flags: List[str] = []
    if evidence.get('source_api_hits') or any(x in feats for x in ['api_location', 'api_device_id', 'cb_onreceive']):
        role_flags.append('source')
    if evidence.get('sink_api_hits') or any(x in feats for x in ['api_network', 'has_url', 'has_ip']):
        role_flags.append('sink')
    if evidence.get('transform_api_hits') or any(x in feats for x in ['api_crypto', 'api_reflection', 'api_dynload']):
        role_flags.append('transform')
    method_name = str(node.get('method_name') or '').lower()
    if method_name.startswith(('oncreate', 'onstart', 'onresume', 'onreceive', 'onaccessibility')) or any(x in feats for x in ['cb_lifecycle', 'cb_onreceive', 'cb_accessibility']):
        role_flags.append('entry')
    if node.get('is_anchor'):
        role_flags.append('anchor')
    primary = role_flags[0] if role_flags else ('anchor' if node.get('is_anchor') else 'helper')
    return primary, _stable_uniq(role_flags, 8)


def _build_behavior_summary(nodes: List[Dict[str, Any]], anchors_meta: Dict[str, Any]) -> Dict[str, Any]:
    source = sum(1 for n in nodes if 'source' in (n.get('role_flags') or []))
    sink = sum(1 for n in nodes if 'sink' in (n.get('role_flags') or []))
    transform = sum(1 for n in nodes if 'transform' in (n.get('role_flags') or []))
    entry = sum(1 for n in nodes if 'entry' in (n.get('role_flags') or []))
    truncated = sum(1 for n in nodes if (n.get('smali_meta') or {}).get('truncated'))
    return {
        'source_like_nodes': source, 'sink_like_nodes': sink, 'transform_like_nodes': transform, 'entry_like_nodes': entry,
        'truncated_methods': truncated, 'num_anchors': len(anchors_meta.get('anchors') or []),
    }


# -------------------------
# GNN encoder (PyG)
# -------------------------
class GNNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, dropout: float, model: str):
        super().__init__()
        self.model = model
        self.dropout = dropout

        convs = []
        if model == "gat":
            convs.append(GATConv(in_dim, hidden_dim // 4, heads=4, dropout=dropout))
            for _ in range(num_layers - 2):
                convs.append(GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout))
            convs.append(GATConv(hidden_dim, out_dim, heads=1, dropout=dropout))
        else:
            convs.append(SAGEConv(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                convs.append(SAGEConv(hidden_dim, hidden_dim))
            convs.append(SAGEConv(hidden_dim, out_dim))

        self.convs = nn.ModuleList(convs)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i != len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


def split_edges(edge_index: torch.Tensor, val_ratio: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    E = edge_index.size(1)
    if E == 0:
        return edge_index, edge_index
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(E, generator=g)
    val_e = int(E * val_ratio)
    val_idx = perm[:val_e]
    tr_idx = perm[val_e:]
    return edge_index[:, tr_idx], edge_index[:, val_idx]


def score_edges(z: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    src, dst = edges[0], edges[1]
    return (z[src] * z[dst]).sum(dim=1)


def train_link_pred(
    graph_pts: List[Path],
    ckpt_path: Path,
    meta_path: Path,
    cfg: dict,
    device: torch.device,
) -> GNNEncoder:
    if SAGEConv is None or negative_sampling is None:
        raise RuntimeError("torch_geometric not available. Please install torch-geometric in your env.")

    torch.manual_seed(int(cfg.get("lp_seed", 7)))

    sample = torch.load(str(graph_pts[0]), map_location="cpu")
    in_dim = int(sample["x"].shape[1])

    enc = GNNEncoder(
        in_dim=in_dim,
        hidden_dim=int(cfg["gnn_hidden_dim"]),
        out_dim=int(cfg["gnn_out_dim"]),
        num_layers=int(cfg["gnn_num_layers"]),
        dropout=float(cfg["gnn_dropout"]),
        model=str(cfg.get("gnn_model", "sage")),
    ).to(device)

    opt = torch.optim.Adam(enc.parameters(), lr=float(cfg["lp_lr"]), weight_decay=float(cfg["lp_weight_decay"]))
    bce = nn.BCEWithLogitsLoss()

    splits = {}
    for gp in graph_pts:
        d = torch.load(str(gp), map_location="cpu")
        ei = d["edge_index"].long()
        tr, va = split_edges(ei, float(cfg["lp_val_ratio"]), int(cfg["lp_seed"]))
        splits[str(gp)] = {"train": tr, "val": va}

    pos_batch = int(cfg["lp_pos_batch_size"])
    epochs = int(cfg["lp_epochs"])

    print(f"[LP] graphs={len(graph_pts)} epochs={epochs} pos_batch={pos_batch} val_ratio={float(cfg['lp_val_ratio'])}")

    for ep in range(1, epochs + 1):
        enc.train()
        total_loss = 0.0
        total_edges = 0

        for gi, gp in enumerate(graph_pts, start=1):
            d = torch.load(str(gp), map_location="cpu")
            x = d["x"].float().to(device)
            ei = d["edge_index"].long().to(device)
            num_nodes = x.size(0)

            tr_edges = splits[str(gp)]["train"].to(device)
            if tr_edges.size(1) == 0:
                continue

            E = tr_edges.size(1)
            if E > pos_batch:
                idx = torch.randint(0, E, (pos_batch,), device=device)
                pos = tr_edges[:, idx]
            else:
                pos = tr_edges

            neg = negative_sampling(
                edge_index=ei,
                num_nodes=num_nodes,
                num_neg_samples=pos.size(1),
                method="sparse",
            ).to(device)

            z = enc(x, ei)
            pos_logits = score_edges(z, pos)
            neg_logits = score_edges(z, neg)

            logits = torch.cat([pos_logits, neg_logits], dim=0)
            labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)

            loss = bce(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * int(pos.size(1))
            total_edges += int(pos.size(1))

        avg = total_loss / max(total_edges, 1)
        print(f"[LP] epoch={ep}/{epochs} loss={avg:.4f}")

    safe_mkdir(ckpt_path.parent)
    torch.save(enc.state_dict(), ckpt_path)

    meta_path.write_text(json.dumps({
        "model": str(cfg.get("gnn_model", "sage")),
        "in_dim": in_dim,
        "hidden_dim": int(cfg["gnn_hidden_dim"]),
        "out_dim": int(cfg["gnn_out_dim"]),
        "num_layers": int(cfg["gnn_num_layers"]),
        "dropout": float(cfg["gnn_dropout"]),
        "epochs": epochs,
        "pos_batch_size": pos_batch,
        "val_ratio": float(cfg["lp_val_ratio"]),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    return enc


def load_encoder(ckpt_path: Path, meta_path: Path, device: torch.device) -> GNNEncoder:
    if SAGEConv is None:
        raise RuntimeError("torch_geometric not available. Please install torch-geometric in your env.")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    enc = GNNEncoder(
        in_dim=int(meta["in_dim"]),
        hidden_dim=int(meta["hidden_dim"]),
        out_dim=int(meta["out_dim"]),
        num_layers=int(meta["num_layers"]),
        dropout=float(meta["dropout"]),
        model=str(meta["model"]),
    ).to(device)
    state = torch.load(str(ckpt_path), map_location=device)
    enc.load_state_dict(state, strict=True)
    enc.eval()
    return enc


# -------------------------
# Graph helpers for mining
# -------------------------
def build_adj(edge_index: torch.Tensor, n: int, undirected: bool) -> List[List[int]]:
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    adj = [[] for _ in range(n)]
    for u, v in zip(src, dst):
        adj[u].append(v)
        if undirected and u != v:
            adj[v].append(u)
    return adj


def bfs_khop_dist(adj: List[List[int]], sources: List[int], k: int) -> Dict[int, int]:
    dist = {s: 0 for s in sources}
    q = deque(sources)
    while q:
        u = q.popleft()
        du = dist[u]
        if du == k:
            continue
        for v in adj[u]:
            if v not in dist:
                dist[v] = du + 1
                q.append(v)
    return dist


def connected_components_subset(adj: List[List[int]], nodes: Set[int]) -> List[List[int]]:
    comps = []
    seen: Set[int] = set()
    for s in nodes:
        if s in seen:
            continue
        q = deque([s])
        seen.add(s)
        comp = [s]
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v in nodes and v not in seen:
                    seen.add(v)
                    q.append(v)
                    comp.append(v)
        comps.append(comp)
    return comps


# -------------------------
# Anchor grouping (same spirit as v2)
# -------------------------
def group_sensitive_anchors(
    adj_u: List[List[int]],
    sens_ids: List[int],
    h: torch.Tensor,
    link_hops: int,
    sim_thr: float,
) -> Tuple[List[List[int]], Dict[int, List[Tuple[int, float]]]]:
    sens_set = set(sens_ids)
    sens_graph: Dict[int, List[Tuple[int, float]]] = {s: [] for s in sens_ids}

    for s in sens_ids:
        dist = bfs_khop_dist(adj_u, [s], link_hops)
        cands = [t for t in dist.keys() if t in sens_set and t != s]
        for t in cands:
            sim = cosine(h[s], h[t])
            if sim >= sim_thr:
                sens_graph[s].append((t, sim))
                sens_graph[t].append((s, sim))

    visited: Set[int] = set()
    groups: List[List[int]] = []
    for s in sens_ids:
        if s in visited:
            continue
        q = deque([s])
        visited.add(s)
        comp = [s]
        while q:
            u = q.popleft()
            for v, _ in sens_graph.get(u, []):
                if v not in visited:
                    visited.add(v)
                    q.append(v)
                    comp.append(v)
        groups.append(sorted(comp))
    return groups, sens_graph


# -------------------------
# v3: PPR + sweep cut
# -------------------------
def ppr_scores(
    adj: List[List[int]],
    seeds: List[int],
    alpha: float,
    iters: int,
    nodes_mask: Optional[Set[int]] = None,
) -> Dict[int, float]:
    if not seeds:
        return {}

    seeds = list(dict.fromkeys(seeds))
    s = {u: 1.0 / len(seeds) for u in seeds}
    p = dict(s)

    for _ in range(iters):
        newp: Dict[int, float] = {}
        for u, val in s.items():
            newp[u] = newp.get(u, 0.0) + alpha * val

        for u, pu in p.items():
            if nodes_mask is not None and u not in nodes_mask:
                continue
            nbrs = adj[u]
            if nodes_mask is not None:
                nbrs = [v for v in nbrs if v in nodes_mask]
            if not nbrs:
                continue
            share = (1.0 - alpha) * pu / len(nbrs)
            for v in nbrs:
                newp[v] = newp.get(v, 0.0) + share

        p = newp

    return p


def sweep_cut(
    adj: List[List[int]],
    cand_nodes: List[int],
    scores: Dict[int, float],
    anchors: Set[int],
    min_nodes: int,
    max_nodes: int,
) -> Tuple[List[int], float]:
    def key(u: int) -> float:
        if u in anchors:
            return float("inf")
        return scores.get(u, 0.0)

    order = sorted(cand_nodes, key=key, reverse=True)

    cand_set = set(cand_nodes)
    deg = {}
    vol_total = 0
    for u in cand_nodes:
        du = 0
        for v in adj[u]:
            if v in cand_set:
                du += 1
        deg[u] = du
        vol_total += du

    inS: Set[int] = set()
    boundary = 0
    volS = 0

    best_phi = 1e9
    best_k = 0

    anchors_needed = set(anchors)

    for i, u in enumerate(order, start=1):
        inS.add(u)
        volS += deg.get(u, 0)

        for v in adj[u]:
            if v not in cand_set:
                continue
            if v in inS:
                boundary -= 1
            else:
                boundary += 1

        if u in anchors_needed:
            anchors_needed.remove(u)

        if anchors_needed:
            continue

        if i < min_nodes:
            continue
        if i > max_nodes:
            break

        denom = min(volS, max(vol_total - volS, 1))
        phi = boundary / max(denom, 1)

        if phi < best_phi:
            best_phi = phi
            best_k = i

    if best_k == 0:
        picked = set(anchors)
        for u in order:
            picked.add(u)
            if len(picked) >= min_nodes:
                break
        return sorted(picked), best_phi

    return sorted(order[:best_k]), best_phi


def mine_units_v3(
    node_sigs: List[str],
    edge_index: torch.Tensor,
    is_external: torch.Tensor,
    is_sensitive: torch.Tensor,
    node_perms: List[List[str]],
    h: torch.Tensor,
    cfg: dict,
) -> Tuple[List[List[int]], List[dict], Dict[int, Dict[int, float]]]:
    n = len(node_sigs)
    adj_u = build_adj(edge_index, n, undirected=True)

    sens_ids = torch.where(is_sensitive == 1)[0].tolist()
    if not sens_ids:
        return [], [], {}

    groups, sens_graph = group_sensitive_anchors(
        adj_u=adj_u,
        sens_ids=sens_ids,
        h=h,
        link_hops=int(cfg.get("v3_anchor_link_hops", cfg.get("mine_link_hops", 3))),
        sim_thr=float(cfg.get("v3_anchor_sim_thr", cfg.get("mine_anchor_sim_thr", 0.55))),
    )

    candidate_hops = int(cfg.get("v3_candidate_hops", 4))
    alpha = float(cfg.get("v3_ppr_alpha", 0.15))
    iters = int(cfg.get("v3_ppr_iters", 25))
    w_ppr = float(cfg.get("v3_w_ppr", 0.55))
    w_sim = float(cfg.get("v3_w_sim", 0.35))
    w_dist = float(cfg.get("v3_w_dist", 0.10))

    min_nodes = int(cfg.get("v3_min_nodes", cfg.get("mine_min_unit_nodes", 8)))
    max_nodes = int(cfg.get("v3_max_nodes", 120))

    keep_external = bool(cfg.get("mine_keep_external", True))
    drop_non_sensitive_external = bool(cfg.get("v3_drop_non_sensitive_external", True))
    split_components = bool(cfg.get("v3_split_components", True))

    units: List[List[int]] = []
    metas: List[dict] = []
    unit_scores: Dict[int, Dict[int, float]] = {}

    for gid, anchors in enumerate(groups):
        anchors_set = set(anchors)

        dist = bfs_khop_dist(adj_u, anchors, candidate_hops)
        cand_nodes = sorted(dist.keys())
        cand_set = set(cand_nodes)

        ppr = ppr_scores(adj_u, anchors, alpha=alpha, iters=iters, nodes_mask=cand_set)

        ha = torch.stack([h[a] for a in anchors], dim=0).mean(dim=0)
        scores: Dict[int, float] = {}
        for u in cand_nodes:
            if (not keep_external) and int(is_external[u].item()) == 1:
                continue
            sim = cosine(h[u], ha)
            dd = dist.get(u, candidate_hops + 1)
            dist_score = 1.0 / (dd + 1.0)
            ppr_score = ppr.get(u, 0.0)
            s = w_ppr * ppr_score + w_sim * sim + w_dist * dist_score
            scores[u] = s

        picked, best_phi = sweep_cut(
            adj=adj_u,
            cand_nodes=[u for u in cand_nodes if u in scores],
            scores=scores,
            anchors=anchors_set,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
        )
        picked_set = set(picked)

        if keep_external and drop_non_sensitive_external:
            to_drop = []
            for u in list(picked_set):
                if u in anchors_set:
                    continue
                if int(is_external[u].item()) == 1 and int(is_sensitive[u].item()) == 0:
                    deg_in = 0
                    for v in adj_u[u]:
                        if v in picked_set and int(is_external[v].item()) == 0:
                            deg_in += 1
                    if deg_in <= 1:
                        to_drop.append(u)
            for u in to_drop:
                picked_set.remove(u)

        comps = [sorted(list(picked_set))]
        if split_components:
            comps = connected_components_subset(adj_u, picked_set)
            comps = [sorted(c) for c in comps if any(a in c for a in anchors_set)]

        for comp in comps:
            if len(comp) < min_nodes:
                continue

            in_comp_anchors = [a for a in anchors if a in comp]
            chain_edges = []
            aset = set(in_comp_anchors)
            for a in in_comp_anchors:
                for b, simab in sens_graph.get(a, []):
                    if b in aset and a < b:
                        chain_edges.append({
                            "src": a, "dst": b, "sim": simab,
                            "src_sig": node_sigs[a], "dst_sig": node_sigs[b],
                            "src_perms": node_perms[a] if a < len(node_perms) else [],
                            "dst_perms": node_perms[b] if b < len(node_perms) else [],
                        })

            uid = len(units)
            units.append(comp)
            unit_scores[uid] = {u: float(scores.get(u, 0.0)) for u in comp}

            metas.append({
                "algo": "v3",
                "group_id": gid,
                "conductance_phi": float(best_phi),
                "anchors": [{"node_id": a, "sig": node_sigs[a], "perms": (node_perms[a] if a < len(node_perms) else [])}
                            for a in in_comp_anchors],
                "anchor_chain_edges": chain_edges,
                "candidate_hops": candidate_hops,
                "ppr_alpha": alpha,
                "ppr_iters": iters,
                "weights": {"ppr": w_ppr, "sim": w_sim, "dist": w_dist},
                "size": len(comp),
            })

    return units, metas, unit_scores


def edges_in_unit(edge_index: torch.Tensor, nodes_set: Set[int]) -> List[Tuple[int, int]]:
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    e = []
    for u, v in zip(src, dst):
        if u in nodes_set and v in nodes_set:
            e.append((u, v))
    return e


# -------------------------
# Export behavior units for LLM
# -------------------------
def export_units_for_graph(
    graph_pt: Path,
    cfg: dict,
    encoder: GNNEncoder,
    baksmali_jar: Path,
    tmp_dir: Path,
) -> None:
    d = torch.load(str(graph_pt), map_location="cpu")

    node_sigs: List[str] = d["node_sigs"]
    edge_index: torch.Tensor = d["edge_index"].long()
    edge_index_fwd: torch.Tensor = d.get("edge_index_fwd", edge_index).long()
    x: torch.Tensor = d["x"].float()
    is_external: torch.Tensor = d["is_external"].long()
    is_sensitive: torch.Tensor = d["is_sensitive"].long()
    node_perms: List[List[str]] = d.get("node_perms", [[] for _ in range(len(node_sigs))])

    apk_path = Path(d.get("apk_path", "")).expanduser()
    graph_meta_obj = {}
    if (graph_pt.parent / "meta.json").exists():
        graph_meta_obj = json.loads((graph_pt.parent / "meta.json").read_text(encoding="utf-8"))
    manifest_obj = {}
    if (graph_pt.parent / "manifest.json").exists():
        manifest_obj = json.loads((graph_pt.parent / "manifest.json").read_text(encoding="utf-8"))
    methods_meta = load_methods_meta(graph_pt.parent / "methods.jsonl")
    if not apk_path.exists() and graph_meta_obj.get("apk_path"):
        apk_path = Path(graph_meta_obj["apk_path"]).expanduser()

    device = next(encoder.parameters()).device
    encoder.eval()
    with torch.no_grad():
        h = encoder(x.to(device), edge_index.to(device)).cpu()

    adj_u = build_adj(edge_index_fwd, len(node_sigs), undirected=True)
    rev_edge_index = torch.stack([edge_index_fwd[1], edge_index_fwd[0]], dim=0) if edge_index_fwd.numel() > 0 else edge_index_fwd
    adj_fwd = build_adj(edge_index_fwd, len(node_sigs), undirected=False)
    adj_bwd = build_adj(rev_edge_index, len(node_sigs), undirected=False)

    algo = str(cfg.get("mine_algo", "v3")).lower()

    if algo == "v2":
        raise RuntimeError("mine_algo=v2 is disabled in this latest script (to avoid giant units). Use v3.")
    else:
        units, metas, unit_scores = mine_units_v3(
            node_sigs=node_sigs,
            edge_index=edge_index,
            is_external=is_external,
            is_sensitive=is_sensitive,
            node_perms=node_perms,
            h=h,
            cfg=cfg,
        )

    out_dir = graph_pt.parent / str(cfg.get("units_subdir", "behavior_units"))
    safe_mkdir(out_dir)
    idx_path = out_dir / "index.json"
    if idx_path.exists() and (not bool(cfg.get("overwrite_units", False))):
        print(f"[SKIP] units exists: {out_dir}")
        return

    needed_internal: Set[str] = set()
    for nodes in units:
        for nid in nodes:
            if int(is_external[nid].item()) == 0:
                needed_internal.add(node_sigs[nid])

    work = tmp_dir / f"unit_export_{graph_pt.parent.parent.name}_{graph_pt.parent.name}"
    smali_root = work / "smali"
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    safe_mkdir(work)

    smali_map = {}
    try:
        disassemble_all_dex(baksmali_jar, apk_path, smali_root)
        smali_map = extract_smali_blocks_for_methods(
            smali_root=smali_root,
            target_method_sigs=needed_internal,
            max_lines_per_method=int(cfg["max_smali_lines_per_method"]),
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)

    max_edges_md = int(cfg.get("max_edges_in_md", 300))
    md_max_internal = int(cfg.get("v3_md_max_internal_nodes", 60))
    md_max_external = int(cfg.get("v3_md_max_external_nodes", 40))

    index = {
        "graph_pt": str(graph_pt),
        "apk_path": str(apk_path),
        "apk_sha256": d.get("apk_sha256", ""),
        "year": d.get("year", ""),
        "num_units": len(units),
        "mine_algo": algo,
        "params": {k: cfg.get(k) for k in cfg.keys() if k.startswith("v3_") or k.startswith("mine_")},
        "units": [],
    }

    if bool(cfg.get("overwrite_units", False)):
        for p in out_dir.glob("unit_*.json"):
            p.unlink(missing_ok=True)
        # for p in out_dir.glob("unit_*.md"):
        #     p.unlink(missing_ok=True)

    for uid, nodes in enumerate(units):
        nodes_set = set(nodes)
        e = edges_in_unit(edge_index, nodes_set)

        score_map = unit_scores.get(uid, {})
        nodes_sorted = sorted(nodes, key=lambda u: score_map.get(u, 0.0), reverse=True)

        internal_nodes = [u for u in nodes_sorted if int(is_external[u].item()) == 0]
        external_nodes = [u for u in nodes_sorted if int(is_external[u].item()) == 1]

        internal_md = internal_nodes[:md_max_internal]
        external_md = external_nodes[:md_max_external]
        md_nodes_set = set(internal_md + external_md)

        node_objs = []
        for nid in nodes:
            sig = node_sigs[nid]
            ext = int(is_external[nid].item())
            sen = int(is_sensitive[nid].item())
            perms = node_perms[nid] if nid < len(node_perms) else []

            smali = ""
            smali_meta = None
            if ext == 0:
                mm = smali_map.get(sig)
                if mm:
                    smali = mm["smali"]
                    smali_meta = {"file": mm["file"], "truncated": mm["truncated"]}

            node_objs.append({
                "node_id": nid,
                "sig": sig,
                "is_external": ext,
                "is_sensitive": sen,
                "perms": perms,
                "score": float(score_map.get(nid, 0.0)),
                "smali": smali,
                "smali_meta": smali_meta,
                "h": h[nid].tolist(),
            })

        unit_json = {
            "unit_id": uid,
            "year": d.get("year", ""),
            "apk_sha256": d.get("apk_sha256", ""),
            "apk_path": str(apk_path),
            "anchors_meta": metas[uid] if uid < len(metas) else {},
            "num_nodes": len(nodes),
            "num_edges": len(e),
            "nodes": node_objs,
            "edges": [{"src": u, "dst": v, "src_sig": node_sigs[u], "dst_sig": node_sigs[v]} for (u, v) in e],
        }
        (out_dir / f"unit_{uid:04d}.json").write_text(json.dumps(unit_json, ensure_ascii=False, indent=2), encoding="utf-8")

        md = []
        md.append(f"# Behavior Unit {uid:04d}")
        md.append(f"- APK: `{apk_path}`")
        md.append(f"- Nodes(full): {len(nodes)}  Edges: {len(e)}")
        md.append(f"- MD shows internal<= {md_max_internal}, external<= {md_max_external}. Full in JSON.")
        md.append("")

        md.append("## Anchor Sensitive APIs")
        am = unit_json["anchors_meta"].get("anchors", [])
        if am:
            for a in am:
                perms_txt = ", ".join(a.get("perms", [])) if a.get("perms") else "(no perms matched)"
                md.append(f"- `{a['sig']}` | perms: {perms_txt}")
        else:
            md.append("- (none)")
        md.append("")

        md.append("## Anchor Chain Edges (sim)")
        ce = unit_json["anchors_meta"].get("anchor_chain_edges", [])
        if ce:
            for x1 in ce[:200]:
                md.append(f"- `{x1['src_sig']}` -> `{x1['dst_sig']}`  (cos={x1['sim']:.3f})")
        else:
            md.append("- (none)")
        md.append("")

        md.append("## Subgraph Edges (sig) [trimmed by shown nodes]")
        shown_edges = []
        for (u, v) in e:
            if u in md_nodes_set and v in md_nodes_set:
                shown_edges.append((u, v))
        for (u, v) in shown_edges[:max_edges_md]:
            md.append(f"- `{node_sigs[u]}` -> `{node_sigs[v]}`")
        if len(shown_edges) > max_edges_md:
            md.append(f"- ... <TRUNCATED edges: {len(shown_edges)} shown> ...")
        md.append("")

        md.append("## Node Details (smali for internal nodes) [trimmed]")
        md_nodes = internal_md + external_md
        for nid in md_nodes:
            sig = node_sigs[nid]
            ext = int(is_external[nid].item())
            sen = int(is_sensitive[nid].item())
            perms = node_perms[nid] if nid < len(node_perms) else []
            md.append(f"### `{sig}`  (score={score_map.get(nid, 0.0):.4f})")
            md.append(f"- external: {ext}   sensitive: {sen}")
            if perms:
                md.append(f"- perms: {', '.join(perms)}")
            if ext == 0:
                mm = smali_map.get(sig)
                md.append("```smali")
                md.append(mm["smali"] if mm else "# <SMALI NOT FOUND>")
                md.append("```")
            else:
                md.append("_External API node: signature only._")
            md.append("")

        (out_dir / f"unit_{uid:04d}.md").write_text("\n".join(md), encoding="utf-8")

        index["units"].append({
            "unit_id": uid,
            "json": f"unit_{uid:04d}.json",
            "md": f"unit_{uid:04d}.md",
            "num_nodes": len(nodes),
            "num_edges": len(e),
            "num_anchors": len(am),
            "phi": unit_json["anchors_meta"].get("conductance_phi", None),
        })

    idx_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OUT] exported units: {out_dir}  num_units={len(units)} algo={algo}")


# -------------------------
# driver
# -------------------------
def run_with_config(CONFIG: dict):
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    def R(p: str) -> Path:
        return (project_root / p).expanduser().resolve()

    out_dir = R(CONFIG.get("out_dir", "out"))
    years = parse_years(CONFIG.get("years", "2014-2024"))
    graph_pts = find_graph_pts(out_dir, years)

    if not graph_pts:
        print("[GNN] no graph.pt found. Run apk_analyze stage first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GNN] device={device} graphs={len(graph_pts)} out_dir={out_dir}")

    ckpt = R(CONFIG.get("gnn_encoder_ckpt", "out/_gnn/gnn_encoder.pt"))
    meta = R(CONFIG.get("gnn_encoder_meta", "out/_gnn/gnn_encoder_meta.json"))

    # train or load
    if bool(CONFIG.get("gnn_train", True)) or (not ckpt.exists()) or (not meta.exists()):
        print(f"[GNN] training link prediction encoder -> {ckpt}")
        enc = train_link_pred(
            graph_pts=graph_pts,
            ckpt_path=ckpt,
            meta_path=meta,
            cfg=CONFIG,
            device=device,
        )
        enc.eval()
    else:
        print(f"[GNN] loading encoder: {ckpt}")
        enc = load_encoder(ckpt, meta, device)

    if not bool(CONFIG.get("mine_enabled", True)) or not bool(CONFIG.get("export_units", True)):
        print("[GNN] mine/export disabled by config.")
        return

    baksmali_jar = R(CONFIG.get("baksmali_jar"))
    tmp_dir = R(CONFIG.get("tmp_dir", "_tmp"))
    safe_mkdir(tmp_dir)

    # if single_apk is set, only export that one graph
    single_apk = (CONFIG.get("single_apk") or "").strip()
    if single_apk:
        sha = Path(single_apk).stem.upper()
        year = Path(single_apk).parent.name
        gp = out_dir / year / sha / "graph.pt"
        graph_pts = [gp] if gp.exists() else []

    show_export = bool(CONFIG.get("progress_show_gnn_export", True))

    for i, gp in enumerate(graph_pts, start=1):
        try:
            if show_export:
                year = gp.parent.parent.name
                sha = gp.parent.name
                print(f"[GNN][{i}/{len(graph_pts)}] export units for {year}/{sha}  graph={gp}")
            export_units_for_graph(
                graph_pt=gp,
                cfg=CONFIG,
                encoder=enc.to(device),
                baksmali_jar=baksmali_jar,
                tmp_dir=tmp_dir,
            )
        except Exception as e:
            print(f"[ERR] {gp} -> {repr(e)}")
