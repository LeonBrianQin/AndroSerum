# proto_db.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    from .utils_io import safe_read_json, l2_normalize, cos, jaccard
except Exception:
    from utils_io import safe_read_json, l2_normalize, cos, jaccard  # type: ignore


@dataclass
class MatchParams:
    tau: float = 0.55
    gamma: float = 0.10
    eta: float = 0.60
    w_unit: float = 0.45
    w_internal: float = 0.20
    w_sensitive: float = 0.20
    w_anchor: float = 0.10
    w_max: float = 0.05


def _vec(unit: Dict[str, Any], key: str) -> Optional[List[float]]:
    sem = unit.get("semantic_info") or {}
    v = sem.get(key)
    if not isinstance(v, list) or not v:
        return None
    out: List[float] = []
    for x in v:
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
    for p in (sym.get("perms_set") or {}).keys():
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
    for r, cnt in (audit.get("role_count") or {}).items():
        try:
            c = int(cnt)
        except Exception:
            c = 0
        if c > 0:
            toks.add(f"ROLE:{str(r).strip()}")
    return toks


class PrototypeDB:
    def __init__(self, proto_root: Path, params: MatchParams):
        self.proto_root = proto_root
        self.index_path = proto_root / "cluster_index.json"
        self.params = params
        self._clusters: Dict[str, Dict[str, Any]] = {}
        self._inv: Dict[str, Set[str]] = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        idx = safe_read_json(self.index_path)
        if not idx:
            raise FileNotFoundError(f"prototype index not found: {self.index_path}")
        self._inv = {str(k): set(v or []) for k, v in (idx.get("inverted_index") or {}).items()}
        for cid in idx.get("cluster_ids") or []:
            cid = str(cid)
            c = safe_read_json(self.proto_root / "clusters" / f"{cid}.json")
            if c:
                self._clusters[cid] = c
        self._loaded = True

    @property
    def clusters(self) -> Dict[str, Dict[str, Any]]:
        if not self._loaded:
            self.load()
        return self._clusters

    @property
    def inverted_index(self) -> Dict[str, Set[str]]:
        if not self._loaded:
            self.load()
        return self._inv

    @staticmethod
    def frequent_psi(cluster: Dict[str, Any], eta: float, cap: int = 256) -> Set[str]:
        size = int((cluster.get("base_info") or {}).get("size", 0) or 0)
        if size <= 0:
            return set()
        psi_count = (cluster.get("symbolic_info") or {}).get("psi_count") or {}
        thr = eta * size
        arr = []
        for k, v in psi_count.items():
            try:
                c = int(v)
            except Exception:
                continue
            if c >= thr:
                arr.append((c, str(k)))
        arr.sort(reverse=True)
        return {k for _, k in arr[:cap]}


class ClusterMatcher:
    def __init__(self, db: PrototypeDB):
        self.db = db

    def _semantic_sim(self, cluster: Dict[str, Any], unit: Dict[str, Any]) -> float:
        center = (cluster.get("semantic_info") or {}).get("center") or {}
        views = [
            ("unit_emb", self.db.params.w_unit),
            ("unit_emb_internal", self.db.params.w_internal),
            ("unit_emb_sensitive", self.db.params.w_sensitive),
            ("unit_emb_anchor", self.db.params.w_anchor),
            ("unit_emb_max", self.db.params.w_max),
        ]
        s = 0.0
        wsum = 0.0
        for key, w in views:
            cv = center.get(key)
            uv = _vec(unit, key)
            if cv and uv and len(cv) == len(uv):
                s += w * cos(uv, cv)
                wsum += w
        if wsum <= 0:
            return -1.0
        return float(s / wsum)

    def rank_clusters(self, kb_unit: Dict[str, Any], topk: int = 3) -> List[Dict[str, Any]]:
        psi_b = _symbolic_tokens(kb_unit)
        cand_ids: Set[str] = set()
        for it in psi_b:
            cand_ids |= self.db.inverted_index.get(it, set())
        rows: List[Dict[str, Any]] = []
        for cid in cand_ids:
            cluster = self.db.clusters.get(cid)
            if not cluster:
                continue
            sem = self._semantic_sim(cluster, kb_unit)
            if sem < self.db.params.tau:
                continue
            freq = PrototypeDB.frequent_psi(cluster, self.db.params.eta, cap=256)
            sym = jaccard(psi_b, freq)
            if sym < self.db.params.gamma:
                continue
            score = 0.85 * sem + 0.15 * sym
            rows.append({
                "cluster_id": cid,
                "score": float(score),
                "semantic_sim": float(sem),
                "symbolic_jaccard": float(sym),
                "cluster": cluster,
            })
        rows.sort(key=lambda x: (x["score"], x["semantic_sim"], x["symbolic_jaccard"]), reverse=True)
        return rows[:max(1, int(topk or 1))]