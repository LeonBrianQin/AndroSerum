# -*- coding: utf-8 -*-
import time
from run_config import CONFIG

import apk_analyze
import gnn_analyze
import kb_builder
import proto_builder
import infer_engine


def _banner(title: str) -> None:
    print("\n" + "=" * 88)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] {title}")
    print("=" * 88)


if __name__ == "__main__":
    progress = bool(CONFIG.get("progress_enable", True))
    if progress:
        _banner("AndroSerum Pipeline START")
        print(f"[CFG] stage_apk_analyze={bool(CONFIG.get('stage_apk_analyze', False))}")
        print(f"[CFG] stage_gnn_analyze={bool(CONFIG.get('stage_gnn_analyze', False))}")
        print(f"[CFG] stage_kb_build={bool(CONFIG.get('stage_kb_build', False))}")
        print(f"[CFG] stage_prototype={bool(CONFIG.get('stage_prototype', False))}")
        print(f"[CFG] stage_infer={bool(CONFIG.get('stage_infer', False))}")
        print(f"[CFG] years={CONFIG.get('years')} single_apk={CONFIG.get('single_apk')!r} limit_apks={CONFIG.get('limit_apks')}")
        print(f"[CFG] out_dir={CONFIG.get('out_dir')} tmp_dir={CONFIG.get('tmp_dir')}")
        print("-" * 88)

    if bool(CONFIG.get("stage_apk_analyze", True)):
        if progress:
            _banner("STAGE 1/5: apk_analyze")
        apk_analyze.run_with_config(CONFIG)

    if bool(CONFIG.get("stage_gnn_analyze", False)):
        if progress:
            _banner("STAGE 2/5: gnn_analyze")
        gnn_analyze.run_with_config(CONFIG)

    if bool(CONFIG.get("stage_kb_build", False)):
        if progress:
            _banner("STAGE 3/5: kb_builder")
        kb_builder.run_with_config(CONFIG)

    if bool(CONFIG.get("stage_prototype", False)):
        if progress:
            _banner("STAGE 4/5: proto_builder")
        proto_builder.run_with_config(CONFIG)

    if bool(CONFIG.get("stage_infer", False)):
        if progress:
            _banner("STAGE 5/5: infer_engine")
        infer_engine.run_with_config(CONFIG)

    if progress:
        _banner("AndroSerum Pipeline DONE")