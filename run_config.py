# run_config.py
# -*- coding: utf-8 -*-

CONFIG = {
    # -------------------------
    # Global switches
    # -------------------------
    "stage_apk_analyze": True,
    "stage_gnn_analyze": True,
    "stage_kb_build": True,
    "stage_prototype": True,
    "stage_infer": True,

    # -------------------------
    # Logging / Progress
    # -------------------------
    "progress_enable": True,
    "progress_every_apk": True,
    "progress_show_apk_embed": True,
    "progress_show_gnn_export": True,
    "progress_show_kb": True,

    # -------------------------
    # Paths (DexBERT)
    # -------------------------
    "bert_weights": "analyze_tools/weights/model_steps_604364.pt",
    "bert_vocab": "analyze_tools/vocab.txt",
    "bert_cfg": "analyze_tools/bert_base.json",
    "baksmali_jar": "analyze_tools/baksmali-2.5.2.jar",
    "pscout_allmappings": "/home/leon/tools/PScout/results/API_22/allmappings",

    # -------------------------
    # Data
    # -------------------------
    "apks_root": "downloaded_samples",
    "years": "2014-2024",
    "single_apk": "",
    "limit_apks": 0,

    # -------------------------
    # Output
    # -------------------------
    "out_dir": "out",
    "tmp_dir": "_tmp",

    # -------------------------
    # APK -> methods/edges/graph
    # -------------------------
    "keep_only_app_code": False,
    "dedup_instructions": False,
    "include_methodname_line": True,
    "invoke_only_for_embedding": False,
    "long_method_mode": "chunk_pool",
    "pool_mode": "mean",
    "max_strings_per_method": 16,
    "record_sensitive_invokes": True,
    "max_sensitive_invokes_per_method": 40,
    "build_fcg": True,
    "embed_external_apis": "signature",
    "fcg_add_reverse_edges": True,
    "fcg_add_self_loops": False,
    "fcg_keep_isolated_nodes": True,
    "force_rebuild_graph": False,

    # -------------------------
    # GNN (self-supervised link prediction)
    # -------------------------
    "gnn_enabled": True,
    "gnn_train": True,
    "gnn_encoder_ckpt": "out/_gnn/gnn_encoder.pt",
    "gnn_encoder_meta": "out/_gnn/gnn_encoder_meta.json",
    "gnn_model": "sage",
    "gnn_hidden_dim": 256,
    "gnn_out_dim": 128,
    "gnn_num_layers": 2,
    "gnn_dropout": 0.2,
    "lp_epochs": 100,
    "lp_lr": 1e-3,
    "lp_weight_decay": 1e-4,
    "lp_val_ratio": 0.1,
    "lp_pos_batch_size": 4096,
    "lp_seed": 7,
    "lp_patience": 12,
    "lp_batches_per_graph": 2,
    "lp_hard_negative_ratio": 0.5,
    "lp_use_forward_edges_only": False,

    # -------------------------
    # Behavior mining/export
    # -------------------------
    "mine_enabled": True,
    "export_units": True,
    "overwrite_units": True,
    "units_subdir": "behavior_units",
    "max_smali_lines_per_method": 160,
    "max_edges_in_md": 300,
    "mine_algo": "v3",
    "mine_link_hops": 3,
    "mine_anchor_sim_thr": 0.55,
    "mine_expand_hops": 3,
    "mine_node_sim_thr": 0.35,
    "mine_min_unit_nodes": 8,
    "mine_keep_external": True,

    # v3
    "v3_candidate_hops": 4,
    "v3_ppr_alpha": 0.15,
    "v3_ppr_iters": 25,
    "v3_w_ppr": 0.45,
    "v3_w_sim": 0.25,
    "v3_w_dist": 0.10,
    "v3_w_dir": 0.20,
    "v3_min_nodes": 8,
    "v3_max_nodes": 120,
    "v3_split_components": True,
    "v3_drop_non_sensitive_external": True,
    "v3_md_max_internal_nodes": 60,
    "v3_md_max_external_nodes": 40,
    "v3_anchor_link_hops": 3,
    "v3_anchor_sim_thr": 0.55,

    # -------------------------
    # KB builder
    # -------------------------
    "kb_units_subdir": "behavior_units_kb",
    "kb_lexical_limit": 200,
    "kb_hint_limit": 256,
    "kb_pool_mode": "score_weighted",
    "kb_keep_nodes_sig": True,
    "kb_keep_edges_sig": True,
    "kb_write_md": True,
    "kb_out_subdir": "behavior_units_kb",
    "kb_in_subdir": "behavior_units",

    # -------------------------
    # Prototype builder
    # -------------------------
    "proto_root_rel": "_prototypes",
    "proto_apk_manifest": "",
    "proto_label_manifest": "",
    "proto_reset": True,

    "proto_tau": 0.70,
    "proto_gamma": 0.25,
    "proto_eta": 0.70,

    "proto_w_unit": 0.25,
    "proto_w_internal": 0.25,
    "proto_w_sensitive": 0.25,
    "proto_w_anchor": 0.20,
    "proto_w_max": 0.05,

    "proto_top_k_members": 20,
    "proto_top_m_exemplars": 8,
    "proto_top_l_psi": 1024,
    "proto_max_center_step": 0.02,
    "proto_alpha_fallback": 0.10,
    "proto_s_exemplar": 0.80,

    # -------------------------
    # Inference / prototype
    # -------------------------
    "infer_in_subdir": "behavior_units_kb",
    "infer_out_subdir": "infer",
    "infer_proto_index": "_prototypes/cluster_index.json",
    "infer_proto_root": "_prototypes",
    "infer_tau": 0.70,
    "infer_gamma": 0.25,
    "infer_eta": 0.70,
    "llm_api_key": "",
    "llm_api_key_env": "OPENROUTER_API_KEY",
    # "llm_proxy": "http://127.0.0.1:7890",
    "llm_proxy": "",
    "llm_provider": "",
    "llm_model": "",
    "llm_base_url": "",
    "llm_timeout_sec": 90,
    "llm_temperature": 0.0,
    "llm_max_output_tokens_bu": 1200,
    "llm_max_output_tokens_final": 1600,
    "llm_max_output_tokens_state": 1600,
    "llm_max_output_tokens_reconcile": 1600,

    "llm_site_url": "http://localhost",
    "llm_app_name": "AndroSerum",

    # retrieve budget
    "infer_topk_bu": 16,
    "infer_max_exemplars": 3,
    "infer_max_ev_nodes": 30,
    "infer_max_ev_edges": 80,
    "infer_max_lex_items": 32,
}

CONFIG.update({
    # ---------- LLM runtime ----------
    "llm_temperature": 0.0,
    "llm_connect_timeout_sec": 10,
    "llm_read_timeout_sec": 180,
    "llm_max_retries": 6,
    "llm_retry_base_sec": 1.5,

    # ---------- inference / prototype matching ----------
    "infer_tau": 0.70,
    "infer_gamma": 0.25,
    "infer_eta": 0.70,

    # ---------- incremental inference policy ----------
    "infer_topk_bu": 8,
    "infer_topk_proto_per_bu": 2,
    "infer_batch_size": 2,
    "infer_final_reconcile": True,

    # ---------- evidence caps ----------
    "infer_max_ev_nodes": 10,
    "infer_max_ev_edges": 40,
    "infer_max_lex_items": 24,
    "infer_max_cluster_freq_psi": 12,

    # ---------- proto weights ----------
    "proto_w_unit": 0.25,
    "proto_w_internal": 0.25,
    "proto_w_sensitive": 0.25,
    "proto_w_anchor": 0.20,
    "proto_w_max": 0.05,
})
