# AndroSerum

AndroSerum is a behavior-centric static analysis framework for Android malware analysis. It processes APK files, reconstructs function call graphs, extracts Behavior Units (BUs), builds reusable cross-sample behavior prototypes, and performs prototype-guided LLM inference to produce evidence-grounded APK-level analysis results.

The current implementation is organized as a five-stage pipeline:

1. `apk_analyze`: disassemble APKs, parse smali code, recover method-level evidence, and build function call graphs.
2. `gnn_analyze`: train or load a GNN encoder and mine Behavior Units from recovered graphs.
3. `kb_builder`: convert raw Behavior Units into compact behavior-unit knowledge.
4. `proto_builder`: align behavior units across samples and construct behavior prototypes.
5. `infer_engine`: retrieve matched prototypes and run LLM-based incremental APK-level reasoning.

The pipeline entry point is:

```bash
python main.py
```

All major runtime options are controlled by `run_config.py`.

---

## 1. Repository Structure

A typical repository layout is:

```text
AndroSerum/
├── main.py
├── run_config.py
├── apk_analyze.py
├── gnn_analyze.py
├── kb_builder.py
├── proto_builder.py
├── proto_db.py
├── infer_engine.py
├── llm_client.py
├── params.py
├── utils_io.py
├── requirements.txt
├── analyze_tools/
│   ├── baksmali-2.5.2.jar
│   ├── bert_base.json
│   ├── vocab.txt
│   ├── tokenization.py
│   ├── models.py
│   ├── dataloader.py
│   └── weights/
│       └── model_steps_604364.pt
├── downloaded_samples/
│   ├── 2014/
│   │   ├── <sha256>.apk
│   │   └── ...
│   └── ...
└── out/
```

`analyze_tools/`, `downloaded_samples/`, and `out/` may need to be created manually depending on how the repository is distributed.

---

## 2. Environment Setup

### 2.1 System Requirements

We recommend using a Linux environment.

Required software:

- Python 3.10
- Java Runtime Environment, required by `baksmali`
- Conda or Miniconda
- Git

Recommended hardware:

- CUDA-compatible GPU for GNN training and DexBERT embedding
- Sufficient disk space for disassembled smali files and intermediate graph outputs

The experiments were developed with Python 3.10 and PyTorch 2.0.0.

### 2.2 Create a Conda Environment

```bash
conda create -n androserum python=3.10 -y
conda activate androserum
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

A minimal `requirements.txt` is:

```txt
numpy==1.23.5
scipy==1.15.3
pandas==2.3.3
scikit-learn==1.7.2
networkx==3.4.2
matplotlib==3.10.8
joblib==1.5.3
tqdm==4.67.3
psutil==7.2.1

torch==2.0.0
torch-geometric==2.7.0
triton==2.0.0
einops==0.8.1
nystrom-attention==0.0.11

datasets==4.6.1
pyarrow==23.0.1

openai==1.109.1
tiktoken==0.12.0
langchain-core==0.3.49
langchain-openai==0.3.11
langsmith==0.3.45

requests==2.32.5
httpx==0.28.1
aiohttp==3.13.3
pydantic==2.12.5
pyyaml==6.0.3
regex==2026.2.28
rich==14.3.3
typer==0.24.1
click==8.3.1
packaging==24.2
pillow==12.1.1
javalang==0.13.0
androguard
```

Check whether PyTorch and CUDA are available:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If CUDA is required, please make sure that the installed PyTorch version matches your local CUDA driver.

---

## 3. Third-Party Tools and Models

The repository does not include APK samples, DexBERT weights, PScout mappings, or other third-party resources that may have separate licenses. Please prepare them manually before running the pipeline.

### 3.1 baksmali

Download `baksmali-2.5.2.jar` and place it at:

```text
analyze_tools/baksmali-2.5.2.jar
```

Check Java and baksmali:

```bash
java -version
java -jar analyze_tools/baksmali-2.5.2.jar --help
```

The path is configured in `run_config.py`:

```python
"baksmali_jar": "analyze_tools/baksmali-2.5.2.jar",
```

### 3.2 DexBERT

AndroSerum uses DexBERT to encode smali method bodies. Please prepare the DexBERT model files and helper code under `analyze_tools/`.

Expected files:

```text
analyze_tools/
├── bert_base.json
├── vocab.txt
├── tokenization.py
├── models.py
├── dataloader.py
└── weights/
    └── model_steps_604364.pt
```

Configure the paths in `run_config.py`:

```python
"bert_weights": "analyze_tools/weights/model_steps_604364.pt",
"bert_vocab": "analyze_tools/vocab.txt",
"bert_cfg": "analyze_tools/bert_base.json",
```

### 3.3 PScout Mapping

AndroSerum uses PScout mappings to identify sensitive Android APIs. Please download the PScout mapping file and set its path in `run_config.py`:

```python
"pscout_allmappings": "/path/to/PScout/results/API_22/allmappings",
```

The file is required by the APK analysis stage.

### 3.4 LLM API Key

The final inference stage requires an OpenAI-compatible API endpoint. We recommend using an environment variable rather than hard-coding keys in source files.


For OpenAI:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Then configure:

```python
"llm_api_key": "",
"llm_api_key_env": "OPENAI_API_KEY",
"llm_provider": "openai",
"llm_model": "gpt-5-chat",
"llm_base_url": "https://api.openai.com/v1",
"llm_proxy": "",
```

Do not commit API keys or local proxy settings to the repository.

---

## 4. Data Preparation

Place APK files under `downloaded_samples/` using year-based subdirectories:

```text
downloaded_samples/
├── 2013/
│   ├── <sha256_1>.apk
│   ├── <sha256_2>.apk
│   └── ...
├── 2015/
│   └── ...
└── 2025/
    └── ...
```

The default data-related configuration is:

```python
"apks_root": "downloaded_samples",
"years": "2013-2025",
"single_apk": "",
"limit_apks": 0,
```

Supported year formats:

```python
"years": "2014-2024"        # range
"years": "2019,2020,2021"   # selected years
"years": ["2019", "2020"]   # list
```

To process one APK:

```python
"single_apk": "downloaded_samples/2024/example.apk",
"limit_apks": 1,
```

To process all APKs in the selected years:

```python
"single_apk": "",
"limit_apks": 0,
```

---

## 5. Configuration

All major options are defined in `run_config.py`.

### 5.1 Stage Switches

```python
"stage_apk_analyze": True,
"stage_gnn_analyze": True,
"stage_kb_build": True,
"stage_prototype": True,
"stage_infer": True,
```

The five switches correspond to the five pipeline stages. `main.py` executes the enabled stages in order.

### 5.2 Output Directories

```python
"out_dir": "out",
"tmp_dir": "_tmp",
```

- `out/` stores graph files, behavior units, prototypes, and inference results.
- `_tmp/` stores temporary disassembly and intermediate files.

### 5.3 APK Analysis Options

```python
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
```

This stage disassembles APKs, parses smali methods, extracts symbolic evidence, builds invoke edges, and writes graph files.

### 5.4 GNN and Behavior Unit Extraction

```python
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

"mine_enabled": True,
"export_units": True,
"overwrite_units": True,
"units_subdir": "behavior_units",
```

If `gnn_train` is `True`, the GNN encoder is trained from recovered graphs. If it is `False`, the pipeline attempts to load the checkpoint from `gnn_encoder_ckpt`.

Behavior unit mining is controlled by:

```python
"mine_algo": "v3",
"v3_candidate_hops": 4,
"v3_ppr_alpha": 0.15,
"v3_ppr_iters": 25,
"v3_min_nodes": 8,
"v3_max_nodes": 120,
"v3_anchor_link_hops": 3,
"v3_anchor_sim_thr": 0.55,
```

### 5.5 Behavior Knowledge Construction

```python
"kb_in_subdir": "behavior_units",
"kb_out_subdir": "behavior_units_kb",
"kb_lexical_limit": 200,
"kb_hint_limit": 256,
"kb_pool_mode": "score_weighted",
"kb_write_md": True,
```

This stage converts raw behavior units into compact behavior-unit knowledge files.

### 5.6 Prototype Construction

```python
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
"proto_s_exemplar": 0.80,
```

The prototype database is written to:

```text
out/_prototypes/
```

The key files are:

```text
out/_prototypes/cluster_index.json
out/_prototypes/cluster_build_summary.json
out/_prototypes/assignments.jsonl
out/_prototypes/clusters/
```

### 5.7 Inference Configuration

```python
"infer_in_subdir": "behavior_units_kb",
"infer_out_subdir": "infer",
"infer_proto_index": "_prototypes/cluster_index.json",
"infer_proto_root": "_prototypes",

"infer_tau": 0.70,
"infer_gamma": 0.25,
"infer_eta": 0.70,

"infer_topk_bu": 8,
"infer_topk_proto_per_bu": 2,
"infer_batch_size": 2,
"infer_final_reconcile": True,

"infer_max_ev_nodes": 10,
"infer_max_ev_edges": 40,
"infer_max_lex_items": 24,
"infer_max_cluster_freq_psi": 12,
```

The inference stage retrieves matched prototypes for each selected BU and asks the LLM to produce local BU summaries, incrementally update APK-level state, and optionally perform final reconciliation.

---

## 6. Running the Pipeline

### 6.1 Full Pipeline

After preparing APKs, DexBERT, baksmali, PScout, and the LLM API key, enable all stages:

```python
"stage_apk_analyze": True,
"stage_gnn_analyze": True,
"stage_kb_build": True,
"stage_prototype": True,
"stage_infer": True,
```

Run:

```bash
python main.py
```

### 6.2 Run a Single APK for a Smoke Test

Set:

```python
"single_apk": "downloaded_samples/2024/example.apk",
"limit_apks": 1,

"stage_apk_analyze": True,
"stage_gnn_analyze": True,
"stage_kb_build": True,
"stage_prototype": True,
"stage_infer": True,
```

Run:

```bash
python main.py
```

This mode is useful for checking whether the environment, paths, and third-party tools are correctly configured.

### 6.3 Extract Behavior Units Without LLM Inference

To stop before the LLM stage:

```python
"stage_apk_analyze": True,
"stage_gnn_analyze": True,
"stage_kb_build": True,
"stage_prototype": True,
"stage_infer": False,
```

Run:

```bash
python main.py
```

### 6.4 Build Historical Prototypes and Infer Future APKs

For evaluation, avoid building prototypes from the same APKs used for test-time inference.

First, build prototypes using historical or training APKs:

```python
"years": "2013-2020",

"stage_apk_analyze": True,
"stage_gnn_analyze": True,
"stage_kb_build": True,
"stage_prototype": True,
"stage_infer": False,

"proto_reset": True,
```

Run:

```bash
python main.py
```

Then infer future APKs using the fixed prototype database:

```python
"years": "2021",

"stage_apk_analyze": True,
"stage_gnn_analyze": True,
"stage_kb_build": True,
"stage_prototype": False,
"stage_infer": True,

"infer_proto_root": "_prototypes",
"infer_proto_index": "_prototypes/cluster_index.json",
```

Run:

```bash
python main.py
```

If the future APKs have already been processed into `behavior_units_kb/`, inference can be run directly:

```python
"stage_apk_analyze": False,
"stage_gnn_analyze": False,
"stage_kb_build": False,
"stage_prototype": False,
"stage_infer": True,
```

Run:

```bash
python main.py
```

---

## 7. Output Structure

After running the pipeline, the output directory is organized as follows:

```text
out/
├── _gnn/
│   ├── gnn_encoder.pt
│   └── gnn_encoder_meta.json
├── _prototypes/
│   ├── cluster_index.json
│   ├── cluster_build_summary.json
│   ├── assignments.jsonl
│   └── clusters/
│       ├── p_000001.json
│       └── ...
├── _exp1_androserum.csv
└── <year>/
    └── <APK_SHA256>/
        ├── methods.jsonl
        ├── edges.jsonl
        ├── graph.pt
        ├── manifest.json
        ├── meta.json
        ├── behavior_units/
        │   ├── index.json
        │   ├── unit_0000.json
        │   ├── unit_0000.md
        │   └── ...
        ├── behavior_units_kb/
        │   ├── index.json
        │   ├── unit_0000.json
        │   └── ...
        └── infer/
            ├── apk_aligned_bus.json
            ├── apk_local_bu_summaries.json
            ├── apk_local_bu_trace.json
            ├── apk_state_trace.json
            ├── apk_result.json
            └── apk_report.txt
```

Important outputs:

- `methods.jsonl`: method-level smali representations, embeddings, and symbolic features.
- `edges.jsonl`: recovered invoke edges.
- `graph.pt`: recovered function call graph.
- `behavior_units/`: raw Behavior Units and optional markdown exports.
- `behavior_units_kb/`: compact behavior-unit knowledge.
- `_prototypes/cluster_index.json`: behavior prototype index.
- `infer/apk_result.json`: final APK-level inference result.
- `infer/apk_report.txt`: readable report containing BU summaries and the final APK state.
- `_exp1_androserum.csv`: CSV summary of APK-level inference results.

---

## 8. Troubleshooting

### 8.1 PScout Mapping Not Found

Check:

```python
"pscout_allmappings": "/path/to/PScout/results/API_22/allmappings",
```

The path must point to the actual `allmappings` file.

### 8.2 DexBERT Files Not Found

Check that the following files exist:

```text
analyze_tools/weights/model_steps_604364.pt
analyze_tools/vocab.txt
analyze_tools/bert_base.json
```

Also check that `analyze_tools/tokenization.py`, `analyze_tools/models.py`, and `analyze_tools/dataloader.py` are available.

### 8.3 baksmali Failure

Check Java:

```bash
java -version
```

Check baksmali:

```bash
java -jar analyze_tools/baksmali-2.5.2.jar --help
```

If an APK fails during disassembly, check the corresponding baksmali log under the output or temporary directory.

### 8.4 No `graph.pt` Found

The GNN stage requires graph files generated by the APK analysis stage.

Make sure:

```python
"stage_apk_analyze": True,
"build_fcg": True,
```

Then rerun:

```bash
python main.py
```

### 8.5 Missing Prototype Index

The inference stage requires:

```text
out/_prototypes/cluster_index.json
```

If this file is missing, run the prototype construction stage first:

```python
"stage_kb_build": True,
"stage_prototype": True,
"stage_infer": False,
```

### 8.6 Missing LLM API Key

Then make sure the corresponding variable name is configured in `run_config.py`:

```python
"llm_api_key": "",
"llm_api_key_env": "OPENROUTER_API_KEY",
```



---

## 9. Reproducibility Notes

For standard evaluation, prototypes should be built only from training or historical APKs. Future or test APKs should not be included in prototype construction.

A typical evaluation workflow is:

1. Run APK analysis, GNN behavior extraction, KB construction, and prototype construction on training years.
2. Freeze the generated prototype database under `out/_prototypes/`.
3. Run APK analysis, GNN behavior extraction, KB construction, and inference on test years.
4. Disable `stage_prototype` during test-time inference.

This prevents test APKs from being used during prototype construction.

For quick debugging, it is acceptable to run all stages on a single APK. However, this setting should not be used for reporting evaluation results.

---

## 10. Security and Artifact Release Notes

This repository is intended for defensive Android malware analysis research.

Before public release:

- Do not commit APK malware samples unless redistribution is allowed.
- Do not commit API keys, local credentials, or private proxy settings.
- Do not include third-party model weights or tools unless their licenses permit redistribution.
- If APK samples cannot be redistributed, provide SHA-256 hashes and instructions for obtaining them from public threat-intelligence sources.

---


