# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Qwen3-8B model server with SAE (Sparse Autoencoder) interpretability research. Provides HTTP endpoints for generation, hidden state extraction, logits lens, SAE feature analysis, and SAE feature intervention.

## Quick Start

```bash
# Activate venv (alias: "llm")
source venv/bin/activate

# 1. Start server
python3 qwen3_server.py

# 2. Load model
python3 qwen3_client.py load /home/shadowmydx/.cache/modelscope/hub/models/Qwen/Qwen3-8B

# 3. Set SAE directory
python3 qwen3_client.py sae_set_dir /home/shadowmydx/.cache/modelscope/hub/models/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100

# 4. Generate
python3 qwen3_client.py generate "how to make a cake"
```

## Architecture

```
qwen3_server.py          — FastAPI server (port 8000), model lives in GPU memory
qwen3_client.py          — CLI tool, uses requests to talk to server
test_qwen3.py            — Integration test (starts server, runs all endpoints)
qwen3_demo.py            — Standalone demo (loads model directly, no server needed)

surgenry/                — SAE interpretability experiments
├── core/                — Reusable library (data+function decoupled)
│   ├── models.py        — Typed dataclasses (ExperimentConfig, RankedFeatures, etc.)
│   ├── client.py        — Server interaction layer (sae_scan, generate, intervene)
│   └── workflows.py     — Business logic (discover, rank, intervention assembly)
├── data/prompts/        — External prompt lists (JSON arrays)
│   ├── france.json
│   └── china.json
├── discover_features.py — Step 1: find features specific to prompt A vs B
├── rank_features.py     — Step 2: rank candidate features by frequency across 50 prompts
├── steer_generation.py  — Step 3: intervene (negate/inject) to steer generation
├── sae_analysis.py      — Quick SAE feature viewer for a single prompt
└── sae_intervene.py     — Quick single-feature intervention tool
```

## Key Paths

- **Model**: `/home/shadowmydx/.cache/modelscope/hub/models/Qwen/Qwen3-8B`
- **SAE**: `/home/shadowmydx/.cache/modelscope/hub/models/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100`
- **SAE format**: `layer{N}.sae.pt` with keys: W_enc(65536,4096), W_dec(4096,65536), b_enc(65536), b_dec(4096)

## Server Endpoints

| POST path | Purpose |
|-----------|---------|
| `/load` | Load model from directory |
| `/generate` | Text generation (supports SSE streaming) |
| `/hidden_states` | Extract residual stream at specified layers |
| `/logits_lens` | Project hidden states through LM head |
| `/sae_set_dir` | Set SAE directory |
| `/sae` | Run SAE to extract top activating features |
| `/sae_intervene` | Generate with SAE feature interventions |
| `/health` (GET) | Check server status |

## Key Technical Details

- **Chat template**: All generation requests must apply `tokenizer.apply_chat_template()` with `add_generation_prompt=True` (already handled by server)
- **SAE mechanism**: Uses `torch.topk(k=100)` to get sparse activations from 65536-dim space; interventions applied via `register_forward_hook` on decoder layers
- **Intervention types**: `zero`, `scale`, `set`, `clamp_max`, `negate`
- **Layer 31** (last hidden layer) is the most effective intervention target
- **SAE weights are lazy-loaded** into `sae_cache` dict on first use

## Running New Experiments

To run a custom experiment (not France vs China), provide prompt data files and use CLI args:

```bash
# Step 1: Discover differential features
python3 surgenry/discover_features.py \
  --prompt-a "prompt for group A" --prompt-b "prompt for group B" \
  -l 24,28,31 --output discovered_myexp.json

# Step 2: Rank by frequency
python3 surgenry/rank_features.py \
  --input discovered_myexp.json \
  --name my_experiment --group-a "label_a" --group-b "label_b" \
  --prompts-a surgenry/data/prompts/group_a.json \
  --prompts-b surgenry/data/prompts/group_b.json \
  --topk 10 --output ranked_myexp.json

# Step 3: Steer interventions
python3 surgenry/steer_generation.py steer-negate \
  -i ranked_myexp.json -l 31 --side a --top 3 --prompt "..."
python3 surgenry/steer_generation.py inject \
  -i ranked_myexp.json -l 31 --side b --top 3 --prompt "..."
```

## Surgenry Experiment Flow (France vs China)

```bash
# Step 1: Discover differential features
python3 surgenry/discover_features.py

# Step 2: Rank by cross-prompt frequency (50 prompts each side, ~300 SAE calls)
python3 surgenry/rank_features.py

# Step 3: Steer interventions
python3 surgenry/steer_generation.py baseline --prompt "The capital of France is"
python3 surgenry/steer_generation.py steer-negate --prompt "..." -i ranked_features.json -l 31 --side a --top 3
python3 surgenry/steer_generation.py inject --prompt "..." -i ranked_features.json -l 31 --side b --top 3
python3 surgenry/steer_generation.py compare --prompt "..." -i ranked_features.json
```

## Running Tests

```bash
python3 test_qwen3.py /home/shadowmydx/.cache/modelscope/hub/models/Qwen/Qwen3-8B
# With SAE:
SAE_DIR=/home/shadowmydx/.cache/modelscope/hub/models/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100 \
  python3 test_qwen3.py /home/shadowmydx/.cache/modelscope/hub/models/Qwen/Qwen3-8B
```

## Known Issues

- Server must be restarted if `qwen3_server.py` is modified (changes don't hot-reload)
- `/sae_intervene` requests use `torch.no_grad()` — the SAE encoder/decoder is not trained during intervention
- rank_features.py outputs JSON to CWD (project root), not the surgenry/ directory
