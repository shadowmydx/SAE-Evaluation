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
│   └── workflows.py     — Business logic (discover, rank, overlap, intervention assembly)
├── data/prompts/        — External prompt lists (JSON arrays, gitignored)
│   ├── france.json      — 50 France-related prompts
│   ├── china.json       — 50 China-related prompts
│   ├── code.json                 — 30 code generation prompts (A)
│   ├── knowledge.json            — 30 knowledge recall prompts (B)
│   ├── descriptions.json         — 15 concept descriptions (D, control set)
│   ├── reasoning_code.json       — 15 code-related reasoning prompts (C, easy)
│   ├── reasoning_pure.json       — 15 pure reasoning prompts (C', easy)
│   ├── hard_reasoning_code.json  — 46 random-param code-related reasoning (hard)
│   ├── hard_reasoning_pure.json  — 24 random-param pure reasoning (hard)
│   ├── procedural.json           — 11 non-code procedural tasks (E, baking/assembly/etc.)
│   └── generate_hard_prompts.py  — Generator for hard reasoning prompts
├── discover_features.py         — Step 1: find features specific to prompt A vs B
├── rank_features.py             — Step 2: rank candidate features by frequency
├── rank_shared.py               — Rank A∩D (concept-shared) features by cross-group frequency
├── steer_generation.py          — Step 3: intervene (negate/inject) to steer
├── sae_analysis.py              — Quick SAE feature viewer for a single prompt
├── sae_intervene.py             — Quick single-feature intervention tool
├── scan_shared_with_tilt.py     — Scan A∩D, output per-feature code/desc frequency and tilt ratio
├── trace_features.py            — Token-level feature activation tracing (uses /sae_trace)
├── eval_benchmark.py            — GSM8K + HumanEval benchmark eval with degradation detection
└── verify_reasoning_overlap.py  — Reasoning overlap experiment (uses A-D, A∩D)
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
| `/sae_trace` | Trace specified feature activations across ALL token positions |
| `/sae_intervene` | Generate with SAE feature interventions |
| `/health` (GET) | Check server status |

## Key Technical Details

- **Chat template**: All generation requests must apply `tokenizer.apply_chat_template()` with `add_generation_prompt=True` (already handled by server)
- **SAE mechanism**: Uses `torch.topk(k=100)` to get sparse activations from 65536-dim space; interventions applied via `register_forward_hook` on decoder layers
- **Intervention types**: `zero`, `scale`, `set`, `clamp_max`, `negate`, `add_direction`
- **`add_direction`**: New action (2026-05-12). Encodes residual to confirm feature is in top-100, then directly adds/subtracts `W_dec[:, fid]` direction vector in residual space: `h' = reconstructed + α × W_dec[:, fid]`. Avoids decode loss, α is arbitrary. Only intervenes if feature is actually active.
  ```python
  {"layer": 20, "feature_id": 34612, "action": "add_direction", "value": -20.0}  # suppress
  {"layer": 20, "feature_id": 34612, "action": "add_direction", "value": 30.0}   # reinforce
  ```
- **Layer selection is critical**: Do NOT default to Layer 31. Use Logits Lens to identify decision burst points. For code/reasoning tasks, Layer 20 (structure decision) is often more effective than Layer 31 (stylistic refinement).
- **SAE weights are lazy-loaded** into `sae_cache` dict on first use
- **W_dec shape is (4096, 65536)** — index as `W_dec[:, fid]`, NOT `W_dec[fid]`.

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

## Reasoning Overlap Experiment (Code vs Knowledge) — 修正版

验证假说：排除概念共享干扰后，代码拆解特征是否仍在推理中激活。

实验结论：**"代码拆解为推理提供 token 桥梁"假说被否决。** 具体证据见 `reports/2026-05-11_推理重叠验证.md`。

### 一键运行（相关性分析）

```bash
python3 surgenry/verify_reasoning_overlap.py
python3 surgenry/verify_reasoning_overlap.py --layers 15,24,31 -n 30 -o overlap_report.json
```

### 因果干预验证

```bash
# A-D: 代码独有特征 → 预期: 无影响（假设被否决）
python3 surgenry/discover_features.py \
  --prompt-a "Write a function to compute the nth Fibonacci number" \
  --prompt-b "The Fibonacci sequence is a series where each number is the sum of the two preceding ones" \
  -l 31 --output discovered_code_vs_desc.json

python3 surgenry/rank_features.py \
  --input discovered_code_vs_desc.json \
  --name code_vs_desc --group-a "code-specific" --group-b "description-specific" \
  --prompts-a surgenry/data/prompts/code.json \
  --prompts-b surgenry/data/prompts/descriptions.json \
  --num-prompts 30 --topk 10 --output ranked_code_vs_desc.json

python3 surgenry/steer_generation.py steer-negate \
  -i ranked_code_vs_desc.json -l 31 --side a --top 87 \
  --prompt "..." -m 100 -t 0.3

# A∩D: 通用语言特征 → 预期: 全面崩溃（控制组）
python3 surgenry/rank_shared.py -l 31 --topk 10 --output ranked_shared.json
python3 surgenry/steer_generation.py steer-negate-shared \
  -i ranked_shared.json -l 31 --top 10 --prompt "..." -m 100 -t 0.3
```

## Code-Tilted A∩D Causal Experiment (2026-05-12)

Key insight: **Layer 31 is too late for intervention.** Logits Lens shows structure decisions happen at L20 (e.g., "using" 18%→80%), not L31 (stylistic refinement).

### Experiment Flow

```bash
# Phase 1: Scan A∩D with per-side frequency, rank by code-tilt
python3 surgenry/scan_shared_with_tilt.py -l 20 --topk 200 --output shared_with_tilt.json

# Phase 2: Intervene with add_direction at decision layer
# α=-20 is the selective dose (code ok, most reasoning broken)
python3 -c "
from core import intervene
# Build interventions from shared_with_tilt.json (tilt>=2.0 features)
interventions = [{'layer': 20, 'feature_id': fid, 'action': 'add_direction', 'value': -20.0} ...]
"
```

### Key Findings

| Finding | Detail |
|---------|--------|
| **Layer specificity** | Only L20 works, not L26 or L31 |
| **Specificity control** | Random features at α=-100 on L20 → no effect |
| **Selective dose** | α=-20: code intact, reasoning broken |
| **Complex code affected** | Quicksort crashes, Fibonacci/Palindrome survive |
| **Reasoning affected** | Stack, Insertion Sort, Water Pouring all crash |
| **Non-monotonic recovery** | α=-30 shows partial recovery on simple tasks (bypass paths?) |
| **Report** | `reports/2026-05-12_特征分离与推理因果实验.md` |

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

## Experiment Report Guidelines

每个实验报告（`reports/YYYY-MM-DD_*.md`）必须包含以下内容：

1. **标题**: 日期 + 实验核心内容
2. **运行日期 & 模型信息**
3. **背景与动机**: 研究问题、前置实验结论
4. **实验方法**:
   - 候选特征集的构建方式（seed prompt、层、筛选条件）
   - 控制组设计（随机特征、D-A、baseline 等）
   - 剂量滴定过程
5. **实验结果**: 表格形式呈现，标注 ✓/✗ 及崩溃模式
6. **分析讨论**: 解释实验现象，提出假说
7. **结论**: 对研究问题的判决
8. **复现命令**: 可以直接执行的 shell 命令（含完整参数），确保其他研究者能复现全部结果

### 编码实验脚本规范

- 实验脚本放在 `surgenry/` 目录，以 `test_*.py` 命名
- 脚本必须包含清晰的注释说明任务分组和判断标准
- 自动分类输出为 PASS/CRASH/LOOP 并统计通过率

## Experiment Reports

| File | Date | Content |
|------|------|---------|
| `reports/2026-05-11_推理重叠验证.md` | 2026-05-11 | Original reasoning overlap (A-D negate, no effect) |
| `reports/2026-05-11_推理重叠验证结论.md` | 2026-05-11 | Summary conclusion of 05-11 experiments |
| `reports/2026-05-12_特征分离与推理因果实验.md` | 2026-05-12 | Code-tilted A∩D causal experiment (breakthrough) |
| `reports/2026-05-13_AD重检验.md` | 2026-05-13 | A-D re-evaluation at L20 + add_direction; Pure reasoning extension |
| `reports/2026-05-14_特征激活模式分析.md` | 2026-05-14 | Code-tilted A∩D per-token activation patterns (/sae_trace) |
| `reports/2026-05-15_正向干预与剂量对称性.md` | 2026-05-15 | Positive intervention α>0, dose symmetry (simple vs complex tasks) |
| `reports/discuss/2026-05-15_标准Benchmark验证策略.md` | 2026-05-15 | GSM8K + HumanEval benchmark verification strategy |
| `reports/discuss/` | — | Research direction discussions and analysis notes |

## Benchmark Evaluation (2026-05-15)

Script: `surgenry/eval_benchmark.py` — runs GSM8K (1319 test), HumanEval (164 problems), and BBH (27 tasks, one-shot) with/without intervention.

**Goal**: NOT correctness checking. Detect **degradation** (empty output, stuttering, bigram/trigram repetition, low diversity, looping). Uses `detect_degradation()` with multi-metric thresholds.

**Default intervention**: L20 + add_direction α=-20, 23 code-tilted A∩D features.

```bash
# Specific benchmarks
python3 surgenry/eval_benchmark.py --gsm8k --both-modes --max-samples 20
python3 surgenry/eval_benchmark.py --humaneval --intervene
python3 surgenry/eval_benchmark.py --bbh --intervene                      # BBH (10/task default)
python3 surgenry/eval_benchmark.py --bbh --intervene --bbh-per-task 25    # BBH more samples

# Default: all 3 benchmarks, both modes
python3 surgenry/eval_benchmark.py --max-samples 50
```

### Time Estimates (measured)

| Config | Per sample | Full run (all 3) |
|--------|:---------:|:----------------:|
| GSM8K baseline | ~32s | ~11.8h |
| GSM8K intervene | ~38s | ~14.1h |
| HumanEval baseline | ~34s | ~1.6h |
| HumanEval intervene | ~25s | ~1.1h |
| BBH (10/task = 270) | ~30s (est.) | ~2.3h |
| BBH (all = 6225) | ~30s (est.) | ~52h |

Strategy doc: `reports/discuss/2026-05-15_标准Benchmark验证策略.md`

## Known Issues

- Server must be restarted if `qwen3_server.py` is modified (changes don't hot-reload)
- `/sae_intervene` requests use `torch.no_grad()` — the SAE encoder/decoder is not trained during intervention
- rank_features.py outputs JSON to CWD (project root), not the surgenry/ directory
