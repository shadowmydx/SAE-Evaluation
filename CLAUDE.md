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
├── data/
│   ├── prompts/            — External prompt lists (JSON arrays, gitignored)
│   │   ├── france.json      — 50 France-related prompts
│   │   ├── china.json       — 50 China-related prompts
│   │   ├── code.json                 — 30 code generation prompts (A)
│   │   ├── knowledge.json            — 30 knowledge recall prompts (B)
│   │   ├── descriptions.json         — 15 concept descriptions (D, control set)
│   │   ├── reasoning_code.json       — 15 code-related reasoning prompts (C, easy)
│   │   ├── reasoning_pure.json       — 15 pure reasoning prompts (C', easy)
│   │   ├── hard_reasoning_code.json  — 46 random-param code-related reasoning (hard)
│   │   ├── hard_reasoning_pure.json  — 24 random-param pure reasoning (hard)
│   │   ├── procedural.json           — 11 non-code procedural tasks (E, baking/assembly/etc.)
│   │   ├── hotpotqa.json             — 30 multi-hop QA questions (HotpotQA-style)
│   │   └── generate_hard_prompts.py  — Generator for hard reasoning prompts
│   └── evaluation/         — Raw experiment outputs (originals, by date)
│       └── YYYY-MM-DD/     — Per-day subdirectories with raw console output + JSON data
├── discover_features.py         — Step 1: find features specific to prompt A vs B
├── rank_features.py             — Step 2: rank candidate features by frequency
├── rank_shared.py               — Rank A∩D (concept-shared) features by cross-group frequency
├── steer_generation.py          — Step 3: intervene (negate/inject) to steer
├── sae_analysis.py              — Quick SAE feature viewer for a single prompt
├── sae_intervene.py             — Quick single-feature intervention tool
├── scan_shared_with_tilt.py     — Scan A∩D, output per-feature code/desc frequency and tilt ratio
├── trace_features.py            — Token-level feature activation tracing (uses /sae_trace)
├── dose_titration.py            — Dose titration (α=-1 to -20 or α=+1 to +20, step=1)
├── eval_benchmark.py            — GSM8K + HumanEval benchmark eval with degradation detection
├── verify_reasoning_overlap.py  — Reasoning overlap experiment (uses A-D, A∩D)
├── layer_ablation.py            — Layer ablation: test each layer's own code-tilted features
├── clean_ablation.py            — Clean ablation: A-only∪D-only vs code-tilted A∩D (full scan)
├── activated_ablation.py        — Ablation from top-200 A∪D pool (early version)
└── test_pure_reasoning.py       — Pure reasoning verification (39 prompts, non-code)
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

## Key Technical Details — Methodological Lessons (from 6 days of experiments)

### Intervention methods (ranked by power)
- **`add_direction` — PREFERRED**: Directly adds/subtracts `W_dec[:, fid]` in residual space. No reconstruction loss, α is arbitrary. Only intervenes if feature is active.
  ```python
  {"layer": 20, "feature_id": 34612, "action": "add_direction", "value": -20.0}  # suppress
  {"layer": 20, "feature_id": 34612, "action": "add_direction", "value": 30.0}   # reinforce
  ```
- **`negate` — WEAK, DEPRECATED**: Requires full SAE encode-decode cycle, reconstruction loss, signal limited by original activation magnitude. L31 + negate produced false negatives (05-11).

### Critical methodological findings

1. **Layer selection must use Logits Lens**: L31 (last layer) has all decisions complete. L20 (structure decision burst point) is correct for code/reasoning tasks.
2. **Tilt ratio > set difference**: `code_freq/desc_freq` (frequency ratio) identifies causally important features. Boolean set difference (A-D) finds correlation, not causation.
3. **Ablation controls must sample from activated space**: Sampling from all 65536 FIDs (mostly dead neurons) is too loose. Use **A-only∪D-only** (full scan of both prompt groups, exclude ALL A∩D) for rigorous specificity testing.
4. **Dose step=1 required**: Coarse sampling (±20, ±50) misses non-monotonicity entirely.
5. **α=0 baseline ≠ no-intervention baseline**: SAE hook changes computation even with zero vector (reproduced twice).
6. **Effect strength ≠ specificity**: L15 is stronger but non-specific (random features crash too). Distinguish by **crash pattern** (selective vs uniform).
- **SAE weights are lazy-loaded** into `sae_cache` dict on first use
- **W_dec shape is (4096, 65536)** — index as `W_dec[:, fid]`, NOT `W_dec[fid]`.

## Core Experimental Pipeline (最终验证版)

### Purpose
Verify causal relationship between code-tilted A∩D features and reasoning. The pipeline produces a reliable feature set (23 features, L20, tilt≥2.0) with the following property: **suppressing them selectively breaks multi-step reasoning but spares short-chain inference**.

### Standard Verification Flow

```bash
# Step 1: Scan A∩D with per-side frequency, rank by tilt
# This replaces the old set-difference approach (A-D) which produced false negatives
python3 surgenry/scan_shared_with_tilt.py -l 20 --topk 200 --output shared_with_tilt.json

# Step 2: Intervene with add_direction at the correct layer
# α=-20 is the verified selective dose
python3 -c "
import json
from surgenry.core.client import intervene
with open('shared_with_tilt.json') as f:
    data = json.load(f)
# 23 code-tilted A∩D features (tilt>=2.0)
fids = [f['feature_id'] for f in data['20']['features'] if f['tilt'] >= 2.0]
iv = [{'layer':20,'feature_id':fid,'action':'add_direction','value':-20.0} for fid in fids]
result = intervene('Simulate a stack. Push 1, Push 2, Pop...', iv, 300, 0.3)
print(result)
"

# Dose titration (negative): α=-1 to -20, step=1
python3 surgenry/dose_titration.py

# Dose titration (positive): α=+1 to +20, step=1
python3 surgenry/dose_titration.py --positive
```

### Essential Controls (required for publication)

```bash
# 1. Layer ablation — test with EACH LAYER's OWN features
python3 surgenry/scan_shared_with_tilt.py -l 15,20,24,26,31 --topk 200 --output layer_ablation_features.json
python3 surgenry/layer_ablation.py

# 2. Clean ablation — A-only∪D-only vs code-tilted A∩D (full scan, exclude ALL A∩D)
# This distinguishes "layer-specific" from "feature-class-specific"
python3 surgenry/clean_ablation.py

# 3. Dose titration (required because coarse ±20 misses non-monotonicity)
python3 surgenry/dose_titration.py
python3 surgenry/dose_titration.py --positive
```

### Core Finding

L20 code-tilted A∩D features (23 features, tilt ≥ 2.0) encode a **"structured output orchestration"** signal. Suppressing them selectively breaks multi-step reasoning while preserving short-chain inference (Algebra). This is verified through:

1. **Dose titration (bidirectional)**: α=-1 to -20 (negative) + α=+1 to +20 (positive), step=1
2. **Layer ablation (L15/L20/L24/L26/L31)**: each layer with its own features
3. **Clean ablation (L20)**: code-tilted A∩D vs random A-only∪D-only → **distinguishable** (code-tilted is task-selective, random is uniform mild interference)
4. **Clean ablation (L15)**: code-tilted A∩D vs random A-only∪D-only → **indistinguishable** (both crash) → L15 = generic language foundation
5. **Random feature control (L20, α=-100)**: no effect from add_direction itself
6. **Pure reasoning extension (39 non-code prompts)**: 60-67% crash → features are NOT code-specific

### Key Findings

| Finding | Detail |
|---------|--------|
| **Layer specificity** | Only L20 works selectively. L15 also crashes but non-specifically (random features crash too). L24+ minimal effect. |
| **Specificity control** | L20 random A-only∪D-only (23 features, α=-20) → only mild uniform UNCLOSED, NOT selective crash |
| **Selective dose** | α=-20: code intact, reasoning broken |
| **Non-monotonicity (bidirectional)** | Both negative and positive show oscillation (crash→recover→crash at adjacent α) |
| **Algebra immune across 40 doses** | (α=-1 to -20 + α=+1 to +20) — shortest-chain deterministic reasoning |
| **Task demand spectrum** | Negative sensitivity: Water > Clock > Capital > Fibonacci > Algebra (immune). Positive sensitivity: Clock > Capital > Fibonacci > Algebra > Water (most robust) — **complete reversal** |
| **Crash level asymmetry** | Negative → word-level loops ("capital of France is capital of France"). Positive → character-level stutter ("fffff...", "\| \| \| \|") |
| **α=0 ≠ baseline** | Repeatedly reproduced. SAE hook itself changes computation even with zero vector. |
| **Report** | `reports/2026-05-16_综合现象解释与理论框架.md` |

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

## Data Retention

- **Experiment outputs** (raw console output) → `surgenry/data/evaluation/YYYY-MM-DD/`
- **Prompt files** → `surgenry/data/prompts/`
- **Feature scan data** (JSON) → project root or `surgenry/data/evaluation/`
- **Reports** → `reports/YYYY-MM-DD_*.md`

When running new experiments, save raw console output to the evaluation directory:
```bash
python3 surgenry/experiment_script.py 2>&1 | tee surgenry/data/evaluation/$(date +%F)/experiment_name_output.txt
# Or copy later:
cp /path/to/background/output surgenry/data/evaluation/2026-05-16/experiment_name_output.txt
```

## Experiment Reports

| File | Date | Content |
|------|------|---------|
| `reports/2026-05-11_推理重叠验证.md` | 2026-05-11 | Original reasoning overlap (A-D negate, no effect — method flawed) |
| `reports/2026-05-11_推理重叠验证结论.md` | 2026-05-11 | Summary conclusion of 05-11 experiments |
| `reports/2026-05-12_特征分离与推理因果实验.md` | 2026-05-12 | Code-tilted A∩D breakthrough (L20, add_direction, α=-20) |
| `reports/2026-05-13_AD重检验.md` | 2026-05-13 | A-D re-evaluation at L20 + add_direction; Pure reasoning extension |
| `reports/2026-05-14_特征激活模式分析.md` | 2026-05-14 | Code-tilted A∩D per-token activation patterns (/sae_trace) |
| `reports/2026-05-15_正向干预与剂量对称性.md` | 2026-05-15 | Positive intervention α=+20/+50, dose symmetry |
| `reports/2026-05-16_负向干预精确剂量滴定.md` | 2026-05-16 | Negative dose titration (α=-1 to -20, step=1) |
| `reports/2026-05-16_正向干预精确剂量滴定.md` | 2026-05-16 | Positive dose titration (α=+1 to +20, step=1) |
| `reports/2026-05-16_层消融实验.md` | 2026-05-16 | Layer ablation L15-L31 + clean ablation (A-only∪D-only control) |
| `reports/2026-05-16_综合现象解释与理论框架.md` | 2026-05-16 | **Comprehensive synthesis: all phenomena + unified framework** |
| `reports/2026-05-16_HotpotQA多跳推理验证.md` | 2026-05-16 | Multi-hop QA (30 questions, 97%→7% accuracy) + activation comparison |
| `reports/discuss/2026-05-14_共享子空间框架.md` | 2026-05-14 | Shared subspace theoretical framework |
| `reports/discuss/2026-05-15_标准Benchmark验证策略.md` | 2026-05-15 | GSM8K + HumanEval benchmark strategy |
| `reports/discuss/2026-05-16_正负干预对比分析.md` | 2026-05-16 | Positive vs negative intervention comparison |

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
- Raw experiment outputs are stored in `surgenry/data/evaluation/YYYY-MM-DD/` (gitignored). Copy important outputs there before the temp dirs are cleaned up.
