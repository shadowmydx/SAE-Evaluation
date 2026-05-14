# 讨论: 标准 Benchmark 验证策略 — GSM8K + HumanEval + BBH

**日期**: 2026-05-15（更新: 新增 BBH）
**背景**: 当前实验结论基于自定义 prompt 集（15-24 条），需要在业界标准 benchmark 上规模化验证。

---

## 一、目的

不在意 correctness（正确性），只在意 degradation（退化）——检查干预后模型输出的推理过程是否出现：
- 空输出 / 截断
- 重复 / stuttering
- 词不达意 / 胡说八道
- 循环 / looping

## 二、Benchmark 选择

| Benchmark | 类型 | 规模 | 加载方式 |
|-----------|------|:----:|---------|
| **GSM8K** | 数学推理 | test: **1319 条** | 本地 arrow 缓存 (`modelscope___gsm8k/`) |
| **HumanEval** | 代码生成 | **164 题** | 本地 gzip JSONL 缓存 (`modelscope/humaneval`) |
| **BBH** | 多领域推理（27 个 one-shot 子任务） | **~6225 条** | HF datasets 本地缓存 (`lukaemon___bbh`) |

**覆盖维度**：

| 维度 | GSM8K | HumanEval | BBH |
|------|:-----:|:---------:|:---:|
| 代码生成 | | ✓ | |
| 数学推理 | ✓ | | ✓ (multistep_arithmetic_two) |
| 逻辑推理 | | | ✓ (boolean, formal_fallacies, etc.) |
| 常识/因果推理 | | | ✓ (causal_judgement, snarks, etc.) |
| 导航/时序 | | | ✓ (navigate, temporal_sequences) |
| 多步结构化输出 | ✓ | ✓ | ✓ |

BBH 补上了**非数学的通用推理**（因果判断、逻辑谬误、讽刺检测等），确保实验结论不局限在数学领域。

## 三、数据源

- **GSM8K**: modelscope 缓存（arrow 文件），无需网络
- **HumanEval**: modelscope 缓存（gzip JSONL），无需网络
- **BBH**: 已通过 HuggingFace 代理下载到本地缓存 (`lukaemon/bbh`)，无需重复下载

BBH 加载方式：
```python
from datasets import load_dataset
for cfg in BBH_CONFIGS:  # 27 config names
    ds = load_dataset('lukaemon/bbh', cfg, split='test')
```

每个子任务有 `input`（问题）和 `target`（答案）两个字段。

## 四、当前脚本

`surgenry/eval_benchmark.py` — 支持所有三个 benchmark：

```bash
# 各取子集测试
python3 surgenry/eval_benchmark.py --gsm8k --both-modes --max-samples 20
python3 surgenry/eval_benchmark.py --humaneval --intervene
python3 surgenry/eval_benchmark.py --bbh --intervene --bbh-per-task 10

# 全部三个一起
python3 surgenry/eval_benchmark.py --all-benchmarks --intervene --max-samples 50
```

**干预配置**：L20 + add_direction α=-20，23 个 code-tilted A∩D 特征（fids 见脚本 `CODE_TILTED_FIDS`）。

## 五、时间估算（实测/估算）

| Config | 单样本耗时 | 全量时间 |
|--------|:---------:|:--------:|
| GSM8K baseline | ~32s | ~11.8h |
| GSM8K intervene | ~38s | ~14.1h |
| HumanEval baseline | ~34s | ~1.6h |
| HumanEval intervene | ~25s | ~1.1h |
| BBH baseline (10/task=270) | ~30s (估) | ~2.3h |
| BBH intervene (10/task=270) | ~30s (估) | ~2.3h |
| BBH intervene (全量 6225) | ~30s (估) | ~52h |

## 六、建议执行方案

### 方案 A：先验效果（推荐，~6h）
```
GSM8K intervene 200 条（~2h）+ HumanEval 全量 intervene（~1h）
+ BBH intervene 10/task（~2h）+ BBH baseline 10/task（~1h）
→ 三个 benchmark 先看干预效果是否一致
```

### 方案 B：夜跑完整干预（~17h）
```
GSM8K intervene 全量 1319 条（~14h）+ HumanEval intervene（~1h）+ BBH 10/task intervene（~2h）
```

## 七、注意事项

- 服务器需先启动、加载模型和 SAE
- 退化检测算法在 `eval_benchmark.py::detect_degradation()` 中，特征包括：stuttering、bigram/trigram 循环、低 diversity、token 聚类、looping 等
- 详细结果保存在 `eval_results_YYYYMMDD_HHMMSS.json`
- 如果干预效果显著（退化率明显高于 baseline 的 few %），可以考虑写到正式实验报告中
- GSM8K 的 baseline 退化率预计在 2-5%（模型本身偶尔会出错/循环），如果干预后到 20-30% 以上就是显著效果
- **BBH 的 diversity 很重要**：27 个任务覆盖不同类型的推理，可以分析哪些类型受干预影响最大，哪些不受影响
