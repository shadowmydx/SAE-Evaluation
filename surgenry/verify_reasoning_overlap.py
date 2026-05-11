"""
推理重叠验证实验（修正版）— 排除概念共享干扰

修正逻辑:
  旧版: C ∩ (A-B) vs C ∩ (B-A)
    问题: B 与 C 不共享领域, 代码特征多于知识特征是必然结果

  新版: C ∩ (A-D) vs C ∩ (A∩D)
    - A = 代码生成, D = 同概念纯描述 (控制概念共享)
    - A-D = 代码独有, 排除概念共享 → "拆解过程特征"
    - A∩D = 代码和描述共有的概念特征 → "语义共享特征"

  关键比较:
    C (代码相关推理) 中 (A-D) 特征数量 → 如果 > 0, 说明推理借用了代码拆解机制
    C' (纯推理) 中 (A-D) 特征数量 → 如果 > 0 但 < C, 说明拆解机制跨领域通用
    D (控制组) 中 (A-D) 特征数量 → 应该接近 0 (D 没做代码也没做推理)

用法:
  python3 surgenry/verify_reasoning_overlap.py
  python3 surgenry/verify_reasoning_overlap.py --layers 15,24,31 --num-prompts 30 -o overlap_report.json
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import collect_feature_union, compute_multi_overlap
from core.client import sae_scan


def _prompt_file(name: str) -> str:
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "prompts")
    return os.path.join(d, name)


def _load_prompts(path: str) -> list[str]:
    with open(path) as f:
        return json.load(f)


def _avg(vals: list[int]) -> float:
    return round(sum(vals) / max(len(vals), 1), 2)


def _report_row(name: str, code_specific: list[int], concept_shared: list[int],
                cs_size: int, sh_size: int, indent: int = 4):
    """Print a formatted row with averages and ratios."""
    prefix = " " * indent
    avg_cs = _avg(code_specific)
    avg_sh = _avg(concept_shared)
    cs_ratio = round(avg_cs / max(cs_size, 1), 4)
    sh_ratio = round(avg_sh / max(sh_size, 1), 4)
    dom = f"{avg_cs / max(avg_sh, 0.01):.2f}x" if avg_sh > 0 else "N/A"
    print(f"{prefix}{name}: avg code-spec={avg_cs} ({cs_ratio:.1%}), "
          f"avg concept={avg_sh} ({sh_ratio:.1%}), dom={dom}")


def main():
    parser = argparse.ArgumentParser(
        description="推理重叠验证 (修正版): 代码拆解特征在推理中的激活, 排除概念共享"
    )
    parser.add_argument("--prompts-code", default=_prompt_file("code.json"))
    parser.add_argument("--prompts-desc", default=_prompt_file("descriptions.json"))
    parser.add_argument("--prompts-reasoning-code", default=_prompt_file("reasoning_code.json"))
    parser.add_argument("--prompts-reasoning-pure", default=_prompt_file("reasoning_pure.json"))
    parser.add_argument("-l", "--layers", type=str, default="15,24,31")
    parser.add_argument("-n", "--num-prompts", type=int, default=30, dest="num_prompts")
    parser.add_argument("-o", "--output", type=str, default="overlap_report.json")
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]

    code_prompts = _load_prompts(args.prompts_code)
    desc_prompts = _load_prompts(args.prompts_desc)
    reasoning_code = _load_prompts(args.prompts_reasoning_code)
    reasoning_pure = _load_prompts(args.prompts_reasoning_pure)

    print("=" * 60)
    print("推理重叠验证实验 (修正版)")
    print("排除概念共享干扰")
    print("=" * 60)
    print(f"Code prompts (A):          {len(code_prompts)}")
    print(f"Description prompts (D):   {len(desc_prompts)}")
    print(f"Reasoning code (C):        {len(reasoning_code)}")
    print(f"Reasoning pure (C'):       {len(reasoning_pure)}")
    print(f"Layers:                    {layers}")
    print()

    # ── Phase 1: Build candidate sets ──
    print("── Phase 1: Collecting feature unions ──")
    A = collect_feature_union(code_prompts, layers, args.num_prompts, "code (A)")
    print()
    D_set = collect_feature_union(desc_prompts, layers, len(desc_prompts), "description (D)")
    print()

    # Compute A-D and A∩D per layer
    A_minus_D: dict[int, set[int]] = {}
    A_cap_D: dict[int, set[int]] = {}
    for layer in layers:
        A_minus_D[layer] = A[layer] - D_set.get(layer, set())
        A_cap_D[layer] = A[layer] & D_set.get(layer, set())

    print("Candidate sets per layer:")
    for layer in layers:
        print(f"  Layer {layer}: |A|={len(A[layer])}, |D|={len(D_set.get(layer, set()))}, "
              f"|A-D|={len(A_minus_D[layer])}, |A∩D|={len(A_cap_D[layer])}")
    print()

    # ── Phase 2: Scan all three target groups ──
    named_sets = {
        "code_specific": A_minus_D,
        "concept_shared": A_cap_D,
    }

    print("── Phase 2a: Scanning code-related reasoning (C) ──")
    result_C = compute_multi_overlap(
        reasoning_code, layers, named_sets,
        prompt_type_label="code-related reasoning (C)",
    )
    print()

    print("── Phase 2b: Scanning pure reasoning (C') ──")
    result_Cprime = compute_multi_overlap(
        reasoning_pure, layers, named_sets,
        prompt_type_label="pure reasoning (C')",
    )
    print()

    print("── Phase 2c: Scanning description control (D control) ──")
    result_D = compute_multi_overlap(
        desc_prompts, layers, named_sets,
        prompt_type_label="description (D control)",
    )
    print()

    # ── REPORT ──
    print("=" * 60)
    print("实验结果报告")
    print("=" * 60)

    summary = {"layers": {}}
    for layer in layers:
        C_data = result_C[layer]
        Cp_data = result_Cprime[layer]
        D_data = result_D[layer]

        cs_size = C_data["set_sizes"]["code_specific"]
        sh_size = C_data["set_sizes"]["concept_shared"]

        C_cs = C_data["per_prompt"]["code_specific"]
        C_sh = C_data["per_prompt"]["concept_shared"]
        Cp_cs = Cp_data["per_prompt"]["code_specific"]
        Cp_sh = Cp_data["per_prompt"]["concept_shared"]
        D_cs = D_data["per_prompt"]["code_specific"]
        D_sh = D_data["per_prompt"]["concept_shared"]

        print(f"\n── Layer {layer} ──")
        print(f"  Candidate: code-specific (A-D)={cs_size}, concept-shared (A∩D)={sh_size}")
        print()

        # Per-prompt for C
        print(f"  Code-related reasoning (C) — {len(C_cs)} prompts:")
        for idx in range(len(C_cs)):
            pm = reasoning_code[idx] if idx < len(reasoning_code) else "?"
            print(f"    [{C_cs[idx]:>2d}cs / {C_sh[idx]:>2d}sh] {pm[:55]}")

        # Aggregates
        print()
        _report_row("C (code-related)", C_cs, C_sh, cs_size, sh_size)
        _report_row("C' (pure)", Cp_cs, Cp_sh, cs_size, sh_size)
        _report_row("D (control)", D_cs, D_sh, cs_size, sh_size)
        print()

        # Key comparisons
        avg_C_cs = _avg(C_cs)
        avg_C_sh = _avg(C_sh)
        avg_Cp_cs = _avg(Cp_cs)
        avg_Cp_sh = _avg(Cp_sh)
        avg_D_cs = _avg(D_cs)

        print(f"  ★ Key metrics:")
        print(f"     C code-spec (A-D):  avg={avg_C_cs} → 推理在代码拆解特征上的激活量")
        print(f"     C' code-spec (A-D): avg={avg_Cp_cs} → 纯推理在代码拆解特征上的激活量")
        print(f"     D code-spec (A-D):  avg={avg_D_cs} → 概念控制组, 应接近0")
        print(f"     C concept (A∩D):    avg={avg_C_sh} → 推理中的概念共享特征")
        print()

        # Conclusion
        conclusion = ""
        if avg_C_cs > avg_C_sh * 1.3:
            conclusion += "C: 代码拆解特征 > 概念共享特征 (支持假说: 推理真的借用了代码拆解)"
        elif avg_C_cs > avg_C_sh:
            conclusion += "C: 代码拆解特征略 > 概念共享特征 (弱支持)"
        else:
            conclusion += "C: 概念共享特征占主导 (不支持假说, 之前结果可能是概念干扰)"

        if avg_D_cs <= 1.0:
            conclusion += " | D控制组: 无代码特征泄漏 (控制有效)"
        else:
            conclusion += f" | D控制组: avg={avg_D_cs} 代码特征泄漏 (可能D中某些概念也涉及代码描述)"

        if avg_Cp_cs > avg_Cp_sh:
            conclusion += " | C': 纯推理中代码特征仍占优 (拆解机制跨领域通用)"
        else:
            conclusion += " | C': 纯推理中概念特征占优 (代码拆解仅在同领域激活)"

        print(f"  → {conclusion}")

        summary["layers"][str(layer)] = {
            "set_sizes": {"code_specific": cs_size, "concept_shared": sh_size},
            "avg_code_specific": {
                "C": avg_C_cs,
                "Cprime": avg_Cp_cs,
                "D": avg_D_cs,
            },
            "avg_concept_shared": {
                "C": avg_C_sh,
                "Cprime": avg_Cp_sh,
            },
            "conclusion": conclusion,
        }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to {args.output}")

    print("\n" + "=" * 60)
    print("验证完成。若代码拆解特征在 C 中显著激活且控制组有效,")
    print("建议进行干预验证确认因果:")
    print("  python3 surgenry/steer_generation.py steer-negate ...")
    print("=" * 60)


if __name__ == "__main__":
    main()
