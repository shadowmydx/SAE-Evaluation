"""
Step 1: 差异特征发现 — 薄 CLI 壳

用法:
  python3 surgenry/discover_features.py [--layer 24,28,31] [--output discovered_features.json]
  python3 surgenry/discover_features.py --prompt-a "..." --prompt-b "..." -l 31
"""

import argparse
import json
from core import ExperimentConfig, discover_differential


def main():
    parser = argparse.ArgumentParser(description="差异特征发现: 两组 prompt 的 SAE 特征差异")
    parser.add_argument("--prompt-a", type=str, default="The capital of France is",
                        help="Prompt A (默认)")
    parser.add_argument("--prompt-b", type=str, default="The capital of China is",
                        help="Prompt B (默认)")
    parser.add_argument("-l", "--layers", type=str, default="24,28,31",
                        help="分析层号逗号分隔 (默认 24,28,31)")
    parser.add_argument("--output", type=str, default="discovered_features.json",
                        help="输出 JSON 路径")
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    config = ExperimentConfig(
        name="cli_discover",
        group_a="group_a",
        group_b="group_b",
        seed_prompt_a=args.prompt_a,
        seed_prompt_b=args.prompt_b,
        prompts_file_a="",
        prompts_file_b="",
        layers=layers,
    )

    result = discover_differential(config)

    print(f"Prompt A: \"{args.prompt_a}\"")
    print(f"Prompt B: \"{args.prompt_b}\"")
    print(f"Layers  : {layers}")
    print()

    for layer, df in result.items():
        print(f"── Layer {layer} ──")
        print(f"   A only: {len(df.a_only)} features")
        print(f"   B only: {len(df.b_only)} features")
        print(f"   Shared: {len(df.shared)} features")
        print(f"   A-only top-10: {df.a_only[:10]}")
        print(f"   B-only top-10: {df.b_only[:10]}")
        print()

    # Serialize and save
    out = {str(l): {"a_only": df.a_only, "b_only": df.b_only, "shared": df.shared}
           for l, df in result.items()}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
