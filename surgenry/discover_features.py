"""
Step 1: 差异特征发现
====================
对给定两个 prompt A (France) 和 B (China)，提取指定层的 SAE 特征，
计算 A-B 和 B-A 的差异集合。

用法:
  python3 surgenry/discover_features.py [--layer 24,28,31]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen3_client import _r


def discover(prompt_a: str, prompt_b: str, layers: list[int]) -> dict:
    """返回 {layer: {"a_only": [fid,...], "b_only": [fid,...], "shared": [fid,...]}}"""
    body_a = {"prompt": prompt_a, "layers": layers, "token_position": -1, "max_features": 100}
    body_b = {"prompt": prompt_b, "layers": layers, "token_position": -1, "max_features": 100}

    ra = _r("post", "/sae", json=body_a).json()
    rb = _r("post", "/sae", json=body_b).json()

    result = {}
    for layer in layers:
        ls = str(layer)
        a_ids = {feat["feature_id"] for feat in ra["layers"][ls]["top_features"]}
        b_ids = {feat["feature_id"] for feat in rb["layers"][ls]["top_features"]}
        result[layer] = {
            "a_only": sorted(a_ids - b_ids),
            "b_only": sorted(b_ids - a_ids),
            "shared": sorted(a_ids & b_ids),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="差异特征发现: 两组 prompt 的 SAE 特征差异")
    parser.add_argument("--prompt-a", type=str, default="The capital of France is",
                        help="Prompt A (默认: France)")
    parser.add_argument("--prompt-b", type=str, default="The capital of China is",
                        help="Prompt B (默认: China)")
    parser.add_argument("-l", "--layers", type=str, default="24,28,31",
                        help="分析层号逗号分隔 (默认深层 24,28,31)")
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    print(f"Prompt A: \"{args.prompt_a}\"")
    print(f"Prompt B: \"{args.prompt_b}\"")
    print(f"Layers  : {layers}")
    print()

    result = discover(args.prompt_a, args.prompt_b, layers)

    for layer, sets in result.items():
        print(f"── Layer {layer} ──")
        print(f"   A only (France 独有): {len(sets['a_only'])} features")
        print(f"   B only (China 独有) : {len(sets['b_only'])} features")
        print(f"   Shared               : {len(sets['shared'])} features")
        print(f"   A-only top-10: {sets['a_only'][:10]}")
        print(f"   B-only top-10: {sets['b_only'][:10]}")
        print()

    # Save to JSON for next step
    import json
    out_path = "discovered_features.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
