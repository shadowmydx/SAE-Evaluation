"""
对 code prompts 和 description prompts 做全量扫描，计算 A∩D（概念共享特征），
按跨两侧的总出现频率排序。

输出 JSON 格式兼容 steer-negate-shared:
  {str(layer): {"shared": [fid1, fid2, ...]}}

用法:
  python3 surgenry/rank_shared.py
  python3 surgenry/rank_shared.py --prompts-a surgenry/data/prompts/code.json \
                                  --prompts-b surgenry/data/prompts/descriptions.json \
                                  --topk 20 --output ranked_shared.json
"""

import argparse
import json
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import collect_feature_union


def _prompt_file(name: str) -> str:
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "prompts")
    return os.path.join(d, name)


def _load(path: str) -> list[str]:
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="扫描 code 和 description prompt, 输出按频率排序的 A∩D 共享特征"
    )
    parser.add_argument("--prompts-a", default=_prompt_file("code.json"),
                        help="Group A prompt 列表 (e.g. code)")
    parser.add_argument("--prompts-b", default=_prompt_file("descriptions.json"),
                        help="Group B prompt 列表 (e.g. descriptions)")
    parser.add_argument("-l", "--layers", type=str, default="15,24,31")
    parser.add_argument("--topk", type=int, default=50,
                        help="每层保留 top-K 共享特征")
    parser.add_argument("--num-prompts", type=int, default=30, dest="num_prompts",
                        help="每侧扫描的 prompt 数量")
    parser.add_argument("--output", type=str, default="ranked_shared.json")
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    prompts_a = _load(args.prompts_a)[:args.num_prompts]
    prompts_b = _load(args.prompts_b)[:args.num_prompts]

    print(f"A prompts: {args.prompts_a} ({len(prompts_a)}), "
          f"B prompts: {args.prompts_b} ({len(prompts_b)})")
    print(f"Layers: {layers}")
    print()

    # Collect per-prompt feature sets (not just union — need counts)
    print("Scanning group A...")
    a_prompt_sets: dict[int, list[set]] = {l: [] for l in layers}
    for i, prompt in enumerate(prompts_a):
        print(f"  [{i+1}/{len(prompts_a)}] A...", end="\r")
        sys.stdout.flush()
        try:
            from core import sae_scan
            data = sae_scan(prompt, layers)
        except Exception as e:
            print(f"\n  Error: {e}")
            continue
        for ls, entry in data["layers"].items():
            layer = int(ls)
            if layer not in layers:
                continue
            a_prompt_sets[layer].append({feat["feature_id"] for feat in entry["top_features"]})

    print("\nScanning group B...")
    b_prompt_sets: dict[int, list[set]] = {l: [] for l in layers}
    for i, prompt in enumerate(prompts_b):
        print(f"  [{i+1}/{len(prompts_b)}] B...", end="\r")
        sys.stdout.flush()
        try:
            data = sae_scan(prompt, layers)
        except Exception as e:
            print(f"\n  Error: {e}")
            continue
        for ls, entry in data["layers"].items():
            layer = int(ls)
            if layer not in layers:
                continue
            b_prompt_sets[layer].append({feat["feature_id"] for feat in entry["top_features"]})

    # Build union per side
    a_union: dict[int, set[int]] = {}
    b_union: dict[int, set[int]] = {}
    for layer in layers:
        a_union[layer] = set().union(*a_prompt_sets[layer]) if a_prompt_sets[layer] else set()
        b_union[layer] = set().union(*b_prompt_sets[layer]) if b_prompt_sets[layer] else set()

    # Intersection = concept-shared
    shared: dict[int, set[int]] = {}
    for layer in layers:
        shared[layer] = a_union[layer] & b_union[layer]

    # Rank by frequency: count appearances in A + appearances in B
    result = {}
    for layer in layers:
        shared_fids = shared[layer]
        freq: dict[int, int] = defaultdict(int)
        for s in a_prompt_sets[layer]:
            for fid in shared_fids & s:
                freq[fid] += 1
        for s in b_prompt_sets[layer]:
            for fid in shared_fids & s:
                freq[fid] += 1

        sorted_fids = sorted(freq.items(), key=lambda x: -x[1])
        top_fids = [fid for fid, cnt in sorted_fids[:args.topk]]

        result[str(layer)] = {
            "shared": top_fids,
            "stats": {
                "a_union": len(a_union[layer]),
                "b_union": len(b_union[layer]),
                "intersection": len(shared_fids),
                "with_frequencies": [(fid, cnt) for fid, cnt in sorted_fids[:args.topk]],
            }
        }

        print(f"\n── Layer {layer} ──")
        print(f"  |A|={len(a_union[layer])}, |B|={len(b_union[layer])}, |A∩D|={len(shared_fids)}")
        print(f"  Top-{args.topk} shared (by combined frequency):")
        for fid, cnt in sorted_fids[:min(args.topk, 10)]:
            print(f"    #{fid:>6d}  total={cnt}")

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
