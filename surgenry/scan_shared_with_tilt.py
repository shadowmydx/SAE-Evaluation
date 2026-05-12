"""
Scan A (code) and D (description) prompts, compute A∩D intersection,
and output per-feature frequencies SEPARATELY for each side.

Rank by code-to-desc tilt ratio (code_freq / desc_freq)
to identify features that are code-skewed within the shared set.

用法:
  python3 surgenry/scan_shared_with_tilt.py -l 31 --topk 200 --output shared_with_tilt.json
"""

import argparse
import json
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import sae_scan


def _prompt_file(name: str) -> str:
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "prompts")
    return os.path.join(d, name)


def _load(path: str) -> list[str]:
    with open(path) as f:
        return json.load(f)


def scan_group(prompts: list[str], layers: list[int], label: str) -> dict[int, list[set[int]]]:
    """Scan prompts and return per-prompt feature sets per layer."""
    prompt_sets: dict[int, list[set[int]]] = {l: [] for l in layers}
    n = len(prompts)
    for i, prompt in enumerate(prompts):
        print(f"  [{i+1}/{n}] scanning {label}...", end="\r")
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
            prompt_sets[layer].append({feat["feature_id"] for feat in entry["top_features"]})
    print()
    return prompt_sets


def main():
    parser = argparse.ArgumentParser(
        description="扫描 code 和 description, 输出 A∩D 各 feature 在每侧的频率和倾斜度"
    )
    parser.add_argument("--prompts-a", default=_prompt_file("code.json"))
    parser.add_argument("--prompts-b", default=_prompt_file("descriptions.json"))
    parser.add_argument("-l", "--layers", type=str, default="31")
    parser.add_argument("--topk", type=int, default=200,
                        help="保留 top-K 最倾斜的 shared features")
    parser.add_argument("--output", type=str, default="shared_with_tilt.json")
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]

    # Load prompts
    prompts_a = _load(args.prompts_a)
    prompts_b = _load(args.prompts_b)
    print(f"A (code): {len(prompts_a)} prompts")
    print(f"B (desc): {len(prompts_b)} prompts")
    print(f"Layers: {layers}")
    print()

    a_prompt_sets = scan_group(prompts_a, layers, "code (A)")
    b_prompt_sets = scan_group(prompts_b, layers, "desc (D)")

    result = {}
    for layer in layers:
        a_union = set().union(*a_prompt_sets[layer]) if a_prompt_sets[layer] else set()
        b_union = set().union(*b_prompt_sets[layer]) if b_prompt_sets[layer] else set()
        shared = a_union & b_union

        print(f"\n── Layer {layer} ──")
        print(f"  |A|={len(a_union)}, |D|={len(b_union)}, |A∩D|={len(shared)}")

        code_freq: dict[int, int] = defaultdict(int)
        desc_freq: dict[int, int] = defaultdict(int)
        n_a = len(a_prompt_sets[layer])
        n_d = len(b_prompt_sets[layer])

        for s in a_prompt_sets[layer]:
            for fid in shared & s:
                code_freq[fid] += 1
        for s in b_prompt_sets[layer]:
            for fid in shared & s:
                desc_freq[fid] += 1

        feature_stats = []
        for fid in shared:
            cf = code_freq.get(fid, 0)
            df = desc_freq.get(fid, 0)
            code_rate = (cf + 0.5) / (n_a + 0.5)
            desc_rate = (df + 0.5) / (n_d + 0.5)
            tilt = code_rate / desc_rate
            feature_stats.append({
                "feature_id": fid,
                "code_count": cf,
                "desc_count": df,
                "code_freq": round(cf / n_a, 3),
                "desc_freq": round(df / n_d, 3),
                "tilt": round(tilt, 3),
            })

        feature_stats.sort(key=lambda x: -x["tilt"])
        top_features = feature_stats[:args.topk]

        result[str(layer)] = {
            "stats": {
                "a_union_size": len(a_union),
                "d_union_size": len(b_union),
                "intersection_size": len(shared),
            },
            "features": top_features,
        }

        print(f"  Top-10 code-tilted features (tilt > 1 = code-skewed):")
        for f in top_features[:10]:
            label = f"code-tilt {f['tilt']:.1f}x" if f['tilt'] > 1.0 else f"desc-tilt {1/f['tilt']:.1f}x"
            print(f"    #{f['feature_id']:>6d}  code={f['code_freq']:.2f}  desc={f['desc_freq']:.2f}  ({label})")

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
