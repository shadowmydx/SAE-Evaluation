"""
Step 2: 跨 prompt 频率排序 — 薄 CLI 壳

基于差异特征发现的结果 (discovered_features.json)，
对每个候选特征在 prompt 文件中统计出现频率，筛选 top-K。

用法:
  python3 surgenry/rank_features.py [--prompts-a surgenry/data/prompts/france.json]
                                    [--prompts-b surgenry/data/prompts/china.json]
                                    [--topk 10] [--num-prompts 50]
"""

import argparse
import json
from core import ExperimentConfig, DiscoveredFeatures, rank_by_frequency


def load_discovered(path: str) -> dict[int, DiscoveredFeatures]:
    with open(path) as f:
        raw = json.load(f)
    return {int(k): DiscoveredFeatures.from_dict(int(k), v) for k, v in raw.items()}


def main():
    parser = argparse.ArgumentParser(description="跨 prompt 频率排序: 筛选高频率差异特征")
    parser.add_argument("--input", type=str, default="discovered_features.json",
                        help="Step 1 的输出 JSON")
    parser.add_argument("--name", type=str, default="cli_rank",
                        help="实验名称")
    parser.add_argument("--group-a", type=str, default="group_a",
                        help="Group A 标签")
    parser.add_argument("--group-b", type=str, default="group_b",
                        help="Group B 标签")
    parser.add_argument("--prompt-a", type=str, default="",
                        help="seed prompt A (用于 info)")
    parser.add_argument("--prompt-b", type=str, default="",
                        help="seed prompt B (用于 info)")
    parser.add_argument("--prompts-a", type=str,
                        default="surgenry/data/prompts/france.json",
                        help="Group A prompt 列表 JSON")
    parser.add_argument("--prompts-b", type=str,
                        default="surgenry/data/prompts/china.json",
                        help="Group B prompt 列表 JSON")
    parser.add_argument("--topk", type=int, default=10,
                        help="每层取 top-K 最高频特征")
    parser.add_argument("--num-prompts", type=int, default=50, dest="num_prompts",
                        help="用于排序的 prompt 数量")
    parser.add_argument("--output", type=str, default="ranked_features.json",
                        help="输出路径")
    args = parser.parse_args()

    discovered = load_discovered(args.input)
    layers = sorted(discovered.keys())

    config = ExperimentConfig(
        name=args.name,
        group_a=args.group_a,
        group_b=args.group_b,
        seed_prompt_a=args.prompt_a,
        seed_prompt_b=args.prompt_b,
        prompts_file_a=args.prompts_a,
        prompts_file_b=args.prompts_b,
        layers=layers,
    )

    print(f"Group A label: {args.group_a}")
    print(f"Group B label: {args.group_b}")
    print(f"Scanning {args.num_prompts} prompts from {args.prompts_a} for set A...")
    ranked = rank_by_frequency(config, discovered, num_prompts=args.num_prompts, topk=args.topk)

    ranked.save(args.output)

    print(f"\nSaved to {args.output}")
    for layer in layers:
        print(f"\n── Layer {layer} ──")
        top_a = ranked.set_a.top(layer, args.topk)
        top_b = ranked.set_b.top(layer, args.topk)
        print(f"  {args.group_a} top-{args.topk}: {[f.feature_id for f in top_a]}")
        if top_a:
            print(f"    frequencies: {[f.frequency for f in top_a]}")
        print(f"  {args.group_b} top-{args.topk}: {[f.feature_id for f in top_b]}")
        if top_b:
            print(f"    frequencies: {[f.frequency for f in top_b]}")


if __name__ == "__main__":
    main()
