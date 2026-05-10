"""
Step 2: 跨 prompt 频率排序
==========================
基于差异特征发现的结果 (discovered_features.json)，
对每个候选特征在 50 个相关 prompt 中统计出现频率，筛选 top-K。

用法:
  python3 surgenry/rank_features.py [--topk 10] [--num-prompts 50]
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen3_client import _r


# 50 个 France 相关 prompt
FRANCE_PROMPTS = [
    "The capital of France is",
    "France is a country in",
    "The Eiffel Tower is located in",
    "French cuisine is famous for",
    "The French Revolution began in",
    "Paris is the capital of",
    "France borders Germany to the",
    "The Louvre Museum is in",
    "French wine regions include",
    "The president of France is",
    "France has a population of",
    "The French language originated from",
    "Normandy is a region in",
    "France colonized Algeria in",
    "The Tour de France is a",
    "French cheese varieties include",
    "France shares a border with",
    "The French Riviera is known for",
    "France was a founding member of",
    "French art movements include",
    "The Seine river flows through",
    "France uses the euro as",
    "French fashion houses include",
    "France has overseas territories in",
    "The French education system is",
    "France is divided into departments",
    "French literature includes works by",
    "The French Resistance fought during",
    "France hosted the World Cup in",
    "French cinema is known for",
    "The Palace of Versailles is in",
    "France has a strong economy in",
    "French philosophy influenced modern",
    "The French Alps are popular for",
    "France has many UNESCO sites",
    "French history includes the reign of",
    "The French flag is blue",
    "France had a monarchy until",
    "French scientists have discovered",
    "The French army was led by",
    "France is known for its",
    "French music genres include",
    "The French government is a",
    "France exports luxury goods like",
    "French architecture styles include",
    "The French island of Corsica is",
    "France has many regional dialects",
    "French holidays include Bastille Day",
    "The French railway system connects",
    "France has Mediterranean beaches along",
]

# 50 个 China 相关 prompt
CHINA_PROMPTS = [
    "The capital of China is",
    "China is a country in",
    "The Great Wall is located in",
    "Chinese cuisine is famous for",
    "The Chinese Communist Party was founded in",
    "Beijing is the capital of",
    "China borders Russia to the",
    "The Forbidden City is in",
    "Chinese tea varieties include",
    "The president of China is",
    "China has a population of",
    "The Chinese language originated from",
    "Yunnan is a province in",
    "China colonized Tibet in",
    "The Dragon Boat Festival is a",
    "Chinese medicine practices include",
    "China shares a border with",
    "The Yangtze River flows through",
    "China was a founding member of",
    "Chinese martial arts include",
    "The Silk Road passed through",
    "China uses the yuan as",
    "Chinese technology companies include",
    "China has autonomous regions in",
    "The Chinese education system is",
    "China is divided into provinces",
    "Chinese literature includes works by",
    "The Chinese Red Army fought during",
    "China hosted the Olympics in",
    "Chinese cinema is known for",
    "The Summer Palace is in",
    "China has a growing economy in",
    "Chinese philosophy influenced modern",
    "The Himalayas are popular for",
    "China has many UNESCO sites",
    "Chinese history includes the reign of",
    "The Chinese flag is red",
    "China had an imperial system until",
    "Chinese scientists have discovered",
    "The Chinese army was led by",
    "China is known for its",
    "Chinese music genres include",
    "The Chinese government is a",
    "China exports electronics like",
    "Chinese architecture styles include",
    "The Chinese island of Hainan is",
    "China has many regional dialects",
    "Chinese holidays include Lunar New Year",
    "The Chinese railway system connects",
    "China has Pacific coastlines along",
]


def scan_features(prompts: list[str], layers: list[int], candidate_fids: dict[int, set]) -> dict:
    """
    对所有 prompt 扫描指定候选特征，统计每个特征在 prompt 中的出现频率。
    candidate_fids: {layer: set(feature_id, ...)}
    返回: {layer: {fid: count, ...}}
    """
    layer_counts: dict[int, dict[int, int]] = {l: defaultdict(int) for l in layers}

    for i, prompt in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}] scanning...", end="\r")
        sys.stdout.flush()
        body = {
            "prompt": prompt,
            "layers": layers,
            "token_position": -1,
            "max_features": 100,
        }
        try:
            data = _r("post", "/sae", json=body).json()
        except Exception as e:
            print(f"\n  Error on prompt {i}: {e}")
            continue

        for ls, entry in data["layers"].items():
            layer = int(ls)
            if layer not in layers:
                continue
            active_fids = {feat["feature_id"] for feat in entry["top_features"]}
            for fid in candidate_fids.get(layer, set()):
                if fid in active_fids:
                    layer_counts[layer][fid] += 1

    return layer_counts


def main():
    parser = argparse.ArgumentParser(description="跨 prompt 频率排序: 筛选高频率差异特征")
    parser.add_argument("--input", type=str, default="discovered_features.json",
                        help="Step 1 的输出 JSON")
    parser.add_argument("--topk", type=int, default=10,
                        help="每层取 top-K 最高频特征")
    parser.add_argument("--num-prompts", type=int, default=50, dest="num_prompts",
                        help="用于排序的 prompt 数量")
    parser.add_argument("--output", type=str, default="ranked_features.json",
                        help="输出路径")
    args = parser.parse_args()

    with open(args.input) as f:
        discovered = json.load(f)

    # 解析 layer (key 是 str)
    layers = [int(k) for k in discovered.keys()]

    # 选择候选集合: A-only (France独有) 作为 C1, B-only (China独有) 作为 C2
    c1_fids: dict[int, set] = {}
    c2_fids: dict[int, set] = {}
    for ls, sets in discovered.items():
        layer = int(ls)
        c1_fids[layer] = set(sets["a_only"])
        c2_fids[layer] = set(sets["b_only"])

    print(f"Scanning {args.num_prompts} France prompts for C1 (France features)...")
    c1_counts = scan_features(FRANCE_PROMPTS[:args.num_prompts], layers, c1_fids)

    print(f"\nScanning {args.num_prompts} China prompts for C2 (China features)...")
    c2_counts = scan_features(CHINA_PROMPTS[:args.num_prompts], layers, c2_fids)

    # 排序取 top-K
    ranked = {
        "info": {
            "prompt_a": "France (C1: France-unique features)",
            "prompt_b": "China (C2: China-unique features)",
            "layers": layers,
            "topk": args.topk,
            "num_prompts": args.num_prompts,
        },
        "c1": {},
        "c2": {},
    }

    for layer in layers:
        sorted_c1 = sorted(c1_counts[layer].items(), key=lambda x: -x[1])
        sorted_c2 = sorted(c2_counts[layer].items(), key=lambda x: -x[1])

        ranked["c1"][str(layer)] = [
            {"feature_id": fid, "count": cnt, "frequency": round(cnt / args.num_prompts, 3)}
            for fid, cnt in sorted_c1[:args.topk]
        ]
        ranked["c2"][str(layer)] = [
            {"feature_id": fid, "count": cnt, "frequency": round(cnt / args.num_prompts, 3)}
            for fid, cnt in sorted_c2[:args.topk]
        ]

    with open(args.output, "w") as f:
        json.dump(ranked, f, indent=2)

    print(f"\nSaved to {args.output}")
    for layer in layers:
        print(f"\n── Layer {layer} ──")
        print(f"  C1 (France) top-{args.topk}: {[e['feature_id'] for e in ranked['c1'][str(layer)]]}")
        if ranked["c1"][str(layer)]:
            print(f"    frequencies: {[e['frequency'] for e in ranked['c1'][str(layer)]]}")
        print(f"  C2 (China) top-{args.topk}: {[e['feature_id'] for e in ranked['c2'][str(layer)]]}")
        if ranked["c2"][str(layer)]:
            print(f"    frequencies: {[e['frequency'] for e in ranked['c2'][str(layer)]]}")


if __name__ == "__main__":
    main()
