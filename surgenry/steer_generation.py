"""
Step 3: SAE 特征 steer 干预验证 — 薄 CLI 壳

基于 rank_features.py 输出的 ranked_features.json:

实验 A (steer-negate): 对 prompt 生成，将某侧特征取反
实验 B (inject):      注入对侧特征的激活值
实验 C (compare):     一键对比 baseline + steer-negate + inject

用法:
  python3 surgenry/steer_generation.py baseline --prompt "..."
  python3 surgenry/steer_generation.py steer-negate --prompt "..." -i ranked_features.json -l 28 --top 3
  python3 surgenry/steer_generation.py inject --prompt "..." -i ranked_features.json -l 28 --top 3
  python3 surgenry/steer_generation.py compare --prompt "..." -i ranked_features.json
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import RankedFeatures, generate, intervene, intervene_stream




def cmd_baseline(args):
    text = generate(args.prompt, args.max_tokens, args.temperature)
    print(text)


def cmd_steer(args):
    ranked = RankedFeatures.load(args.input)
    side = args.side  # "a" or "b"
    source = ranked.set_a if side == "a" else ranked.set_b
    ls = args.layer

    features = source.top(ls, args.top)
    if not features:
        print(f"Layer {ls} not found in set {side}")
        sys.exit(1)

    interventions = [
        {"layer": ls, "feature_id": f.feature_id, "action": "negate", "value": 0.0}
        for f in features
    ]

    label = ranked.set_a.group_label if side == "a" else ranked.set_b.group_label
    print(f"Steer-negate: negating {label} features in layer {ls}")
    print(f"  Features: {[f.feature_id for f in features]}")
    print(f"  Freq:     {[f.frequency for f in features]}")
    print()

    if args.stream:
        for token in intervene_stream(args.prompt, interventions, args.max_tokens, args.temperature):
            print(token, end="", flush=True)
        print()
    else:
        text = intervene(args.prompt, interventions, args.max_tokens, args.temperature)
        print(text)


def cmd_inject(args):
    ranked = RankedFeatures.load(args.input)
    side = args.side  # which side's features to inject
    source = ranked.set_a if side == "a" else ranked.set_b
    seed = ranked.info.seed_prompt_a if side == "a" else ranked.info.seed_prompt_b
    ls = args.layer

    features = source.top(ls, args.top)
    if not features:
        print(f"Layer {ls} not found in set {side}")
        sys.exit(1)

    # Get activation values for these features
    from core import scan_activation
    fids = [f.feature_id for f in features]
    act_vals = scan_activation(seed, ls, fids)

    interventions = []
    for f in features:
        val = act_vals.get(f.feature_id, 0.0)
        if val == 0.0:
            print(f"  Warning: feature {f.feature_id} not active in seed prompt, skipping")
            continue
        interventions.append({"layer": ls, "feature_id": f.feature_id, "action": "set", "value": val})

    if not interventions:
        print("No features to intervene. Try --top with a smaller number.")
        sys.exit(1)

    label = ranked.set_b.group_label if side == "a" else ranked.set_a.group_label
    print(f"Inject: injecting {label} features into prompt")
    print(f"  Layer {ls}: injections = {[(iv['feature_id'], round(iv['value'], 2)) for iv in interventions]}")
    print()

    if args.stream:
        for token in intervene_stream(args.prompt, interventions, args.max_tokens, args.temperature):
            print(token, end="", flush=True)
        print()
    else:
        text = intervene(args.prompt, interventions, args.max_tokens, args.temperature)
        print(text)


def cmd_compare(args):
    ranked = RankedFeatures.load(args.input)
    layers = ranked.info.layers

    print(f"=== Comparison: \"{args.prompt}\" ===")
    print(f"Layers: {layers}")
    print()

    # Baseline
    print("── Baseline ──")
    text = generate(args.prompt, args.max_tokens, args.temperature)
    print(text[:200])
    print()

    # Steer-negate set A
    for layer in layers:
        features = ranked.set_a.top(layer, args.top)
        if not features:
            continue
        interventions = [
            {"layer": layer, "feature_id": f.feature_id, "action": "negate", "value": 0.0}
            for f in features
        ]
        text = intervene(args.prompt, interventions, args.max_tokens, args.temperature)
        print(f"── Steer-negate {ranked.set_a.group_label} (layer {layer}) ──")
        print(text[:200])
        print()

    # Inject set B
    for layer in layers:
        features = ranked.set_b.top(layer, args.top)
        if not features:
            continue
        seed = ranked.info.seed_prompt_b
        from core import scan_activation
        fids = [f.feature_id for f in features]
        act_vals = scan_activation(seed, layer, fids)
        interventions = []
        for f in features:
            val = act_vals.get(f.feature_id, 0.0)
            if val > 0:
                interventions.append({"layer": layer, "feature_id": f.feature_id, "action": "set", "value": val})
        if not interventions:
            continue
        text = intervene(args.prompt, interventions, args.max_tokens, args.temperature)
        print(f"── Inject {ranked.set_b.group_label} (layer {layer}) ──")
        print(text[:200])
        print()


def main():
    parser = argparse.ArgumentParser(description="SAE 特征 steer 干预验证")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, desc in [("baseline", "无干预基线输出"), ("steer-negate", "反方向 steer"),
                        ("inject", "注入目标特征激活"), ("compare", "一键对比")]:
        p = sub.add_parser(name, help=desc)
        p.add_argument("--prompt", type=str, default="The capital of France is")
        p.add_argument("-m", "--max-tokens", type=int, default=256, dest="max_tokens")
        p.add_argument("-t", "--temperature", type=float, default=0.3)
        if name == "baseline":
            continue
        p.add_argument("-i", "--input", type=str, default="ranked_features.json")
        p.add_argument("-l", "--layer", type=int, default=28)
        p.add_argument("--top", type=int, default=3)
        p.add_argument("--side", type=str, default="a", choices=["a", "b"],
                       help="For steer-negate: which side to negate. For inject: which side to inject from.")
        if name == "compare":
            continue
        p.add_argument("-s", "--stream", action="store_true")

    args = parser.parse_args()
    dispatch = {
        "baseline": cmd_baseline,
        "steer-negate": cmd_steer,
        "inject": cmd_inject,
        "compare": cmd_compare,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
