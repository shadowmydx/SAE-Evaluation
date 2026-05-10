"""
Step 3: SAE 特征 steer 干预验证
===============================
基于 rank_features.py 输出的 C1 (France) 和 C2 (China) 特征集合：

实验 A (France prompt → 反方向 steer):
  对 France 原始 prompt 生成，将 C1 特征值取反 (negate)，
  观察输出是否不再趋近 France 相关内容。

实验 B (France prompt → China 特征注入):
  对 France 原始 prompt 生成，将属于 C2 的特征激活值
  set 为它们在中国 prompt 下的激活值，观察输出是否转向 China。
  需要先扫描热点量值。

用法:
  # 实验 A: 反向 steer C1
  python3 surgenry/steer_generation.py baseline --prompt "The capital of France is"
  python3 surgenry/steer_generation.py steer-negate --prompt "The capital of France is" -i ranked_features.json -t c1 -l 28 --top 3

  # 实验 B: 注入 C2
  python3 surgenry/steer_generation.py inject --prompt "The capital of France is" -i ranked_features.json -t c2 -l 28 --top 3

  # 对比输出
  python3 surgenry/steer_generation.py compare --prompt "The capital of France is" -i ranked_features.json
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen3_client import _r


def cmd_baseline(args):
    """无干预基线输出"""
    body = {"prompt": args.prompt, "max_new_tokens": args.max_tokens, "temperature": args.temperature}
    data = _r("post", "/generate", json=body).json()
    print(data["response"])


def cmd_steer(args):
    """实验 A: 反方向 steer (negate)，对抗 Hydra 效应"""
    with open(args.input) as f:
        ranked = json.load(f)

    target_name = "c1"  # steer-negate always operates on C1
    target = ranked[target_name]
    ls = str(args.layer)
    if ls not in target:
        print(f"Layer {args.layer} not found in {target_name}")
        sys.exit(1)

    features = target[ls][:args.top]
    interventions = [
        {"layer": args.layer, "feature_id": feat["feature_id"],
         "action": "negate", "value": 0.0}
        for feat in features
    ]

    body = {
        "prompt": args.prompt,
        "max_new_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": args.stream,
        "interventions": interventions,
    }
    print(f"Experiment A: Negating C1 features in layer {args.layer}")
    print(f"  Features: {[f['feature_id'] for f in features]}")
    print(f"  Freq:     {[f['frequency'] for f in features]}")
    print()

    if args.stream:
        r = _r("post", "/sae_intervene", json=body, stream=True)
        for line in r.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    print()
                    break
                try:
                    obj = json.loads(payload)
                    print(obj["token"], end="", flush=True)
                except json.JSONDecodeError:
                    pass
    else:
        data = _r("post", "/sae_intervene", json=body).json()
        print(data["response"])


def _scan_activation(prompt: str, layer: int, feature_ids: list[int]) -> dict[int, float]:
    """获取指定特征在某个 prompt 下的激活值"""
    body = {"prompt": prompt, "layers": [layer], "token_position": -1, "max_features": 100}
    data = _r("post", "/sae", json=body).json()
    ls = str(layer)
    values = {}
    for feat in data["layers"][ls]["top_features"]:
        if feat["feature_id"] in feature_ids:
            values[feat["feature_id"]] = feat["activation"]
    return values


def cmd_inject(args):
    """实验 B: 把 China 特征的激活值注入到 France prompt 的生成中"""
    with open(args.input) as f:
        ranked = json.load(f)

    ls = str(args.layer)
    if ls not in ranked["c2"]:
        print(f"Layer {args.layer} not found in C2")
        sys.exit(1)

    c2_features = ranked["c2"][ls][:args.top]
    c2_fids = [feat["feature_id"] for feat in c2_features]

    # 获取 China prompt 下这些特征的激活值
    china_prompt = ranked["info"].get("prompt_b", "The capital of China is").split(" (")[0]
    print(f"Scanning activation values from: \"{china_prompt}\"")
    china_vals = _scan_activation(china_prompt, args.layer, c2_fids)

    interventions = []
    for feat in c2_features:
        fid = feat["feature_id"]
        val = china_vals.get(fid, 0.0)
        if val == 0.0:
            # 候选特征未被扫描到，拿第一个非零值
            print(f"  Warning: feature {fid} not active in China prompt, skipping")
            continue
        interventions.append({
            "layer": args.layer,
            "feature_id": fid,
            "action": "set",
            "value": val,
        })

    if not interventions:
        print("No features to intervene. Try --top with a smaller number.")
        sys.exit(1)

    body = {
        "prompt": args.prompt,
        "max_new_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": args.stream,
        "interventions": interventions,
    }
    print(f"Experiment B: Injecting C2 features activation into France prompt")
    print(f"  Layer {args.layer}: injections = {[(iv['feature_id'], round(iv['value'], 2)) for iv in interventions]}")
    print()

    if args.stream:
        r = _r("post", "/sae_intervene", json=body, stream=True)
        for line in r.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    print()
                    break
                try:
                    obj = json.loads(payload)
                    print(obj["token"], end="", flush=True)
                except json.JSONDecodeError:
                    pass
    else:
        data = _r("post", "/sae_intervene", json=body).json()
        print(data["response"])


def cmd_compare(args):
    """一键对比: baseline vs steer-negate C1 vs inject C2"""
    with open(args.input) as f:
        ranked = json.load(f)

    layers = ranked["info"]["layers"]
    print(f"=== Comparison: \"{args.prompt}\" ===")
    print(f"Top-K: {ranked['info']['topk']}, Layers: {layers}")
    print()

    # Baseline
    print("── Baseline ──")
    body = {"prompt": args.prompt, "max_new_tokens": args.max_tokens, "temperature": args.temperature}
    data = _r("post", "/generate", json=body).json()
    baseline_text = data["response"]
    print(baseline_text[:200])
    print()

    # Steer-negate C1
    for layer in layers:
        ls = str(layer)
        if ls not in ranked["c1"]:
            continue
        features = ranked["c1"][ls][:args.top]
        interventions = [
            {"layer": layer, "feature_id": feat["feature_id"], "action": "negate", "value": 0.0}
            for feat in features
        ]
        body = {
            "prompt": args.prompt, "max_new_tokens": args.max_tokens,
            "temperature": args.temperature, "interventions": interventions,
        }
        data = _r("post", "/sae_intervene", json=body).json()
        print(f"── Steer-negate C1 (layer {layer}) ──")
        print(data["response"][:200])
        print()

    # Inject C2
    for layer in layers:
        ls = str(layer)
        if ls not in ranked["c2"]:
            continue
        features = ranked["c2"][ls][:args.top]
        china_prompt = ranked["info"].get("prompt_b", "The capital of China is").split(" (")[0]
        china_vals = _scan_activation(china_prompt, layer, [feat["feature_id"] for feat in features])
        interventions = []
        for feat in features:
            val = china_vals.get(feat["feature_id"], 0.0)
            if val > 0:
                interventions.append({"layer": layer, "feature_id": feat["feature_id"], "action": "set", "value": val})
        if not interventions:
            continue
        body = {
            "prompt": args.prompt, "max_new_tokens": args.max_tokens,
            "temperature": args.temperature, "interventions": interventions,
        }
        data = _r("post", "/sae_intervene", json=body).json()
        print(f"── Inject C2 (layer {layer}) ──")
        print(data["response"][:200])
        print()


def main():
    parser = argparse.ArgumentParser(description="SAE 特征 steer 干预验证")
    sub = parser.add_subparsers(dest="command", required=True)

    gen_args = {"help": "生成参数"}
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
