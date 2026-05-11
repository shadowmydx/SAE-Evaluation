"""
SAE 特征分析客户端 — 薄 CLI 壳
用法: python3 surgenry/sae_analysis.py [--prompt "..."] [--layer 15] [--topk 20]
"""

import argparse
from core import sae_scan


def main():
    parser = argparse.ArgumentParser(description="SAE 特征分析")
    parser.add_argument("--prompt", type=str, default="The capital of France is",
                        help="分析文本")
    parser.add_argument("-l", "--layers", type=str, default="0,15,31",
                        help="逗号分隔的层号")
    parser.add_argument("-n", "--top-n", type=int, default=20, dest="top_n",
                        help="显示 top-N 特征")
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    data = sae_scan(args.prompt, layers, max_features=args.top_n, include_reconstruction=True)

    first = str(layers[0])
    print(f"Prompt: \"{data['layers'][first]['prompt']}\"")
    print(f"Tokens: {data['layers'][first]['num_tokens']}")
    print()

    for layer_str, entry in data["layers"].items():
        print(f"── Layer {layer_str} (residual_norm={entry['residual_norm']}) ──")
        print(f"   Total active features: {entry['feature_count']}")
        print(f"   Top {len(entry['top_features'])} features:")
        for feat in entry["top_features"]:
            bar = "▓" * int(min(feat["activation"] / 5, 40))
            print(f"     #{feat['feature_id']:>6d}  {feat['activation']:.4f}  {bar}")
        rec = entry["reconstruction"]
        print(f"   Reconstruction MSE: {rec['mse_loss']:.6f}  (recon_norm={rec['reconstructed_norm']})")
        print()


if __name__ == "__main__":
    main()
