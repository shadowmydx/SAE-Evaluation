"""
SAE 特征分析客户端
==================
基于 Qwen3 Model Server 的 SAE 分析，功能等同 surgenry/test.py。

用法：
  1. 先启动服务器并加载模型
     python3 qwen3_server.py &
     python3 qwen3_client.py load /path/to/Qwen3-8B
     python3 qwen3_client.py sae_set_dir /path/to/SAE-Res-Qwen3-8B-Base-W64K-L0_100

  2. 运行分析
     python3 surgenry/sae_analysis.py [--prompt "..."] [--layer 15] [--topk 20]
"""

import argparse
import sys
import os

# Add project root to path so we can import the client module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen3_client import _r, SERVER_URL


def analyze(prompt: str, layers: list[int], top_n: int, show_all: bool):
    body = {
        "prompt": prompt,
        "layers": layers,
        "token_position": -1,
        "max_features": top_n,
        "include_reconstruction": True,
    }
    r = _r("post", "/sae", json=body)
    data = r.json()

    print(f"Prompt: \"{data['layers'][str(layers[0])]['prompt']}\"")
    print(f"Tokens: {data['layers'][str(layers[0])]['num_tokens']}")
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
    analyze(args.prompt, layers, args.top_n)


if __name__ == "__main__":
    main()
