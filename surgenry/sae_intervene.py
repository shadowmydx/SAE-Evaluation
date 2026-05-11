"""
SAE 特征干预实验客户端 — 薄 CLI 壳

用法:
  python3 surgenry/sae_intervene.py "The capital of France is" -l 15 -f 45231 -a zero
  python3 surgenry/sae_intervene.py "The capital of France is" --baseline
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import generate, intervene, intervene_stream


def main():
    parser = argparse.ArgumentParser(description="SAE 特征干预实验")
    parser.add_argument("prompt", type=str, help="生成 prompt")
    parser.add_argument("-l", "--layer", type=int, help="干预的目标层")
    parser.add_argument("-f", "--feature-id", type=int, dest="feature_id",
                        help="SAE 特征 ID")
    parser.add_argument("-a", "--action", type=str, default="zero",
                        choices=["zero", "scale", "set", "clamp_max"],
                        help="干预类型")
    parser.add_argument("-v", "--value", type=float, default=0.0, help="干预值")
    parser.add_argument("-m", "--max-tokens", type=int, default=256, dest="max_tokens")
    parser.add_argument("-t", "--temperature", type=float, default=0.6)
    parser.add_argument("-s", "--stream", action="store_true", help="流式输出")
    parser.add_argument("--baseline", action="store_true", help="输出 baseline（无干预）")
    args = parser.parse_args()

    if args.baseline:
        text = generate(args.prompt, args.max_tokens, args.temperature)
        print(text)
    elif args.layer is not None and args.feature_id is not None:
        interventions = [{"layer": args.layer, "feature_id": args.feature_id,
                          "action": args.action, "value": args.value}]
        if args.stream:
            for token in intervene_stream(args.prompt, interventions, args.max_tokens, args.temperature):
                print(token, end="", flush=True)
            print()
        else:
            text = intervene(args.prompt, interventions, args.max_tokens, args.temperature)
            print(text)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
