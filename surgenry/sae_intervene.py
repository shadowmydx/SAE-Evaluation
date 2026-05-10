"""
SAE 特征干预实验客户端
=====================
基于 Qwen3 Model Server 的 SAE 特征干预实验。

工作流：
  1. 先用 sae_analysis.py 发现特定 prompt 下激活的特征
  2. 选取特征 ID，用本脚本干预生成
  3. 对比干预前后输出差异

用法：
  # 发现特征
  python3 surgenry/sae_analysis.py --prompt "The capital of France is" -l 15

  # 消融特征 #45231
  python3 surgenry/sae_intervene.py "The capital of France is" -l 15 -f 45231 -a zero

  # 对比 baseline
  python3 surgenry/sae_intervene.py "The capital of France is" --baseline

  # 放大特征 3 倍
  python3 surgenry/sae_intervene.py "The capital of France is" -l 15 -f 45231 -a scale -v 3.0
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen3_client import _r, SERVER_URL


def intervene(prompt: str, layer: int, feature_id: int,
              action: str, value: float,
              max_tokens: int, temperature: float,
              stream: bool):
    body = {
        "prompt": prompt,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
        "interventions": [
            {"layer": layer, "feature_id": feature_id,
             "action": action, "value": value}
        ],
    }
    r = _r("post", "/sae_intervene", json=body, stream=stream)
    if stream:
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
                    import json
                    obj = json.loads(payload)
                    print(obj["token"], end="", flush=True)
                except json.JSONDecodeError:
                    pass
    else:
        data = r.json()
        print(data["response"])


def baseline(prompt: str, max_tokens: int, temperature: float):
    body = {
        "prompt": prompt,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
    }
    r = _r("post", "/generate", json=body)
    data = r.json()
    print(data["response"])


def main():
    parser = argparse.ArgumentParser(description="SAE 特征干预实验")
    parser.add_argument("prompt", type=str, help="生成 prompt")
    parser.add_argument("-l", "--layer", type=int, help="干预的目标层")
    parser.add_argument("-f", "--feature-id", type=int, dest="feature_id",
                        help="SAE 特征 ID")
    parser.add_argument("-a", "--action", type=str, default="zero",
                        choices=["zero", "scale", "set", "clamp_max"],
                        help="干预类型")
    parser.add_argument("-v", "--value", type=float, default=0.0,
                        help="干预值")
    parser.add_argument("-m", "--max-tokens", type=int, default=256,
                        dest="max_tokens")
    parser.add_argument("-t", "--temperature", type=float, default=0.6)
    parser.add_argument("-s", "--stream", action="store_true",
                        help="流式输出")
    parser.add_argument("--baseline", action="store_true",
                        help="输出 baseline（无干预）")
    args = parser.parse_args()

    if args.baseline:
        baseline(args.prompt, args.max_tokens, args.temperature)
    elif args.layer is not None and args.feature_id is not None:
        intervene(args.prompt, args.layer, args.feature_id,
                  args.action, args.value,
                  args.max_tokens, args.temperature, args.stream)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
