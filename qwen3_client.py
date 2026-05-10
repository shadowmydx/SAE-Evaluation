"""
Qwen3 Client
============
CLI tool to interact with the Qwen3 Model Server.

Commands:
  load <model_dir>                            — load model into server
  generate <prompt>                           — text generation (non-streaming)
  stream <prompt>                             — text generation (SSE streaming)
  hidden <prompt> -l LAYERS                   — extract hidden states
  logits <prompt> -l LAYERS                   — logits lens
  sae_set_dir <sae_dir>                       — set SAE model directory
  sae <prompt> -l LAYERS                      — run SAE on given layers
  intervene <prompt> -l LAYER -f FEATURE_ID    — intervene on a SAE feature during generation
  health                                      — check server status

Examples:
  python qwen3_client.py load /path/to/Qwen3-8B
  python qwen3_client.py generate "how to make a cake" -m 512 -t 0.7
  python qwen3_client.py stream "tell me a story"
  python qwen3_client.py hidden "the cat sat" -l 0,10,20,31
  python qwen3_client.py logits "the cat sat" -l 10,20,31 -k 5
"""

import argparse
import json
import os
import sys
import requests

SERVER_URL = "http://127.0.0.1:8000"


class ServerError(Exception):
    pass


def _r(method: str, path: str, **kwargs) -> requests.Response:
    url = f"{SERVER_URL}{path}"
    try:
        return getattr(requests, method)(url, **kwargs, timeout=300)
    except requests.ConnectionError:
        raise ServerError(f"Cannot connect to {SERVER_URL}. Is the server running?")
    except requests.Timeout:
        raise ServerError("Request timed out.")


# ================================================================

def cmd_health(args):
    r = _r("get", "/health")
    print(json.dumps(r.json(), indent=2))


def cmd_load(args):
    body = {"model_dir": args.model_dir}
    r = _r("post", "/load", json=body)
    print(json.dumps(r.json(), indent=2))


def cmd_generate(args):
    body = {
        "prompt": args.prompt,
        "max_new_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "stream": False,
    }
    r = _r("post", "/generate", json=body)
    data = r.json()
    print(data["response"])


def cmd_stream(args):
    body = {
        "prompt": args.prompt,
        "max_new_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "stream": True,
    }
    r = _r("post", "/generate", json=body, stream=True)
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


def cmd_hidden(args):
    body = {
        "prompt": args.prompt,
        "layers": [int(l) for l in args.layers.split(",")],
        "token_position": args.position,
    }
    r = _r("post", "/hidden_states", json=body)
    data = r.json()

    print(f"Prompt  : {data['prompt']}")
    print(f"Tokens  : {data['num_tokens']}")
    print(f"Position: {data['token_position']}")
    print(f"Hidden  : {data['hidden_size']}-d vectors")
    for layer_str, vec in data["layers"].items():
        # Print shape + first 5 values
        prefix = ", ".join(f"{v:.4f}" for v in vec[:5])
        print(f"  Layer {layer_str}: [{prefix}, ...]  (len={len(vec)})")


def cmd_logits(args):
    body = {
        "prompt": args.prompt,
        "layers": [int(l) for l in args.layers.split(",")],
        "top_k": args.top_k,
        "token_position": args.position,
    }
    r = _r("post", "/logits_lens", json=body)
    data = r.json()

    print(f"Prompt  : {data['prompt']}")
    print(f"Tokens  : {data['num_tokens']}")
    print(f"Position: {data['token_position']}")
    print()
    for layer_str, tokens in data["layers"].items():
        print(f"── Layer {layer_str} ──")
        for t in tokens:
            bar = "█" * int(t["prob"] * 100)
            print(f"  {t['token']:>12s}  {t['prob']:.4f}  {bar}")
        print()


def cmd_sae_set_dir(args):
    abs_path = os.path.abspath(args.sae_dir)
    r = _r("post", "/sae_set_dir", json={"sae_dir": abs_path})
    print(json.dumps(r.json(), indent=2))


def cmd_sae(args):
    body = {
        "prompt": args.prompt,
        "layers": [int(l) for l in args.layers.split(",")],
        "token_position": args.position,
        "max_features": args.max_features,
        "include_reconstruction": args.reconstruction,
    }
    r = _r("post", "/sae", json=body)
    data = r.json()

    for layer_str, entry in data["layers"].items():
        print(f"── Layer {layer_str} ({entry['num_tokens']} tokens, pos={entry['token_position']}) ──")
        print(f"   Residual norm: {entry['residual_norm']}")
        print(f"   Active features: {entry['feature_count']}")
        print(f"   Top {len(entry['top_features'])} features:")
        for feat in entry["top_features"]:
            bar = "▓" * int(min(feat["activation"] / 5, 40))
            print(f"     #{feat['feature_id']:>6d}  {feat['activation']:.4f}  {bar}")
        if "reconstruction" in entry:
            rec = entry["reconstruction"]
            print(f"   Reconstruction MSE: {rec['mse_loss']:.6f}")
        print()


def cmd_intervene(args):
    body = {
        "prompt": args.prompt,
        "max_new_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "stream": args.stream,
        "interventions": [
            {"layer": args.layer, "feature_id": args.feature_id,
             "action": args.action, "value": args.value}
        ],
    }
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
        r = _r("post", "/sae_intervene", json=body)
        data = r.json()
        print(data["response"])


# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Qwen3 Client")
    sub = parser.add_subparsers(dest="command", required=True)

    # health
    sub.add_parser("health", help="Check server status")

    # load
    p_load = sub.add_parser("load", help="Load model into server")
    p_load.add_argument("model_dir", type=str, help="Path to model directory")

    # generate
    p_gen = sub.add_parser("generate", help="Generate text (non-streaming)")
    _add_gen_args(p_gen)

    # stream
    p_str = sub.add_parser("stream", help="Generate text (SSE streaming)")
    _add_gen_args(p_str)

    # hidden_states
    p_hid = sub.add_parser("hidden", help="Extract hidden states")
    p_hid.add_argument("prompt", type=str)
    p_hid.add_argument("-l", "--layers", type=str, required=True, help="Comma-separated layers, e.g. 0,10,20,31")
    p_hid.add_argument("-p", "--position", type=int, default=-1, help="Token position (-1 = last)")

    # logits_lens
    p_log = sub.add_parser("logits", help="Logits lens projection")
    p_log.add_argument("prompt", type=str)
    p_log.add_argument("-l", "--layers", type=str, required=True, help="Comma-separated layers, e.g. 10,20,31")
    p_log.add_argument("-k", "--top-k", type=int, default=10, dest="top_k", help="Top-K tokens to show")
    p_log.add_argument("-p", "--position", type=int, default=-1, help="Token position (-1 = last)")

    # sae_set_dir
    p_sae_dir = sub.add_parser("sae_set_dir", help="Set SAE model directory")
    p_sae_dir.add_argument("sae_dir", type=str, help="Path to SAE directory")

    # sae
    p_sae = sub.add_parser("sae", help="Run Sparse Autoencoder on hidden states")
    p_sae.add_argument("prompt", type=str)
    p_sae.add_argument("-l", "--layers", type=str, required=True, help="Comma-separated layers, e.g. 10,20,31")
    p_sae.add_argument("-p", "--position", type=int, default=-1, help="Token position (-1 = last)")
    p_sae.add_argument("-n", "--max-features", type=int, default=20, dest="max_features", help="Top-N features to show")
    p_sae.add_argument("-r", "--reconstruction", action="store_true", help="Include reconstruction metrics")

    # intervene
    p_int = sub.add_parser("intervene", help="Generate with SAE feature intervention")
    _add_gen_args(p_int)
    p_int.add_argument("-s", "--stream", action="store_true", help="Enable SSE streaming")
    p_int.add_argument("-l", "--layer", type=int, required=True, help="Layer to intervene on")
    p_int.add_argument("-f", "--feature-id", type=int, required=True, dest="feature_id", help="Feature ID to intervene on")
    p_int.add_argument("-a", "--action", type=str, default="zero", choices=["zero", "negate", "scale", "set", "clamp_max"], help="Intervention type")
    p_int.add_argument("-v", "--value", type=float, default=0.0, help="Intervention value (for scale/set/clamp_max)")

    args = parser.parse_args()

    dispatch = {
        "health": cmd_health,
        "load": cmd_load,
        "generate": cmd_generate,
        "stream": cmd_stream,
        "hidden": cmd_hidden,
        "logits": cmd_logits,
        "sae_set_dir": cmd_sae_set_dir,
        "sae": cmd_sae,
        "intervene": cmd_intervene,
    }
    try:
        dispatch[args.command](args)
    except ServerError as e:
        print(f"Error: {e}")
        sys.exit(1)


def _add_gen_args(p):
    p.add_argument("prompt", type=str)
    p.add_argument("-m", "--max-tokens", type=int, default=1024, dest="max_tokens")
    p.add_argument("-t", "--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)


if __name__ == "__main__":
    main()
