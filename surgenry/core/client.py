"""
Server interaction layer.
Single source of truth for HTTP calls and SSE parsing.
"""

import json
import sys
import os
from typing import Generator, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from qwen3_client import _r


def sae_scan(prompt: str, layers: list[int], token_position: int = -1,
             max_features: int = 100, include_reconstruction: bool = False) -> dict:
    """Run SAE on a prompt and return the raw JSON response."""
    body = {
        "prompt": prompt,
        "layers": layers,
        "token_position": token_position,
        "max_features": max_features,
    }
    if include_reconstruction:
        body["include_reconstruction"] = True
    return _r("post", "/sae", json=body).json()


def generate(prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
    """Non-streaming generation."""
    body = {
        "prompt": prompt,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
    }
    data = _r("post", "/generate", json=body).json()
    return data["response"]


def generate_stream(prompt: str, max_tokens: int = 256,
                    temperature: float = 0.7) -> Generator[str, None, None]:
    """Streaming generation, yields individual tokens."""
    body = {
        "prompt": prompt,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
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
                return
            try:
                obj = json.loads(payload)
                yield obj["token"]
            except json.JSONDecodeError:
                pass


def intervene(prompt: str, interventions: list[dict],
              max_tokens: int = 256, temperature: float = 0.7) -> str:
    """Non-streaming generation with SAE feature interventions."""
    body = {
        "prompt": prompt,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "interventions": interventions,
        "stream": False,
    }
    data = _r("post", "/sae_intervene", json=body).json()
    return data["response"]


def intervene_stream(prompt: str, interventions: list[dict],
                     max_tokens: int = 256,
                     temperature: float = 0.7) -> Generator[str, None, None]:
    """Streaming generation with SAE feature interventions."""
    body = {
        "prompt": prompt,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "interventions": interventions,
        "stream": True,
    }
    r = _r("post", "/sae_intervene", json=body, stream=True)
    for line in r.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if line.startswith("data: "):
            payload = line[6:]
            if payload.strip() == "[DONE]":
                print()
                return
            try:
                obj = json.loads(payload)
                yield obj["token"]
            except json.JSONDecodeError:
                pass


def scan_activation(prompt: str, layer: int, feature_ids: list[int]) -> dict[int, float]:
    """Get activation values for specific features from a single SAE scan."""
    data = sae_scan(prompt, [layer])
    ls = str(layer)
    values = {}
    for feat in data["layers"][ls]["top_features"]:
        if feat["feature_id"] in feature_ids:
            values[feat["feature_id"]] = feat["activation"]
    return values
