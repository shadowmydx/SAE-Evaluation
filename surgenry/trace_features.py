"""
Trace 23 code-tilted A∩D features across ALL token positions
for code and pure reasoning prompts. Visualize as heatmap.

Usage:
  python3 surgenry/trace_features.py
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _r(method, path, **kwargs):
    import requests
    BASE = "http://localhost:8000"
    return getattr(requests, method)(BASE + path, **kwargs)

# ── Load the 23 code-tilted A∩D features (L20, tilt>=2.0) ──
with open("shared_with_tilt_l20_l26.json") as f:
    data = json.load(f)

ct_fids = sorted(set(f["feature_id"] for f in data["20"]["features"] if f["tilt"] >= 2.0))
print(f"Code-tilted A∩D features (L20, tilt>=2.0): {len(ct_fids)}")
print(f"  FIDs: {ct_fids}")
print()

# ── Prompts to trace ──
PROMPTS = {
    # Code prompts
    "Code: Fibonacci function": "Write a function to compute the nth Fibonacci number",
    "Code: Quicksort": "Write a function that implements quicksort",

    # Code reasoning
    "CR: Stack LIFO": "Simulate a stack. Push 1, Push 2, Pop. Push 3. Pop. Pop. What is the final stack?",

    # Pure reasoning (should also activate these features)
    "PR: Train speed": "A train leaves station A at 3:00 PM traveling at 60 mph. Another train leaves station B at 4:00 PM traveling at 80 mph. The stations are 300 miles apart. When do they meet?",
    "PR: Chicken/rabbit": "A farmer has chickens and rabbits. He counts 20 heads and 56 legs. How many chickens and how many rabbits does he have?",
    "PR: Clock angle": "A clock shows 3:15. What is the angle between the hour hand and the minute hand?",
    "PR: Water pouring": "You have a 5-liter jug and a 3-liter jug. How do you measure exactly 4 liters?",

    # Simple knowledge (should NOT activate)
    "KN: Capital": "The capital of France is",
}

for label, prompt in PROMPTS.items():
    print(f"═══ {label} ═══")
    print(f"  Prompt: {prompt[:60]}...")

    resp = _r("post", "/sae_trace", json={
        "prompt": prompt,
        "layers": [20],
        "feature_ids": ct_fids,
    })
    data = resp.json()
    layer_data = data["layers"]["20"]
    tokens = layer_data["tokens"]
    activations = layer_data["activations"]

    # Compute aggregate stats per token position
    n_fids = len(ct_fids)
    n_tokens = len(tokens)

    # For each token: total activation across all 23 features, number of active features
    total_acts = []
    active_counts = []
    for pos in range(n_tokens):
        total = sum(activations[str(fid)][pos] for fid in ct_fids)
        active = sum(1 for fid in ct_fids if activations[str(fid)][pos] > 0)
        total_acts.append(total)
        active_counts.append(active)

    # Find top-5 activating tokens
    top_positions = sorted(range(n_tokens), key=lambda i: -total_acts[i])[:5]

    print(f"  Tokens: {n_tokens}")
    print(f"  Top-5 activating tokens:")
    for pos in top_positions:
        if total_acts[pos] > 0:
            print(f"    [{pos}] '{tokens[pos]}'  sum_act={total_acts[pos]:.4f}  active_fids={active_counts[pos]}/{n_fids}")

    # Show all tokens with activation heat (filter near-zero)
    print(f"  All tokens with activation > 0.5:")
    for pos in range(n_tokens):
        if total_acts[pos] > 0.5:
            tok = tokens[pos].replace('Ġ', ' ').replace('Ċ', '\\n')
            print(f"    [{pos:3d}] '{tok:>15s}'  sum={total_acts[pos]:7.4f}  n_active={active_counts[pos]:2d}")
    print()
