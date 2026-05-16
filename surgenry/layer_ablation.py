"""
Layer ablation: test each layer's own code-tilted A∩D features
with add_direction α=-20 on 5 tasks.

Usage: python3 surgenry/layer_ablation.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surgenry.core.client import generate, intervene

MAX_TOKENS = 1024
TEMPERATURE = 0.3
ALPHA = -20.0

TASKS = [
    ("Capital", "The capital of France is"),
    ("Algebra", "If 3x + 7 = 22, what is the value of 2x + 5?"),
    ("Clock", "A clock shows 3:15. What is the angle between the hour hand and the minute hand?"),
    ("Water", "You have a 11-liter jug and a 4-liter jug, both empty. Can you measure exactly 8 liters? Answer yes or no and explain how."),
    ("Fibonacci", "Write a function to compute the nth Fibonacci number."),
]

LAYERS = [15, 20, 24, 26, 31]

def load_tilt_features(path, layer, min_tilt=2.0):
    with open(path) as f:
        data = json.load(f)
    return [f["feature_id"] for f in data[str(layer)]["features"] if f["tilt"] >= min_tilt]

def get_visible(out):
    out = out.strip()
    if '</think>' in out:
        return out.split('</think>')[-1].strip()
    if '<think>' in out:
        body = out.split('<think>')[-1]
        return f"[UNCLOSED THINK, len={len(body)}] ...{body[-100:]}"
    return out

print("=" * 60)
print("Layer Ablation: code-tilted A∩D + add_direction α=-20")
print("=" * 60)

for layer in LAYERS:
    fids = load_tilt_features("layer_ablation_features.json", layer, min_tilt=2.0)
    print(f"\n{'─' * 60}")
    print(f"Layer {layer}: {len(fids)} code-tilted A∩D features")
    print(f"  IDs: {fids}")
    print()

    iv = [{"layer": layer, "feature_id": fid, "action": "add_direction", "value": ALPHA}
          for fid in fids]

    for tid, prompt in TASKS:
        out = intervene(prompt, iv, MAX_TOKENS, TEMPERATURE)
        vis = get_visible(out)
        short = vis[:80].replace("\n", " ")
        print(f"  [{tid}] {short}", flush=True)
        time.sleep(0.5)

# ── Random feature ablation control ──
print(f"\n{'=' * 60}")
print("Control: 23 Random Features at L20, add_direction α=-20")
import random
random.seed(42)
# Generate 23 random FIDs not in the code-tilted set
used_fids = set()
for layer in LAYERS:
    used_fids.update(load_tilt_features("layer_ablation_features.json", layer, min_tilt=2.0))
all_possible = list(set(range(1, 65537)) - used_fids)
random_fids = random.sample(all_possible, 23)
print(f"  Random FIDs: {random_fids[:5]}...")
iv_rand = [{"layer": 20, "feature_id": fid, "action": "add_direction", "value": -100.0}
           for fid in random_fids]
for tid, prompt in TASKS:
    out = intervene(prompt, iv_rand, MAX_TOKENS, TEMPERATURE)
    vis = get_visible(out)
    short = vis[:80].replace("\n", " ")
    print(f"  [{tid}] {short}", flush=True)
    time.sleep(0.5)
