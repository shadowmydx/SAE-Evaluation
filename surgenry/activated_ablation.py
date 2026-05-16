"""
Activated feature ablation: sample random features from the activated union (A∪D),
excluding code-tilted features, at L15 and L20 with add_direction α=-20.
"""
import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surgenry.core.client import intervene

random.seed(42)

with open('layer_ablation_features.json') as f:
    data = json.load(f)

TASKS = [
    ("Capital", "The capital of France is"),
    ("Algebra", "If 3x + 7 = 22, what is the value of 2x + 5?"),
    ("Clock", "A clock shows 3:15. What is the angle between the hour hand and the minute hand?"),
    ("Water", "You have a 11-liter jug and a 4-liter jug, both empty. Can you measure exactly 8 liters? Answer yes or no and explain how."),
    ("Fibonacci", "Write a function to compute the nth Fibonacci number."),
]

for layer, n_samples in [(15, 21), (20, 23)]:
    ls = str(layer)
    all_fids = [f["feature_id"] for f in data[ls]["features"]]
    tilt20 = set(f["feature_id"] for f in data[ls]["features"] if f["tilt"] >= 2.0)
    pool = list(set(all_fids) - tilt20)
    random.shuffle(pool)
    sampled = pool[:n_samples]

    print(f"=== L{layer}: {n_samples} random A∪D features (excluding tilt≥2.0) ===", flush=True)

    iv = [{"layer": layer, "feature_id": fid, "action": "add_direction", "value": -20.0}
          for fid in sampled]

    for tid, prompt in TASKS:
        out = intervene(prompt, iv, 1024, 0.3)
        vis = out.strip()
        if "</think>" in vis:
            vis = vis.split("</think>")[-1].strip()
        elif "<think>" in vis:
            body = vis.split("<think>")[-1]
            vis = f"[UNCLOSED THINK, len={len(body)}] ...{body[-100:]}"
        short = vis[:80].replace("\n", " ")
        print(f"  [{tid}] {short}", flush=True)
    print()
