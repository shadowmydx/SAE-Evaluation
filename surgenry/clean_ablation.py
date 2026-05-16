"""
Clean ablation: compare code-tilted A∩D vs random A-only∪D-only at L15 and L20.
Full scan of all prompts to build complete A-only∪D-only pool for each layer.
"""
import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surgenry.core.client import sae_scan, intervene

random.seed(42)

LAYERS = [15, 20]

TASKS = [
    ("Capital", "The capital of France is"),
    ("Algebra", "If 3x + 7 = 22, what is the value of 2x + 5?"),
    ("Clock", "A clock shows 3:15. What is the angle between the hour hand and the minute hand?"),
    ("Water", "You have a 11-liter jug and a 4-liter jug, both empty. Can you measure exactly 8 liters? Answer yes or no and explain how."),
    ("Fibonacci", "Write a function to compute the nth Fibonacci number."),
]

def _prompt_file(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "prompts", name)

with open(_prompt_file("code.json")) as f:
    code_prompts = json.load(f)[:30]
with open(_prompt_file("descriptions.json")) as f:
    desc_prompts = json.load(f)[:15]

# Scan ALL prompts at ALL layers
a_sets = {l: [] for l in LAYERS}
for i, p in enumerate(code_prompts):
    print(f"  scan code [{i+1}/{len(code_prompts)}]...", end="\r", flush=True)
    data = sae_scan(p, LAYERS)
    for l in LAYERS:
        fids = {f["feature_id"] for f in data["layers"][str(l)]["top_features"]}
        a_sets[l].append(fids)
print()

d_sets = {l: [] for l in LAYERS}
for i, p in enumerate(desc_prompts):
    print(f"  scan desc [{i+1}/{len(desc_prompts)}]...", end="\r", flush=True)
    data = sae_scan(p, LAYERS)
    for l in LAYERS:
        fids = {f["feature_id"] for f in data["layers"][str(l)]["top_features"]}
        d_sets[l].append(fids)
print()

# Load code-tilted features from previous data
with open("layer_ablation_features.json") as f:
    prev = json.load(f)

for LA in LAYERS:
    ls = str(LA)
    a_union = set().union(*a_sets[LA])
    d_union = set().union(*d_sets[LA])
    shared = a_union & d_union
    a_only = a_union - d_union
    d_only = d_union - a_union
    a_or_d_only = a_only | d_only

    print(f"\nL{LA}: |A|={len(a_union)}, |D|={len(d_union)}, |A∩D|={len(shared)}, |A-only∪D-only|={len(a_or_d_only)}")

    # Code-tilted set
    code_tilted = [f["feature_id"] for f in prev[ls]["features"] if f["tilt"] >= 2.0]
    n_ct = len(code_tilted)
    print(f"  code-tilted A∩D (tilt≥2.0): {n_ct} features")
    print(f"    IDs: {code_tilted}")

    # Random from A-only ∪ D-only, same count
    pool = list(a_or_d_only)
    random.shuffle(pool)
    rand_fids = pool[:n_ct]
    print(f"  random A-only∪D-only: {len(rand_fids)} features")
    print(f"    IDs: {rand_fids}")

    # Test code-tilted
    print(f"\n=== L{LA}: {n_ct} code-tilted A∩D (tilt≥2.0) ===", flush=True)
    iv_ct = [{"layer": LA, "feature_id": fid, "action": "add_direction", "value": -20.0}
             for fid in code_tilted[:n_ct]]
    for tid, prompt in TASKS:
        out = intervene(prompt, iv_ct, 1024, 0.3)
        vis = out.strip()
        if "</think>" in vis:
            vis = vis.split("</think>")[-1].strip()
        elif "<think>" in vis:
            body = vis.split("<think>")[-1]
            vis = f"[UNCLOSED THINK, len={len(body)}] ...{body[-100:]}"
        print(f"  [{tid}] {vis[:80].replace(chr(10),' ')}", flush=True)

    # Test random A-only ∪ D-only
    print(f"\n=== L{LA}: {n_ct} random A-only ∪ D-only (excl. ALL A∩D) ===", flush=True)
    iv_rand = [{"layer": LA, "feature_id": fid, "action": "add_direction", "value": -20.0}
               for fid in rand_fids]
    for tid, prompt in TASKS:
        out = intervene(prompt, iv_rand, 1024, 0.3)
        vis = out.strip()
        if "</think>" in vis:
            vis = vis.split("</think>")[-1].strip()
        elif "<think>" in vis:
            body = vis.split("<think>")[-1]
            vis = f"[UNCLOSED THINK, len={len(body)}] ...{body[-100:]}"
        print(f"  [{tid}] {vis[:80].replace(chr(10),' ')}", flush=True)
