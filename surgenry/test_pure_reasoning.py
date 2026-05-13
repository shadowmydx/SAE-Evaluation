"""
Test code-tilted A∩D (L20, α=-20) on PURE reasoning (no code concepts).

Groups:
  1. Pure reasoning (easy)    — 15 prompts, no code concepts
  2. Pure reasoning (hard)    — 24 prompts (water pouring + scheduling + arithmetic)
  3. Code reasoning (easy)    — 15 prompts, positive control (should crash)
  4. Code generation          — 30 prompts, baseline control (should be ok)

判断标准:
  CRASH: 输出 < 20 tokens
  LOOP:  一个 10-token 片段重复 >= 4 次
  PASS:  以上皆否
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surgenry.core import generate, intervene

def _load(name):
    path = os.path.join("surgenry", "data", "prompts", name)
    with open(path) as f:
        return json.load(f)

def classify(text):
    """Return 'PASS', 'CRASH', or 'LOOP'."""
    if len(text) < 20:
        return "CRASH"
    # Check for loop: a 10-char substring repeated >= 4 times
    for i in range(len(text) - 10):
        snippet = text[i:i+10]
        if snippet.strip() and text.count(snippet) >= 4:
            return "LOOP"
    return "PASS"

# ── Load groups ──
groups = {
    "Pure Reasoning (easy)": _load("reasoning_pure.json"),
    "Pure Reasoning (hard)": _load("hard_reasoning_pure.json"),
    "Code Reasoning (easy)": _load("reasoning_code.json"),
    "Code Generation":       _load("code.json")[:10],  # subset, time
}

# ── Code-tilted A∩D features (L20, tilt>=2.0) ──
with open("shared_with_tilt_l20_l26.json") as f:
    shared_data = json.load(f)

ct_fids = [f["feature_id"] for f in shared_data["20"]["features"] if f["tilt"] >= 2.0]
print(f"Code-tilted A∩D features (L20, tilt>=2.0): {len(ct_fids)}")
print(f"  FIDs: {ct_fids}")
print()

interventions = [
    {"layer": 20, "feature_id": fid, "action": "add_direction", "value": -20.0}
    for fid in ct_fids
]

# ── Run ──
for group_name, prompts in groups.items():
    print(f"═══ {group_name} ({len(prompts)} prompts) ═══")
    pass_c, crash_c, loop_c = 0, 0, 0

    for i, prompt in enumerate(prompts):
        # Baseline
        base_text = generate(prompt, 100, 0.3)
        base_verdict = classify(base_text)

        # Intervention
        intr_text = intervene(prompt, interventions, 100, 0.3)
        intr_verdict = classify(intr_text)

        if intr_verdict == "PASS":
            pass_c += 1
        elif intr_verdict == "CRASH":
            crash_c += 1
        else:
            loop_c += 1

        # Print first 5 + any crash/loop
        if i < 5 or intr_verdict != "PASS" or base_verdict != "PASS":
            short_prompt = prompt[:60].replace("\n", " ")
            print(f"  [{i+1}] {short_prompt}...")
            print(f"        base={base_verdict}  intr={intr_verdict}")
            if intr_verdict != "PASS":
                print(f"        intr: {intr_text[:80]}")
        else:
            print(f"  [{i+1}] ✓ PASS")

    total = len(prompts)
    print(f"  => PASS={pass_c}/{total}  CRASH={crash_c}  LOOP={loop_c}  FAIL_RATE={(1-pass_c/total)*100:.0f}%")
    print()
