"""
HotpotQA-style multi-hop QA: baseline vs α=-20 intervention.
30 two-hop questions, each needs two facts integrated.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surgenry.core.client import generate, intervene

FIDS = [48490, 44619, 30516, 28894, 12894, 53330, 677, 32701,
        44187, 32000, 63452, 23719, 44704, 62121, 60695, 46062,
        42571, 37931, 10155, 11188, 60881, 45687, 55738]

IV = [{"layer": 20, "feature_id": fid, "action": "add_direction", "value": -20.0}
      for fid in FIDS]

with open("surgenry/data/prompts/hotpotqa.json") as f:
    questions = json.load(f)

MAX_TOKENS = 200
TEMPERATURE = 0.3

def answer_in_output(answer: str, output: str) -> bool:
    """Check if the expected answer appears in the generated output."""
    return answer.lower().strip() in output.lower()

print(f"Testing {len(questions)} multi-hop QA questions\n", flush=True)

results = []
for i, q in enumerate(questions):
    prompt = q["question"]
    answer = q["answer"]

    # Baseline
    out_bl = generate(prompt, MAX_TOKENS, TEMPERATURE)
    has_bl = answer_in_output(answer, out_bl)
    vis_bl = out_bl.strip().split("</think>")[-1].strip() if "</think>" in out_bl else out_bl[:120]

    # Intervene
    out_iv = intervene(prompt, IV, MAX_TOKENS, TEMPERATURE)
    has_iv = answer_in_output(answer, out_iv)
    vis_iv = out_iv.strip().split("</think>")[-1].strip() if "</think>" in out_iv else out_iv[:120]

    correct_bl = "✓" if has_bl else "✗"
    correct_iv = "✓" if has_iv else "✗"

    results.append({"question": prompt, "answer": answer,
                    "baseline_correct": has_bl, "intervene_correct": has_iv})

    print(f"[{i+1}/30] {correct_bl}→{correct_iv} | {prompt[:50]}...", flush=True)
    print(f"  BL: {vis_bl[:80]}")
    print(f"  IV: {vis_iv[:80]}")
    print()
    time.sleep(0.3)

# Summary
bl_correct = sum(1 for r in results if r["baseline_correct"])
iv_correct = sum(1 for r in results if r["intervene_correct"])
print("=" * 60)
print(f"BASELINE: {bl_correct}/{len(results)} correct ({100*bl_correct/len(results):.0f}%)")
print(f"INTERVENE: {iv_correct}/{len(results)} correct ({100*iv_correct/len(results):.0f}%)")
print(f"DEGRADATION: {bl_correct - iv_correct}/{len(results)} ({(bl_correct-iv_correct)/len(results)*100:.0f}%)")

# Detailed failures
print("\nCases where answer present in baseline but NOT in intervene:")
for r in results:
    if r["baseline_correct"] and not r["intervene_correct"]:
        print(f"  Q: {r['question']}  A: {r['answer']}")
