"""
Dose titration: add_direction α=-1 to α=-20, 5 tasks.
Output: per-dose table with collapsed answer text.
Strips think, no auto-classification — raw output for manual judgment.
"""

import sys, os, time, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surgenry.core.client import generate, intervene

MAX_TOKENS = 1024

FIDS = [48490, 44619, 30516, 28894, 12894, 53330, 677, 32701,
        44187, 32000, 63452, 23719, 44704, 62121, 60695, 46062,
        42571, 37931, 10155, 11188, 60881, 45687, 55738]

TASKS = [
    ("Capital", "The capital of France is"),
    ("Algebra", "If 3x + 7 = 22, what is the value of 2x + 5?"),
    ("Clock", "A clock shows 3:15. What is the angle between the hour hand and the minute hand?"),
    ("Water", "You have a 11-liter jug and a 4-liter jug, both empty. Can you measure exactly 8 liters? Answer yes or no and explain how."),
    ("Fibonacci", "Write a function to compute the nth Fibonacci number."),
]

def think_closed(out):
    """Return True if </think> is present."""
    return "</think>" in out


def think_len(out):
    """Return length of the think block, or 0 if none."""
    if "<think>" in out and "</think>" in out:
        return len(out.split("<think>")[-1].split("</think>")[0])
    return 0


def get_visible(out):
    """Get content after </think>; if no close tag, show last part."""
    out = out.strip()
    if '</think>' in out:
        return out.split('</think>')[-1].strip()
    if '<think>' in out:
        # never closed — show the tail (last 150 chars) as evidence of looping
        body = out.split('<think>')[-1]
        return f"[UNCLOSED THINK, len={len(body)}] ...{body[-150:]}"
    return out

def run(positive=False):
    direction = "positive" if positive else "negative"
    sign = +1 if positive else -1

    # Baseline: generate (no SAE)
    print("## Baseline (generate, no SAE)\n", flush=True)
    for tid, prompt in TASKS:
        out = generate(prompt, MAX_TOKENS, 0.3)
        vis = get_visible(out)
        closed = think_closed(out)
        print(f"**{tid}**: closed={closed}", flush=True)
        print(f"  {vis}", flush=True)
        print(flush=True)
    print("---\n", flush=True)

    # α=0 baseline using intervene empty []
    print("## α=0 (add_direction, 23 features, zeroed)\n", flush=True)
    iv0 = [{"layer":20,"feature_id":fid,"action":"add_direction","value":0.0} for fid in FIDS]
    for tid, prompt in TASKS:
        out = intervene(prompt, iv0, MAX_TOKENS, 0.3)
        vis = get_visible(out)
        closed = think_closed(out)
        print(f"**{tid}**: closed={closed}", flush=True)
        print(f"  {vis}", flush=True)
        print(flush=True)
    print("---\n", flush=True)

    # Dose scan
    print(f"## Dose Titration ({direction}, α={sign}1 to α={sign}20)\n")
    row_tpl = "| {:>3} | " + " | ".join(["{}"] * len(TASKS)) + " |"
    print(row_tpl.format("α", *[t[0] for t in TASKS]))
    print("|" + "---|" * (len(TASKS) + 1))

    for i in range(1, 21):
        alpha = sign * i
        print(f"\n=== α={alpha} ===", file=sys.stderr, flush=True)
        iv = [{"layer":20,"feature_id":fid,"action":"add_direction","value":float(alpha)} for fid in FIDS]
        cells = []
        closed_count = 0
        for tid, prompt in TASKS:
            out = intervene(prompt, iv, MAX_TOKENS, 0.3)
            vis = get_visible(out)
            closed = think_closed(out)
            if closed:
                closed_count += 1
            short = vis[:60].replace("\n", " ")
            cells.append(short)
            print(f"  [{alpha}] {tid}: closed={closed}  {short}", file=sys.stderr, flush=True)
            time.sleep(0.3)
        print(f"  >> think closed: {closed_count}/5", file=sys.stderr, flush=True)
        print(row_tpl.format(alpha, *cells))
        sys.stdout.flush()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive", action="store_true", help="Run positive dose titration (α=+1 to +20)")
    args = parser.parse_args()
    run(positive=args.positive)
