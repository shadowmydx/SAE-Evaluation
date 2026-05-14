"""
Evaluate model output degradation on GSM8K and HumanEval benchmarks
under SAE feature intervention.

We do NOT check correctness. We check for reasoning process degradation:
- Empty / truncated output
- Repetition / stuttering
- Gibberish / low token diversity
- Looping behavior

Usage:
  # Specific benchmarks
  python3 surgenry/eval_benchmark.py --gsm8k --both-modes --max-samples 20
  python3 surgenry/eval_benchmark.py --humaneval --intervene
  python3 surgenry/eval_benchmark.py --bbh --one-shot --intervene

  # Default: run all 3 benchmarks, baseline + intervention
  python3 surgenry/eval_benchmark.py --max-samples 50

  # Intervention only (saves time)
  python3 surgenry/eval_benchmark.py --intervene

Default intervention: L20 + add_direction α=-20, 23 code-tilted A∩D features (fids in CODE_TILTED_FIDS).
"""

import json
import sys
import os
import time
import re
import math
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surgenry.core.client import generate, intervene

# ── Config ──────────────────────────────────────────────────────────────

CODE_TILTED_FIDS = [48490, 44619, 30516, 28894, 12894, 53330, 677, 32701,
                    44187, 32000, 63452, 23719, 44704, 62121, 60695, 46062,
                    42571, 37931, 10155, 11188, 60881, 45687, 55738]

INTERVENTION = [
    {"layer": 20, "feature_id": fid, "action": "add_direction", "value": -20.0}
    for fid in CODE_TILTED_FIDS
]

MAX_TOKENS = 300
TEMPERATURE = 0.3


# ── Degradation Detection ───────────────────────────────────────────────

def detect_degradation(text: str) -> dict:
    """Analyze output for signs of degradation."""
    result = {"degraded": False, "reason": "", "metrics": {}}
    if not text or not text.strip():
        result["degraded"] = True
        result["reason"] = "empty_output"
        return result

    words = text.split()
    chars = text.strip()
    n_chars = len(chars)
    n_words = len(words)

    # 1. Very short output
    if n_chars < 10:
        result["degraded"] = True
        result["reason"] = "too_short"
        return result

    # 2. Stuttering: character-level repetition
    stutter_chars = set()
    i = 0
    while i < len(chars):
        j = i
        while j < len(chars) and chars[j] == chars[i]:
            j += 1
        if j - i >= 4:
            stutter_chars.add(chars[i])
        i = j
    stutter_ratio = sum(len(c) for c in stutter_chars) / max(n_chars, 1)

    # 3. Word diversity
    unique_words = set(w.lower() for w in words)
    word_diversity = len(unique_words) / max(len(words), 1)

    # 4. N-gram repetition
    bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
    trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]

    def repetition_coefficient(ngrams):
        if not ngrams:
            return 0.0
        total = len(ngrams)
        unique = len(set(ngrams))
        return unique / max(total, 1)

    bigram_div = repetition_coefficient(bigrams)
    trigram_div = repetition_coefficient(trigrams)

    # 5. Looping: detect repeating large block
    looping = False
    if len(words) >= 20:
        half = len(words) // 2
        first_half = ' '.join(words[:half])
        second_half = ' '.join(words[half:half+half])
        overlap = len(set(first_half.lower().split()) & set(second_half.lower().split()))
        if len(set(second_half.lower().split())) > 0:
            overlap_ratio = overlap / max(len(set(second_half.lower().split())), 1)
            if overlap_ratio > 0.7 and len(words) >= 40:
                looping = True

    # 6. Top token clustering
    top_5_tokens = Counter(w.lower() for w in words).most_common(5)
    top_5_total = sum(c for _, c in top_5_tokens)
    top_5_ratio = top_5_total / max(len(words), 1)

    metrics = {
        "n_chars": n_chars,
        "n_words": n_words,
        "word_diversity": round(word_diversity, 4),
        "bigram_diversity": round(bigram_div, 4),
        "trigram_diversity": round(trigram_div, 4),
        "stutter_ratio": round(stutter_ratio, 4),
        "top5_ratio": round(top_5_ratio, 4),
        "looping": looping,
    }
    result["metrics"] = metrics

    reasons = []
    if stutter_ratio > 0.3:
        reasons.append("stuttering")
    if bigram_div < 0.3:
        reasons.append("bigram_loop")
    if trigram_div < 0.2:
        reasons.append("trigram_loop")
    if word_diversity < 0.2 and n_words >= 15:
        reasons.append("low_diversity")
    if top_5_ratio > 0.8 and n_words >= 15:
        reasons.append("token_clustering")
    if looping:
        reasons.append("looping")

    alpha_chars = sum(1 for c in text if c.isalnum())
    if alpha_chars / max(n_chars, 1) < 0.3 and n_chars > 10:
        reasons.append("mostly_punctuation")

    # Known crash patterns: short fragments that end in a line break / "\n" only
    if re.match(r'^\s*\n+\s*$', text):
        reasons.append("blank_lines_only")

    if reasons:
        result["degraded"] = True
        result["reason"] = "+".join(reasons)

    return result


# ── Data Loading ────────────────────────────────────────────────────────

def load_gsm8k() -> list[dict]:
    """Load GSM8K test set from local arrow cache."""
    import pyarrow as pa
    path = ("/home/shadowmydx/.cache/modelscope/hub/datasets/"
            "modelscope___gsm8k/main/1.1.0/"
            "9bba91b8d001ff705b591a67e4783fd12b387d20c4ce7fd719351e07f488382e/"
            "gsm8k-test.arrow")
    with open(path, 'rb') as f:
        reader = pa.ipc.open_stream(f)
        table = reader.read_all()
    samples = []
    for i in range(table.num_rows):
        samples.append({
            "id": f"gsm8k_{i}",
            "prompt": table.column("question")[i].as_py(),
            "reference": table.column("answer")[i].as_py(),
        })
    return samples


def load_humaneval() -> list[dict]:
    """Load HumanEval from cached gzipped JSONL."""
    import gzip
    path = ("/home/shadowmydx/.cache/modelscope/hub/datasets/downloads/"
            "6446f27f3bdeb99626030cbb73b87450601708115f4bde4292d233e14041d297")
    with gzip.open(path, 'rt') as f:
        raw = [json.loads(line) for line in f if line.strip()]
    samples = []
    for item in raw:
        samples.append({
            "id": item["task_id"],
            "prompt": item["prompt"],
            "entry_point": item.get("entry_point", ""),
            "reference": item.get("canonical_solution", ""),
        })
    return samples


def load_bbh(max_per_task: int = 0) -> list[dict]:
    """Load BBH from local HF datasets cache. One-shot tasks only."""
    import os
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from datasets import load_dataset

    CONFIGS = ['boolean_expressions', 'causal_judgement', 'date_understanding',
               'disambiguation_qa', 'dyck_languages', 'formal_fallacies',
               'geometric_shapes', 'hyperbaton', 'logical_deduction_five_objects',
               'logical_deduction_seven_objects', 'logical_deduction_three_objects',
               'movie_recommendation', 'multistep_arithmetic_two', 'navigate',
               'object_counting', 'penguins_in_a_table', 'reasoning_about_colored_objects',
               'ruin_names', 'salient_translation_error_detection', 'snarks',
               'sports_understanding', 'temporal_sequences',
               'tracking_shuffled_objects_five_objects',
               'tracking_shuffled_objects_seven_objects',
               'tracking_shuffled_objects_three_objects', 'web_of_lies', 'word_sorting']

    samples = []
    for cfg in CONFIGS:
        ds = load_dataset('lukaemon/bbh', cfg, split='test')
        task_samples = min(max_per_task, len(ds)) if max_per_task else len(ds)
        for i in range(task_samples):
            samples.append({
                "id": f"bbh_{cfg}_{i}",
                "prompt": ds[i]["input"],
                "task": cfg,
                "reference": ds[i]["target"],
            })
    return samples


# ── Prompt Formatting ───────────────────────────────────────────────────

def format_gsm8k_prompt(question: str) -> str:
    return f"Question: {question}\nLet's work through this step by step:\n"


def format_humaneval_prompt(prompt: str) -> str:
    return prompt


def format_bbh_prompt(input_text: str) -> str:
    """BBH is already a self-contained question. Just add a gentle reasoning prompt."""
    return f"Question: {input_text}\nLet's work through this step by step:\n"


# ── Evaluation Runner ───────────────────────────────────────────────────

def evaluate_sample(prompt: str, use_intervention: bool,
                    max_tokens: int = MAX_TOKENS,
                    temperature: float = TEMPERATURE) -> dict:
    """Run one sample and return output + degradation analysis."""
    start = time.time()
    try:
        if use_intervention:
            output = intervene(prompt, INTERVENTION, max_tokens, temperature)
        else:
            output = generate(prompt, max_tokens, temperature)
        elapsed = time.time() - start
    except Exception as e:
        return {
            "output": "",
            "error": str(e),
            "elapsed": time.time() - start,
            "degraded": True,
            "degradation_reason": f"exception: {e}",
            "metrics": {},
        }

    analysis = detect_degradation(output)
    return {
        "output": output,
        "error": None,
        "elapsed": elapsed,
        "degraded": analysis["degraded"],
        "degradation_reason": analysis["reason"],
        "metrics": analysis["metrics"],
    }


def run_benchmark(samples: list[dict], use_intervention: bool,
                  max_samples: int = 0, label: str = "") -> dict:
    """Run benchmark on a list of samples."""
    if max_samples and max_samples < len(samples):
        samples = samples[:max_samples]

    results = []
    degraded_count = 0
    total_elapsed = 0.0
    degradation_reasons = Counter()

    mode = "intervention" if use_intervention else "baseline"
    print(f"\n{'='*60}")
    print(f"  {label} — {mode} ({len(samples)} samples)")
    print(f"{'='*60}")

    for idx, sample in enumerate(samples):
        if label == "GSM8K":
            prompt = format_gsm8k_prompt(sample["prompt"])
        elif label == "BBH" or sample.get("task", ""):
            prompt = format_bbh_prompt(sample["prompt"])
        else:
            prompt = format_humaneval_prompt(sample["prompt"])

        result = evaluate_sample(prompt, use_intervention)

        degraded_mark = "DEGRADED" if result["degraded"] else "OK"
        print(f"  [{idx+1}/{len(samples)}] {sample['id']} [{degraded_mark}] "
              f"({result['elapsed']:.1f}s)"
              + (f" [{result['degradation_reason']}]" if result['degraded'] else ""))

        if result["degraded"] and result["output"]:
            output_preview = result["output"][:80].replace('\n', '\\n')
            print(f"          output: \"{output_preview}...\"")

        entry = {"id": sample["id"], "prompt": prompt, **result}
        results.append(entry)

        if result["degraded"]:
            degraded_count += 1
            if result["degradation_reason"]:
                degradation_reasons[result["degradation_reason"]] += 1
        total_elapsed += result["elapsed"]

    total = len(samples)
    pass_count = total - degraded_count
    print(f"\n  ── {label} {mode} summary ──")
    print(f"  Total: {total} | Pass: {pass_count} ({pass_count/total*100:.1f}%) "
          f"| Degraded: {degraded_count} ({degraded_count/total*100:.1f}%)")
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/total:.1f}s/sample)")

    if degradation_reasons:
        print(f"  Degradation breakdown:")
        for reason, count in degradation_reasons.most_common():
            print(f"    {reason}: {count} ({count/degraded_count*100:.1f}%)")

    return {
        "label": label,
        "mode": mode,
        "total": total,
        "passed": pass_count,
        "degraded": degraded_count,
        "pass_rate": pass_count / total * 100 if total > 0 else 0,
        "degradation_rate": degraded_count / total * 100 if total > 0 else 0,
        "total_time": total_elapsed,
        "avg_time_per_sample": total_elapsed / total if total > 0 else 0,
        "degradation_reasons": dict(degradation_reasons),
        "results": results,
    }


# ── Main ────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark eval for degradation")
    parser.add_argument("--gsm8k", action="store_true", help="Run GSM8K")
    parser.add_argument("--humaneval", action="store_true", help="Run HumanEval")
    parser.add_argument("--bbh", action="store_true", help="Run BBH")
    parser.add_argument("--all-benchmarks", action="store_true", help="Run all three benchmarks")
    parser.add_argument("--baseline", action="store_true", help="Run baseline (no intervention)")
    parser.add_argument("--intervene", action="store_true", help="Run with intervention")
    parser.add_argument("--both-modes", action="store_true", help="Run both baseline and intervention")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples per benchmark (default: all)")
    parser.add_argument("--bbh-per-task", type=int, default=10,
                        help="Max samples per BBH task (default: 10, use 0 for all)")
    parser.add_argument("--output", type=str, default="",
                        help="Output JSON file for aggregated results")
    args = parser.parse_args()

    # Determine which benchmarks to run
    no_benchmark_specified = not (args.gsm8k or args.humaneval or args.bbh or args.all_benchmarks)
    run_gsm = args.gsm8k or args.all_benchmarks or no_benchmark_specified
    run_he = args.humaneval or args.all_benchmarks or no_benchmark_specified
    run_bbh = args.bbh or args.all_benchmarks

    run_baseline = args.baseline or args.both_modes
    run_intervene = args.intervene or args.both_modes
    if not run_baseline and not run_intervene:
        run_baseline = run_intervene = True

    if run_gsm:
        gsm_samples = load_gsm8k()
        print(f"Loaded GSM8K: {len(gsm_samples)} samples (test)")
    if run_he:
        he_samples = load_humaneval()
        print(f"Loaded HumanEval: {len(he_samples)} samples")
    if run_bbh:
        bbh_per_task = args.bbh_per_task if not args.max_samples else args.max_samples
        bbh_samples = load_bbh(max_per_task=bbh_per_task)
        print(f"Loaded BBH: {len(bbh_samples)} samples ({bbh_per_task}/task)")

    all_reports = []

    if run_gsm:
        if run_baseline:
            all_reports.append(run_benchmark(gsm_samples, False, args.max_samples, "GSM8K"))
        if run_intervene:
            all_reports.append(run_benchmark(gsm_samples, True, args.max_samples, "GSM8K"))

    if run_he:
        if run_baseline:
            all_reports.append(run_benchmark(he_samples, False, args.max_samples, "HumanEval"))
        if run_intervene:
            all_reports.append(run_benchmark(he_samples, True, args.max_samples, "HumanEval"))

    if run_bbh:
        if run_baseline:
            all_reports.append(run_benchmark(bbh_samples, False, 0, "BBH"))
        if run_intervene:
            all_reports.append(run_benchmark(bbh_samples, True, 0, "BBH"))

    # Summary
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Benchmark':<15} {'Mode':<15} {'Pass':>6} {'Degraded':>10} {'Degrade%':>9}")
    print(f"  {'-'*55}")
    for r in all_reports:
        print(f"  {r['label']:<15} {r['mode']:<15} {r['passed']:>6} "
              f"{r['degraded']:>10} {r['degradation_rate']:>8.1f}%")
        if r['degradation_reasons']:
            top_reason = max(r['degradation_reasons'], key=r['degradation_reasons'].get)
            print(f"  {'':15} {'':15} top: {top_reason}")

    if args.output:
        clean = [{k: v for k, v in r.items() if k != 'results'} for r in all_reports]
        with open(args.output, 'w') as f:
            json.dump(clean, f, indent=2)
        print(f"\n  Results saved to {args.output}")

    detailed_path = f"eval_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(detailed_path, 'w') as f:
        json.dump(all_reports, f, indent=2)
    print(f"  Detailed results saved to {detailed_path}")


if __name__ == "__main__":
    main()
