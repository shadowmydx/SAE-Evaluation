"""
Pure business logic for the 3-step experiment pipeline.
Each function takes typed data and returns typed data — no HTTP, no argparse, no print.
"""

import json
from collections import defaultdict
from .models import (
    ExperimentConfig,
    DiscoveredFeatures,
    serialize_discovered,
    RankedFeature,
    RankedSet,
    RankedFeatures,
    Intervention,
)
from .client import sae_scan, scan_activation


def discover_differential(config: ExperimentConfig) -> dict[int, DiscoveredFeatures]:
    """
    Step 1: Find features unique to group A vs group B (and shared).
    Uses seed prompts from the config.
    Returns per-layer DiscoveredFeatures objects.
    """
    body_a = {
        "prompt": config.seed_prompt_a,
        "layers": config.layers,
        "token_position": -1,
        "max_features": 100,
    }
    body_b = {
        "prompt": config.seed_prompt_b,
        "layers": config.layers,
        "token_position": -1,
        "max_features": 100,
    }
    ra = sae_scan(config.seed_prompt_a, config.layers)
    rb = sae_scan(config.seed_prompt_b, config.layers)

    result = {}
    for layer in config.layers:
        ls = str(layer)
        a_ids = {feat["feature_id"] for feat in ra["layers"][ls]["top_features"]}
        b_ids = {feat["feature_id"] for feat in rb["layers"][ls]["top_features"]}
        result[layer] = DiscoveredFeatures(
            layer=layer,
            a_only=sorted(a_ids - b_ids),
            b_only=sorted(b_ids - a_ids),
            shared=sorted(a_ids & b_ids),
        )
    return result


def rank_by_frequency(
    config: ExperimentConfig,
    discovered: dict[int, DiscoveredFeatures],
    num_prompts: int = 50,
    topk: int = 10,
) -> RankedFeatures:
    """
    Step 2: Rank candidate features by cross-prompt frequency.
    Candidate set A = a_only features; candidate set B = b_only features.
    Scans num_prompts prompts from each side, counts occurrences.
    """
    prompts_a = config.load_prompts_a()[:num_prompts]
    prompts_b = config.load_prompts_b()[:num_prompts]

    # Build candidate sets per layer
    c1_fids: dict[int, set[int]] = {}
    c2_fids: dict[int, set[int]] = {}
    for layer in config.layers:
        df = discovered[layer]
        c1_fids[layer] = df.candidate_a()
        c2_fids[layer] = df.candidate_b()

    c1_counts = _scan_group(prompts_a, config.layers, c1_fids, config.group_a)
    c2_counts = _scan_group(prompts_b, config.layers, c2_fids, config.group_b)

    # Build typed results
    set_a = RankedSet(group_label=config.group_a)
    set_b = RankedSet(group_label=config.group_b)
    for layer in config.layers:
        sorted_c1 = sorted(c1_counts[layer].items(), key=lambda x: -x[1])
        sorted_c2 = sorted(c2_counts[layer].items(), key=lambda x: -x[1])
        set_a.by_layer[layer] = [
            RankedFeature(fid, cnt, round(cnt / num_prompts, 3))
            for fid, cnt in sorted_c1[:topk]
        ]
        set_b.by_layer[layer] = [
            RankedFeature(fid, cnt, round(cnt / num_prompts, 3))
            for fid, cnt in sorted_c2[:topk]
        ]

    return RankedFeatures(info=config, set_a=set_a, set_b=set_b)


def _scan_group(
    prompts: list[str],
    layers: list[int],
    candidate_fids: dict[int, set[int]],
    label: str,
) -> dict[int, dict[int, int]]:
    """Scan all prompts in a group and count candidate feature occurrences."""
    layer_counts: dict[int, dict[int, int]] = {l: defaultdict(int) for l in layers}
    n = len(prompts)
    for i, prompt in enumerate(prompts):
        print(f"  [{i+1}/{n}] scanning {label}...", end="\r")
        import sys as _sys
        _sys.stdout.flush()
        try:
            data = sae_scan(prompt, layers)
        except Exception as e:
            print(f"\n  Error on prompt {i}: {e}")
            continue
        for ls, entry in data["layers"].items():
            layer = int(ls)
            if layer not in layers:
                continue
            active_fids = {feat["feature_id"] for feat in entry["top_features"]}
            for fid in candidate_fids.get(layer, set()):
                if fid in active_fids:
                    layer_counts[layer][fid] += 1
    print()
    return layer_counts


def prepare_negate_interventions(
    ranked: RankedFeatures,
    target_side: str,
    layer: int,
    topk: int = 3,
) -> list[Intervention]:
    """
    Prepare negate interventions for top-K features of a side.
    target_side: "a" or "b".
    """
    source = ranked.set_a if target_side == "a" else ranked.set_b
    features = source.top(layer, topk)
    return [
        Intervention(layer=layer, feature_id=f.feature_id, action="negate", value=0.0)
        for f in features
    ]


def prepare_inject_interventions(
    ranked: RankedFeatures,
    source_side: str,
    layer: int,
    topk: int = 3,
) -> list[Intervention]:
    """
    Prepare inject (set) interventions using activation values from the source side's seed prompt.
    source_side: "a" or "b" — whose features to inject.
    """
    source = ranked.set_a if source_side == "a" else ranked.set_b
    seed_prompt = ranked.info.seed_prompt_a if source_side == "a" else ranked.info.seed_prompt_b
    features = source.top(layer, topk)
    fids = [f.feature_id for f in features]

    # Scan activation on the source seed prompt
    act_values = scan_activation(seed_prompt, layer, fids)

    interventions = []
    for f in features:
        val = act_values.get(f.feature_id, 0.0)
        if val == 0.0:
            continue
        interventions.append(
            Intervention(layer=layer, feature_id=f.feature_id, action="set", value=val)
        )
    return interventions


# ── Overlap analysis ────────────────────────────────────────────────────────

def collect_feature_union(
    prompts: list[str],
    layers: list[int],
    max_prompts: int = 30,
    label: str = "",
) -> dict[int, set[int]]:
    """
    Scan prompts and collect the union of all top-100 feature IDs per layer.

    Returns {layer: {fid, ...}}.
    """
    result: dict[int, set[int]] = {l: set() for l in layers}
    n = min(max_prompts, len(prompts))
    for i, prompt in enumerate(prompts[:max_prompts]):
        if label:
            print(f"  [{i+1}/{n}] scanning {label}...", end="\r")
            import sys as _sys
            _sys.stdout.flush()
        try:
            data = sae_scan(prompt, layers)
        except Exception as e:
            print(f"\n  Error on prompt {i}: {e}")
            continue
        for ls, entry in data["layers"].items():
            layer = int(ls)
            if layer not in layers:
                continue
            for feat in entry["top_features"]:
                result[layer].add(feat["feature_id"])
    if label:
        print()
    return result


def compute_multi_overlap(
    target_prompts: list[str],
    layers: list[int],
    named_sets: dict[str, dict[int, set[int]]],
    prompt_type_label: str = "",
    output_file: str = "",
) -> dict[int, dict]:
    """
    Scan target prompts and compute overlap with multiple named feature sets.

    named_sets: {"set_name": {layer: {fid, ...}}, ...}

    Returns {layer: {set_name: [overlap_counts_per_prompt], "set_sizes": {set_name: size}}}
    """
    # Pre-compute set sizes per layer
    set_sizes: dict[int, dict[str, int]] = {}
    for layer in layers:
        set_sizes[layer] = {}
        for set_name, fset in named_sets.items():
            set_sizes[layer][set_name] = len(fset.get(layer, set()))

    # Overlap accumulators per layer per set
    overlaps: dict[int, dict[str, list[int]]] = {}
    for layer in layers:
        overlaps[layer] = {sn: [] for sn in named_sets}

    n = len(target_prompts)
    for i, prompt in enumerate(target_prompts):
        if prompt_type_label:
            print(f"  [{i+1}/{n}] scanning {prompt_type_label}...", end="\r")
            import sys as _sys
            _sys.stdout.flush()
        try:
            data = sae_scan(prompt, layers)
        except Exception as e:
            print(f"\n  Error: {e}")
            continue
        for ls, entry in data["layers"].items():
            layer = int(ls)
            if layer not in layers:
                continue
            active_fids = {feat["feature_id"] for feat in entry["top_features"]}
            for set_name, fset in named_sets.items():
                cnt = len(active_fids & fset.get(layer, set()))
                overlaps[layer][set_name].append(cnt)
    if prompt_type_label:
        print()

    # Build output
    result = {}
    for layer in layers:
        result[layer] = {"set_sizes": set_sizes[layer], "per_prompt": {}}
        for set_name in named_sets:
            result[layer]["per_prompt"][set_name] = overlaps[layer][set_name]

    if output_file:
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Overlap details saved to {output_file}")

    return result
