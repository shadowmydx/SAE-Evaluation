"""
Hard reasoning prompt generator — produce multi-step reasoning tasks
that cannot be solved by memorization (each instance is randomized).

Outputs JSON arrays compatible with verify_reasoning_overlap.py:
  surgenry/data/prompts/hard_reasoning_code.json
  surgenry/data/prompts/hard_reasoning_pure.json

Usage:
  python3 surgenry/data/prompts/generate_hard_prompts.py --seed 42 --num-per-type 8
"""

import argparse
import json
import random
import os
from typing import Callable


def make_stack_prompts(rng: random.Random, n: int) -> list[str]:
    """Multi-step stack simulation with random push/pop sequences."""
    prompts = []
    for _ in range(n):
        seq = []
        for _ in range(rng.randint(3, 6)):
            seq.append(f"push {rng.randint(1, 99)}")
            if rng.random() < 0.5:
                seq.append("pop")
        seq.append("pop")
        ops = ", ".join(seq)
        prompts.append(f"A stack starts empty. {ops}. What is the stack after these operations?")
    return prompts


def make_insertion_sort_prompts(rng: random.Random, n: int) -> list[str]:
    """Sorting simulation: show array after inserting element k during insertion sort."""
    prompts = []
    for _ in range(n):
        k = rng.randint(2, 4)
        arr = [rng.randint(1, 50) for _ in range(k + 2)]
        after_k = arr[:k]
        after_k.sort()
        remaining = arr[k:]
        final = after_k + remaining
        prompts.append(
            f"Sorting list [{', '.join(str(x) for x in arr)}] using insertion sort. "
            f"After inserting the first {k} elements into the sorted portion, what is the array?"
        )
    return prompts


def make_selection_sort_prompts(rng: random.Random, n: int) -> list[str]:
    """Selection sort: show array after k passes."""
    prompts = []
    for _ in range(n):
        k = rng.randint(1, 3)
        arr = [rng.randint(1, 50) for _ in range(rng.randint(5, 7))]
        prompts.append(
            f"Sorting list [{', '.join(str(x) for x in arr)}] using selection sort. "
            f"After {k} full passes (each pass selects the minimum and swaps it to the front), "
            f"what is the list?"
        )
    return prompts


def make_sequence_induction_prompts(rng: random.Random, n: int) -> list[str]:
    """
    Induce the rule from examples, then compute next term.
    Rules: x*2+1, x*2-1, x+prime_position, x+prev, fib-like, etc.
    """
    prompts = []

    rules: list[Callable[[list[int]], list[int]]] = [
        lambda s: [s[-1] * 2 + 1],                  # 2,5,11,23,? → 47 (2x+1)
        lambda s: [s[-1] * 2 - 1],                  # 3,5,9,17,? → 33 (2x-1)
        lambda s: [s[-1] + len(s) + 2],             # 3,6,10,15,? → 21 (+3,+4,+5...)
        lambda s: [s[-1] + s[-2] + 1],              # 1,2,4,7,? → 12 (a+b+1)
        lambda s: [s[-2] + s[-1]],                  # fib-like: 3,5,8,13,? → 21
        lambda s: [s[-1] * 3 - s[-2]],              # 2,5,13,34,? → 89 (a*3-b)
    ]

    for rule in rules:
        for _ in range(n // len(rules)):
            # Generate seed differently for each rule
            if rule == rules[0]:
                seed = rng.randint(1, 10)
                seq = [seed, seed * 2 + 1]
            elif rule == rules[1]:
                seed = rng.randint(3, 15)
                seq = [seed, seed * 2 - 1]
            elif rule == rules[2]:
                base = rng.randint(1, 5)
                seq = [base, base + 3]
            elif rule == rules[3]:
                a, b = rng.randint(1, 5), rng.randint(2, 6)
                seq = [a, b, a + b + 1]
            elif rule == rules[4]:
                a, b = rng.randint(1, 5), rng.randint(2, 7)
                seq = [a, b, a + b]
            else:
                a, b = rng.randint(1, 5), rng.randint(3, 8)
                seq = [a, b, b * 3 - a]
            # Extend to 4 terms
            while len(seq) < 4:
                nxt = rule(seq)
                seq.extend(nxt)
            shown = seq[:4]
            prompts.append(
                f"Sequence: {', '.join(str(x) for x in shown)}, ?. "
                f"What is the next number in this sequence?"
            )
    return prompts


def make_graph_path_prompts(rng: random.Random, n: int) -> list[str]:
    """Random graph with edges, find shortest path length or specific path."""
    prompts = []
    for _ in range(n):
        nodes = ["A", "B", "C", "D", "E"]
        rng.shuffle(nodes)
        # Build random DAG
        edges = []
        for i, src in enumerate(nodes[:-1]):
            for dst in nodes[i + 1:]:
                if rng.random() < 0.4:
                    if rng.random() < 0.4:
                        weight = rng.randint(1, 9)
                        edges.append((src, dst, weight))
                    else:
                        edges.append((src, dst, 1))
        if len(edges) < 3:
            edges = [(nodes[0], nodes[1], 1), (nodes[1], nodes[2], 1), (nodes[0], nodes[2], 2)]
        src, dst = nodes[0], nodes[-1]
        edge_str = ", ".join(f"{s}→{d}" for s, d, _ in edges)
        prompts.append(
            f"A graph has nodes with edges: {edge_str}. "
            f"What is the number of edges on the shortest path from {src} to {dst}?"
        )
    return prompts


def make_multi_step_arithmetic_prompts(rng: random.Random, n: int) -> list[str]:
    """Multi-step arithmetic where intermediate results must be maintained."""
    prompts = []
    for _ in range(n // 2):
        a, b, c = rng.randint(10, 99), rng.randint(5, 20), rng.randint(2, 9)
        prompts.append(
            f"Start with {a}. Add {b}, then divide by {c}, then multiply by 3, then subtract {b // 2}. "
            f"What is the final result?"
        )
    for _ in range(n // 2):
        a, b, c = rng.randint(2, 9), rng.randint(10, 99), rng.randint(1, 5)
        prompts.append(
            f"Compute: ({a} × {b} + {c}) ÷ {max(a, 2)} − {c}. "
            f"What is the result?"
        )
    return prompts


def make_water_pour_prompts(rng: random.Random, n: int) -> list[str]:
    """Water pouring / jug problem reasoning (pure reasoning, no code concepts)."""
    prompts = []
    for _ in range(n):
        big = rng.randint(8, 15)
        small = rng.randint(3, big - 2)
        target = rng.randint(1, big)
        prompts.append(
            f"You have a {big}-liter jug and a {small}-liter jug, both empty. "
            f"You can fill a jug completely, empty a jug completely, "
            f"or pour water from one jug to the other until one is full or empty. "
            f"Can you measure exactly {target} liters? Answer yes or no and explain how."
        )
    return prompts


def make_scheduling_prompts(rng: random.Random, n: int) -> list[str]:
    """Scheduling / constraint satisfaction (pure reasoning)."""
    prompts = []
    people = ["Alice", "Bob", "Carol", "Dave"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    for _ in range(n):
        rng.shuffle(people)
        rng.shuffle(days)
        p1, p2, p3 = people[:3]
        d1, d2, d3 = days[:3]
        # Build constraint
        constraint = rng.choice([
            f"{p1} must meet before {p2}, and {p2} must meet before {p3}",
            f"{p1} cannot meet on {d1}, {p2} can only meet on {d2} or {d3}",
            f"{p1} meets on {d1}, {p2} meets 2 days after {p1}, {p3} meets 1 day after {p2}",
        ])
        prompts.append(
            f"Three people {p1}, {p2}, {p3} each need to schedule a meeting on different days. "
            f"Constraint: {constraint}. "
            f"What day does each person meet on?"
        )
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Generate hard reasoning prompts with randomization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num-per-type", type=int, default=8, help="Prompts per reasoning type")
    parser.add_argument("--output-dir", type=str, default=os.path.dirname(os.path.abspath(__file__)),
                        help="Output directory for JSON files")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Code-related reasoning categories (share concepts with code generation)
    code_reasoning_types = {
        "stack_trace": make_stack_prompts,
        "insertion_sort": make_insertion_sort_prompts,
        "selection_sort": make_selection_sort_prompts,
        "sequence_induction": make_sequence_induction_prompts,
        "graph_path": make_graph_path_prompts,
        "multi_step_arithmetic": make_multi_step_arithmetic_prompts,
    }

    # Pure reasoning categories (no code concept overlap)
    pure_reasoning_types = {
        "water_pour": make_water_pour_prompts,
        "scheduling": make_scheduling_prompts,
        "multi_step_arithmetic": make_multi_step_arithmetic_prompts,  # same arithmetic but framed as word problem
    }

    hard_code = []
    hard_pure = []

    print(f"Generating hard reasoning prompts (seed={args.seed}, num_per_type={args.num_per_type})")
    print()

    for name, maker_fn in code_reasoning_types.items():
        generated = maker_fn(rng, args.num_per_type)
        hard_code.extend(generated)
        print(f"  {name}: {len(generated)} prompts")
        for p in generated[:2]:
            print(f"    → {p[:80]}...")

    for name, maker_fn in pure_reasoning_types.items():
        generated = maker_fn(rng, args.num_per_type)
        hard_pure.extend(generated)
        print(f"  {name} (pure): {len(generated)} prompts")
        for p in generated[:2]:
            print(f"    → {p[:80]}...")

    # Write code-related reasoning
    code_path = os.path.join(args.output_dir, "hard_reasoning_code.json")
    with open(code_path, "w") as f:
        json.dump(hard_code, f, indent=2)
    print(f"\nCode-related: {len(hard_code)} prompts → {code_path}")

    # Write pure reasoning
    pure_path = os.path.join(args.output_dir, "hard_reasoning_pure.json")
    with open(pure_path, "w") as f:
        json.dump(hard_pure, f, indent=2)
    print(f"Pure reasoning: {len(hard_pure)} prompts → {pure_path}")


if __name__ == "__main__":
    main()
