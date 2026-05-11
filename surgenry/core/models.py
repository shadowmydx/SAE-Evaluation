"""
Typed data structures for SAE interpretability experiments.
All serialization/deserialization lives here so CLIs don't touch JSON directly.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


# ── Config ──────────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """High-level experiment definition. This is the only thing a new experiment needs to write."""
    name: str                      # "france_vs_china", "safety_vs_capability", ...
    group_a: str                   # human-readable label, e.g. "france-related"
    group_b: str                   # human-readable label, e.g. "china-related"
    seed_prompt_a: str             # seed prompt used in Step 1 (differential discovery)
    seed_prompt_b: str             # seed prompt used in Step 1
    prompts_file_a: str            # path to JSON array of 50 prompts for frequency scan
    prompts_file_b: str            # path to JSON array of 50 prompts for frequency scan
    layers: list[int] = field(default_factory=lambda: [24, 28, 31])

    def load_prompts_a(self) -> list[str]:
        with open(self.prompts_file_a) as f:
            return json.load(f)

    def load_prompts_b(self) -> list[str]:
        with open(self.prompts_file_b) as f:
            return json.load(f)


# ── Step 1: discovered features ────────────────────────────────────────────

@dataclass
class DiscoveredFeatures:
    """Per-layer sets from Step 1."""
    layer: int
    a_only: list[int]       # feature ids present in group A but not B
    b_only: list[int]       # feature ids present in group B but not A
    shared: list[int]       # feature ids present in both

    def to_dict(self) -> dict:
        return {"a_only": self.a_only, "b_only": self.b_only, "shared": self.shared}

    @classmethod
    def from_dict(cls, layer: int, d: dict) -> "DiscoveredFeatures":
        return cls(layer=layer, a_only=d["a_only"], b_only=d["b_only"], shared=d["shared"])

    def candidate_a(self) -> set[int]:
        return set(self.a_only)

    def candidate_b(self) -> set[int]:
        return set(self.b_only)


def serialize_discovered(layer_map: dict[int, DiscoveredFeatures]) -> dict:
    """Serialize to the JSON-compatible format expected by CLI pipelines."""
    return {str(l): v.to_dict() for l, v in layer_map.items()}


def deserialize_discovered(data: dict) -> dict[int, DiscoveredFeatures]:
    """Load from the JSON format produced by Step 1."""
    return {int(k): DiscoveredFeatures.from_dict(int(k), v) for k, v in data.items()}


# ── Step 2: ranked features ────────────────────────────────────────────────

@dataclass
class FeatureInfo:
    """A single feature's raw info from the SAE endpoint."""
    feature_id: int
    activation: float = 0.0


@dataclass
class RankedFeature:
    """A feature with its cross-prompt frequency count."""
    feature_id: int
    count: int
    frequency: float  # ratio count / total_prompts

    def to_dict(self) -> dict:
        return {"feature_id": self.feature_id, "count": self.count, "frequency": self.frequency}

    @classmethod
    def from_dict(cls, d: dict) -> "RankedFeature":
        return cls(feature_id=d["feature_id"], count=d["count"], frequency=d["frequency"])


@dataclass
class RankedSet:
    """Ranked features for one side (group A / group B) across layers."""
    group_label: str
    by_layer: dict[int, list[RankedFeature]] = field(default_factory=dict)

    def top(self, layer: int, k: int = 3) -> list[RankedFeature]:
        return self.by_layer.get(layer, [])[:k]

    def to_dict(self) -> dict:
        return {
            "group_label": self.group_label,
            "by_layer": {str(l): [f.to_dict() for f in flist] for l, flist in self.by_layer.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RankedSet":
        return cls(
            group_label=d.get("group_label", ""),
            by_layer={int(k): [RankedFeature.from_dict(f) for f in v] for k, v in d["by_layer"].items()},
        )


@dataclass
class RankedFeatures:
    """Complete output of Step 2, wrapping both ranked sets."""
    info: ExperimentConfig
    set_a: RankedSet
    set_b: RankedSet

    def to_dict(self) -> dict:
        return {
            "info": {
                "name": self.info.name,
                "group_a": self.info.group_a,
                "group_b": self.info.group_b,
                "seed_prompt_a": self.info.seed_prompt_a,
                "seed_prompt_b": self.info.seed_prompt_b,
                "layers": self.info.layers,
            },
            "set_a": self.set_a.to_dict(),
            "set_b": self.set_b.to_dict(),
        }

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "RankedFeatures":
        with open(path) as f:
            data = json.load(f)
        info_data = data["info"]
        config = ExperimentConfig(
            name=info_data["name"],
            group_a=info_data["group_a"],
            group_b=info_data["group_b"],
            seed_prompt_a=info_data["seed_prompt_a"],
            seed_prompt_b=info_data["seed_prompt_b"],
            layers=info_data["layers"],
            prompts_file_a="",
            prompts_file_b="",
        )
        return cls(
            info=config,
            set_a=RankedSet.from_dict(data["set_a"]),
            set_b=RankedSet.from_dict(data["set_b"]),
        )


# ── Intervention ────────────────────────────────────────────────────────────

@dataclass
class Intervention:
    """A single SAE feature intervention spec."""
    layer: int
    feature_id: int
    action: str     # "zero" | "scale" | "set" | "clamp_max" | "negate"
    value: float = 0.0

    def to_dict(self) -> dict:
        return {"layer": self.layer, "feature_id": self.feature_id,
                "action": self.action, "value": self.value}
