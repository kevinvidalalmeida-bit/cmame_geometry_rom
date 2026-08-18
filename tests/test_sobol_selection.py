"""Deterministic design checks for the fixed full-rank Sobol pipeline."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "scripts" / "cmame_interpretable_pipeline"
for path in (ROOT, ROOT / "scripts", ROOT / "src", PIPELINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

spec = importlib.util.spec_from_file_location(
    "cmame_sobol_pod_pipeline",
    PIPELINE_DIR / "04_sobol_pod_pipeline.py",
)
assert spec is not None and spec.loader is not None
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def test_affine_maximin_and_warm_route_preserve_fixed_design():
    candidates = pipeline.validate._build_independent_materials(64, 20260821).copy()
    candidates.insert(0, "candidate_id", np.arange(len(candidates), dtype=int))

    selected_a = pipeline.affine_maximin_sequence(candidates, 10)
    selected_b = pipeline.affine_maximin_sequence(candidates, 10)
    assert selected_a["candidate_id"].is_unique
    assert selected_a["candidate_id"].tolist() == selected_b["candidate_id"].tolist()

    routed = pipeline.fixed_warm_start_route(selected_a)
    assert set(routed["candidate_id"]) == set(selected_a["candidate_id"])
    assert routed["solve_position"].tolist() == list(range(10))
    assert sorted(routed["sobol_set_position"].tolist()) == list(range(10))


def test_default_protocol_uses_only_fixed_training_and_final_validation():
    config = json.loads(
        (PIPELINE_DIR / "campaign_config.json").read_text(encoding="utf-8")
    )["sobol_pod_pipeline"]

    assert pipeline.resolve_training_protocol(
        config["adaptive"], config["monitor_count"], config["training_limit"]
    ) == "fixed"
    assert config["training_limit"] == 14
    assert config["adaptive_training_limit"] == 0
    assert config["adaptive"] is False
    assert config["monitor_count"] == 5
    assert config["final_validation_count"] == 5
    assert config["ritz_contraction_dtype"] == "float32"
    assert config["ritz_gram_rank_rtol"] == pytest.approx(1.0e-6)


def test_fixed_protocol_requires_an_explicit_training_budget():
    with pytest.raises(ValueError, match="positive training_limit"):
        pipeline.resolve_training_protocol(False, 5, 0)


def test_adaptive_protocol_uses_a_positive_monitor_pool():
    assert pipeline.resolve_training_protocol(True, 5, 0) == "adaptive"
    with pytest.raises(ValueError, match="positive monitor_count"):
        pipeline.resolve_training_protocol(True, 0, 14)
