"""Deterministic design checks for the fixed full-rank Sobol pipeline."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


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
