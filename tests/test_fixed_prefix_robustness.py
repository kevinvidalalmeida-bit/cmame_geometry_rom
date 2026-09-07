from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "scripts" / "cmame_interpretable_pipeline"
for path in (ROOT, ROOT / "scripts", ROOT / "src", PIPELINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
MODULE_PATH = PIPELINE_DIR / "11_fixed_prefix_seed_robustness.py"
SPEC = importlib.util.spec_from_file_location("fixed_prefix_robustness", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
robustness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(robustness)


def _settings() -> tuple[dict, dict]:
    campaign = robustness.load_json(PIPELINE_DIR / "campaign_config.json")
    study = robustness.load_json(
        PIPELINE_DIR / "fixed_prefix_robustness_config.json"
    )
    return campaign, study


def test_default_study_has_no_monitor_and_unique_seed_runs() -> None:
    campaign, study = _settings()
    robustness.validate_settings(campaign, study)
    specs = robustness.run_specs(study)

    assert len(specs) == 80
    assert study["checkpoints"][:9] == list(range(2, 11))
    assert study["checkpoints"][-1] == study["max_training_materials"] == 10
    assert len(study["training_seeds"]) == 8
    assert len({spec["run_name"] for spec in specs}) == len(specs)
    for geometry_id in study["geometry_ids"]:
        geometry = [spec for spec in specs if spec["geometry_id"] == geometry_id]
        assert sum(bool(spec["validation_owner"]) for spec in geometry) == 1


def test_commands_are_fixed_incremental_and_share_validation() -> None:
    campaign, study = _settings()
    specs = robustness.run_specs(study)
    args = SimpleNamespace(
        config=PIPELINE_DIR / "campaign_config.json",
        smoke=False,
        overwrite=False,
    )
    destination = ROOT / "results" / "test_fixed_prefix"
    owner_command = robustness.command_for_spec(
        args=args,
        campaign=campaign,
        study=study,
        destination=destination,
        spec=specs[0],
    )
    follower_command = robustness.command_for_spec(
        args=args,
        campaign=campaign,
        study=study,
        destination=destination,
        spec=specs[1],
    )

    assert "--no-adaptive" in owner_command
    assert "--record-fixed-prefixes" in owner_command
    assert "--no-warm-start-route" in owner_command
    assert owner_command[owner_command.index("--monitor-count") + 1] == "0"
    assert owner_command[owner_command.index("--training-limit") + 1] == "10"
    assert owner_command[owner_command.index("--fixed-prefix-start") + 1] == "2"
    assert owner_command[owner_command.index("--final-validation-count") + 1] == "100"
    assert follower_command[
        follower_command.index("--final-validation-count") + 1
    ] == "1"


def test_incompatible_run_is_archived_without_deleting_it(tmp_path: Path) -> None:
    existing = tmp_path / "incomplete"
    existing.mkdir()
    (existing / "partial.csv").write_text("a\n1\n", encoding="utf-8")

    archived = robustness.archive_incompatible_run(existing, tmp_path / "summary")

    assert not existing.exists()
    assert (archived / "partial.csv").is_file()


def test_completed_owner_run_accepts_shared_validation_cache(tmp_path: Path) -> None:
    run_dir = tmp_path / "owner"
    run_dir.mkdir()
    (run_dir / "sobol_pod_summary.json").write_text(
        '{"status":"complete","adaptive":false,'
        '"record_fixed_prefixes":true,"candidate_seed":10,'
        '"training_limit":10,"final_validation_count":1}',
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        '{"solve_order_policy":"sobol_prefix_order"}', encoding="utf-8"
    )
    (run_dir / "reduced_operators.npz").touch()
    study = {"validation_count": 100, "non_owner_validation_count": 1,
             "max_training_materials": 10}

    assert robustness.completed_run(
        run_dir,
        {"training_seed": 10, "validation_owner": True,
         "shared_validation_cached": True},
        study,
    )
    assert not robustness.completed_run(
        run_dir,
        {"training_seed": 10, "validation_owner": True,
         "shared_validation_cached": False},
        study,
    )


def test_prefix_timing_uses_measured_cumulative_rows() -> None:
    frame = pd.DataFrame(
        {
            "snapshot_step_wall_s": [1.0, 2.0, 3.0],
            "solve_wall_s": [0.5, 1.0, 1.5],
            "ritz_contraction_wall_s": [0.0, 0.2, 0.4],
            "operator_contraction_workspace_peak_bytes": [0, 100, 200],
        }
    )
    row = robustness.prefix_timing(
        frame,
        2,
        {
            "geometry_load_wall_s": 0.1,
            "affine_setup_wall_s": 0.2,
            "pipeline_wall_s_before_final_write": 20.0,
        },
        recompile_wall_s=0.3,
        evaluation_wall_s=0.4,
    )

    assert row["snapshot_step_wall_s_cumulative"] == 3.0
    assert row["solve_wall_s_cumulative"] == 1.5
    assert row["ritz_contraction_wall_s_cumulative"] == 0.2
    assert row["operator_contraction_workspace_peak_bytes_maximum"] == 100
    assert row["prefix_compile_wall_s"] == pytest.approx(3.6)


def test_required_materials_reports_crossings_and_censoring() -> None:
    summary = pd.DataFrame(
        {
            "geometry_id": [3, 3, 3],
            "training_seed": [10, 10, 10],
            "training_materials": [2, 4, 6],
            "error_p95": [2.0e-3, 8.0e-4, 8.0e-5],
            "error_max": [3.0e-3, 2.0e-3, 2.0e-4],
        }
    )
    required = robustness.required_materials(summary, [2, 4, 6], 1.0e-4).iloc[0]

    assert required["first_Ns_p95_target"] == 6
    assert np.isnan(required["first_Ns_max_target"])
    assert bool(required["censored_max_target"])
    assert required["first_Ns_p95_1e3"] == 4
    assert required["first_Ns_max_1e3"] == 6
