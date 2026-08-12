from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cmame_rank_scalability as rank_scaling


def test_required_rank_reports_first_and_stable_crossings() -> None:
    curve = pd.DataFrame(
        {
            "rank": [1, 2, 3, 4, 5],
            "error_max": [2.0e-2, 8.0e-3, 1.2e-2, 7.0e-3, 6.0e-3],
        }
    )

    result = rank_scaling.required_rank(curve, 1.0e-2)

    assert result["first_rank"] == 2
    assert result["stable_rank"] == 4
    assert result["achieved"] is True


def test_required_rank_marks_censored_threshold() -> None:
    curve = pd.DataFrame(
        {"rank": [1, 2, 3], "error_max": [1.0e-2, 2.0e-3, 4.0e-4]}
    )

    result = rank_scaling.required_rank(curve, 1.0e-4)

    assert result["first_rank"] is None
    assert result["stable_rank"] is None
    assert result["achieved"] is False
    assert result["max_tested_rank"] == 3
    assert np.isclose(result["error_at_max_rank"], 4.0e-4)


def test_descriptor_correlations_keep_censoring_explicit() -> None:
    summary = pd.DataFrame(
        {
            "case_kind": ["geometry"] * 4 + ["voxel_grid"],
            "interface_density": [0.1, 0.2, 0.3, 0.4, 0.5],
            "r_1e-4": [10.0, 20.0, 30.0, np.nan, 99.0],
        }
    )

    correlations = rank_scaling.descriptor_correlations(summary)
    row = correlations.loc[
        (correlations["rank_column"] == "r_1e-4")
        & (correlations["descriptor"] == "interface_density")
    ].iloc[0]

    assert int(row["geometry_count"]) == 3
    assert int(row["censored_count"]) == 1
    assert np.isclose(float(row["spearman_rho"]), 1.0)
    assert row["interpretation"] == "exploratory_n10_not_universal_law"
