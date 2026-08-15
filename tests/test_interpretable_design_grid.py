from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "scripts" / "cmame_interpretable_pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

from design_grid import rounded_grid_size
from validation_reporting import empirical_coverage


@pytest.mark.parametrize(
    ("requested", "expected_nominal", "expected_odd"),
    [(60.0, 60, 61), (150.0, 150, 151), (240.0, 240, 241)],
)
def test_odd_up_preserves_or_increases_resolution(
    requested: float,
    expected_nominal: int,
    expected_odd: int,
) -> None:
    nominal, adjusted = rounded_grid_size(requested, multiple=1, parity="odd_up")
    assert nominal == expected_nominal
    assert adjusted == expected_odd
    assert adjusted >= requested
    assert adjusted % 2 == 1


def test_any_parity_preserves_existing_rounding() -> None:
    assert rounded_grid_size(60.0, multiple=1, parity="any") == (60, 60)


def test_odd_grid_rejects_even_multiple() -> None:
    with pytest.raises(ValueError, match="odd grid"):
        rounded_grid_size(60.0, multiple=2, parity="odd_up")


def test_empirical_coverage_reports_counts_and_percent_units() -> None:
    summary = empirical_coverage([1.0e-5, 1.0e-4, 2.0e-4], 1.0e-4)
    assert summary["observed_count"] == 3
    assert summary["observed_below_threshold_count"] == 2
    assert summary["observed_above_threshold_count"] == 1
    assert summary["observed_below_threshold_percent"] == pytest.approx(200.0 / 3.0)
    assert summary["report_threshold_percent"] == pytest.approx(0.01)
