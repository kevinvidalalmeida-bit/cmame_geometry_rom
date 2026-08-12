from __future__ import annotations

from scripts.applications.cmame_inclusion_benchmark import hashin_shtrikman_bounds, isotropic_moduli


def test_hashin_shtrikman_bounds_are_ordered_and_between_phases() -> None:
    K1, G1 = isotropic_moduli(4.0, 0.30)
    K2, G2 = isotropic_moduli(20.0, 0.25)
    lower_K, upper_K, lower_G, upper_G = hashin_shtrikman_bounds(K1, G1, K2, G2, 0.1)
    assert K1 < lower_K <= upper_K < K2
    assert G1 < lower_G <= upper_G < G2
