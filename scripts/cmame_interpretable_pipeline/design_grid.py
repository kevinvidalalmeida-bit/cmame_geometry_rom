"""Grid-size rules shared by the interpretable geometry campaign."""

from __future__ import annotations


GRID_PARITIES = ("any", "odd_up")


def rounded_grid_size(value: float, multiple: int, parity: str = "any") -> tuple[int, int]:
    """Return the nominal and parity-adjusted grid sizes.

    ``odd_up`` chooses the first admissible odd multiple at or above the
    nominal rounded size. This preserves, or slightly increases, the requested
    voxel resolution.
    """
    multiple = max(1, int(multiple))
    nominal = max(8, int(round(float(value) / float(multiple)) * multiple))
    parity = str(parity).lower()
    if parity not in GRID_PARITIES:
        raise ValueError(f"grid_parity must be one of {GRID_PARITIES}, got {parity!r}.")
    if parity == "any":
        return nominal, nominal
    if multiple % 2 == 0:
        raise ValueError("An odd grid cannot also be a multiple of an even nvox_multiple.")

    adjusted = nominal
    while adjusted % 2 == 0:
        adjusted += multiple
    return nominal, adjusted
