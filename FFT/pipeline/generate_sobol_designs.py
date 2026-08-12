"""Genera los puntos Sobol que usa la campana FFT del main."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import math
import os
import json
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import qmc


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "si", "sí", "on"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not str(value).strip():
        return int(default)
    return int(value)


def env_optional_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or not str(value).strip():
        return None
    return int(value)


def env_optional_float(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or not str(value).strip():
        return None
    return float(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not str(value).strip():
        return float(default)
    return float(value)


STUDY_NAME = os.environ.get("STUDY_NAME", "Estudio_Sobol_Continuo")
PHASE_LABEL = os.environ.get("PHASE_LABEL", "sobol_stage1_simple")

WORKSPACE_DIR = Path(__file__).resolve().parents[1]
RESULTS_ROOT = WORKSPACE_DIR / "results"

N_SOBOL_POINTS = env_int("N_SOBOL_POINTS", 60)
GLOBAL_SEED = env_int("GLOBAL_SEED", 21062026)
SCRAMBLE = env_bool("SOBOL_SCRAMBLE", True)
SOBOL_SCRAMBLE_SEED = env_optional_int("SOBOL_SCRAMBLE_SEED")
if SOBOL_SCRAMBLE_SEED is None:
    SOBOL_SCRAMBLE_SEED = GLOBAL_SEED

BOX_FACTOR = env_float("BOX_FACTOR", 2.0)
DOMAIN_LENGTH_UM = env_optional_float("DOMAIN_LENGTH_UM")
DF_VOXEL_TARGET = env_float("DF_VOXEL_TARGET", 6.0)
NVOX_MULTIPLE = env_int("NVOX_MULTIPLE", 1)
FIBER_DIAMETER_UM = env_float("FIBER_DIAMETER_UM", 1.0)

BOUNDS: Dict[str, Tuple[float, float]] = {
    "AR": (
        env_float("SOBOL_AR_MIN", 5.0),
        env_float("SOBOL_AR_MAX", 30.0),
    ),
    "Vf_target": (
        env_float("SOBOL_VF_TARGET_MIN", 0.050),
        env_float("SOBOL_VF_TARGET_MAX", 0.30),
    ),
    "Em": (
        env_float("SOBOL_EM_MIN", 1.1),
        env_float("SOBOL_EM_MAX", 4.4),
    ),
    "Ef_L": (
        env_float("SOBOL_EF_L_MIN", 72.0),
        env_float("SOBOL_EF_L_MAX", 290.0),
    ),
    "Ef_T": (
        env_float("SOBOL_EF_T_MIN", 10.0),
        env_float("SOBOL_EF_T_MAX", 23.0),
    ),
    "G_LT": (
        env_float("SOBOL_G_LT_MIN", 9.0),
        env_float("SOBOL_G_LT_MAX", 30.0),
    ),
    "nu_LT": (
        env_float("SOBOL_NU_LT_MIN", 0.20),
        env_float("SOBOL_NU_LT_MAX", 0.26),
    ),
    "nu_TT": (
        env_float("SOBOL_NU_TT_MIN", 0.35),
        env_float("SOBOL_NU_TT_MAX", 0.40),
    ),
    "nu_m": (
        env_float("SOBOL_NU_M_MIN", 0.35),
        env_float("SOBOL_NU_M_MAX", 0.42),
    ),
}

OUTPUT_COLUMNS = [
    "config_id",
    "label",
    "sobol_index",
    "design_id",
    "is_operable",
    "reject_reason",

    "BOX_FACTOR",
    "DF_VOXEL_TARGET",
    "NVOX_MULTIPLE",
    "caja_um",
    "nvox",
    "nvox_ref",
    "res",
    "res_ref",
    "voxel_um",
    "df_voxel",
    "d_um",
    "L_um",
    "fiber_length_lf",
    "AR",
    "Lf_Ldom",
    "Vf_target",

    "a11",
    "a22",
    "a33",

    "Em",
    "nu_m",
    "Gm",
    "Ef_L_over_Em",
    "Ef_T_over_Em",
    "G_LT_over_Gm",
    "Ef_L",
    "Ef_T",
    "nu_LT",
    "nu_TT",
    "G_LT",
    "G_TT",
]


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_study_dir() -> Path:
    study_id = f"{STUDY_NAME}_{PHASE_LABEL}_{now_tag()}"
    study_dir = RESULTS_ROOT / study_id
    study_dir.mkdir(parents=True, exist_ok=True)
    return study_dir


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def round_value(value: float, ndigits: int = 6) -> float:
    return round(float(value), ndigits)


def round_up_to_multiple(value: float, multiple: int = 1) -> int:
    multiple = max(1, int(multiple))
    return int(math.ceil(float(value) / float(multiple)) * multiple)


def compute_box_and_resolution(
    fiber_length_um: float,
    d_um: float,
) -> Tuple[float, int, float, float, float, float]:
    if DOMAIN_LENGTH_UM is None:
        caja_um = float(BOX_FACTOR) * float(fiber_length_um)
    else:
        caja_um = float(DOMAIN_LENGTH_UM)

    nvox_raw = float(DF_VOXEL_TARGET) * caja_um / float(d_um)
    nvox = round_up_to_multiple(nvox_raw, NVOX_MULTIPLE)

    res = float(nvox) / caja_um
    voxel_um = caja_um / float(nvox)
    df_voxel = float(d_um) * res
    Lf_Ldom = float(fiber_length_um) / caja_um

    return caja_um, nvox, res, voxel_um, df_voxel, Lf_Ldom


def derive_material_properties(sampled: Dict[str, float]) -> Dict[str, float]:
    Em = float(sampled["Em"])
    nu_m = float(sampled["nu_m"])
    Gm = Em / (2.0 * (1.0 + nu_m))

    Ef_L = float(sampled["Ef_L"])
    Ef_T = float(sampled["Ef_T"])
    G_LT = float(sampled["G_LT"])
    nu_LT = float(sampled["nu_LT"])
    nu_TT = float(sampled["nu_TT"])
    G_TT = Ef_T / (2.0 * (1.0 + nu_TT))

    return {
        "Em": Em,
        "nu_m": nu_m,
        "Gm": Gm,
        "Ef_L_over_Em": Ef_L / Em,
        "Ef_T_over_Em": Ef_T / Em,
        "G_LT_over_Gm": G_LT / Gm,
        "Ef_L": Ef_L,
        "Ef_T": Ef_T,
        "nu_LT": nu_LT,
        "nu_TT": nu_TT,
        "G_LT": G_LT,
        "G_TT": G_TT,
    }


def validate_material_properties(material: Dict[str, float]) -> Tuple[bool, str]:
    Em = float(material["Em"])
    nu_m = float(material["nu_m"])
    Ef_L = float(material["Ef_L"])
    Ef_T = float(material["Ef_T"])
    nu_LT = float(material["nu_LT"])
    nu_TT = float(material["nu_TT"])
    G_LT = float(material["G_LT"])

    if Em <= 0.0:
        return False, "Em_non_positive"
    if not (-1.0 < nu_m < 0.5):
        return False, "nu_m_out_of_isotropic_range"
    if Ef_L <= 0.0 or Ef_T <= 0.0 or G_LT <= 0.0:
        return False, "fiber_modulus_non_positive"
    if Ef_L < Ef_T:
        return False, "Ef_L_less_than_Ef_T"
    if not (-1.0 < nu_TT < 0.5):
        return False, "nu_TT_out_of_ti_range"

    G_TT = Ef_T / (2.0 * (1.0 + nu_TT))
    if G_TT <= 0.0:
        return False, "G_TT_non_positive"

    S = np.zeros((6, 6), dtype=float)
    S[0, 0] = 1.0 / Ef_L
    S[1, 1] = 1.0 / Ef_T
    S[2, 2] = 1.0 / Ef_T
    S[0, 1] = S[1, 0] = -nu_LT / Ef_L
    S[0, 2] = S[2, 0] = -nu_LT / Ef_L
    S[1, 2] = S[2, 1] = -nu_TT / Ef_T
    S[3, 3] = 1.0 / G_TT
    S[4, 4] = 1.0 / G_LT
    S[5, 5] = 1.0 / G_LT

    if np.min(np.linalg.eigvalsh(S)) <= 1e-12:
        return False, "fiber_compliance_not_positive_definite"

    return True, ""


def sobol_to_orientation_triangle(u: float, v: float) -> Tuple[float, float, float]:
    u = float(np.clip(u, 0.0, 1.0))
    v = float(np.clip(v, 0.0, 1.0))

    if u + v > 1.0:
        u = 1.0 - u
        v = 1.0 - v

    a11 = u
    a22 = v
    a33 = 1.0 - a11 - a22

    a11 = round_value(a11)
    a22 = round_value(a22)
    a33 = round_value(a33)

    if a33 < 0.0 and abs(a33) < 1e-8:
        a33 = 0.0

    return a11, a22, a33


def validate_orientation(a11: float, a22: float, a33: float) -> Tuple[bool, str]:
    if a11 < 0.0:
        return False, "a11_negative"

    if a22 < 0.0:
        return False, "a22_negative"

    if a33 < 0.0:
        return False, "a33_negative"

    if a11 > 1.0:
        return False, "a11_greater_than_one"

    if a22 > 1.0:
        return False, "a22_greater_than_one"

    if a33 > 1.0:
        return False, "a33_greater_than_one"

    if a11 + a22 > 1.0 + 1e-6:
        return False, "a11_plus_a22_greater_than_one"

    if not np.isclose(a11 + a22 + a33, 1.0, atol=1e-6):
        return False, "orientation_trace_not_one"

    return True, ""


def generate_sobol_dataframe() -> pd.DataFrame:
    if int(N_SOBOL_POINTS) < 1:
        raise ValueError("N_SOBOL_POINTS debe ser >= 1.")

    parameter_names = list(BOUNDS.keys())

    n_dims = len(parameter_names) + 2

    sampler = qmc.Sobol(
        d=n_dims,
        scramble=SCRAMBLE,
        seed=SOBOL_SCRAMBLE_SEED,
    )

    sobol_power = int(math.ceil(math.log2(int(N_SOBOL_POINTS))))
    x_unit = sampler.random_base2(m=sobol_power)[: int(N_SOBOL_POINTS)]

    x_params = x_unit[:, :len(parameter_names)]
    x_orientation = x_unit[:, len(parameter_names):]

    lower = [BOUNDS[key][0] for key in parameter_names]
    upper = [BOUNDS[key][1] for key in parameter_names]

    x_scaled = qmc.scale(x_params, lower, upper)

    d_um = float(FIBER_DIAMETER_UM)

    rows: List[Dict[str, Any]] = []

    for sobol_index, sampled_row in enumerate(x_scaled):
        sampled = {
            key: float(value)
            for key, value in zip(parameter_names, sampled_row)
        }

        u = float(x_orientation[sobol_index, 0])
        v = float(x_orientation[sobol_index, 1])

        a11, a22, a33 = sobol_to_orientation_triangle(u, v)
        orientation_ok, orientation_reason = validate_orientation(a11, a22, a33)
        material = derive_material_properties(sampled)
        material_ok, material_reason = validate_material_properties(material)
        ok = bool(orientation_ok and material_ok)
        reject_reason = ""
        if not orientation_ok:
            reject_reason = orientation_reason
        elif not material_ok:
            reject_reason = material_reason

        AR = float(sampled["AR"])
        L_um = AR * d_um
        caja_um, nvox, res, voxel_um, df_voxel, Lf_Ldom = compute_box_and_resolution(
            fiber_length_um=L_um,
            d_um=d_um,
        )

        row = {
            "config_id": "cfg_simple_continuous_orientation",
            "label": "simple_continuous_orientation",
            "sobol_index": int(sobol_index),
            "design_id": pd.NA,
            "is_operable": bool(ok),
            "reject_reason": reject_reason,

            "BOX_FACTOR": round_value(caja_um / L_um),
            "DF_VOXEL_TARGET": round_value(DF_VOXEL_TARGET),
            "NVOX_MULTIPLE": int(NVOX_MULTIPLE),
            "caja_um": round_value(caja_um),
            "nvox": int(nvox),
            "nvox_ref": int(nvox),
            "res": round_value(res),
            "res_ref": round_value(res),
            "voxel_um": round_value(voxel_um),
            "df_voxel": round_value(df_voxel),
            "d_um": round_value(d_um),
            "L_um": round_value(L_um),
            "fiber_length_lf": round_value(L_um),
            "AR": round_value(AR),
            "Lf_Ldom": round_value(Lf_Ldom),
            "Vf_target": round_value(sampled["Vf_target"]),

            "a11": a11,
            "a22": a22,
            "a33": a33,

            "Em": round_value(material["Em"]),
            "nu_m": round_value(material["nu_m"]),
            "Gm": round_value(material["Gm"]),
            "Ef_L_over_Em": round_value(material["Ef_L_over_Em"]),
            "Ef_T_over_Em": round_value(material["Ef_T_over_Em"]),
            "G_LT_over_Gm": round_value(material["G_LT_over_Gm"]),
            "Ef_L": round_value(material["Ef_L"]),
            "Ef_T": round_value(material["Ef_T"]),
            "nu_LT": round_value(material["nu_LT"]),
            "nu_TT": round_value(material["nu_TT"]),
            "G_LT": round_value(material["G_LT"]),
            "G_TT": round_value(material["G_TT"]),
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[OUTPUT_COLUMNS].copy()

    valid_mask = df["is_operable"].fillna(False)

    ids = np.full(len(df), pd.NA, dtype=object)
    ids[valid_mask.values] = np.arange(0, int(valid_mask.sum()))

    df["design_id"] = pd.array(ids, dtype="Int64")

    return df


def build_catalog_dataframe() -> pd.DataFrame:
    row = {
        "study_name": STUDY_NAME,
        "phase_label": PHASE_LABEL,
        "orientation_mode": "continuous_triangle",
        "n_sobol_points": int(N_SOBOL_POINTS),
        "global_seed": int(GLOBAL_SEED),
        "scramble": bool(SCRAMBLE),
        "sobol_scramble_seed": SOBOL_SCRAMBLE_SEED,
        "box_factor": float(BOX_FACTOR),
        "domain_length_um": DOMAIN_LENGTH_UM,
        "df_voxel_target": float(DF_VOXEL_TARGET),
        "nvox_multiple": int(NVOX_MULTIPLE),
        "fiber_diameter_um": float(FIBER_DIAMETER_UM),
        "bounds": json.dumps(BOUNDS, ensure_ascii=False, sort_keys=True),
    }

    return pd.DataFrame([row])


def generate_designs() -> Path:
    study_dir = build_study_dir()
    df = generate_sobol_dataframe()
    valid_df = df.loc[df["is_operable"].fillna(False)].copy()
    rejected_df = df.loc[~df["is_operable"].fillna(False)].copy()

    catalog_df = build_catalog_dataframe()

    catalog_path = study_dir / "sobol_config_catalog.xlsx"
    raw_path = study_dir / "sobol_points_raw.xlsx"
    valid_path = study_dir / "sobol_points_valid.xlsx"
    rejected_path = study_dir / "sobol_points_rejected.xlsx"

    catalog_df.to_excel(catalog_path, index=False)
    df.to_excel(raw_path, index=False)
    valid_df.to_excel(valid_path, index=False)
    rejected_df.to_excel(rejected_path, index=False)

    manifest = {
        "study_name": STUDY_NAME,
        "phase_label": PHASE_LABEL,
        "study_dir": str(study_dir),
        "orientation_mode": "continuous_triangle",
        "n_sobol_points": int(N_SOBOL_POINTS),
        "n_raw_rows": int(len(df)),
        "n_valid_rows": int(len(valid_df)),
        "n_rejected_rows": int(len(rejected_df)),
        "global_seed": int(GLOBAL_SEED),
        "scramble": bool(SCRAMBLE),
        "sobol_scramble_seed": SOBOL_SCRAMBLE_SEED,
        "box_factor": float(BOX_FACTOR),
        "domain_length_um": DOMAIN_LENGTH_UM,
        "df_voxel_target": float(DF_VOXEL_TARGET),
        "nvox_multiple": int(NVOX_MULTIPLE),
        "fiber_diameter_um": float(FIBER_DIAMETER_UM),
        "bounds": BOUNDS,
        "catalog_xlsx": str(catalog_path),
        "raw_xlsx": str(raw_path),
        "valid_xlsx": str(valid_path),
        "rejected_xlsx": str(rejected_path),
    }

    write_json(study_dir / "study_manifest_stage1.json", manifest)

    print(
        "[SOBOL] "
        f"raw={len(df)} | validos={len(valid_df)} | rejected={len(rejected_df)} | "
        f"salida={study_dir}"
    )

    return valid_path


def main() -> int:
    generate_designs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
