#!/usr/bin/env python3
"""Build a causally enriched auxiliary space above the production r=168 ROM."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "scripts", ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from constitutive_transfer.schur_estimator import IncrementalTwoKernelCompiler
import cmame_campaign_common as common
import schur_estimator_compiler as compile_estimator
import schur_energy_indicators as qoi
import rom_reduced_operator as reduced


RUN = (
    ROOT
    / "results"
    / "fixed_geometry_ffthompy"
    / "fixed_geometry_ar15_vf20_sobol8_center_fields"
)
BASE_ROM = RUN / (
    "rom_tangential_r168_center_m1_m2_m3_m5_m7_v13_v3_v15_qoi64_m1_v1_"
    "v6_v7_v8_v9_v10_v11_v12_v13_v14_v16_v17_v22_v23_v24_v25_v30_v31_basis"
)
QOI_METHOD = (
    ROOT
    / "results"
    / "cmame_method"
    / "causal_comparator"
    / "methods"
    / "qoi_schur_greedy"
    / "replicate_00"
)
OUT = ROOT / "results" / "cmame_method" / "final_bound_auxiliary_r168"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_npz_fields(path: Path) -> list[np.ndarray]:
    with np.load(path) as payload:
        names = sorted(name for name in payload.files if name.startswith("basis_"))
        return [np.asarray(payload[name], dtype=np.float64) for name in names]


def _anchor_base_blocks(
    *, base_path: Path, auxiliary_path: Path, base_rank: int
) -> dict[str, float]:
    """Copy the production-ROM blocks exactly into the nested auxiliary arrays."""
    with np.load(base_path) as source, np.load(auxiliary_path) as target:
        payload = {name: np.asarray(target[name]) for name in target.files}
        diagnostics: dict[str, float] = {}
        for name in source.files:
            if name not in payload:
                continue
            base = np.asarray(source[name])
            auxiliary = payload[name]
            if name == "Kq":
                view = auxiliary[:, :base_rank, :base_rank]
            elif name == "Bq":
                view = auxiliary[:, :base_rank, :]
            elif name == "BK":
                view = auxiliary[..., :base_rank]
            elif name == "KK":
                view = auxiliary[..., :base_rank, :base_rank]
            elif name in {"Dq", "BB"}:
                view = auxiliary
            else:
                continue
            diagnostics[f"{name}_max_abs_before_anchor"] = float(
                np.max(np.abs(view - base))
            )
            view[...] = base
        np.savez_compressed(auxiliary_path, **payload)
    return diagnostics


def _prepare_nested_basis(
    *,
    base_rom: Path,
    method_dir: Path,
    additional_materials: int,
    tolerance: float,
    out_dir: Path,
) -> tuple[list[np.ndarray], list[int], pd.DataFrame]:
    selection = pd.read_csv(method_dir / "selection.csv").sort_values("offline_index")
    selected = selection.iloc[: int(additional_materials)].copy()
    if len(selected) != int(additional_materials):
        raise RuntimeError("The frozen QoI sequence is shorter than the requested auxiliary budget.")
    if selected["selection_uses_fom_error"].astype(bool).any():
        raise RuntimeError("Auxiliary enrichment must not use validation FOM errors.")

    base = _load_npz_fields(base_rom / "basis_fields.npz")
    base_rank = len(base)
    appended_path = out_dir / "appended_basis_fields.npz"
    rank_path = out_dir / "auxiliary_rank_schedule.csv"
    if appended_path.is_file() and rank_path.is_file():
        appended = _load_npz_fields(appended_path)
        schedule = pd.read_csv(rank_path)
        return base + appended, schedule["auxiliary_rank"].astype(int).tolist(), selected

    basis = list(base)
    rank_schedule: list[int] = []
    appended_all: list[np.ndarray] = []
    basis_paths = sorted((method_dir / "basis_fields").glob("basis_*.npy"))
    previous = 0
    for _, row in selected.iterrows():
        expected_rank = int(row["basis_rank"])
        source_fields = [np.load(path).astype(np.float64) for path in basis_paths[previous:expected_rank]]
        appended = common._append_orthonormal(
            basis,
            source_fields,
            tolerance=float(tolerance),
        )
        appended_all.extend(appended)
        rank_schedule.append(len(basis))
        previous = expected_rank
        print(
            f"[FINAL-BOUND] auxiliary material {int(row['offline_index'])}/"
            f"{additional_materials} | added={len(appended)} | rank={len(basis)}",
            flush=True,
        )

    np.savez_compressed(
        appended_path,
        **{
            f"basis_{index:04d}": field.astype(np.float32)
            for index, field in enumerate(appended_all)
        },
    )
    pd.DataFrame(
        {
            "offline_index": selected["offline_index"].astype(int).to_numpy(),
            "candidate_id": selected["candidate_id"].astype(int).to_numpy(),
            "auxiliary_rank": rank_schedule,
        }
    ).to_csv(rank_path, index=False)
    if len(base) != base_rank:
        raise RuntimeError("The base r=168 space changed while preparing the auxiliary basis.")
    return basis, rank_schedule, selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-rom", type=Path, default=BASE_ROM)
    parser.add_argument("--method-dir", type=Path, default=QOI_METHOD)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--additional-materials", type=int, default=10)
    parser.add_argument("--orthonormal-tol", type=float, default=1.0e-10)
    parser.add_argument("--atom-batch-size", type=int, default=6)
    parser.add_argument("--feature-block", type=int, default=131_072)
    parser.add_argument("--fft-workers", type=int, default=16)
    parser.add_argument("--validation-points", type=int, default=2)
    parser.add_argument("--validation-seed", type=int, default=20261129)
    parser.add_argument("--validation-rtol", type=float, default=1.0e-8)
    parser.add_argument(
        "--parent-certificate",
        type=Path,
        help="Optional prior online_benchmark.json that causally triggered continuation.",
    )
    parser.add_argument("--skip-estimator", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--keep-compiler-work", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_rom = args.base_rom.resolve()
    method_dir = args.method_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if int(args.additional_materials) < 1:
        raise ValueError("additional_materials must be positive.")

    started = time.perf_counter()
    basis, rank_schedule, selected = _prepare_nested_basis(
        base_rom=base_rom,
        method_dir=method_dir,
        additional_materials=int(args.additional_materials),
        tolerance=float(args.orthonormal_tol),
        out_dir=out_dir,
    )
    with np.load(base_rom / "reduced_operators.npz") as base_operators:
        base_rank = int(base_operators["Kq"].shape[1])
    phase = np.load(RUN / "_fixed_geometry" / "phase.npy").astype(np.uint8)
    ori = np.load(RUN / "_fixed_geometry" / "ori.npy").astype(np.float64)
    operator_path = out_dir / "reduced_operators.npz"
    assembly_meta: dict[str, Any] = {"reused": operator_path.is_file()}
    if not operator_path.is_file():
        Kq, Bq, Dq, assembly_meta = reduced._assemble_reduced_operators(
            phase=phase,
            ori=ori,
            basis=basis,
        )
        np.savez_compressed(
            operator_path,
            Kq=Kq,
            Bq=Bq,
            Dq=Dq,
            coefficient_names=np.asarray(reduced.COEFF_NAMES),
        )
    operator_anchor = _anchor_base_blocks(
        base_path=base_rom / "reduced_operators.npz",
        auxiliary_path=operator_path,
        base_rank=base_rank,
    )

    estimator_path = out_dir / "online_estimator.npz"
    compiler_meta: dict[str, Any] = {"skipped": bool(args.skip_estimator)}
    if not args.skip_estimator and not estimator_path.is_file():
        matrix_idx, fiber_groups = qoi._geometry_groups(phase, ori)
        work_dir = out_dir / "incremental_estimator_work"
        compiler = IncrementalTwoKernelCompiler(
            work_dir=work_dir,
            shape=tuple(int(value) for value in phase.shape),
            coefficient_names=reduced.COEFF_NAMES,
            affine_stress_batch=compile_estimator._affine_stress_factory(matrix_idx, fiber_groups),
            max_rank=len(basis),
            atom_batch_size=int(args.atom_batch_size),
            feature_block=int(args.feature_block),
            fft_workers=int(args.fft_workers),
            resume=work_dir.is_dir(),
        )
        append_started = time.perf_counter()
        while compiler.rank < len(basis):
            stop = min(compiler.rank + int(args.atom_batch_size), len(basis))
            compiler.append(
                np.stack(basis[compiler.rank:stop]).reshape(stop - compiler.rank, 6, -1)
            )
            print(f"[FINAL-BOUND] estimator rank={compiler.rank}/{len(basis)}", flush=True)
        compiler.save(estimator_path)
        compiler_meta = {
            "basis_rank": compiler.rank,
            "append_wall_s": float(time.perf_counter() - append_started),
        }
        compiler.close()
        if not args.keep_compiler_work:
            compiler.cleanup()

    estimator_anchor: dict[str, float] = {}
    if estimator_path.is_file():
        estimator_anchor = _anchor_base_blocks(
            base_path=base_rom / "online_estimator.npz",
            auxiliary_path=estimator_path,
            base_rank=base_rank,
        )

    validation_meta: dict[str, Any] = {"skipped": bool(args.skip_validation)}
    validation_path = out_dir / "online_estimator_validation.json"
    if estimator_path.is_file() and not args.skip_validation:
        estimator = compile_estimator.TwoKernelEstimator.load(estimator_path)
        with np.load(operator_path) as payload:
            operators = {
                key: np.asarray(payload[key], dtype=np.float64)
                for key in ("Kq", "Bq", "Dq")
            }
        matrix_idx, fiber_groups = qoi._geometry_groups(phase, ori)
        basis_array = np.stack(basis).reshape(len(basis), 6, phase.size)
        records = compile_estimator._validate(
            estimator=estimator,
            basis=basis_array,
            operators=operators,
            shape=tuple(int(value) for value in phase.shape),
            matrix_idx=matrix_idx,
            fiber_groups=fiber_groups,
            points=int(args.validation_points),
            seed=int(args.validation_seed),
        )
        max_relative_error = max(float(row["relative_error"]) for row in records)
        validation_meta = {
            "records": records,
            "max_relative_error": max_relative_error,
            "threshold": float(args.validation_rtol),
        }
        _write_json(validation_path, validation_meta)
        if max_relative_error >= float(args.validation_rtol):
            raise RuntimeError(
                "Auxiliary dense/direct estimator mismatch: "
                f"{max_relative_error:.3e} >= {float(args.validation_rtol):.3e}."
            )

    selection_path = method_dir / "selection.csv"
    continuation: dict[str, Any] = {}
    if args.parent_certificate is not None:
        parent_path = args.parent_certificate.resolve()
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        continuation = {
            "parent_certificate": str(parent_path),
            "parent_certificate_sha256": _sha256(parent_path),
            "rule": "continue fixed ten-material batches while any candidate relative width exceeds 1e-4",
            "parent_query_count": int(parent["query_count"]),
            "parent_relative_width_below_1e-4_count": int(
                parent["relative_width_below_1e-4_count"]
            ),
            "parent_relative_width_max": float(parent["relative_width_max"]),
            "uses_validation_fom_error": False,
        }
    manifest = {
        "base_rom": str(base_rom),
        "base_rank": base_rank,
        "auxiliary_rank": len(basis),
        "rank_schedule": rank_schedule,
        "additional_material_budget": int(args.additional_materials),
        "selected_candidate_ids": selected["candidate_id"].astype(int).tolist(),
        "selection_uses_fom_error": False,
        "continuation": continuation,
        "selection_sha256": _sha256(selection_path),
        "operator_assembly": assembly_meta,
        "operator_anchor": operator_anchor,
        "estimator_compilation": compiler_meta,
        "estimator_anchor": estimator_anchor,
        "estimator_validation": validation_meta,
        "wall_s": float(time.perf_counter() - started),
    }
    _write_json(out_dir / "auxiliary_manifest.json", manifest)
    print(
        f"[FINAL-BOUND] listo | base=168 | auxiliary={len(basis)} | out={out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
