#!/usr/bin/env python3
"""Single entry point for the interpretable 10-geometry campaign."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from env_bootstrap import ensure_configured_venv


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CONFIG_DEFAULT = HERE / "campaign_config.json"
ensure_configured_venv(CONFIG_DEFAULT)


def append_geometry_ids(command: list[str], geometry_ids: list[int] | None) -> None:
    if geometry_ids is None:
        return
    command.append("--geometry-ids")
    command.extend(str(int(value)) for value in geometry_ids)


def run(command: list[str]) -> None:
    print("[MAIN] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=str(ROOT), check=True)


def script(name: str) -> str:
    return str(HERE / name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("design", "geometries", "pipeline", "all"), default="all")
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--geometry-ids", type=int, nargs="*", default=None)
    parser.add_argument("--overwrite-geometries", action="store_true")
    parser.add_argument("--pipeline-stage", choices=("all", "sobol_pod"), default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--overwrite-pipeline", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def base_command(script_name: str, args: argparse.Namespace) -> list[str]:
    command = [sys.executable, script(script_name), "--config", str(args.config)]
    if args.out_root is not None:
        command.extend(["--out-root", str(args.out_root)])
    return command


def main() -> int:
    args = parse_args()
    geometry_ids = args.geometry_ids
    if args.smoke and geometry_ids is None:
        geometry_ids = [8]

    if args.stage in {"design", "all"}:
        run(base_command("01_design_cases.py", args))

    if args.stage in {"geometries", "all"}:
        command = base_command("02_generate_geometries.py", args)
        append_geometry_ids(command, geometry_ids)
        if args.overwrite_geometries:
            command.append("--overwrite")
        run(command)

    if args.stage in {"pipeline", "all"}:
        command = base_command("03_run_pipeline.py", args)
        append_geometry_ids(command, geometry_ids)
        if args.pipeline_stage is not None:
            command.extend(["--stage", args.pipeline_stage])
        if args.run_name is not None:
            command.extend(["--run-name", args.run_name])
        if args.overwrite_pipeline:
            command.append("--overwrite")
        if args.smoke:
            command.append("--smoke")
        run(command)

    print("[MAIN] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
