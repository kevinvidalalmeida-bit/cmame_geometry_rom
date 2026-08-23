"""Relaunch campaign scripts inside the configured virtual environment."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def _arg_value(flag: str, default: Path) -> Path:
    argv = list(sys.argv[1:])
    for index, value in enumerate(argv):
        if value == flag and index + 1 < len(argv):
            return Path(argv[index + 1]).expanduser()
        prefix = f"{flag}="
        if value.startswith(prefix):
            return Path(value[len(prefix) :]).expanduser()
    return default


def _project_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def ensure_configured_venv(default_config: Path) -> None:
    """Exec the current script with the campaign venv before third-party imports."""
    root = Path(__file__).resolve().parents[2]
    config_path = _arg_value("--config", Path(default_config)).resolve()
    if not config_path.is_file():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    venv_path = _project_path(root, config["paths"]["venv_path"]).resolve()
    venv_python = venv_path / "bin" / "python"
    if not venv_python.is_file():
        return
    if Path(sys.prefix).resolve() == venv_path:
        return
    if os.environ.get("CMAME_INTERPRETABLE_VENV_BOOTSTRAPPED") == str(venv_path):
        raise RuntimeError(
            "Tried to enter the campaign virtualenv, but the process is still "
            f"outside it. Expected sys.prefix={venv_path}, got {sys.prefix}."
        )

    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(venv_path)
    env["CMAME_INTERPRETABLE_VENV_BOOTSTRAPPED"] = str(venv_path)
    bin_dir = str(venv_path / "bin")
    current_path = env.get("PATH", "")
    env["PATH"] = bin_dir if not current_path else f"{bin_dir}:{current_path}"
    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    nvidia_root = venv_path / "lib" / python_version / "site-packages" / "nvidia"
    nvidia_library_dirs = sorted(
        str(path.resolve())
        for path in nvidia_root.glob("*/lib")
        if path.is_dir()
    )
    if nvidia_library_dirs:
        current_library_path = env.get("LD_LIBRARY_PATH", "")
        entries = [*nvidia_library_dirs]
        if current_library_path:
            entries.append(current_library_path)
        env["LD_LIBRARY_PATH"] = os.pathsep.join(entries)
    os.execve(str(venv_python), [str(venv_python), *sys.argv], env)
