"""Read-only checks for a self-contained checkout and its bundled artifacts."""

import hashlib
import importlib.metadata
import json
import platform
import sys

from . import __version__
from .scene import ROOT


def check():
    versions = {}
    errors = []
    for package in (
        "numpy",
        "torch",
        "mujoco",
        "onnxruntime",
        "better-actuator-models",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"Missing dependency: {package}. Run uv sync --locked.")
    manifest_path = ROOT / "third_party/manifest.json"
    checked = 0
    if manifest_path.exists():
        for name, expected in json.loads(manifest_path.read_text())["sha256"].items():
            path = ROOT / "third_party" / name
            if not path.is_file():
                errors.append(f"Missing bundled file: third_party/{name}")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                errors.append(f"Modified bundled file: third_party/{name}")
            checked += 1
    else:
        errors.append("Missing third_party/manifest.json")
    release_manifest = ROOT / "outputs/release/manifest.json"
    if not release_manifest.is_file():
        errors.append("Missing trained supervisor release")
    else:
        manifest = json.loads(release_manifest.read_text())
        for name, expected in manifest.get("sha256", {}).items():
            path = release_manifest.parent / name
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected
            ):
                errors.append(f"Release integrity check failed: {name}")
    import torch

    return dict(
        ok=not errors,
        version=__version__,
        python=sys.version.split()[0],
        platform=platform.system(),
        cuda_available=torch.cuda.is_available(),
        dependencies=versions,
        bundled_files_checked=checked,
        errors=errors,
        scope="Simulation research release. Physical hardware is not validated.",
    )
