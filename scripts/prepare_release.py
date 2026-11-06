"""Freeze a trained supervisor and its provenance for distribution."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch

from duck_search import __version__
from duck_search.scene import ROOT


def prepare(run, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("best.pt", "search_policy.onnx", "training.csv", "evaluation.json"):
        source = run / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination / name)
    checkpoint = torch.load(
        destination / "best.pt", map_location="cpu", weights_only=True
    )
    latest = torch.load(run / "latest.pt", map_location="cpu", weights_only=True)
    # Host paths are not part of the model contract or a reproducible config.
    for data in (checkpoint,):
        config = data.get("config", {})
        for field in ("output", "resume"):
            if config.get(field):
                config[field] = Path(config[field]).name
    torch.save(checkpoint, destination / "best.pt")
    hashes = {
        name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
        for name in ("best.pt", "search_policy.onnx")
    }
    manifest = dict(
        version=__version__,
        kind="supervisory_search",
        input=[1, 45],
        output=[1, 6],
        direct_robotctl_load=False,
        training_simulator="SE2 proxy, PPO",
        training_steps=latest["steps"],
        selected_checkpoint_steps=checkpoint["steps"],
        training_seed=checkpoint["seed"],
        validation_seed=937,
        motor_policies=["progresion_4_mejor.onnx", "velstand.onnx", "alpha_stand.onnx"],
        actor_inputs="Forward depth, visible RGB detections, odometry and visit memory; no hidden target position",
        sha256=hashes,
    )
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--destination", type=Path, default=ROOT / "outputs/release")
    args = p.parse_args()
    prepare(args.run, args.destination)
