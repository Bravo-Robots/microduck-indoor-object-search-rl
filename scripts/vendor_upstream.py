"""Bundle the already present workshop dependencies, preserving their licences."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def bundle(source):
    root = Path(__file__).resolve().parents[1]
    workshop = source.resolve()
    vendor = root / "third_party"
    vendor.mkdir(exist_ok=True)
    shutil.copytree(workshop / "robot", vendor / "robot", dirs_exist_ok=True)
    shutil.copy2(workshop / "scripts/infer_policy.py", vendor / "infer_policy.py")
    (vendor / "policies").mkdir(exist_ok=True)
    for name in ["progresion_4_mejor.onnx", "velstand.onnx", "alpha_stand.onnx"]:
        shutil.copy2(workshop / "policies" / name, vendor / "policies" / name)
    shutil.copy2(workshop / "policies/NOTICE.md", vendor / "policies/NOTICE.md")
    hashes = {
        str(p.relative_to(vendor)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in vendor.rglob("*")
        if p.is_file() and "__pycache__" not in str(p) and p.name != "manifest.json"
    }
    (vendor / "manifest.json").write_text(
        json.dumps(
            dict(
                source="https://github.com/pollen-robotics/microduck_rl",
                official_policies="https://huggingface.co/pollen-robotics/microduck-policies",
                workshop_gait="progresion_4_mejor.onnx, existing local PPO run model_1499.pt; NOT an official Pollen checkpoint",
                robot_mesh_licence="CC BY-NC-SA 4.0; see robot/LICENSE-MODELS.md",
                adapter="Workshop's Apache-2.0 adaptation of Pollen inference; copied unchanged",
                sha256=hashes,
            ),
            indent=2,
        )
    )
    print(f"Bundled {len(hashes)} files with hashes and original licences")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="Workshop checkout to vendor"
    )
    bundle(parser.parse_args().source)
