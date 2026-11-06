"""Publish only validated release files to a Hugging Face model repository.

Run with: uv run --with huggingface-hub python scripts/publish_huggingface.py
Use --publish to upload; the default only validates and prints the file count.
Authentication uses the standard Hugging Face credential cache or HF_TOKEN.
"""

import argparse
import json
import shutil
import tempfile
import tomllib
from pathlib import Path

from check_repo import ROOT, publication_files, validate

VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]

METADATA = """---
language:
- en
- es
license: other
license_name: mixed-apache-2.0-and-cc-by-nc-sa-4.0
license_link: https://github.com/Bravo-Robots/microduck-room-search/blob/main/THIRD_PARTY.md
library_name: pytorch
pipeline_tag: reinforcement-learning
tags:
- robotics
- microduck
- mujoco
- ppo
- onnx
- simulation
---

"""


def publish(args):
    report = validate()
    if not report["ok"]:
        raise ValueError(report["errors"])
    files = publication_files()
    print(
        json.dumps(
            {"repo_id": args.repo_id, "files": len(files), "publish": args.publish}
        )
    )
    if not args.publish:
        return
    from huggingface_hub import HfApi

    with tempfile.TemporaryDirectory(prefix="microduck-hub-") as temporary:
        folder = Path(temporary)
        for source in files:
            destination = folder / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        readme = folder / "README.md"
        readme.write_text(METADATA + readme.read_text())
        api = HfApi()
        api.create_repo(args.repo_id, repo_type="model", private=False, exist_ok=True)
        commit = api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=folder,
            commit_message=f"Release MicroDuck Room Search v{VERSION}",
        )
        api.create_tag(
            args.repo_id, tag=f"v{VERSION}", revision=commit.oid, exist_ok=True
        )
        print(
            json.dumps(
                {"url": f"https://huggingface.co/{args.repo_id}", "commit": commit.oid}
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Owner/model-name")
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Create or update the public model repository",
    )
    publish(parser.parse_args())
