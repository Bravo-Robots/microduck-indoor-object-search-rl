"""Validate exactly the files that will be included in a public source archive."""

import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
TOP_FILES = {
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
    "LICENSE",
    "README.md",
    "THIRD_PARTY.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "pyproject.toml",
    "uv.lock",
    "run.sh",
}
TOP_DIRS = {
    ".github",
    "duck_search",
    "scripts",
    "tests",
    "third_party",
    "assets",
    "docs",
    "benchmarks",
}


def publication_files(root=ROOT):
    if (root / ".git").exists():
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        files = [root / name for name in result.stdout.decode().split("\0") if name]
    else:
        files = []
        for path in root.rglob("*"):
            rel = path.relative_to(root)
            if not path.is_file():
                continue
            if (
                rel.parts[0] in TOP_DIRS
                or rel.as_posix() in TOP_FILES
                or rel.parts[:2] == ("outputs", "release")
            ):
                files.append(path)
    excluded = {"__pycache__", ".ruff_cache", ".venv"}
    return sorted(
        p
        for p in files
        if not any(part in excluded for part in p.parts)
        and p.suffix not in {".pyc", ".blend1"}
        and not p.name.endswith((".scene.xml", ".confirmation.png"))
        and p.name not in {"robot_blender_meshes.json", "make_reference_sheet.py"}
        and p.relative_to(root).as_posix() != "assets/room.xml"
        and not (
            p.relative_to(root).parts[0] == "benchmarks" and p.name == "checkpoint.pt"
        )
    )


def validate(root=ROOT):
    files = publication_files(root)
    errors = []
    for path in files:
        rel = path.relative_to(root)
        if path.stat().st_size > 95 * 1024 * 1024:
            errors.append(f"File too large for ordinary Git: {rel}")
        if (
            path.suffix in {".md", ".py", ".toml", ".json", ".yml", ".sh"}
            and rel.parts[0] != "third_party"
        ):
            text = path.read_text()
            if re.search(r"/home/(robot|antoine)/|/tmp/" + "browser-use/", text):
                errors.append(f"Host-specific path in {rel}")
        if path.suffix == ".md":
            for link in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", path.read_text()):
                link = link.strip("<>").split("#")[0]
                if not link or ":" in link:
                    continue
                if not (path.parent / unquote(link)).exists():
                    errors.append(f"Broken local link in {rel}: {link}")
    for name in (
        "README.md",
        "LICENSE",
        "THIRD_PARTY.md",
        "uv.lock",
        ".github/workflows/ci.yml",
        "outputs/release/MODEL_CARD.md",
    ):
        if not (root / name).is_file():
            errors.append(f"Missing publication file: {name}")
    report = dict(
        ok=not errors,
        file_count=len(files),
        total_bytes=sum(p.stat().st_size for p in files),
        errors=errors,
    )
    return report


if __name__ == "__main__":
    report = validate()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ok"] else 1)
