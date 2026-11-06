"""Summarize measured results; never invent test counts or benchmark outcomes."""

import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def summarize():
    folder = ROOT / "benchmarks/v0.2"
    result = {
        "version": tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
            "version"
        ],
        "release": json.loads((ROOT / "outputs/release/manifest.json").read_text()),
    }
    for key, name in [
        ("proxy", "proxy.json"),
        ("mujoco_rl", "rl/summary.json"),
        ("mujoco_baseline", "baseline/summary.json"),
    ]:
        metric = json.loads((folder / name).read_text())
        if key.startswith("mujoco_"):
            reports = [
                json.loads((folder / Path(name).parent / r["report"]).read_text())
                for r in metric["results"]
            ]
            expected = {
                "episodes": len(reports),
                "successes": sum(r["success"] for r in reports),
                "falls": sum(r["fall"] for r in reports),
                "false_confirmations": sum(r["false_confirmations"] for r in reports),
                "episodes_with_contacts": sum(r["contact_steps"] > 0 for r in reports),
            }
            if any(metric[k] != v for k, v in expected.items()):
                raise ValueError(f"Stale benchmark summary: {name}")
            if (
                key == "mujoco_rl"
                and metric["checkpoint_sha256"]
                != result["release"]["sha256"]["best.pt"]
            ):
                raise ValueError("Benchmark checkpoint does not match the release")
        metric.pop("results", None)
        metric.pop("episodes_detail", None)
        result[key] = metric
    (ROOT / "outputs/release/RESULTS.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    summarize()
