"""Public command-line interface. Run from the checkout, or through run.sh."""

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .scene import ROOT, build


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def seed_value(value):
    result = int(value)
    if result < 0 or result > 2**32 - 1:
        raise argparse.ArgumentTypeError("seed must be between 0 and 4294967295")
    return result


def positive_seconds(value):
    result = float(value)
    if not math.isfinite(result) or result < 0.02:
        raise argparse.ArgumentTypeError("seconds must be finite and at least 0.02")
    return result


def parser():
    p = argparse.ArgumentParser(
        description="MicroDuck Indoor Object Search RL — simulated object search with PPO"
    )
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check dependencies, bundled assets and model hashes")
    b = sub.add_parser("build", help="Generate the seeded room and its MJCF")
    b.add_argument("--seed", type=seed_value, default=0)
    t = sub.add_parser(
        "train", help="Train a new PPO supervisor in the fast SE(2) proxy"
    )
    t.add_argument("--steps", type=positive_int, default=3_000_000)
    t.add_argument("--envs", type=positive_int, default=256)
    t.add_argument("--seed", type=seed_value, default=42)
    t.add_argument("--horizon", type=positive_int, default=180)
    t.add_argument("--device", choices=["cpu", "cuda"], default=None)
    t.add_argument("--resume", type=Path, default=None)
    t.add_argument(
        "--output",
        type=Path,
        default=None,
        help="New directory; existing checkpoints are never overwritten",
    )
    e = sub.add_parser("evaluate", help="Evaluate a checkpoint on new proxy episodes")
    e.add_argument("checkpoint", type=Path)
    e.add_argument("--episodes", type=positive_int, default=128)
    e.add_argument("--seed", type=seed_value, default=3030)
    e.add_argument("--output", type=Path, default=ROOT / "outputs/heldout.json")
    r = sub.add_parser(
        "rehearse", help="Run the actual MuJoCo body with RGB perception"
    )
    r.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/release/best.pt")
    r.add_argument("--seed", type=seed_value, default=0)
    r.add_argument(
        "--goal",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="0=red ball, 1=blue box, 2=yellow cylinder",
    )
    r.add_argument("--seconds", type=positive_seconds, default=108)
    r.add_argument(
        "--viewer",
        action="store_true",
        help="Open the interactive MuJoCo viewer (requires a desktop)",
    )
    r.add_argument("--video", type=Path, default=ROOT / "outputs/rehearsal.mp4")
    r.add_argument("--no-video", action="store_true")
    r.add_argument("--controller", choices=["rl", "baseline"], default="rl")
    r.add_argument("--output", type=Path, default=ROOT / "outputs/rehearsal.json")
    sub.add_parser(
        "preview", help="Render overview, RGB, depth and segmentation images"
    )
    sub.add_parser(
        "blender",
        help="Regenerate the editable scene (requires the separate bpy environment)",
    )
    suite = sub.add_parser(
        "benchmark", help="Run a reproducible full-body comparison suite"
    )
    suite.add_argument("--seeds", type=positive_int, default=20)
    suite.add_argument("--start-seed", type=seed_value, default=0)
    suite.add_argument("--seconds", type=positive_seconds, default=108)
    suite.add_argument(
        "--checkpoint", type=Path, default=ROOT / "outputs/release/best.pt"
    )
    suite.add_argument("--controller", choices=["rl", "baseline"], default="rl")
    suite.add_argument("--output", type=Path, default=ROOT / "outputs/benchmark")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        dispatch(args)
    except (FileNotFoundError, ValueError, ImportError) as error:
        p.exit(2, f"error: {error}\n")


def dispatch(args):
    if args.command == "doctor":
        from .doctor import check

        report = check()
        print(json.dumps(report, indent=2))
        if not report["ok"]:
            raise SystemExit(1)
    elif args.command == "build":
        scene = build(args.seed)
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/room.xml"))
        print(
            json.dumps(
                dict(
                    scene="assets/room.xml",
                    bodies=model.nbody,
                    geoms=model.ngeom,
                    actuators=model.nu,
                    targets=scene["targets"],
                )
            )
        )
    elif args.command == "train":
        from .ppo import train

        if args.output is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            args.output = ROOT / "outputs/runs" / stamp
        args.output = str(args.output)
        args.resume = str(args.resume) if args.resume else None
        train(args)
    elif args.command == "evaluate":
        from .ppo import evaluate, load

        agent, _ = load(args.checkpoint)
        metrics = evaluate(agent, args.seed, args.episodes)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps({k: v for k, v in metrics.items() if k != "episodes_detail"}))
    elif args.command in ("rehearse", "preview"):
        from .physical import preview, rehearse

        preview() if args.command == "preview" else rehearse(args)
    elif args.command == "benchmark":
        from .benchmark import run

        run(args)
    elif args.command == "blender":
        from .blender import export

        export()


if __name__ == "__main__":
    main()
