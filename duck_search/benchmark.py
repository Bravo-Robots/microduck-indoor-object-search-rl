"""Fixed-seed full-body evaluation, with the same depth guard for both agents."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from . import __version__
from .physical import rehearse


def run(args):
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    checkpoint = Path(args.checkpoint)
    checkpoint_hash = None
    if args.controller == "rl":
        payload = checkpoint.read_bytes()
        checkpoint_hash = hashlib.sha256(payload).hexdigest()
        checkpoint = folder / "checkpoint.pt"
        checkpoint.write_bytes(payload)
    for seed in range(args.start_seed, args.start_seed + args.seeds):
        for goal in range(3):
            output = folder / f"seed{seed}_goal{goal}.json"
            result = rehearse(
                SimpleNamespace(
                    seed=seed,
                    goal=goal,
                    seconds=args.seconds,
                    no_video=True,
                    viewer=False,
                    checkpoint=checkpoint,
                    controller=args.controller,
                    output=output,
                    quiet=True,
                )
            )
            record = {k: v for k, v in result.items() if k != "trace"}
            record["report"] = output.name
            results.append(record)
            print(
                json.dumps(
                    dict(
                        completed=len(results),
                        total=args.seeds * 3,
                        seed=seed,
                        goal=goal,
                        status=result["termination_reason"],
                    )
                ),
                flush=True,
            )
    successes = sum(r["success"] for r in results)
    summary = dict(
        version=__version__,
        episodes=len(results),
        successes=successes,
        success_rate=successes / len(results),
        falls=sum(r["fall"] for r in results),
        false_confirmations=sum(r["false_confirmations"] for r in results),
        episodes_with_contacts=sum(r["contact_steps"] > 0 for r in results),
        controller=args.controller,
        start_seed=args.start_seed,
        seed_count=args.seeds,
        seconds_per_episode=args.seconds,
        checkpoint_sha256=checkpoint_hash,
        simulator="MuJoCo full body + BAM + RGB detector; ideal odometry/depth; depth guard enabled",
        results=results,
    )
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}), flush=True)
    return summary
