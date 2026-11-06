# Contributing

Install with `uv sync --locked`. Work from the checkout root.

```bash
uv run ruff check duck_search scripts tests
uv run ruff format --check duck_search scripts tests
uv run python -m unittest discover -s tests -v
bash run.sh doctor
```

Before a long training run, run the CPU smoke test:

```bash
bash run.sh train --device cpu --steps 1536 --envs 8 --output outputs/my-smoke
```

Use a new output directory for each experiment. Preserve the supervisor's 45-input,
6-action contract or version the model interface. Motor policies use a different
61-input, 14-output contract and embed their observation normalizer.

Keep sensor inputs separate from evaluator ground truth. Use the same seeds,
goal classes and depth guard for controller comparisons. Report failures,
contacts, falls and false confirmations alongside success counts. Never promote
proxy results as hardware results.

Third-party robot assets are kept unchanged with their licences and hashes. Put
new original props in the room description and regenerate them with the documented
Blender workflow. Do not add local environments, video-reference screenshots or
temporary experiment outputs to Git.
