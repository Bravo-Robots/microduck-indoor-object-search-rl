# Troubleshooting

| Symptom | Action |
|---|---|
| `Environment missing` | Run `uv sync --locked` in the checkout. |
| OpenGL/EGL initialization fails | On CPU Linux install `libosmesa6`, then set `MUJOCO_GL=osmesa`. On a desktop `MUJOCO_GL=glfw` is another option. |
| Viewer cannot open | Run inside a graphical desktop or omit `--viewer` for headless execution. |
| CUDA is unavailable | Inference and short training support CPU. Use `train --device cpu`. |
| Checkpoint dimension mismatch | Use the bundled 45-input supervisor, not a 61-input motor policy. |
| Training directory already has checkpoints | Choose a new `--output`; use `--resume` to warm-start a previous run. |
| `doctor` reports a modified bundled file | Compare with `third_party/manifest.json` or restore the original asset. Regenerate the manifest only for an intentional vendored update. |
| Blender module missing | Use the separate Python 3.11 `bpy==4.5.15` environment from the README. |
| A search falls or times out | Keep its JSON report and seed. These are evaluated failure modes, not installation errors. |

For a useful bug report include the command, `bash run.sh doctor` output, renderer,
seed, goal and checkpoint SHA-256. Remove unrelated local credentials from any
logs before attaching them. The project itself requires no model API keys.
