# Changelog

## 0.2.1

- Add a sensor-to-actuator integration test requiring visible-goal approach,
  measured forward motion, independent confirmation, no contact and no fall.
- Separate that controlled integration test from cluttered search performance.
  A 20-second mission budget proved sensitive to host numerical differences.
- Keep training weights and published benchmark measurements unchanged.
- Publish synchronized GitHub mirrors, Hugging Face model metadata and ZIP hashes.

## 0.2.0

- Update visit memory during full-body execution, matching training observations.
- Reject brown furniture in the yellow-object RGB detector; add a regression test.
- Sample distinct spawn and object sites; remove trivial overlapping tasks.
- Add the same depth-only braking guard to RL and reactive physical controllers.
- Preserve training step counts when resuming and reject accidental checkpoint overwrite.
- Add validated CLI arguments, nested output creation and clear termination reasons.
- Remove dependencies on adjacent workshop folders; verify bundled artifact hashes.
- Retrain the supervisor and evaluate a larger fixed-seed suite; see benchmark reports.
- Add locked installation, formatting, tests, CI and reproducible publication packaging.

## 0.1.0

Initial local prototype: workshop room, Blender scene, PPO supervisor and 12-case
full-body comparison. Historical results are retained under `benchmarks/v0.1/`.
