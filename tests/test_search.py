import unittest

import mujoco
import numpy as np
import torch

from duck_search.env import OBS_SIZE, SearchEnv
from duck_search.physical import command, detect, robot, tick
from duck_search.ppo import Agent, load
from duck_search.scene import ROOT, build


class SearchContracts(unittest.TestCase):
    def test_hidden_goal_is_not_an_actor_input(self):
        e = SearchEnv(1, seed=11)
        e.pos[:] = 0
        e.yaw[:] = 0
        e.seen[:] = False
        e.memory[:] = 0
        e.visits[:] = 0
        # Two invisible positions behind the same robot: observations must agree.
        e.targets[0, e.goal[0]] = torch.tensor([-1.0, -0.5])
        a = e.observe().clone()
        e.targets[0, e.goal[0]] = torch.tensor([-1.0, 0.5])
        b = e.observe().clone()
        self.assertTrue(torch.equal(a, b))

    def test_obstacle_occludes_target(self):
        e = SearchEnv(1, seed=11)
        e.pos[:] = 0
        e.yaw[:] = 0
        e.boxes[:] = torch.tensor([10.0, 10.0, 11.0, 11.0])
        e.boxes[0, 0] = torch.tensor([0.4, -0.3, 0.6, 0.3])
        e.targets[0, e.goal[0]] = torch.tensor([1.0, 0.0])
        self.assertFalse(bool(e.sensing()[0][0]))
        e.targets[0, e.goal[0]] = torch.tensor([0.2, 0.0])
        self.assertTrue(bool(e.sensing()[0][0]))

    def test_observation_is_finite_and_step_does_not_tunnel(self):
        e = SearchEnv(8, seed=7)
        for _ in range(30):
            obs, r, done, _ = e.step(torch.zeros(8, dtype=torch.long))
            self.assertEqual(tuple(obs.shape), (8, OBS_SIZE))
            self.assertTrue(torch.isfinite(obs).all())
            self.assertTrue(torch.isfinite(r).all())
            self.assertFalse(e.collides(e.pos).any())

    def test_confirmation_requires_visible_near_target(self):
        a = Agent()
        obs = torch.zeros(1, OBS_SIZE)
        obs[:, :12] = 1
        logits, _ = a(obs)
        self.assertLess(float(logits[0, 5].detach()), -1e8)
        obs[0, 12] = 1
        obs[0, 14] = 0.1
        logits, _ = a(obs)
        self.assertGreater(float(logits[0, 5].detach()), -1e8)

    def test_rgb_detector_ignores_hidden_or_wrong_colour(self):
        rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        dep = np.ones((240, 320), dtype=np.float32)
        self.assertFalse(detect(rgb, dep, 0)[0])
        rgb[100:120, 150:170, 0] = 220
        self.assertTrue(detect(rgb, dep, 0)[0])
        self.assertFalse(detect(rgb, dep, 1)[0])

    def test_yellow_detector_rejects_brown_furniture(self):
        rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        depth = np.ones((240, 320), dtype=np.float32)
        # Measured rendered colours: wooden treads versus the yellow target.
        rgb[100:140, 140:180] = [129, 86, 45]
        self.assertFalse(detect(rgb, depth, 2)[0])
        rgb[100:140, 140:180] = [183, 138, 6]
        self.assertTrue(detect(rgb, depth, 2)[0])

    def test_scene_loads_with_real_robot(self):
        build(0)
        m = mujoco.MjModel.from_xml_path(str(ROOT / "assets/room.xml"))
        self.assertEqual(m.nu, 14)
        self.assertGreater(m.ngeom, 100)

    def test_motor_policy_actually_moves_free_floating_robot(self):
        build(0)
        m, d, b, p = robot(ROOT / "assets/room.xml")
        start = d.qpos[:2].copy()
        command(p, 0.2, 0)
        for _ in range(150):
            tick(m, d, b, p)
        self.assertGreater(np.linalg.norm(d.qpos[:2] - start), 0.15)
        self.assertGreater(d.qpos[2], 0.07)

    def test_release_onnx_matches_torch(self):
        import onnxruntime as ort

        release = ROOT / "outputs/release"
        if not (release / "best.pt").exists():
            self.skipTest("No trained release available")
        agent, _ = load(release / "best.pt")
        obs = SearchEnv(16, seed=2027).observe()
        session = ort.InferenceSession(
            str(release / "search_policy.onnx"), providers=["CPUExecutionProvider"]
        )
        with torch.no_grad():
            expected = agent(obs)[0].numpy()
        actual = np.concatenate(
            [
                session.run(None, {session.get_inputs()[0].name: o[None]})[0]
                for o in obs.numpy()
            ]
        )
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
