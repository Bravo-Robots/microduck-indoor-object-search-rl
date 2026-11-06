"""GPU-vectorized SE(2) search curriculum (NOT a rigid-body simulator).

Actor sees idealized forward depth, visible colour-object detections and an
odometry-based visitation memory. Hidden object coordinates are never exposed.
This trains the supervisory policy; MuJoCo separately tests locomotion transfer.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .scene import free_points, obstacles, room

DT = 0.6
MAX_RANGE = 2.5
FOV = math.radians(90)
# vx, yaw rate. No sideways primitive: easier to transfer and avoids side drift.
COMMANDS = np.array(
    [[0.20, 0], [0, 0.9], [0, -0.9], [0.12, 0.65], [0.12, -0.65], [0, 0]],
    dtype=np.float32,
)
OBS_SIZE = 45


def wrap(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


class SearchEnv:
    def __init__(self, n=256, device="cpu", seed=0, max_steps=180):
        self.n = n
        self.device = torch.device(device)
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        layouts = []
        samples = []
        # Pool deterministic layouts so resets do not cross CPU/GPU boundaries.
        for k in range(128):
            b = obstacles(room(seed * 1000 + k))
            layouts.append(b)
            samples.append(free_points(self.rng, b, 64))
        self.layouts = torch.tensor(np.array(layouts), device=self.device)
        self.samples = torch.tensor(np.array(samples), device=self.device)
        self.boxes = torch.empty(n, *self.layouts.shape[1:], device=self.device)
        self.pos = torch.zeros(n, 2, device=self.device)
        self.yaw = torch.zeros(n, device=self.device)
        self.targets = torch.zeros(n, 3, 2, device=self.device)
        self.goal = torch.zeros(n, dtype=torch.long, device=self.device)
        self.steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self.seen = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.memory = torch.zeros(n, 2, device=self.device)
        self.visits = torch.zeros(n, 16, 16, device=self.device)
        self.last = torch.zeros(n, 6, device=self.device)
        self.ever_visible = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.ids = torch.arange(n, device=self.device)
        self.ray_angles = torch.linspace(-FOV / 2, FOV / 2, 12, device=self.device)
        self.commands = torch.tensor(COMMANDS, device=self.device)
        self.reset()

    def rand(self, *shape):
        return torch.rand(*shape, device=self.device, generator=self.generator)

    def reset(self, ids=None, easy=0.0):
        ids = self.ids if ids is None else ids
        k = len(ids)
        li = torch.randint(128, (k,), device=self.device, generator=self.generator)
        # Distinct sites: never place the robot and target on the same sampled
        # point, or two objects at an indistinguishable position.
        si = torch.multinomial(
            torch.ones(k, 64, device=self.device),
            4,
            replacement=False,
            generator=self.generator,
        )
        points = self.samples[li[:, None], si]
        self.boxes[ids] = self.layouts[li]
        self.pos[ids] = points[:, 0]
        self.targets[ids] = points[:, 1:]
        self.goal[ids] = torch.randint(
            3, (k,), device=self.device, generator=self.generator
        )
        yaw = (self.rand(k) * 2 - 1) * math.pi
        if easy:
            delta = self.targets[ids, self.goal[ids]] - self.pos[ids]
            direction = torch.atan2(delta[:, 1], delta[:, 0])
            yaw = torch.where(
                self.rand(k) < easy, direction + (self.rand(k) - 0.5) * 0.5, yaw
            )
        self.yaw[ids] = yaw
        self.steps[ids] = 0
        self.seen[ids] = False
        self.ever_visible[ids] = False
        self.memory[ids] = 0
        self.visits[ids] = 0
        self.last[ids] = 0
        g = self.grid(self.pos[ids])
        self.visits[ids, g[:, 0], g[:, 1]] = 1
        return self.observe()

    def raycast(self, angles):
        """Horizontal rays against exact furniture footprint + room walls."""
        dirs = torch.stack([torch.cos(angles), torch.sin(angles)], -1)
        # Signed epsilon preserves intersections for nearly parallel rays.
        epsilon = torch.where(dirs >= 0, 1e-7, -1e-7)
        safe = torch.where(dirs.abs() < 1e-7, epsilon, dirs)
        low = (self.boxes[:, None, :, :2] - self.pos[:, None, None, :]) / safe[
            :, :, None, :
        ]
        high = (self.boxes[:, None, :, 2:] - self.pos[:, None, None, :]) / safe[
            :, :, None, :
        ]
        enter = torch.minimum(low, high).amax(-1)
        leave = torch.maximum(low, high).amin(-1)
        dist = torch.where(
            (leave >= enter) & (leave > 0), enter.clamp_min(0), MAX_RANGE
        ).amin(-1)
        wall = torch.where(
            safe > 0,
            torch.tensor([2.5, 2.0], device=self.device),
            torch.tensor([-2.5, -2.0], device=self.device),
        )
        wall_dist = ((wall - self.pos[:, None, :]) / safe).amin(-1).clamp_min(0)
        return torch.minimum(dist, wall_dist).clamp(0, MAX_RANGE)

    def sensing(self):
        target = self.targets[self.ids, self.goal]
        delta = target - self.pos
        dist = torch.linalg.vector_norm(delta, dim=-1)
        angle = torch.atan2(delta[:, 1], delta[:, 0])
        bearing = wrap(angle - self.yaw)
        unobstructed = self.raycast(angle[:, None])[:, 0] + 0.03 >= dist
        visible = (bearing.abs() < FOV / 2) & (dist < MAX_RANGE) & unobstructed
        return visible, bearing, dist

    def grid(self, points):
        scale = torch.tensor([5.0, 4.0], device=self.device)
        return (((points + scale / 2) / scale) * 16).long().clamp(0, 15)

    def observe(self, external_detection=None, external_depth=None):
        # External physical rehearsal injects RGB detector and MuJoCo depth.
        if external_detection is None:
            visible, bearing, dist = self.sensing()
        else:
            visible, bearing, dist = external_detection
        angle = self.yaw + bearing
        detected = (
            self.pos
            + torch.stack([torch.cos(angle), torch.sin(angle)], -1) * dist[:, None]
        )
        self.memory = torch.where(visible[:, None], detected, self.memory)
        self.seen |= visible
        delta = self.memory - self.pos
        memdist = torch.linalg.vector_norm(delta, dim=-1)
        membearing = wrap(torch.atan2(delta[:, 1], delta[:, 0]) - self.yaw)
        targetobs = torch.stack(
            [
                visible.float(),
                torch.where(visible, bearing / math.pi, 0),
                torch.where(visible, dist / 3, 0),
            ],
            -1,
        )
        memoryobs = torch.stack(
            [
                self.seen.float(),
                torch.where(self.seen, membearing / math.pi, 0),
                torch.where(self.seen, memdist.clamp_max(6) / 3, 0),
            ],
            -1,
        )
        angles = (
            self.yaw[:, None]
            + torch.arange(8, device=self.device)[None, :] * math.pi / 4
        )
        offsets = torch.stack([torch.cos(angles), torch.sin(angles)], -1)
        probes = torch.cat(
            [
                self.pos[:, None, :] + offsets * 0.35,
                self.pos[:, None, :] + offsets * 0.7,
            ],
            1,
        )
        ix = self.grid(probes)
        history = (
            self.visits[self.ids[:, None], ix[:, :, 0], ix[:, :, 1]].clamp_max(4) / 4
        )
        depth = (
            self.raycast(self.yaw[:, None] + self.ray_angles)
            if external_depth is None
            else external_depth
        )
        pose = torch.stack(
            [
                self.pos[:, 0] / 2.5,
                self.pos[:, 1] / 2,
                torch.sin(self.yaw),
                torch.cos(self.yaw),
            ],
            -1,
        )
        obs = torch.cat(
            [
                depth / MAX_RANGE,
                targetobs,
                memoryobs,
                history,
                pose,
                self.last,
                self.steps[:, None] / self.max_steps,
            ],
            -1,
        )
        assert obs.shape == (self.n, OBS_SIZE)
        return obs

    def collides(self, pos):
        inside = (
            (
                (pos[:, None, :] >= self.boxes[:, :, :2] - 0.13)
                & (pos[:, None, :] <= self.boxes[:, :, 2:] + 0.13)
            )
            .all(-1)
            .any(-1)
        )
        return inside | (pos[:, 0].abs() > 2.35) | (pos[:, 1].abs() > 1.85)

    def record_visit(self):
        """Record a control-step visit, also used by the physical executor."""
        cell = self.grid(self.pos)
        novel = self.visits[self.ids, cell[:, 0], cell[:, 1]] == 0
        self.visits[self.ids, cell[:, 0], cell[:, 1]] += 1
        return novel

    def step(self, action, easy=0.0):
        before_seen = self.seen.clone()
        previous_dist = torch.linalg.vector_norm(self.memory - self.pos, dim=-1)
        cmd = self.commands[action]
        # Substeps prevent tunnelling through thin table legs/screens.
        collision = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        for _ in range(3):
            self.yaw = wrap(self.yaw + cmd[:, 1] * DT / 3)
            proposal = (
                self.pos
                + torch.stack([torch.cos(self.yaw), torch.sin(self.yaw)], -1)
                * cmd[:, :1]
                * DT
                / 3
            )
            hit = self.collides(proposal)
            collision |= hit
            self.pos = torch.where(hit[:, None], self.pos, proposal)
        self.steps += 1
        self.last = torch.nn.functional.one_hot(action, 6).float()
        visible, bearing, dist = self.sensing()
        success = (action == 5) & visible & (dist < 0.55)
        discovery = visible & ~self.ever_visible
        self.ever_visible |= visible
        novel = self.record_visit()
        new_dist = torch.linalg.vector_norm(self.memory - self.pos, dim=-1)
        progress = torch.where(
            before_seen, (previous_dist - new_dist).clamp(-0.15, 0.15), 0
        )
        reward = (
            -0.008
            + novel.float() * 0.06
            + discovery.float() * 1.5
            + progress * 2.5
            - collision.float() * 0.15
            + success.float() * 15
        )
        reward -= ((action == 5) & ~success).float() * 0.1
        done = success | (self.steps >= self.max_steps)
        info = {
            "success": success.clone(),
            "done": done.clone(),
            "collision": collision,
            "steps": self.steps.clone(),
            "visible": visible,
        }
        # Capture terminal observations before autoreset for correct timeout bootstrapping.
        obs = self.observe()
        info["terminal_obs"] = obs.clone()
        if done.any():
            self.reset(self.ids[done], easy=easy)
            obs = self.observe()
        return obs, reward, done, info
