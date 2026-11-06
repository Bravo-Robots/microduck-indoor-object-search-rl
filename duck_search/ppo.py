"""Small inspectable PPO trainer; categorical high-level actions, GAE, clipping."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .env import OBS_SIZE, SearchEnv


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_SIZE, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh()
        )
        self.actor = nn.Linear(128, 6)
        self.critic = nn.Linear(128, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.orthogonal_(self.critic.weight, 1.0)

    def forward(self, x):
        h = self.net(x)
        logits = self.actor(h)
        # Explicit depth/confirmation constraints, used in training AND replay.
        # They cannot reveal a hidden target: all features are sensor inputs.
        clearance = x[:, 2:10].amin(-1) * 2.5
        allowed = torch.ones_like(logits, dtype=torch.bool)
        allowed[:, 0] = clearance > 0.29
        allowed[:, 3] = clearance > 0.25
        allowed[:, 4] = clearance > 0.25
        allowed[:, 5] = (x[:, 12] > 0.5) & (x[:, 14] * 3 < 0.55)
        logits = logits.masked_fill(~allowed, -1e9)
        return logits, self.critic(h).squeeze(-1)


def load(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("observation_size") != OBS_SIZE:
        raise ValueError(f"Expected a {OBS_SIZE}-observation search checkpoint: {path}")
    agent = Agent().to(device)
    agent.load_state_dict(checkpoint["model"])
    agent.eval()
    return agent, checkpoint


def evaluate(agent, seed=937, n=128):
    device = next(agent.parameters()).device
    env = SearchEnv(n, str(device), seed=seed)
    obs = env.observe()
    done_once = torch.zeros(n, dtype=torch.bool, device=device)
    results = []
    collisions = 0
    steps = 0
    with torch.inference_mode():
        for _ in range(env.max_steps):
            action = agent(obs)[0].argmax(-1)
            obs, _, done, info = env.step(action)
            active = ~done_once
            collisions += int((info["collision"] & active).sum())
            steps += int(active.sum())
            for i in torch.where(done & active)[0].tolist():
                results.append(
                    dict(success=bool(info["success"][i]), steps=int(info["steps"][i]))
                )
            done_once |= done
            if done_once.all():
                break
    return dict(
        episodes=len(results),
        success_rate=sum(r["success"] for r in results) / max(1, len(results)),
        mean_steps=float(np.mean([r["steps"] for r in results])),
        collision_step_rate=collisions / max(1, steps),
        seed=seed,
        simulator="SE2 search proxy, ideal forward-depth and object detector",
        episodes_detail=results,
    )


def train(args):
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = (
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out = Path(args.output)
    if any((out / name).exists() for name in ("latest.pt", "best.pt")):
        raise ValueError(
            f"Training output already contains checkpoints; choose a new --output: {out}"
        )
    out.mkdir(parents=True, exist_ok=True)
    previous_steps = 0
    agent = Agent().to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5)
    if args.resume:
        old = torch.load(args.resume, map_location=device, weights_only=True)
        agent.load_state_dict(old["model"])
        optimizer.load_state_dict(old["optimizer"])
        previous_steps = int(old.get("steps", 0))
    env = SearchEnv(args.envs, device, args.seed, max_steps=args.horizon)
    obs = env.reset(easy=0.8)
    nsteps = 96
    n = args.envs
    updates = max(1, args.steps // (n * nsteps))
    global_step = previous_steps
    start = time.time()
    best = -1.0
    shape = (nsteps, n)
    ob = torch.empty(*shape, OBS_SIZE, device=device)
    ac = torch.empty(*shape, dtype=torch.long, device=device)
    lp = torch.empty(*shape, device=device)
    rw = torch.empty_like(lp)
    dn = torch.empty_like(lp)
    vl = torch.empty_like(lp)
    logs = []
    for update in range(updates):
        successes = []
        lens = []
        easy = max(0.0, 0.8 * (1 - update / max(1, updates * 0.35)))
        for t in range(nsteps):
            with torch.no_grad():
                logits, value = agent(obs)
                dist = Categorical(logits=logits)
                action = dist.sample()
            ob[t] = obs
            ac[t] = action
            lp[t] = dist.log_prob(action)
            vl[t] = value
            obs, reward, done, info = env.step(action, easy=easy)
            # Truncations bootstrap the terminal state, true successes do not.
            with torch.no_grad():
                terminal_value = agent(info["terminal_obs"])[1]
            rw[t] = reward + 0.99 * terminal_value * (done & ~info["success"]).float()
            dn[t] = done.float()
            if done.any():
                successes.extend(info["success"][done].float().tolist())
                lens.extend(info["steps"][done].tolist())
        with torch.no_grad():
            next_value = agent(obs)[1]
        adv = torch.zeros_like(rw)
        last = torch.zeros(n, device=device)
        for t in reversed(range(nsteps)):
            nv = next_value if t == nsteps - 1 else vl[t + 1]
            delta = rw[t] + 0.99 * nv * (1 - dn[t]) - vl[t]
            last = delta + 0.99 * 0.95 * (1 - dn[t]) * last
            adv[t] = last
        returns = adv + vl
        flatob = ob.reshape(-1, OBS_SIZE)
        flatac = ac.flatten()
        flatlp = lp.flatten()
        flatadv = adv.flatten()
        flatret = returns.flatten()
        flatadv = (flatadv - flatadv.mean()) / (flatadv.std() + 1e-8)
        total = n * nsteps
        for epoch in range(4):
            inds = torch.randperm(total, device=device)
            for ids in inds.split(min(4096, total // 4)):
                logits, value = agent(flatob[ids])
                dist = Categorical(logits=logits)
                ratio = (dist.log_prob(flatac[ids]) - flatlp[ids]).exp()
                policy = torch.maximum(
                    -flatadv[ids] * ratio, -flatadv[ids] * ratio.clamp(0.8, 1.2)
                ).mean()
                value_loss = 0.5 * ((value - flatret[ids]) ** 2).mean()
                loss = policy + 0.5 * value_loss - 0.015 * dist.entropy().mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
        global_step += total
        row = dict(
            update=update + 1,
            steps=global_step,
            reward=float(rw.mean()),
            success=float(np.mean(successes)) if successes else 0.0,
            episode_steps=float(np.mean(lens)) if lens else 0.0,
            fps=int((global_step - previous_steps) / (time.time() - start)),
            easy=easy,
        )
        logs.append(row)
        checkpoint = dict(
            model=agent.state_dict(),
            optimizer=optimizer.state_dict(),
            steps=global_step,
            seed=args.seed,
            observation_size=OBS_SIZE,
            level="supervisory-search-SE2",
            config={
                **vars(args),
                "output": out.name,
                "resume": Path(args.resume).name if args.resume else None,
            },
        )
        if update % 5 == 0 or update == updates - 1:
            print(json.dumps(row), flush=True)
            torch.save(checkpoint, out / "latest.pt")
        if (update + 1) % 25 == 0 or update == updates - 1:
            metrics = evaluate(agent, n=64)
            print(
                "HELDOUT "
                + json.dumps(
                    {k: v for k, v in metrics.items() if k != "episodes_detail"}
                ),
                flush=True,
            )
            if metrics["success_rate"] > best:
                best = metrics["success_rate"]
                torch.save(checkpoint, out / "best.pt")
                (out / "evaluation.json").write_text(json.dumps(metrics, indent=2))
            with (out / "training.csv").open("w") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(logs)
    # Separate high-level ONNX. It is intentionally NOT a 61->14 motor policy.
    agent, _ = load(out / "best.pt")

    class Policy(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, obs):
            return self.model(obs)[0]

    torch.onnx.export(
        Policy(agent),
        torch.zeros(1, OBS_SIZE),
        str(out / "search_policy.onnx"),
        input_names=["search_observation"],
        output_names=["action_logits"],
        opset_version=17,
        dynamo=False,
    )
    (out / "manifest.json").write_text(
        json.dumps(
            dict(
                kind="supervisory_search",
                input=[1, OBS_SIZE],
                output=[1, 6],
                locomotion_policy="pollen-robotics/microduck-policies",
                direct_robotctl_load=False,
                training="SE2 proxy; see separate MuJoCo rehearsal report",
            ),
            indent=2,
        )
    )
    return out
