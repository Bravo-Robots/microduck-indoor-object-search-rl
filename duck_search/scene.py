"""One scene description drives collisions, visuals, training and Blender export.

All room geometry is original. Robot mesh licensing remains upstream's.
Dimensions are metres, box sizes here are full extents, z is up.
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "third_party"
ROBOT = Path(os.environ.get("MICRODUCK_ROBOT_DIR", VENDOR / "robot"))
POLICIES = Path(os.environ.get("MICRODUCK_POLICY_DIR", VENDOR / "policies"))
SIZE = (5.0, 4.0)
COLORS = [(0.90, 0.07, 0.06, 1), (0.04, 0.20, 0.92, 1), (0.95, 0.72, 0.03, 1)]
OBJECTS = ["pelota_roja", "caja_azul", "cilindro_amarillo"]


def room(seed=0):
    rng = np.random.default_rng(seed)
    parts = []

    def box(name, pos, size, color, collision=True, category="furniture"):
        parts.append(
            dict(
                name=name,
                pos=list(pos),
                size=list(size),
                color=list(color),
                collision=collision,
                category=category,
                type="box",
            )
        )

    wood = (0.57, 0.38, 0.20, 1)
    black = (0.055, 0.065, 0.08, 1)
    # The front wall is kept low for the overview; it is a physical barrier.
    box(
        "floor",
        (0, 0, -0.04),
        (5.1, 4.1, 0.08),
        (0.43, 0.46, 0.49, 1),
        category="floor",
    )
    for name, pos, size in [
        ("wall_n", (0, 2.04, 0.65), (5.16, 0.08, 1.3)),
        ("wall_s", (0, -2.04, 0.08), (5.16, 0.08, 0.16)),
        ("wall_w", (-2.54, 0, 0.65), (0.08, 4.0, 1.3)),
        ("wall_e", (2.54, 0, 0.65), (0.08, 4.0, 1.3)),
    ]:
        box(name, pos, size, (0.78, 0.80, 0.79, 1), category="wall")
    # Video-inspired workshop props. The footprint list is derived from the
    # exact physical boxes at the sensor/collision height, never hand duplicated.
    for idx, (x, y, sx, sy, h) in enumerate(
        [
            (-1.1, 0.4, 0.72, 0.48, 0.35),
            (0.65, 0.25, 0.65, 0.43, 0.40),
            (1.65, 1.25, 0.58, 0.50, 0.32),
            (-1.65, 1.35, 0.55, 0.40, 0.25),
            (0.10, 1.50, 0.45, 0.35, 0.30),
        ]
    ):
        x, y = np.array([x, y]) + rng.uniform(-0.12, 0.12, 2)
        box(f"flight_case_{idx}", (x, y, h / 2), (sx, sy, h), black)
        box(
            f"case_lid_{idx}",
            (x, y, h + 0.008),
            (sx + 0.015, sy + 0.015, 0.016),
            (0.20, 0.23, 0.25, 1),
            False,
        )
        for dx in [-sx * 0.35, sx * 0.35]:
            box(
                f"case_latch_{idx}_{dx:.2f}",
                (x + dx, y - sy / 2 - 0.006, h * 0.70),
                (0.045, 0.018, 0.065),
                (0.66, 0.58, 0.33, 1),
                False,
            )
    box("table_top", (0.85, -1.12, 0.68), (1.20, 0.65, 0.055), wood)
    for i, dx in enumerate([-0.50, 0.50]):
        for j, dy in enumerate([-0.23, 0.23]):
            box(
                f"table_leg_{i}_{j}",
                (0.85 + dx, -1.12 + dy, 0.325),
                (0.06, 0.06, 0.65),
                black,
            )
    box("low_screen", (-0.55, -0.65, 0.23), (0.10, 0.75, 0.46), (0.34, 0.39, 0.42, 1))
    # Decorative worklight: monochromatic lamp avoids target colour aliases.
    box("light_base", (-2.08, 0.15, 0.025), (0.32, 0.25, 0.05), black)
    box("light_post", (-2.08, 0.15, 0.38), (0.035, 0.035, 0.70), (0.48, 0.49, 0.50, 1))
    box("worklight", (-2.08, 0.15, 0.77), (0.30, 0.08, 0.22), black)
    box(
        "worklight_face",
        (-2.08, 0.103, 0.77),
        (0.25, 0.012, 0.17),
        (0.92, 0.92, 0.86, 1),
        False,
    )
    # Three 5 mm treads: bypassable, initially excluded from the
    # search proxy's obstacle footprints. Never claim domestic-stair ability.
    for i in range(3):
        box(
            f"step_{i}",
            (-0.30 + i * 0.14, 1.00, 0.0025 * (i + 1)),
            (0.14, 0.40, 0.005 * (i + 1)),
            wood,
            category="terrain",
        )
    return dict(seed=seed, size=list(SIZE), parts=parts)


def obstacles(scene):
    result = []
    for p in scene["parts"]:
        if (
            p["collision"]
            and p["category"] == "furniture"
            and p["pos"][2] - p["size"][2] / 2 < 0.25
        ):
            x, y, _ = p["pos"]
            sx, sy, _ = p["size"]
            result.append([x - sx / 2, y - sy / 2, x + sx / 2, y + sy / 2])
    return np.asarray(result, dtype=np.float32)


def free_points(rng, boxes, n, clearance=0.22):
    pts = []
    for _ in range(10000):
        p = rng.uniform([-2.20, -1.70], [2.20, 1.70])
        inside = np.any(
            np.all(p >= boxes[:, :2] - clearance, axis=1)
            & np.all(p <= boxes[:, 2:] + clearance, axis=1)
        )
        if not inside and all(np.linalg.norm(p - q) > 0.25 for q in pts):
            pts.append(p)
            if len(pts) == n:
                return np.asarray(pts, dtype=np.float32)
    raise RuntimeError("Unable to sample connected free scene")


def scene_xml(scene, targets, spawn=(-1.9, -1.3, 0.0)):
    """Generate portable MJCF using the existing official mesh assets."""
    root = ET.parse(ROBOT / "scene.xml").getroot()
    root.find("include").set("file", str(ROBOT / "robot_groundcontact.xml"))
    # Included model's relative meshdir resolves against the generated main file.
    ET.SubElement(root, "compiler", meshdir=str(ROBOT / "assets"))
    visual = root.find("visual")
    ET.SubElement(visual, "quality", shadowsize="2048")
    ET.SubElement(visual, "map", znear="0.005")
    ET.SubElement(visual, "global", offwidth="1280", offheight="720") if visual.find(
        "global"
    ) is None else visual.find("global").attrib.update(offwidth="1280", offheight="720")
    world = root.find("worldbody")
    for node in list(world):
        world.remove(node)
    ET.SubElement(
        world,
        "light",
        pos="0 -1 3",
        dir="0 0 -1",
        diffuse="0.8 0.8 0.8",
        castshadow="true",
    )
    ET.SubElement(world, "light", pos="-2 2 2", dir="0 -1 -1", diffuse="0.45 0.45 0.45")
    # Fixed overview. Camera's local -z points down and local y points north.
    ET.SubElement(
        world,
        "camera",
        name="overview",
        pos="0 -3.6 5.0",
        xyaxes="1 0 0 0 0.81 0.586",
        fovy="56",
    )
    for p in scene["parts"]:
        ET.SubElement(
            world,
            "geom",
            name=p["name"],
            type="box",
            pos=fmt(p["pos"]),
            size=fmt(np.array(p["size"]) / 2),
            rgba=fmt(p["color"]),
            contype="1" if p["collision"] else "0",
            conaffinity="1" if p["collision"] else "0",
            friction="0.8 0.005 0.0001",
        )
    for i, (x, y) in enumerate(targets):
        typ = ["sphere", "box", "cylinder"][i]
        sz = ["0.045", "0.05 0.035 0.04", "0.035 0.055"][i]
        ET.SubElement(
            world,
            "geom",
            name=f"search_target_{i}",
            type=typ,
            pos=f"{x} {y} {(0.045, 0.04, 0.055)[i]}",
            size=sz,
            rgba=fmt(COLORS[i]),
            contype="0",
            conaffinity="0",
            group="1",
        )
    key = root.find("keyframe/key[@name='STAND']")
    q = np.fromstring(key.get("qpos"), sep=" ")
    q[:2] = spawn[:2]
    q[3:7] = [np.cos(spawn[2] / 2), 0, 0, np.sin(spawn[2] / 2)]
    key.set("qpos", fmt(q))
    return ET.tostring(root, encoding="unicode")


def fmt(xs):
    return " ".join(str(float(x)) for x in xs)


def scenario(seed=0):
    scene = room(seed)
    rng = np.random.default_rng(seed + 183)
    pts = free_points(rng, obstacles(scene), 4)
    # Keep the showroom scene repeatable; randomized training uses its own seeds.
    scene["targets"] = pts[:3].tolist()
    scene["spawn"] = [*pts[3].tolist(), 0.0]
    return scene


def build(seed=0):
    scene = scenario(seed)
    dest = ROOT / "assets"
    dest.mkdir(exist_ok=True)
    (dest / "room.json").write_text(json.dumps(scene, indent=2))
    (dest / "room.xml").write_text(scene_xml(scene, scene["targets"], scene["spawn"]))
    return scene
