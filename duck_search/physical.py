"""Actual free-floating MuJoCo robot with upstream BAM + 50 Hz motor policy.

The supervisory search policy is separate from the 61->14 motor ONNX.
RGB colour detection is intentionally a narrow baseline, not open-vocabulary AI.
Perfect simulator odometry and horizontal depth rays are explicitly labelled.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import os
import time
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageDraw

from .env import COMMANDS, FOV, MAX_RANGE, SearchEnv
from .guard import guarded_action
from .ppo import load
from .scene import OBJECTS, POLICIES, ROOT, build, obstacles, scenario, scene_xml


def upstream():
    path = ROOT / "third_party/infer_policy.py"
    if not path.exists():
        raise FileNotFoundError(f"Official inference adapter missing: {path}")
    spec = importlib.util.spec_from_file_location("microduck_official_inference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def robot(scene_path):
    up = upstream()
    with contextlib.redirect_stdout(io.StringIO()):
        bm = up.load_bam_model(200.0, 7.4, None)
        model, data, bam, _ = up.load_mujoco_with_bam(
            str(scene_path), bm, 0.005, 0.1, 6.0
        )
        # The workshop's 1499-iteration gait passes the displacement calibration
        # on this model. The bundled alpha gait barely moves in this checkout.
        gait = Path(
            os.environ.get(
                "MICRODUCK_WALK_POLICY", POLICIES / "progresion_4_mejor.onnx"
            )
        )
        policy = up.PolicyInference(
            model,
            data,
            walking_onnx_path=str(gait),
            standing_onnx_path=str(POLICIES / "alpha_stand.onnx"),
            bam_ctrl=bam,
            use_projected_gravity=True,
            new_cmd_obs=True,
        )
        mujoco.mj_resetDataKeyframe(
            model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
        )
        data.qpos[2] = 0.125
        bam.reset(data.qpos)
        policy.set_position_targets(policy.default_pose)
        mujoco.mj_forward(model, data)
    # Match 90 degree horizontal sensing model for 4:3 camera images.
    camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    model.cam_fovy[camera] = math.degrees(2 * math.atan(math.tan(FOV / 2) * 240 / 320))
    # The upstream display camera points into the head in this checkout.
    # Calibrate its optical axes at HOME: +x robot forward, image right=-y,
    # image up=+z. Keep the mount on the articulated head thereafter.
    yaw = yaw_of(data)
    forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    up_axis = np.array([0.0, 0.0, 1.0])
    desired = np.column_stack([right, up_axis, -forward])
    body = int(model.cam_bodyid[camera])
    local = data.xmat[body].reshape(3, 3).T @ desired
    mujoco.mju_mat2Quat(model.cam_quat[camera], local.flatten())
    mujoco.mj_forward(model, data)
    policy.search_gait_path = str(gait)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    policy.search_forward_session = policy.walking_session
    policy.search_turn_session = ort.InferenceSession(
        str(POLICIES / "velstand.onnx"),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    return model, data, bam, policy


def command(policy, vx, yaw):
    policy.vel_cmd = np.array([vx, 0, yaw], dtype=np.float32)
    # Suppress upstream's per-command logs; no control code is changed.
    with contextlib.redirect_stdout(io.StringIO()):
        policy._update_policy_session()
        policy._update_command()


def primitive_command(policy, action):
    """Calibrated command adapters, verified by the motor displacement test.

    These compensate the static-friction dead zone; they do not move/teleport
    the simulated body. Every displacement still comes from motor torques.
    """
    _, yaw = COMMANDS[action]
    policy.walking_session = (
        policy.search_turn_session
        if action in (1, 2)
        else policy.search_forward_session
    )
    if policy.current_policy == "walking":
        policy.ort_session = policy.walking_session
    motor_vx = 0.30 if action == 0 else 0.24 if action in (3, 4) else 0.0
    command(policy, motor_vx, float(yaw))


def tick(model, data, bam, policy):
    action = policy.infer()
    policy.apply_action(action)
    for _ in range(4):
        bam.update()
        mujoco.mj_step(model, data)


def yaw_of(data):
    w, x, y, z = data.qpos[3:7]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def depths(model, data, yaw):
    out = []
    geom = np.array([-1], dtype=np.int32)
    # Group 0 = room; groups 1/2/3 = targets/robot visuals/robot collisions.
    groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    pos = np.array([data.qpos[0], data.qpos[1], 0.16])
    for a in np.linspace(-FOV / 2, FOV / 2, 12) + yaw:
        d = mujoco.mj_ray(
            model, data, pos, np.array([np.cos(a), np.sin(a), 0.0]), groups, 1, -1, geom
        )
        out.append(MAX_RANGE if d < 0 else min(MAX_RANGE, d))
    return np.asarray(out, dtype=np.float32)


def detect(rgb, depth, goal):
    """Only rendered RGB/depth enters detector: never segmentation or geom IDs."""
    r, g, b = rgb.astype(float).transpose(2, 0, 1)
    masks = [
        (r > 85) & (r > g * 2.7) & (r > b * 2.7),
        (b > 85) & (b > r * 1.7) & (b > g * 1.4),
        (r > 100) & (g > 85) & (b < g * 0.25) & (g > r * 0.70),
    ]
    # Fixed robot-body exclusion at <22 cm prevents the orange beak becoming
    # a false red-object detection. This is a geometric near-field mask.
    mask = masks[goal] & (depth < MAX_RANGE) & (depth > 0.22)
    ys, xs = np.where(mask)
    if len(xs) < 12:
        return False, 0.0, 0.0
    # Pinhole horizontal bearing and range from the same visible pixels.
    u = float(np.median(xs))
    cx = (rgb.shape[1] - 1) / 2
    fx = rgb.shape[1] / (2 * math.tan(FOV / 2))
    bearing = -math.atan((u - cx) / fx)
    distance = float(np.median(depth[mask])) / max(0.5, math.cos(bearing))
    return True, bearing, distance


def sensor_frame(renderer, model, data, goal):
    renderer.update_scene(data, camera="head_camera")
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    return rgb, depth, detect(rgb, depth, goal)


def preview():
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    # Regenerate include paths when this repository has moved to another host.
    build(0)
    m, d, b, p = robot(ROOT / "assets/room.xml")
    command(p, 0, 0)
    for _ in range(50):
        tick(m, d, b, p)
    renderer = mujoco.Renderer(m, height=720, width=1280)
    renderer.update_scene(d, camera="overview")
    Image.fromarray(renderer.render()).save(ROOT / "outputs/room_preview.png")
    renderer.close()
    # Exact compiled mesh coordinates + poses for the editable Blender scene.
    meshes = []
    for geom in range(m.ngeom):
        mesh = int(m.geom_dataid[geom])
        if m.geom_bodyid[geom] == 0 or m.geom_group[geom] != 2 or mesh < 0:
            continue
        v = int(m.mesh_vertadr[mesh])
        nv = int(m.mesh_vertnum[mesh])
        f = int(m.mesh_faceadr[mesh])
        nf = int(m.mesh_facenum[mesh])
        mat = int(m.geom_matid[geom])
        rgba = m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[geom]
        meshes.append(
            dict(
                name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mesh),
                vertices=m.mesh_vert[v : v + nv].tolist(),
                faces=m.mesh_face[f : f + nf].tolist(),
                position=d.geom_xpos[geom].tolist(),
                rotation=d.geom_xmat[geom].reshape(3, 3).tolist(),
                rgba=rgba.tolist(),
            )
        )
    (ROOT / "assets/robot_blender_meshes.json").write_text(json.dumps(meshes))
    cam = mujoco.Renderer(m, height=240, width=320)
    rgb, depth, det = sensor_frame(cam, m, d, 0)
    Image.fromarray(rgb).save(ROOT / "outputs/head_rgb.png")
    Image.fromarray(np.uint8(np.clip(depth / 3, 0, 1) * 255)).save(
        ROOT / "outputs/head_depth.png"
    )
    cam.enable_segmentation_rendering()
    cam.update_scene(d, camera="head_camera")
    seg = cam.render()
    cam.disable_segmentation_rendering()
    palette = np.random.default_rng(7).integers(
        40, 230, (m.ngeom + 1, 3), dtype=np.uint8
    )
    Image.fromarray(palette[(seg[:, :, 0] + 1).clip(0, m.ngeom)]).save(
        ROOT / "outputs/head_segmentation.png"
    )
    cam.close()
    print(
        json.dumps(
            dict(z=float(d.qpos[2]), detection=det, output="outputs/room_preview.png")
        )
    )


def rehearse(args):
    torch.set_num_threads(1)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if not args.no_video:
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
    scene = scenario(args.seed)
    scene_path = Path(args.output).with_suffix(".scene.xml")
    scene_path.write_text(scene_xml(scene, scene["targets"], scene["spawn"]))
    env = SearchEnv(1, seed=args.seed)
    # Exact same seed/layout/objects/spawn for abstract and physical geometry.
    boxes = obstacles(scene)
    env.boxes = torch.tensor(boxes[None])
    env.targets = torch.tensor(np.array(scene["targets"], dtype=np.float32)[None])
    env.goal[:] = args.goal
    env.pos[:] = torch.tensor(scene["spawn"][:2])
    env.yaw[:] = scene["spawn"][2]
    env.seen[:] = False
    env.memory[:] = 0
    env.visits[:] = 0
    m, d, bam, policy = robot(scene_path)
    agent = None
    if args.controller == "rl":
        agent, _ = load(args.checkpoint)
    head = mujoco.Renderer(m, height=240, width=320)
    renderer = mujoco.Renderer(m, height=480, width=800) if not args.no_video else None
    writer = None
    if renderer:
        import imageio.v2 as imageio

        writer = imageio.get_writer(
            args.video, fps=12, codec="libx264", quality=7, macro_block_size=1
        )
    viewer = None
    if args.viewer:
        import mujoco.viewer as mjviewer

        viewer = mjviewer.launch_passive(m, d)
    trace = []
    success = False
    fall = False
    contacts = 0
    action = 5
    requested_action = 5
    guard_overrides = 0
    false_confirmations = 0
    start = time.time()
    next_render = 0
    seen_frames = 0
    command(policy, 0, 0)
    for _ in range(50):
        tick(m, d, bam, policy)
    origin_time = float(d.time)
    playback_start = time.monotonic()
    termination_reason = "timeout"
    try:
        for frame in range(int(args.seconds * 50)):
            simtime = frame * 0.02
            if frame % 30 == 0:
                rgb, dep, det = sensor_frame(head, m, d, args.goal)
                visible, bearing, distance = det
                seen_frames += int(visible)
                yaw = yaw_of(d)
                env.pos[:] = torch.tensor(d.qpos[:2].copy(), dtype=torch.float32)
                env.yaw[:] = yaw
                env.steps[:] = frame // 30
                env.record_visit()
                obs = env.observe(
                    tuple(
                        torch.tensor(
                            [v], dtype=(torch.bool if i == 0 else torch.float32)
                        )
                        for i, v in enumerate(det)
                    ),
                    torch.tensor(depths(m, d, yaw)[None]),
                )
                if agent is not None:
                    with torch.inference_mode():
                        action = int(agent(obs)[0].argmax(-1)[0])
                else:
                    # Reactive baseline uses exactly the same detections/ranges.
                    if visible and distance < 0.55:
                        action = 5
                    elif visible:
                        action = 1 if bearing > 0.18 else 2 if bearing < -0.18 else 0
                    else:
                        action = 1
                env.last[:] = torch.nn.functional.one_hot(
                    torch.tensor([action]), 6
                ).float()
                declared = action == 5 and visible and distance < 0.55
                true_distance = float(
                    np.linalg.norm(d.qpos[:2] - np.asarray(scene["targets"][args.goal]))
                )
                # Independent evaluator only: actor never receives geom IDs or
                # the hidden object's true location. Verify declarations against
                # the rendered target and distance from trunk (camera is offset).
                target_pixels = 0
                if declared:
                    head.enable_segmentation_rendering()
                    head.update_scene(d, camera="head_camera")
                    seg = head.render()
                    head.disable_segmentation_rendering()
                    target_id = mujoco.mj_name2id(
                        m, mujoco.mjtObj.mjOBJ_GEOM, f"search_target_{args.goal}"
                    )
                    target_pixels = int(np.sum(seg[:, :, 0] == target_id))
                    Image.fromarray(rgb).save(
                        Path(args.output).with_suffix(".confirmation.png")
                    )
                success = declared and true_distance < 0.75 and target_pixels >= 5
                false_confirmations += int(declared and not success)
                requested_action = action
                action = guarded_action(requested_action, depths(m, d, yaw))
                env.last[:] = torch.nn.functional.one_hot(
                    torch.tensor([action]), 6
                ).float()
                primitive_command(policy, action)
                trace.append(
                    dict(
                        time=simtime,
                        xy=d.qpos[:2].tolist(),
                        yaw=yaw,
                        action=action,
                        requested_action=requested_action,
                        visible=visible,
                        bearing=bearing,
                        detected_distance=distance,
                        z=float(d.qpos[2]),
                        evaluator_true_distance=true_distance,
                        evaluator_target_pixels=target_pixels,
                    )
                )
            # Brake at 10 Hz between the supervisor's 0.6 s decisions.
            if frame % 5 == 0:
                safe_action = guarded_action(requested_action, depths(m, d, yaw_of(d)))
                if safe_action != action:
                    action = safe_action
                    primitive_command(policy, action)
                guard_overrides += int(action != requested_action)
                env.last[:] = torch.nn.functional.one_hot(
                    torch.tensor([action]), 6
                ).float()
            tick(m, d, bam, policy)
            # Finite-state and tilt checks use truth only for evaluation.
            if not np.isfinite(d.qpos).all():
                raise RuntimeError("Nonfinite physical state")
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, d.qpos[3:7])
            fall = float(rotation[8]) < 0.35 or d.qpos[2] < 0.055
            robot_world_contact = False
            for contact in d.contact[: d.ncon]:
                a, c = int(contact.geom1), int(contact.geom2)
                if (m.geom_bodyid[a] == 0) != (m.geom_bodyid[c] == 0):
                    world_geom = a if m.geom_bodyid[a] == 0 else c
                    name = (
                        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, world_geom) or ""
                    )
                    if name not in ["floor"] and not name.startswith("step_"):
                        robot_world_contact = True
            contacts += int(robot_world_contact)
            if renderer and simtime >= next_render:
                renderer.update_scene(d, camera="overview")
                overview = renderer.render().copy()
                # RGB inset and labels are evidence from this exact rollout.
                canvas = Image.fromarray(overview)
                canvas.paste(Image.fromarray(rgb).resize((240, 180)), (550, 42))
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((0, 0, 800, 35), fill=(15, 22, 30))
                draw.text(
                    (12, 10),
                    f"MicroDuck Search | {OBJECTS[args.goal]} | {simtime:5.1f}s | {args.controller.upper()} | visible={visible}",
                    fill="white",
                )
                draw.text((555, 225), "CAMARA RGB REAL DEL SIM", fill="white")
                writer.append_data(np.array(canvas))
                next_render += 1 / 12
            if viewer:
                viewer.sync()
                if not viewer.is_running():
                    termination_reason = "viewer_closed"
                    break
                remaining = simtime + 0.02 - (time.monotonic() - playback_start)
                if remaining > 0:
                    time.sleep(min(remaining, 0.02))
            if success or fall or false_confirmations:
                termination_reason = (
                    "success" if success else "fall" if fall else "false_confirmation"
                )
                break
    finally:
        if writer:
            writer.close()
        if renderer:
            renderer.close()
        head.close()
        if viewer:
            viewer.close()
    report = dict(
        success=bool(success),
        fall=bool(fall),
        simulated_seconds=float(d.time - origin_time),
        wall_seconds=time.time() - start,
        goal=OBJECTS[args.goal],
        seed=args.seed,
        contact_steps=contacts,
        detector_visible_observations=seen_frames,
        controller=args.controller,
        false_confirmations=false_confirmations,
        depth_guard_overrides=guard_overrides,
        depth_guard="10 Hz, depth-only advance-to-turn override, identical for RL and baseline",
        termination_reason=termination_reason,
        scene_file=str(scene_path.relative_to(ROOT))
        if scene_path.is_relative_to(ROOT)
        else scene_path.name,
        success_criterion="Actor declares visible target within 0.55 m estimated camera range; evaluator verifies target pixels and trunk distance <0.75 m",
        physics="MuJoCo 3.10, free floating MicroDuck, BAM M6, ONNX motor policies at 50 Hz",
        walking_policy=Path(policy.search_gait_path).name,
        turning_policy="velstand.onnx",
        perception="RGB colour detector + simulated depth; segmentation is not an actor input",
        limitations=[
            "Perfect simulator odometry",
            "Idealized planar forward depth rays",
            "No physical robot validation",
            "Fixed colour classes; not open-vocabulary search",
            "Supervisory policy trained in an SE2 proxy",
        ],
        trace=trace,
    )
    Path(args.output).write_text(json.dumps(report, indent=2))
    if not getattr(args, "quiet", False):
        print(json.dumps({k: v for k, v in report.items() if k != "trace"}), flush=True)
    return report
