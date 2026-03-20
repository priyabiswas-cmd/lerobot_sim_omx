"""
Franka Panda – Pick-and-Place Dataset Generator for SmolVLA / LeRobot
======================================================================
Runs N episodes with randomised cube positions, records observations and
actions, then saves everything in the LeRobot / HuggingFace dataset format
that SmolVLA expects.

Requirements:
    pip install pybullet numpy opencv-python datasets pyarrow pillow

Output structure:
    dataset/
        data/
            episode_000000.parquet   ← one file per episode
            episode_000001.parquet
            ...
        videos/
            episode_000000_cam_high.mp4
            episode_000000_cam_wrist.mp4
            ...
        meta/
            info.json
            episodes.jsonl
            stats.json

Each parquet row contains:
    observation.state          – 9-dim joint positions  [7 arm + 2 fingers]
    observation.images.top     – RGB image bytes (top camera)
    observation.images.wrist   – RGB image bytes (wrist camera)
    action                     – 9-dim target joint positions
    episode_index              – int
    frame_index                – int within episode
    timestamp                  – float seconds
    task_description           – natural language string

Usage:
    python generate_dataset.py                  # 100 episodes, headless
    python generate_dataset.py --gui            # show PyBullet window
    python generate_dataset.py --episodes 10    # quick test
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pybullet as p
import pybullet_data
from PIL import Image

# ─────────────────────────────────────────────────────────────────
# Dataset / simulation config
# ─────────────────────────────────────────────────────────────────

NUM_EPISODES   = 2
OUTPUT_DIR     = Path("dataset")
SIM_HZ         = 240
SLEEP_DT       = 1.0 / SIM_HZ
CONTROL_HZ     = 10                     # how often we record a frame
CONTROL_DT     = 1.0 / CONTROL_HZ
STEPS_PER_CTRL = SIM_HZ // CONTROL_HZ  # sim steps between recorded frames

IMG_W, IMG_H   = 224, 224               # SmolVLA default input resolution

# Table surface Z and safe cube spawn region (x, y)
TABLE_Z        = 0.625
CUBE_X_RANGE   = (0.35, 0.65)
CUBE_Y_RANGE   = (-0.25, 0.25)

TASK_DESCRIPTION = "Pick up the red cube and place it to the right."

# Set to True at runtime if --gui is passed; controls sleep + renderer
GUI_MODE = False

# Franka home config
HOME_ANGLES = [0, -math.pi/4, 0, -3*math.pi/4, 0, math.pi/2, math.pi/4]
EE_LINK     = 11   # panda_hand

# ─────────────────────────────────────────────────────────────────
# Joint discovery (same robust approach as before)
# ─────────────────────────────────────────────────────────────────

ARM_JOINTS    = []
FINGER_JOINTS = []


def discover_joints(robot):
    global ARM_JOINTS, FINGER_JOINTS
    arm, fingers = [], []
    for i in range(p.getNumJoints(robot)):
        info  = p.getJointInfo(robot, i)
        jtype = info[2]
        jname = info[1].decode()
        if jtype == p.JOINT_FIXED:
            continue
        if "finger" in jname:
            fingers.append(i)
        elif "joint" in jname:
            arm.append(i)
    ARM_JOINTS    = arm[:7]
    FINGER_JOINTS = fingers[:2]


# ─────────────────────────────────────────────────────────────────
# Simulation helpers
# ─────────────────────────────────────────────────────────────────

def setup_sim(gui=True):
    mode   = p.GUI if gui else p.DIRECT
    client = p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(SLEEP_DT, physicsClientId=client)
    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=client)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=client)
    p.loadURDF("plane.urdf", physicsClientId=client)
    p.loadURDF(
        "table/table.urdf",
        basePosition=[0.5, 0.0, 0.0],
        baseOrientation=p.getQuaternionFromEuler([0, 0, math.pi / 2]),
        physicsClientId=client,
    )
    robot = p.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=[0, 0, TABLE_Z],
        useFixedBase=True,
        physicsClientId=client,
    )
    discover_joints(robot)
    if gui:
        p.resetDebugVisualizerCamera(1.5, 45, -30, [0.4, 0, 0.6], physicsClientId=client)
    return client, robot


def reset_robot(client, robot):
    all_angles = HOME_ANGLES + [0.04, 0.04]
    for i, a in enumerate(all_angles):
        p.resetJointState(robot, i, a, physicsClientId=client)
    _send_arm(client, robot, HOME_ANGLES)
    _send_gripper(client, robot, open_=True, steps=0)


def spawn_cube(client, x, y, half=0.025):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half]*3, physicsClientId=client)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half]*3,
                               rgbaColor=[0.85, 0.12, 0.12, 1], physicsClientId=client)
    return p.createMultiBody(
        baseMass=0.08,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=[x, y, TABLE_Z + half],
        physicsClientId=client,
    )


def remove_body(client, body_id):
    p.removeBody(body_id, physicsClientId=client)


# ── Low-level actuators ────────────────────────────────────────

def _send_arm(client, robot, angles, speed=0.8, torque=87):
    for idx, ang in zip(ARM_JOINTS, angles):
        p.setJointMotorControl2(
            robot, idx, p.POSITION_CONTROL,
            targetPosition=ang, maxVelocity=speed, force=torque,
            physicsClientId=client,
        )


def _send_gripper(client, robot, open_=True, steps=60):
    target = 0.04 if open_ else 0.0
    for _ in range(max(steps, 1)):
        for fi in FINGER_JOINTS:
            p.setJointMotorControl2(
                robot, fi, p.POSITION_CONTROL,
                targetPosition=target, maxVelocity=0.05, force=200,
                physicsClientId=client,
            )
        if steps > 0:
            p.stepSimulation(physicsClientId=client)
            if GUI_MODE:
                time.sleep(SLEEP_DT)


def _step(client, n=1):
    for _ in range(n):
        p.stepSimulation(physicsClientId=client)
        if GUI_MODE:
            time.sleep(SLEEP_DT)


def _get_joint_states(robot):
    """Return [pos x9] for all arm+finger joints."""
    positions = []
    for idx in ARM_JOINTS + FINGER_JOINTS:
        positions.append(p.getJointState(robot, idx)[0])
    return np.array(positions, dtype=np.float32)


def compute_ik(client, robot, pos, orn):
    result = p.calculateInverseKinematics(
        robot, EE_LINK, pos, orn,
        lowerLimits=[-2.9, -1.76, -2.9, -3.07, -2.9, -0.02, -2.9],
        upperLimits=[ 2.9,  1.76,  2.9, -0.07,  2.9,  3.75,  2.9],
        jointRanges=[5.8, 3.52, 5.8, 3.0, 5.8, 3.77, 5.8],
        restPoses=HOME_ANGLES,
        maxNumIterations=200,
        residualThreshold=0.005,
        physicsClientId=client,
    )
    return list(result[:7])


# ── Cameras ────────────────────────────────────────────────────

def _render(client, eye, target, up=(0, 0, 1)):
    """Render a single RGB frame from the given viewpoint."""
    view_mat = p.computeViewMatrix(eye, target, up, physicsClientId=client)
    proj_mat = p.computeProjectionMatrixFOV(
        fov=60, aspect=1.0, nearVal=0.01, farVal=10.0, physicsClientId=client)
    renderer = p.ER_BULLET_HARDWARE_OPENGL if GUI_MODE else p.ER_TINY_RENDERER
    _, _, rgb, _, _ = p.getCameraImage(
        IMG_W, IMG_H,
        viewMatrix=view_mat,
        projectionMatrix=proj_mat,
        renderer=renderer,
        physicsClientId=client,
    )
    return np.array(rgb, dtype=np.uint8)[:, :, :3]   # drop alpha


def get_frames(client, robot):
    """Return top-down and wrist camera RGB frames."""
    # Top / overhead camera
    top = _render(client, eye=[0.5, 0.0, 1.4], target=[0.5, 0.0, TABLE_Z])

    # Wrist camera: attached to EE link
    ee  = p.getLinkState(robot, EE_LINK, physicsClientId=client)
    ee_pos = np.array(ee[0])
    ee_mat = np.array(p.getMatrixFromQuaternion(ee[1])).reshape(3, 3)
    cam_pos   = ee_pos + ee_mat @ np.array([0, 0, -0.1])   # 10 cm behind EE
    look_at   = ee_pos + ee_mat @ np.array([0, 0,  0.1])
    wrist = _render(client, eye=cam_pos.tolist(), target=look_at.tolist(),
                    up=ee_mat[:, 1].tolist())

    return top, wrist


# ─────────────────────────────────────────────────────────────────
# Motion primitives  (each returns a list of (joint_state, action) tuples)
# ─────────────────────────────────────────────────────────────────

def execute_phase(client, robot, target_pos, target_orn,
                  gripper_open, n_steps, speed=0.8,
                  grasp_cid=None):
    """
    Move arm toward target_pos/orn for n_steps simulation steps.
    Records one (obs, action) pair every STEPS_PER_CTRL steps.
    Returns list of (joint_state_9, action_9, top_rgb, wrist_rgb).
    """
    angles = compute_ik(client, robot, target_pos, target_orn)
    finger_target = 0.04 if gripper_open else 0.0
    action_9 = np.array(angles + [finger_target, finger_target], dtype=np.float32)

    records = []
    for step in range(n_steps):
        _send_arm(client, robot, angles, speed=speed)
        _send_gripper(client, robot, open_=gripper_open, steps=0)
        p.stepSimulation(physicsClientId=client)

        if GUI_MODE:
            time.sleep(SLEEP_DT)   # real-time pacing so GUI window is visible

        if step % STEPS_PER_CTRL == 0:
            obs_state          = _get_joint_states(robot)
            top_rgb, wrist_rgb = get_frames(client, robot)
            records.append((obs_state.copy(), action_9.copy(), top_rgb, wrist_rgb))

    return records# ─────────────────────────────────────────────────────────────────
# Full episode
# ─────────────────────────────────────────────────────────────────

def run_episode(client, robot, cube_x, cube_y):
    """
    Execute one full pick-and-place episode.
    Returns list of frame dicts.
    """
    cube = spawn_cube(client, cube_x, cube_y)
    _step(client, 60)   # let cube settle

    orn_down  = p.getQuaternionFromEuler([math.pi, 0, math.pi / 4])
    cube_z    = TABLE_Z + 0.025   # top face of cube
    hover_z   = cube_z + 0.15
    place_x   = cube_x
    place_y   = cube_y - 0.30    # place 30 cm in -Y direction

    frames = []

    def collect(phase_records, phase_name):
        for obs, act, top, wrist in phase_records:
            frames.append({
                "obs_state": obs,
                "action":    act,
                "top_rgb":   top,
                "wrist_rgb": wrist,
                "phase":     phase_name,
            })

    # 1. Hover above cube
    collect(execute_phase(client, robot,
                          [cube_x, cube_y, hover_z], orn_down,
                          gripper_open=True, n_steps=300),
            "approach")

    # 2. Open gripper explicitly
    _send_gripper(client, robot, open_=True, steps=60)

    # 3. Descend to grasp
    collect(execute_phase(client, robot,
                          [cube_x, cube_y, cube_z + 0.01], orn_down,
                          gripper_open=True, n_steps=200, speed=0.4),
            "descend")

    # 4. Close gripper
    _send_gripper(client, robot, open_=False, steps=80)

    # Attach cube via constraint
    cb_pos, cb_orn = p.getBasePositionAndOrientation(cube, physicsClientId=client)
    ee_s           = p.getLinkState(robot, EE_LINK, physicsClientId=client)
    inv_p, inv_o   = p.invertTransform(ee_s[0], ee_s[1])
    lp, lo         = p.multiplyTransforms(inv_p, inv_o, cb_pos, cb_orn)
    cid = p.createConstraint(
        robot, EE_LINK, cube, -1,
        p.JOINT_FIXED, [0, 0, 0], lp, [0, 0, 0], lo,
        physicsClientId=client,
    )
    p.changeConstraint(cid, maxForce=300, physicsClientId=client)

    # 5. Lift
    collect(execute_phase(client, robot,
                          [cube_x, cube_y, hover_z + 0.05], orn_down,
                          gripper_open=False, n_steps=250),
            "lift")

    # 6. Transit
    collect(execute_phase(client, robot,
                          [place_x, place_y, hover_z + 0.05], orn_down,
                          gripper_open=False, n_steps=400),
            "transit")

    # 7. Lower to place
    collect(execute_phase(client, robot,
                          [place_x, place_y, cube_z + 0.01], orn_down,
                          gripper_open=False, n_steps=200, speed=0.4),
            "lower")

    # 8. Release
    p.removeConstraint(cid, physicsClientId=client)
    _send_gripper(client, robot, open_=True, steps=80)
    _step(client, 60)

    # 9. Retract
    collect(execute_phase(client, robot,
                          [place_x, place_y, hover_z], orn_down,
                          gripper_open=True, n_steps=200),
            "retract")

    remove_body(client, cube)
    return frames


# ─────────────────────────────────────────────────────────────────
# Dataset saving  (LeRobot / SmolVLA format)
# ─────────────────────────────────────────────────────────────────

def frames_to_parquet(episode_idx, frames, out_dir: Path):
    """Save one episode as a parquet file."""
    rows = []
    for fi, f in enumerate(frames):
        top_bytes   = cv2.imencode(".jpg", cv2.cvtColor(f["top_rgb"],   cv2.COLOR_RGB2BGR))[1].tobytes()
        wrist_bytes = cv2.imencode(".jpg", cv2.cvtColor(f["wrist_rgb"], cv2.COLOR_RGB2BGR))[1].tobytes()
        rows.append({
            "observation.state":          f["obs_state"].tolist(),
            "observation.images.top":     top_bytes,
            "observation.images.wrist":   wrist_bytes,
            "action":                     f["action"].tolist(),
            "episode_index":              episode_idx,
            "frame_index":                fi,
            "timestamp":                  round(fi * CONTROL_DT, 4),
            "task_description":           TASK_DESCRIPTION,
            "phase":                      f["phase"],
        })

    schema = pa.schema([
        pa.field("observation.state",        pa.list_(pa.float32())),
        pa.field("observation.images.top",   pa.binary()),
        pa.field("observation.images.wrist", pa.binary()),
        pa.field("action",                   pa.list_(pa.float32())),
        pa.field("episode_index",            pa.int32()),
        pa.field("frame_index",              pa.int32()),
        pa.field("timestamp",                pa.float32()),
        pa.field("task_description",         pa.string()),
        pa.field("phase",                    pa.string()),
    ])

    table = pa.Table.from_pydict(
        {k: [r[k] for r in rows] for k in rows[0]},
        schema=schema,
    )
    fname = out_dir / f"episode_{episode_idx:06d}.parquet"
    pq.write_table(table, fname)
    return len(rows)


def save_meta(out_dir: Path, episode_infos: list):
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(exist_ok=True)

    info = {
        "dataset_type":      "LeRobotDataset",
        "task":              TASK_DESCRIPTION,
        "robot":             "franka_panda",
        "num_episodes":      len(episode_infos),
        "total_frames":      sum(e["num_frames"] for e in episode_infos),
        "fps":               CONTROL_HZ,
        "image_width":       IMG_W,
        "image_height":      IMG_H,
        "observation_space": {
            "state_dim": 9,
            "cameras":   ["top", "wrist"],
        },
        "action_space": {
            "action_dim": 9,
            "description": "9-dim joint position targets [7 arm + 2 fingers]",
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2))

    with open(meta_dir / "episodes.jsonl", "w") as f:
        for e in episode_infos:
            f.write(json.dumps(e) + "\n")

    print(f"  Meta saved → {meta_dir}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=NUM_EPISODES)
    ap.add_argument("--no-gui",   action="store_true", help="Run headless (no window)")
    ap.add_argument("--out",      type=str, default=str(OUTPUT_DIR))
    args = ap.parse_args()

    show_gui = not args.no_gui    # GUI is ON by default

    out      = Path(args.out)
    data_dir = out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.episodes} episodes → {out}")
    print(f"  Control Hz : {CONTROL_HZ}  |  Image : {IMG_W}x{IMG_H}")
    print(f"  GUI        : {show_gui}\n")

    global GUI_MODE
    GUI_MODE = show_gui

    client, robot = setup_sim(gui=show_gui)

    episode_infos = []
    total_frames  = 0

    for ep in range(args.episodes):
        # Randomise cube position on the table
        cx = random.uniform(*CUBE_X_RANGE)
        cy = random.uniform(*CUBE_Y_RANGE)

        print(f"Episode {ep+1:03d}/{args.episodes}  cube=({cx:.3f}, {cy:.3f})", end="  ", flush=True)

        reset_robot(client, robot)
        _step(client, 120)   # settle after reset

        t0 = time.time()
        frames = run_episode(client, robot, cx, cy)
        elapsed = time.time() - t0

        n_frames = frames_to_parquet(ep, frames, data_dir)
        total_frames += n_frames

        info = {
            "episode_index": ep,
            "cube_start_x":  round(cx, 4),
            "cube_start_y":  round(cy, 4),
            "num_frames":    n_frames,
            "duration_s":    round(elapsed, 2),
        }
        episode_infos.append(info)
        print(f"frames={n_frames}  time={elapsed:.1f}s")

    p.disconnect(client)

    save_meta(out, episode_infos)

    print(f"\n✓ Dataset complete")
    print(f"  Episodes : {args.episodes}")
    print(f"  Frames   : {total_frames}")
    print(f"  Location : {out.resolve()}")
    print(f"\nLoad in Python:")
    print(f'  import datasets')
    print(f'  ds = datasets.load_dataset("parquet", data_files="{out}/data/*.parquet")')


if __name__ == "__main__":
    main()