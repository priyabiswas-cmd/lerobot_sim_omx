#!/usr/bin/env python
"""
Franka Panda – Pick-and-Place Dataset Generator for SmolVLA / LeRobot
======================================================================
Generates a dataset fully compatible with LeRobot's training pipeline
(train.py / make_dataset / EpisodeAwareSampler).

Directory layout produced:
    dataset/
        data/
            train/
                episode_000000.parquet
                episode_000001.parquet
                ...
        videos/
            chunk-000/
                observation.images.top/
                    episode_000000.mp4
                observation.images.wrist/
                    episode_000000.mp4
        meta/
            info.json          ← LeRobotDataset metadata
            episodes.jsonl     ← per-episode rows  (dataset_from_index / dataset_to_index)
            tasks.jsonl        ← task index → description
            stats.json         ← per-feature mean/std/min/max  (used by normalizer)
            modality.json      ← optional feature-type hints

Requirements:
    pip install pybullet numpy opencv-python pyarrow pillow

Usage:
    python generate_dataset.py                   # 100 episodes, with GUI
    python generate_dataset.py --no-gui          # headless / server
    python generate_dataset.py --episodes 5      # quick smoke-test
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

# ─────────────────────────────────────────────────────────────────
# Global config
# ─────────────────────────────────────────────────────────────────

NUM_EPISODES     = 200
OUTPUT_DIR       = Path("dataset")

SIM_HZ           = 240
CONTROL_HZ       = 10
STEPS_PER_CTRL   = SIM_HZ // CONTROL_HZ   # 24

IMG_W, IMG_H     = 224, 224

TABLE_Z          = 0.625
CUBE_X_RANGE     = (0.35, 0.65)
CUBE_Y_RANGE     = (-0.25, 0.25)

TASK_DESCRIPTION = "Pick up the red cube and place it to the right."
TASK_INDEX       = 0

HOME_ANGLES      = [0, -math.pi/4, 0, -3*math.pi/4, 0, math.pi/2, math.pi/4]
EE_LINK          = 11   # panda_hand

GUI_MODE         = False   # set by argparse at runtime

# ─────────────────────────────────────────────────────────────────
# Joint discovery
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

def setup_sim(gui=False):
    mode   = p.GUI if gui else p.DIRECT
    client = p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(1.0 / SIM_HZ, physicsClientId=client)
    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=client)
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
    return client, robot


def reset_robot(client, robot):
    all_angles = HOME_ANGLES + [0.04, 0.04]
    for i, a in enumerate(all_angles):
        p.resetJointState(robot, i, a, physicsClientId=client)
    _send_arm(client, robot, HOME_ANGLES)
    _send_gripper(client, robot, open_=True, steps=0)


def spawn_cube(client, x, y, half=0.025):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half]*3, physicsClientId=client)
    vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[half]*3,
        rgbaColor=[0.85, 0.12, 0.12, 1], physicsClientId=client,
    )
    return p.createMultiBody(
        baseMass=0.08,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=[x, y, TABLE_Z + half],
        physicsClientId=client,
    )


def remove_body(client, body_id):
    p.removeBody(body_id, physicsClientId=client)


# ── Actuators ────────────────────────────────────────────────────

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
                time.sleep(1.0 / SIM_HZ)


def _step(client, n=1):
    for _ in range(n):
        p.stepSimulation(physicsClientId=client)
        if GUI_MODE:
            time.sleep(1.0 / SIM_HZ)


def _get_joint_states(robot):
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


# ── Cameras ──────────────────────────────────────────────────────

def _render(client, eye, target, up=(0, 0, 1)):
    view_mat = p.computeViewMatrix(eye, target, up, physicsClientId=client)
    proj_mat = p.computeProjectionMatrixFOV(
        fov=60, aspect=1.0, nearVal=0.01, farVal=10.0, physicsClientId=client,
    )
    renderer = p.ER_BULLET_HARDWARE_OPENGL if GUI_MODE else p.ER_TINY_RENDERER
    _, _, rgb, _, _ = p.getCameraImage(
        IMG_W, IMG_H,
        viewMatrix=view_mat,
        projectionMatrix=proj_mat,
        renderer=renderer,
        physicsClientId=client,
    )
    return np.array(rgb, dtype=np.uint8)[:, :, :3]


def get_frames(client, robot):
    top = _render(client, eye=[0.5, 0.0, 1.4], target=[0.5, 0.0, TABLE_Z])
    ee  = p.getLinkState(robot, EE_LINK, physicsClientId=client)
    ee_pos = np.array(ee[0])
    ee_mat = np.array(p.getMatrixFromQuaternion(ee[1])).reshape(3, 3)
    cam_pos  = ee_pos + ee_mat @ np.array([0,  0, -0.1])
    look_at  = ee_pos + ee_mat @ np.array([0,  0,  0.1])
    wrist = _render(client, eye=cam_pos.tolist(), target=look_at.tolist(),
                    up=ee_mat[:, 1].tolist())
    return top, wrist


# ─────────────────────────────────────────────────────────────────
# Motion primitive
# ─────────────────────────────────────────────────────────────────

def execute_phase(client, robot, target_pos, target_orn,
                  gripper_open, n_steps, speed=0.8):
    angles       = compute_ik(client, robot, target_pos, target_orn)
    finger_t     = 0.04 if gripper_open else 0.0
    action_9     = np.array(angles + [finger_t, finger_t], dtype=np.float32)
    records      = []

    for step in range(n_steps):
        _send_arm(client, robot, angles, speed=speed)
        _send_gripper(client, robot, open_=gripper_open, steps=0)
        p.stepSimulation(physicsClientId=client)
        if GUI_MODE:
            time.sleep(1.0 / SIM_HZ)

        if step % STEPS_PER_CTRL == 0:
            obs   = _get_joint_states(robot)
            t, w  = get_frames(client, robot)
            records.append((obs.copy(), action_9.copy(), t, w))

    return records


# ─────────────────────────────────────────────────────────────────
# Full episode
# ─────────────────────────────────────────────────────────────────

def run_episode(client, robot, cube_x, cube_y):
    cube     = spawn_cube(client, cube_x, cube_y)
    _step(client, 60)

    orn_down = p.getQuaternionFromEuler([math.pi, 0, math.pi / 4])
    cube_z   = TABLE_Z + 0.025
    hover_z  = cube_z + 0.15
    place_x  = cube_x
    place_y  = cube_y - 0.30

    frames = []

    def collect(recs, phase):
        for obs, act, top, wrist in recs:
            frames.append({"obs": obs, "act": act, "top": top, "wrist": wrist, "phase": phase})

    collect(execute_phase(client, robot, [cube_x, cube_y, hover_z],
                          orn_down, True, 300), "approach")
    _send_gripper(client, robot, open_=True,  steps=60)
    collect(execute_phase(client, robot, [cube_x, cube_y, cube_z + 0.01],
                          orn_down, True, 200, speed=0.4), "descend")
    _send_gripper(client, robot, open_=False, steps=80)

    cb_pos, cb_orn = p.getBasePositionAndOrientation(cube, physicsClientId=client)
    ee_s           = p.getLinkState(robot, EE_LINK, physicsClientId=client)
    inv_p, inv_o   = p.invertTransform(ee_s[0], ee_s[1])
    lp, lo         = p.multiplyTransforms(inv_p, inv_o, cb_pos, cb_orn)
    cid = p.createConstraint(robot, EE_LINK, cube, -1,
                             p.JOINT_FIXED, [0,0,0], lp, [0,0,0], lo,
                             physicsClientId=client)
    p.changeConstraint(cid, maxForce=300, physicsClientId=client)

    collect(execute_phase(client, robot, [cube_x, cube_y, hover_z + 0.05],
                          orn_down, False, 250), "lift")
    collect(execute_phase(client, robot, [place_x, place_y, hover_z + 0.05],
                          orn_down, False, 400), "transit")
    collect(execute_phase(client, robot, [place_x, place_y, cube_z + 0.01],
                          orn_down, False, 200, speed=0.4), "lower")

    p.removeConstraint(cid, physicsClientId=client)
    _send_gripper(client, robot, open_=True, steps=80)
    _step(client, 60)

    collect(execute_phase(client, robot, [place_x, place_y, hover_z],
                          orn_down, True, 200), "retract")

    remove_body(client, cube)
    return frames


# ─────────────────────────────────────────────────────────────────
# Video writer helper
# ─────────────────────────────────────────────────────────────────

def write_video(frames_rgb, path: Path, fps=CONTROL_HZ):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(str(path), fourcc, fps, (IMG_W, IMG_H))
    for f in frames_rgb:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()


# ─────────────────────────────────────────────────────────────────
# Parquet writer  – LeRobot column schema
# ─────────────────────────────────────────────────────────────────

def frames_to_parquet(episode_idx: int, frames: list, global_frame_offset: int,
                      data_dir: Path) -> int:
    """
    Write one episode parquet.
    Returns number of rows written.

    Column layout matches what LeRobot's make_dataset / DataLoader expects:
        index                  – absolute frame index across whole dataset
        episode_index          – episode number
        frame_index            – frame within episode
        timestamp              – float seconds
        task_index             – int  (maps to tasks.jsonl)
        observation.state      – list[float32]  shape (9,)
        action                 – list[float32]  shape (9,)
        next.done              – bool  (True only on last frame)
    Images are stored as videos (mp4) separately; the parquet stores no pixel data.
    """
    n = len(frames)
    rows = {
        "index":              list(range(global_frame_offset, global_frame_offset + n)),
        "episode_index":      [episode_idx] * n,
        "frame_index":        list(range(n)),
        "timestamp":          [round(i / CONTROL_HZ, 4) for i in range(n)],
        "task_index":         [TASK_INDEX] * n,
        "observation.state":  [f["obs"].tolist() for f in frames],
        "action":             [f["act"].tolist() for f in frames],
        "next.done":          [False] * (n - 1) + [True],
    }

    schema = pa.schema([
        pa.field("index",             pa.int64()),
        pa.field("episode_index",     pa.int32()),
        pa.field("frame_index",       pa.int32()),
        pa.field("timestamp",         pa.float32()),
        pa.field("task_index",        pa.int32()),
        pa.field("observation.state", pa.list_(pa.float32())),
        pa.field("action",            pa.list_(pa.float32())),
        pa.field("next.done",         pa.bool_()),
    ])

    table = pa.Table.from_pydict(rows, schema=schema)
    out_path = data_dir / f"episode_{episode_idx:06d}.parquet"
    pq.write_table(table, out_path)
    return n


# ─────────────────────────────────────────────────────────────────
# Stats accumulator  (online Welford for mean/var)
# ─────────────────────────────────────────────────────────────────

class StatsAccumulator:
    def __init__(self, dim):
        self.n   = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2   = np.zeros(dim, dtype=np.float64)
        self.mn   = np.full(dim,  np.inf,  dtype=np.float64)
        self.mx   = np.full(dim, -np.inf,  dtype=np.float64)

    def update(self, x: np.ndarray):
        x = x.astype(np.float64)
        self.n += 1
        delta      = x - self.mean
        self.mean += delta / self.n
        self.M2   += delta * (x - self.mean)
        self.mn    = np.minimum(self.mn, x)
        self.mx    = np.maximum(self.mx, x)

    def to_dict(self):
        std = np.sqrt(self.M2 / max(self.n - 1, 1))
        return {
            "mean": self.mean.tolist(),
            "std":  std.tolist(),
            "min":  self.mn.tolist(),
            "max":  self.mx.tolist(),
            "count": self.n,
        }


# ─────────────────────────────────────────────────────────────────
# Meta writers
# ─────────────────────────────────────────────────────────────────

def save_info_json(meta_dir: Path, num_episodes: int, total_frames: int,
                   chunks_size: int = 1000):
    """
    info.json  –  consumed by LeRobotDataset.__init__ to know feature shapes,
    fps, splits, etc.
    """
    info = {
        "codebase_version": "v2.1",
        "robot_type":       "panda",
        "total_episodes":   num_episodes,
        "total_frames":     total_frames,
        "total_tasks":      1,
        "total_videos":     num_episodes * 2,     # top + wrist per episode
        "total_chunks":     math.ceil(num_episodes / chunks_size),
        "chunks_size":      chunks_size,
        "fps":              CONTROL_HZ,
        "splits":           {"train": f"0:{num_episodes}"},
        "data_path":        "data/train/episode_{episode_index:06d}.parquet",
        "video_path":       "videos/chunk-{chunk_index:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            # ── observations ──────────────────────────────────────
            "observation.state": {
                "dtype":    "float32",
                "shape":    [9],
                "names":    {
                    "joints": [
                        "panda_joint1","panda_joint2","panda_joint3",
                        "panda_joint4","panda_joint5","panda_joint6",
                        "panda_joint7","panda_finger_joint1","panda_finger_joint2"
                    ]
                },
            },
            "observation.images.top": {
                "dtype":    "video",
                "shape":    [IMG_H, IMG_W, 3],
                "names":    ["height","width","channel"],
                "video_info": {
                    "video.fps":      float(CONTROL_HZ),
                    "video.codec":    "mp4v",
                    "video.pix_fmt":  "yuv420p",
                    "video.is_depth_map": False,
                },
            },
            "observation.images.wrist": {
                "dtype":    "video",
                "shape":    [IMG_H, IMG_W, 3],
                "names":    ["height","width","channel"],
                "video_info": {
                    "video.fps":      float(CONTROL_HZ),
                    "video.codec":    "mp4v",
                    "video.pix_fmt":  "yuv420p",
                    "video.is_depth_map": False,
                },
            },
            # ── actions ───────────────────────────────────────────
            "action": {
                "dtype":    "float32",
                "shape":    [9],
                "names":    {
                    "joints": [
                        "panda_joint1","panda_joint2","panda_joint3",
                        "panda_joint4","panda_joint5","panda_joint6",
                        "panda_joint7","panda_finger_joint1","panda_finger_joint2"
                    ]
                },
            },
            # ── bookkeeping ───────────────────────────────────────
            "timestamp":      {"dtype": "float32", "shape": [1], "names": None},
            "frame_index":    {"dtype": "int64",   "shape": [1], "names": None},
            "episode_index":  {"dtype": "int64",   "shape": [1], "names": None},
            "index":          {"dtype": "int64",   "shape": [1], "names": None},
            "task_index":     {"dtype": "int64",   "shape": [1], "names": None},
            "next.done":      {"dtype": "bool",    "shape": [1], "names": None},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2))


def save_episodes_parquet(meta_dir, episode_infos):
    import pandas as pd
    df = pd.DataFrame(episode_infos)

    out_dir = meta_dir / "episodes"
    out_dir.mkdir(exist_ok=True)

    df.to_parquet(out_dir / "episodes.parquet")

def save_tasks_parquet(meta_dir):
    import pandas as pd

    df = pd.DataFrame([{
        "task_index": TASK_INDEX,
        "task": TASK_DESCRIPTION
    }])

    df.to_parquet(meta_dir / "tasks.parquet")

def save_stats_json(meta_dir: Path, state_acc: StatsAccumulator,
                    action_acc: StatsAccumulator):
    """
    stats.json  –  per-feature normalisation stats consumed by the
    normalizer_processor in train.py.
    """
    stats = {
        "observation.state": state_acc.to_dict(),
        "action":            action_acc.to_dict(),
        # Image stats: assume normalised [0,1]; full calculation would
        # require iterating all video frames which is expensive.
        "observation.images.top": {
            "mean":  [[[0.5]], [[0.5]], [[0.5]]],
            "std":   [[[0.5]], [[0.5]], [[0.5]]],
            "min":   [[[0.0]], [[0.0]], [[0.0]]],
            "max":   [[[1.0]], [[1.0]], [[1.0]]],
            "count": 1,
        },
        "observation.images.wrist": {
            "mean":  [[[0.5]], [[0.5]], [[0.5]]],
            "std":   [[[0.5]], [[0.5]], [[0.5]]],
            "min":   [[[0.0]], [[0.0]], [[0.0]]],
            "max":   [[[1.0]], [[1.0]], [[1.0]]],
            "count": 1,
        },
    }
    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=2))


def save_modality_json(meta_dir: Path):
    """Optional but helpful for LeRobot feature routing."""
    modality = {
        "observation": {
            "state":  ["observation.state"],
            "images": ["observation.images.top", "observation.images.wrist"],
        },
        "action": ["action"],
    }
    (meta_dir / "modality.json").write_text(json.dumps(modality, indent=2))


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes",    type=int,  default=NUM_EPISODES)
    ap.add_argument("--no-gui",      action="store_true")
    ap.add_argument("--out",         type=str,  default=str(OUTPUT_DIR))
    ap.add_argument("--chunks-size", type=int,  default=1000)
    args = ap.parse_args()

    global GUI_MODE
    GUI_MODE = not args.no_gui

    out       = Path(args.out)
    data_dir  = out / "data" / "train"
    meta_dir  = out / "meta"
    video_dir = out / "videos"
    for d in [data_dir, meta_dir, video_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.episodes} episodes → {out.resolve()}")
    print(f"  Control Hz : {CONTROL_HZ}  |  Image : {IMG_W}x{IMG_H}  |  GUI : {GUI_MODE}\n")

    client, robot = setup_sim(gui=GUI_MODE)

    episode_infos  = []
    total_frames   = 0
    state_acc      = StatsAccumulator(9)
    action_acc     = StatsAccumulator(9)

    for ep in range(args.episodes):
        cx = random.uniform(*CUBE_X_RANGE)
        cy = random.uniform(*CUBE_Y_RANGE)
        print(f"Episode {ep+1:03d}/{args.episodes}  cube=({cx:.3f},{cy:.3f})", end="  ", flush=True)

        reset_robot(client, robot)
        _step(client, 120)

        t0     = time.time()
        frames = run_episode(client, robot, cx, cy)
        t1     = time.time()

        # ── accumulate stats ─────────────────────────────────────
        for f in frames:
            state_acc.update(f["obs"])
            action_acc.update(f["act"])

        # ── write parquet ────────────────────────────────────────
        n = frames_to_parquet(ep, frames, total_frames, data_dir)

        # ── write videos ─────────────────────────────────────────
        chunk_idx = ep // args.chunks_size
        vid_base  = video_dir / f"chunk-{chunk_idx:03d}"

        write_video(
            [f["top"]   for f in frames],
            vid_base / "observation.images.top"   / f"episode_{ep:06d}.mp4",
        )
        write_video(
            [f["wrist"] for f in frames],
            vid_base / "observation.images.wrist" / f"episode_{ep:06d}.mp4",
        )

        # ── episode meta ─────────────────────────────────────────
        episode_infos.append({
            "episode_index":     ep,
            "tasks":             [TASK_DESCRIPTION],
            "length":            n,
            "dataset_from_index": total_frames,
            "dataset_to_index":   total_frames + n,   # exclusive
            "chunk_index":       chunk_idx,
        })

        total_frames += n
        print(f"frames={n}  time={t1-t0:.1f}s")

    p.disconnect(client)

    # ── write meta files ─────────────────────────────────────────
    print("\nWriting metadata …")
    save_info_json(meta_dir, args.episodes, total_frames, args.chunks_size)
    save_episodes_parquet(meta_dir, episode_infos)
    save_tasks_parquet(meta_dir)
    save_stats_json(meta_dir, state_acc, action_acc)
    save_modality_json(meta_dir)

    print(f"\n✓  Dataset complete")
    print(f"   Episodes : {args.episodes}")
    print(f"   Frames   : {total_frames}")
    print(f"   Location : {out.resolve()}")
    print(f"\nLoad for training:")
    print(f"  lerobot train policy=smolvla dataset_repo_id={out.resolve()}")


if __name__ == "__main__":
    main()