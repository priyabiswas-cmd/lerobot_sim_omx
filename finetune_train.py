"""
SmolVLA Finetuned Model Inference in PyBullet Simulation
---------------------------------------------------------
Runs the finetuned SmolVLA model to control the Franka Panda robot
in a PyBullet pick-and-place task.

Requirements:
    pip install pybullet numpy opencv-python torch

Run:
    python smolvla_pybullet_inference.py
    python smolvla_pybullet_inference.py --episodes 5 --gui
"""

import sys
sys.path.insert(0, '/home/priya/lerobot/src')

import argparse
import math
import time
import numpy as np
import torch
import pybullet as p
import pybullet_data
import cv2

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
CHECKPOINT_PATH = "/home/priya/outputs/train/smolvla_franka/checkpoints/015000/pretrained_model"
TASK_DESCRIPTION = "Pick up the red cube and place it to the right."

SIM_HZ      = 240
SLEEP_DT    = 1.0 / SIM_HZ
EE_LINK     = 11
IMG_W       = 224
IMG_H       = 224

# Cube spawn range (matches training distribution)
CUBE_X_RANGE = (0.35, 0.65)
CUBE_Y_RANGE = (-0.25, 0.25)
CUBE_Z      = 0.63

PLACE_TARGET = [0.5, -0.35, 0.63]

PANDA_ARM_JOINTS    = []
PANDA_FINGER_JOINTS = []


# ─────────────────────────────────────────────
# Simulation setup
# ─────────────────────────────────────────────

def discover_joints(robot):
    global PANDA_ARM_JOINTS, PANDA_FINGER_JOINTS
    n = p.getNumJoints(robot)
    arm, fingers = [], []
    for i in range(n):
        info  = p.getJointInfo(robot, i)
        jtype = info[2]
        jname = info[1].decode()
        if jtype == p.JOINT_FIXED:
            continue
        if "finger" in jname:
            fingers.append(i)
        elif "joint" in jname:
            arm.append(i)
    PANDA_ARM_JOINTS    = arm[:7]
    PANDA_FINGER_JOINTS = fingers[:2]


def setup_sim(gui=True):
    mode   = p.GUI if gui else p.DIRECT
    client = p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(SLEEP_DT,    physicsClientId=client)

    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=client)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=client)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5, cameraYaw=45, cameraPitch=-30,
            cameraTargetPosition=[0.4, 0.0, 0.6],
            physicsClientId=client,
        )

    p.loadURDF("plane.urdf", physicsClientId=client)
    p.loadURDF(
        "table/table.urdf",
        basePosition=[0.5, 0.0, 0.0],
        baseOrientation=p.getQuaternionFromEuler([0, 0, math.pi / 2]),
        physicsClientId=client,
    )
    robot = p.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=[0, 0, 0.625],
        useFixedBase=True,
        physicsClientId=client,
    )
    discover_joints(robot)
    return client, robot


def spawn_cube(client, x=None, y=None):
    x = x if x is not None else np.random.uniform(*CUBE_X_RANGE)
    y = y if y is not None else np.random.uniform(*CUBE_Y_RANGE)
    pos = [x, y, CUBE_Z]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.025]*3, physicsClientId=client)
    vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.025]*3,
        rgbaColor=[0.8, 0.1, 0.1, 1], physicsClientId=client,
    )
    cube = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos,
        physicsClientId=client,
    )
    return cube, pos


def reset_robot(client, robot, gui=False):
    home = [0, -math.pi/4, 0, -3*math.pi/4, 0, math.pi/2, math.pi/4, 0.04, 0.04]
    for i, angle in enumerate(home):
        p.resetJointState(robot, i, angle, physicsClientId=client)
    for _ in range(120):
        p.stepSimulation(physicsClientId=client)
        if gui:
            time.sleep(SLEEP_DT)


def step_sim(client, n=4, gui=False):
    for _ in range(n):
        p.stepSimulation(physicsClientId=client)
        if gui:
            time.sleep(SLEEP_DT)


# ─────────────────────────────────────────────
# Camera rendering
# ─────────────────────────────────────────────

def render_camera(client, eye, target, up=[0, 0, 1]):
    """Render an RGB image from a given viewpoint."""
    view_matrix = p.computeViewMatrix(eye, target, up, physicsClientId=client)
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=1.0, nearVal=0.01, farVal=10.0, physicsClientId=client
    )
    _, _, rgb, _, _ = p.getCameraImage(
        IMG_W, IMG_H,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=client,
    )
    img = np.array(rgb, dtype=np.uint8).reshape(IMG_H, IMG_W, 4)[:, :, :3]  # drop alpha
    # Convert HWC → CHW for torch
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3, H, W) float32 in [0,1]


def get_top_camera(client):
    """Overhead camera above the workspace."""
    return render_camera(
        client,
        eye=[0.5, 0.0, 1.4],
        target=[0.5, 0.0, 0.63],
    )


def get_wrist_camera(client, robot):
    """Camera attached to the end-effector (wrist view)."""
    ee_state = p.getLinkState(robot, EE_LINK, physicsClientId=client)
    ee_pos   = list(ee_state[0])
    ee_orn   = ee_state[1]

    # Wrist camera looks forward/down from EE
    rot_mat = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)
    forward = rot_mat @ np.array([0, 0, 1])
    eye     = np.array(ee_pos) + rot_mat @ np.array([0, 0, -0.1])
    target  = np.array(ee_pos) + forward * 0.3

    return render_camera(client, eye.tolist(), target.tolist(), up=[0, 0, 1])


# ─────────────────────────────────────────────
# Robot state
# ─────────────────────────────────────────────

def get_joint_states(client, robot):
    """Return first 6 arm joint positions as a float32 tensor."""
    states = [p.getJointState(robot, j, physicsClientId=client)[0]
              for j in PANDA_ARM_JOINTS[:6]]
    return torch.tensor(states, dtype=torch.float32)


def apply_action(client, robot, action_tensor, gui=False):
    """
    Apply the 6-DOF action predicted by SmolVLA to the robot.
    action_tensor: shape (6,) — target joint positions for first 6 arm joints.
    """
    action = action_tensor.cpu().numpy()
    for i, (joint_idx, target) in enumerate(zip(PANDA_ARM_JOINTS[:6], action)):
        p.setJointMotorControl2(
            robot, joint_idx,
            controlMode=p.POSITION_CONTROL,
            targetPosition=float(target),
            maxVelocity=0.5,
            force=87,
            physicsClientId=client,
        )
    step_sim(client, n=8, gui=gui)


# ─────────────────────────────────────────────
# Load SmolVLA model
# ─────────────────────────────────────────────

def load_policy(device):
    print(f"Loading finetuned SmolVLA from: {CHECKPOINT_PATH}")
    policy = SmolVLAPolicy.from_pretrained(CHECKPOINT_PATH).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        CHECKPOINT_PATH,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    print("Model loaded successfully!")
    return policy, preprocess, postprocess


# ─────────────────────────────────────────────
# Single episode inference
# ─────────────────────────────────────────────

def run_episode(client, robot, policy, preprocess, postprocess, device, gui=False, max_steps=150):
    """
    Run one pick-and-place episode using SmolVLA to predict actions.
    Returns True if the cube was moved to roughly the target area.
    """
    # Spawn cube at random position
    cube, cube_start = spawn_cube(client)
    print(f"  Cube spawned at: {[round(v,3) for v in cube_start]}")

    # Let cube settle
    step_sim(client, n=60, gui=gui)

    policy.reset()  # reset action chunking buffer

    success = False
    for step in range(max_steps):
        # ── Get observations ──
        top_img   = get_top_camera(client)
        wrist_img = get_wrist_camera(client, robot)
        state     = get_joint_states(client, robot)

        # ── Build frame dict ──
        frame = {
            "observation.images.camera1": top_img,
            "observation.images.camera2": wrist_img,
            "observation.images.camera3": top_img,   # no 3rd camera, duplicate top
            "observation.state":          state,
            "task":                       TASK_DESCRIPTION,
        }

        # ── Preprocess ──
        batch = preprocess(frame)

        # ── Inference ──
        with torch.inference_mode():
            pred_action = policy.select_action(batch)

        # ── Postprocess & apply ──
        pred_action = postprocess(pred_action)
        if isinstance(pred_action, dict):
            action_tensor = pred_action.get("action", list(pred_action.values())[0])
        else:
            action_tensor = pred_action

        # select_action may return (1, 6) or (6,)
        if action_tensor.dim() > 1:
            action_tensor = action_tensor[0]

        apply_action(client, robot, action_tensor, gui=gui)

        # ── Check success: cube moved close to place target ──
        cube_pos, _ = p.getBasePositionAndOrientation(cube, physicsClientId=client)
        dist = math.sqrt(
            (cube_pos[0] - PLACE_TARGET[0])**2 +
            (cube_pos[1] - PLACE_TARGET[1])**2
        )
        if dist < 0.08 and cube_pos[2] > 0.60:
            print(f"  ✓ SUCCESS at step {step+1}! Cube at {[round(v,3) for v in cube_pos]}")
            success = True
            break

        if step % 30 == 0:
            print(f"  Step {step+1}/{max_steps} | cube dist to target: {dist:.3f}m")

    if not success:
        cube_pos, _ = p.getBasePositionAndOrientation(cube, physicsClientId=client)
        dist = math.sqrt(
            (cube_pos[0] - PLACE_TARGET[0])**2 +
            (cube_pos[1] - PLACE_TARGET[1])**2
        )
        print(f"  ✗ Episode ended. Final cube dist to target: {dist:.3f}m")

    # Clean up cube
    p.removeBody(cube, physicsClientId=client)
    step_sim(client, n=30, gui=gui)

    return success


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to run")
    parser.add_argument("--gui", action="store_true", default=True,
                        help="Show PyBullet GUI (default: on)")
    parser.add_argument("--no-gui", dest="gui", action="store_false",
                        help="Run headless (faster)")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH,
                        help="Path to SmolVLA checkpoint")
    parser.add_argument("--max-steps", type=int, default=150,
                        help="Max steps per episode")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load policy
    policy, preprocess, postprocess = load_policy(device)

    # Setup simulation
    client, robot = setup_sim(gui=args.gui)

    results = []
    for ep in range(args.episodes):
        print(f"\n{'='*50}")
        print(f"Episode {ep+1}/{args.episodes}")
        print(f"{'='*50}")

        reset_robot(client, robot, gui=args.gui)
        success = run_episode(
            client, robot,
            policy, preprocess, postprocess,
            device,
            gui=args.gui,
            max_steps=args.max_steps,
        )
        results.append(success)

    # Summary
    n_success = sum(results)
    print(f"\n{'='*50}")
    print(f"RESULTS: {n_success}/{args.episodes} episodes successful")
    print(f"Success rate: {100*n_success/args.episodes:.1f}%")
    print(f"{'='*50}")

    if args.gui:
        print("\nKeeping window open. Press Ctrl-C to quit.")
        try:
            while p.isConnected(client):
                p.stepSimulation(physicsClientId=client)
                time.sleep(SLEEP_DT)
        except (Exception, KeyboardInterrupt):
            pass

    p.disconnect(client)


if __name__ == "__main__":
    main()