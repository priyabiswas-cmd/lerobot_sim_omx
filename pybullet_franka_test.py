"""
Franka Panda Pick and Place Simulation in PyBullet
----------------------------------------------------
Requirements:
    pip install pybullet numpy

Run:
    python franka_pick_place.py

The script will:
  1. Load the Franka Panda robot from the built-in pybullet_data URDF
  2. Spawn a small red cube on a table
  3. Move the robot to a pre-grasp pose above the cube
  4. Lower down to grasp
  5. Close the gripper
  6. Lift the cube
  7. Move to a target location
  8. Place the cube down
  9. Open the gripper and retract
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import math


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
CUBE_START   = [0.5,  0.0,  0.63]   # cube resting position (x, y, z)
PLACE_TARGET = [0.5, -0.35, 0.63]   # where to put it down
SIM_HZ       = 240
SLEEP_DT     = 1.0 / SIM_HZ

# End-effector link index in the Franka URDF (panda_hand link = 11)
EE_LINK = 11

# IK residual threshold
IK_THRESHOLD = 0.01

# Discovered at runtime via get_joint_info
PANDA_ARM_JOINTS    = []
PANDA_FINGER_JOINTS = []


def discover_joints(robot):
    """
    Walk every joint in the URDF and classify arm vs finger joints by name.
    This is robust to URDF versions that may order joints differently.
    """
    global PANDA_ARM_JOINTS, PANDA_FINGER_JOINTS
    n = p.getNumJoints(robot)
    arm, fingers = [], []
    for i in range(n):
        info = p.getJointInfo(robot, i)
        jtype = info[2]          # JOINT_REVOLUTE=0, JOINT_PRISMATIC=1, JOINT_FIXED=4
        jname = info[1].decode() # e.g. "panda_joint1", "panda_finger_joint1"
        if jtype == p.JOINT_FIXED:
            continue
        if "finger" in jname:
            fingers.append(i)
        elif "joint" in jname:
            arm.append(i)
    PANDA_ARM_JOINTS    = arm[:7]   # first 7 revolute = arm
    PANDA_FINGER_JOINTS = fingers[:2]
    print(f"  Arm joints    : {PANDA_ARM_JOINTS}")
    print(f"  Finger joints : {PANDA_FINGER_JOINTS}")


# ─────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────

def setup_simulation(gui=True):
    """Initialise physics client, gravity, and load ground plane + robot."""
    mode = p.GUI if gui else p.DIRECT
    client = p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(SLEEP_DT, physicsClientId=client)

    # Ground plane
    p.loadURDF("plane.urdf", physicsClientId=client)

    # Simple table
    table = p.loadURDF(
        "table/table.urdf",
        basePosition=[0.5, 0.0, 0.0],
        baseOrientation=p.getQuaternionFromEuler([0, 0, math.pi / 2]),
        physicsClientId=client,
    )

    # Franka Panda
    robot = p.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=[0, 0, 0.625],
        useFixedBase=True,
        physicsClientId=client,
    )
    discover_joints(robot)

    return client, robot, table



def spawn_cube(client, position, half_ext=0.025):
    """Create a small coloured cube at the given position."""
    col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half_ext]*3, physicsClientId=client)
    vis_shape = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[half_ext]*3,
        rgbaColor=[0.8, 0.1, 0.1, 1],
        physicsClientId=client,
    )
    cube = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=col_shape,
        baseVisualShapeIndex=vis_shape,
        basePosition=position,
        physicsClientId=client,
    )
    return cube


def reset_robot(client, robot):
    """Put the robot in a safe 'home' configuration and enable position control."""
    home = [0, -math.pi/4, 0, -3*math.pi/4, 0, math.pi/2, math.pi/4, 0.04, 0.04]
    for i, angle in enumerate(home):
        p.resetJointState(robot, i, angle, physicsClientId=client)
    set_arm_pose(client, robot, home[:7])
    set_gripper(client, robot, open=True)


def set_arm_pose(client, robot, joint_angles, speed=0.5):
    """Send position-control targets to the 7 arm joints."""
    for i, angle in zip(PANDA_ARM_JOINTS, joint_angles):
        p.setJointMotorControl2(
            robot, i,
            controlMode=p.POSITION_CONTROL,
            targetPosition=angle,
            maxVelocity=speed,
            force=87,
            physicsClientId=client,
        )


def set_gripper(client, robot, open=True):
    """
    Open or close the Franka gripper.

    Each finger travels 0 → 0.04 m (fully open).
    Force must be high enough to both move the fingers AND resist gravity on the cube.
    We also re-apply the command every step for `settle_steps` to make sure it registers.
    """
    target = 0.04 if open else 0.0          # metres per finger
    for _ in range(60):                      # hold command for 60 steps so it actuates
        for fi in PANDA_FINGER_JOINTS:
            p.setJointMotorControl2(
                robot, fi,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target,
                maxVelocity=0.05,            # slow, realistic finger speed
                force=200,                   # Franka fingers rated ~70 N – use 200 to overcome sim friction
                physicsClientId=client,
            )
        p.stepSimulation(physicsClientId=client)
        time.sleep(SLEEP_DT)


def compute_ik(client, robot, target_pos, target_orn):
    """Compute IK for the end-effector link."""
    joint_poses = p.calculateInverseKinematics(
        robot,
        EE_LINK,
        target_pos,
        target_orn,
        lowerLimits=[-2.9, -1.76, -2.9, -3.07, -2.9, -0.02, -2.9],
        upperLimits=[ 2.9,  1.76,  2.9, -0.07,  2.9,  3.75,  2.9],
        jointRanges=[5.8, 3.52, 5.8, 3.0, 5.8, 3.77, 5.8],
        restPoses=[0, -math.pi/4, 0, -3*math.pi/4, 0, math.pi/2, math.pi/4],
        maxNumIterations=200,
        residualThreshold=IK_THRESHOLD,
        physicsClientId=client,
    )
    return list(joint_poses[:7])


def move_to_pose(client, robot, target_pos, target_orn, steps=300, speed=0.5):
    """Move arm to a Cartesian pose by running IK and simulating steps."""
    angles = compute_ik(client, robot, target_pos, target_orn)
    set_arm_pose(client, robot, angles, speed=speed)
    for _ in range(steps):
        p.stepSimulation(physicsClientId=client)
        time.sleep(SLEEP_DT)


def wait(client, steps=120):
    for _ in range(steps):
        p.stepSimulation(physicsClientId=client)
        time.sleep(SLEEP_DT)


# ─────────────────────────────────────────────
# Pick-and-place routine
# ─────────────────────────────────────────────

def pick_and_place(client, robot, cube, pick_pos, place_pos):
    """
    Execute a full pick-and-place sequence.
    A fixed constraint is created between the EE and the cube after gripping
    so the cube is carried reliably regardless of friction tuning.
    """

    # Downward-facing end-effector orientation
    orn_down = p.getQuaternionFromEuler([math.pi, 0, math.pi / 4])

    hover_offset = 0.15    # metres above cube for pre-grasp / post-grasp
    grasp_z_off  = 0.01    # how far above cube centre to target (finger thickness)

    pre_grasp  = [pick_pos[0],  pick_pos[1],  pick_pos[2] + hover_offset]
    grasp_pos  = [pick_pos[0],  pick_pos[1],  pick_pos[2] + grasp_z_off]
    post_lift  = [pick_pos[0],  pick_pos[1],  pick_pos[2] + hover_offset + 0.05]
    pre_place  = [place_pos[0], place_pos[1], place_pos[2] + hover_offset]
    place_pos_ = [place_pos[0], place_pos[1], place_pos[2] + grasp_z_off]

    grasp_constraint = None

    print("[1/8] Moving to pre-grasp hover …")
    move_to_pose(client, robot, pre_grasp, orn_down, steps=400)

    print("[2/8] Opening gripper wide …")
    set_gripper(client, robot, open=True)

    print("[3/8] Lowering to grasp position …")
    move_to_pose(client, robot, grasp_pos, orn_down, steps=250, speed=0.3)

    print("[4/8] Closing gripper …")
    set_gripper(client, robot, open=False)

    # ── Grasp constraint: rigidly attach cube to EE so it won't slip ──
    cube_pos, cube_orn = p.getBasePositionAndOrientation(cube, physicsClientId=client)
    ee_state           = p.getLinkState(robot, EE_LINK, physicsClientId=client)
    ee_pos, ee_orn     = ee_state[0], ee_state[1]

    # Compute cube position relative to EE frame
    inv_ee_pos, inv_ee_orn = p.invertTransform(ee_pos, ee_orn)
    local_pos, local_orn   = p.multiplyTransforms(inv_ee_pos, inv_ee_orn, cube_pos, cube_orn)

    grasp_constraint = p.createConstraint(
        robot,          # parentBodyUniqueId
        EE_LINK,        # parentLinkIndex
        cube,           # childBodyUniqueId
        -1,             # childLinkIndex (-1 = base)
        p.JOINT_FIXED,  # jointType
        [0, 0, 0],      # jointAxis
        local_pos,      # parentFramePosition
        [0, 0, 0],      # childFramePosition
        local_orn,      # parentFrameOrientation
        physicsClientId=client,
    )
    # Allow some compliance so the constraint doesn't explode
    p.changeConstraint(grasp_constraint, maxForce=500, physicsClientId=client)
    print("    ✓ Grasp constraint created")

    print("[5/8] Lifting object …")
    move_to_pose(client, robot, post_lift, orn_down, steps=300)

    print("[6/8] Moving to place hover …")
    move_to_pose(client, robot, pre_place, orn_down, steps=500)

    print("[7/8] Lowering to place position …")
    move_to_pose(client, robot, place_pos_, orn_down, steps=250, speed=0.3)

    print("[8/8] Opening gripper and releasing …")
    # Remove constraint BEFORE opening so physics takes over naturally
    p.removeConstraint(grasp_constraint, physicsClientId=client)
    grasp_constraint = None
    set_gripper(client, robot, open=True)
    wait(client, steps=120)

    print("    Retracting …")
    move_to_pose(client, robot, pre_place, orn_down, steps=300)

    print("✓ Pick-and-place complete!")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("Starting Franka Panda pick-and-place simulation …")
    print("Close the PyBullet window or press Ctrl-C to quit.\n")

    client, robot, table = setup_simulation(gui=True)

    # Configure camera for a better view
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0.4, 0.0, 0.6],
        physicsClientId=client,
    )

    # Home the robot
    reset_robot(client, robot)
    wait(client, steps=200)

    # Spawn cube
    cube = spawn_cube(client, CUBE_START)
    wait(client, steps=100)

    # Execute pick-and-place
    pick_and_place(client, robot, cube, CUBE_START, PLACE_TARGET)

    # Keep window open
    print("\nSimulation finished. Keeping window open …")
    try:
        while p.isConnected(client):
            p.stepSimulation(physicsClientId=client)
            time.sleep(SLEEP_DT)
    except (p.error, KeyboardInterrupt):
        pass

    p.disconnect(client)


if __name__ == "__main__":
    main()