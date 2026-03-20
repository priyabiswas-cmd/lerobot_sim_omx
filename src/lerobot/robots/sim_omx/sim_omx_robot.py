#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0

import logging
import math
import time
from functools import cached_property

import numpy as np
import pybullet as p
import pybullet_data
from . import sim_state  # shared state for SimOMXRobot and ScriptedOMXTeleop
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_sim_omx import SimOMXRobotConfig

logger = logging.getLogger(__name__)

# ── Motor definitions (mirrors OmxFollower motor order exactly) ──
MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "gripper",
]

# Home position in radians for each arm joint (gripper open)
HOME_ANGLES = [0.0, -1.0, 0.3, 0.7] 
GRIPPER_OPEN   = 0.019   # metres, OMX gripper open
GRIPPER_CLOSED = 0.0     # metres, OMX gripper closed
EE_LINK        = 7       # end_effector_joint

TABLE_Z  = 0.626
IMG_W    = 224
IMG_H    = 224


class SimOMXRobot(Robot):
    """
    PyBullet simulation of the OpenManipulator-X.
    Mirrors OmxFollower interface exactly so it can be dropped into
    record.py without any changes to that file.

    observation keys : shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos,
                       wrist_flex.pos,   wrist_roll.pos,    gripper.pos,
                       + one key per camera defined in config.cameras
    action keys      : same as motor observation keys (no camera keys)
    """

    config_class = SimOMXRobotConfig
    name = "sim_omx"

    def __init__(self, config: SimOMXRobotConfig):
        super().__init__(config)
        self.config = config

        # set after connect()
        self._client       = None
        self._robot        = None
        self._arm_joints   = []      # pybullet joint indices for 5 arm DOF
        self._finger_joints = None    # pybullet joint index for gripper
        self._ee_link      = None    # end-effector link index
        self._cube = None

    # ── Feature dicts (mirror OmxFollower property names) ────────

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in MOTOR_NAMES}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height,
                  self.config.cameras[cam].width, 3)
            for cam in self.config.cameras
        }

    @cached_property
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict:
        # Exactly like OmxFollower — only motor positions, no cameras
        return self._motors_ft

    # ── Connection state ──────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    @property
    def is_calibrated(self) -> bool:
        return True   # sim never needs calibration

    # ── Lifecycle ─────────────────────────────────────────────────

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        mode = p.GUI if self.config.gui else p.DIRECT
        self._client = p.connect(mode)

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self._client
        )
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.setTimeStep(
            1.0 / self.config.sim_hz, physicsClientId=self._client
        )

        # Scene
        p.loadURDF("plane.urdf", physicsClientId=self._client)
        p.loadURDF(
            "table/table.urdf",
            basePosition=[0.5, 0.0, 0.0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, math.pi / 2]),
            physicsClientId=self._client,
        )
        
       
        # table_id = p.loadURDF(
        #     "table/table.urdf",
        #     basePosition=[0.5, 0.0, 0.0],
        #     baseOrientation=p.getQuaternionFromEuler([0, 0, math.pi / 2]),
        #     physicsClientId=self._client,
        # )
        # print("Table AABB:", p.getAABB(table_id, physicsClientId=self._client))
        self._robot = p.loadURDF(
            self.config.urdf_path,
            basePosition=[0.5, 0, TABLE_Z],
            useFixedBase=True,
            physicsClientId=self._client,
        )

        p.changeVisualShape(
        self._robot, 7,
        rgbaColor=[0, 0, 0, 0],
        physicsClientId=self._client,
        )
        self._discover_joints()
        self._reset_to_home()

                # after self._reset_to_home()
        self._cube = self._spawn_cube()   #on only when eval mode, not during training to avoid 2 cube spawns

        
        if self.config.gui:
            p.resetDebugVisualizerCamera(
                1.2, 45, -30, [0.4, 0, TABLE_Z + 0.2],
                physicsClientId=self._client,
            )
        
        # Store in shared state for teleop to access
        sim_state.pybullet_client  = self._client
        sim_state.pybullet_robot_id = self._robot
        sim_state.ee_link          = self._ee_link
        sim_state.arm_joints       = self._arm_joints   
        self._connected = True  
        logger.info(f"{self} connected.")

    def calibrate(self) -> None:
        pass   # no-op — sim needs no calibration

    def configure(self) -> None:
        pass   # no-op — no motor registers to write

    def disconnect(self) -> None:
        if self._client is not None:
            p.disconnect(self._client)
            self._client = None
        logger.info(f"{self} disconnected.")

    # ── Core interface (mirrors OmxFollower exactly) ──────────────

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs = {}

        # Motor positions — identical key format to OmxFollower
        start = time.perf_counter()
        raw = self._read_joint_positions()   # dict motor_name → float
        for motor, val in raw.items():
            obs[f"{motor}.pos"] = val
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Camera frames — keyed by camera name from config
        for cam_key in self.config.cameras:
            start = time.perf_counter()
            obs[cam_key] = self._render_camera(cam_key)
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """
        Accepts action dict with keys like 'shoulder_pan.pos'.
        Converts normalized values → radians/metres, sends to PyBullet,
        steps the simulation one control tick, returns action as sent.
        Mirrors OmxFollower.send_action() signature exactly.
        """
        # Strip .pos suffix — same pattern as OmxFollower
        goal_pos = {
            key.removesuffix(".pos"): val
            for key, val in action.items()
            if key.endswith(".pos")
        }

        self._write_joint_positions(goal_pos)

        # Advance sim one control tick
        steps = self.config.sim_hz // self.config.control_hz
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self._client)
            if self.config.gui:
                time.sleep(1.0 / self.config.sim_hz)

        # Return in same format as OmxFollower
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    # ── PyBullet helpers ──────────────────────────────────────────

    def _spawn_cube(self):
            import random
            from lerobot.teleoperators.sim_omx_teleop.sim_omx_teleop import CUBE_POSITIONS
            pos = CUBE_POSITIONS[random.randint(0, len(CUBE_POSITIONS)-1)]
            x, y = pos
            # x = random.uniform(0.62, 0.72)
            # y = random.uniform(-0.10, 0.10)
            half = 0.025
            col = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[half]*3,
                physicsClientId=self._client,
            )
            vis = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[half]*3,
                rgbaColor=[0.85, 0.12, 0.12, 1],
                physicsClientId=self._client,
            )
            cube = p.createMultiBody(
                baseMass=0.08,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[x, y, 0.651],
                physicsClientId=self._client,
            )
            p.changeDynamics(
                cube, -1,
                lateralFriction=5.0,
                spinningFriction=1.0,
                rollingFriction=1.0,
                restitution=0.0,
                physicsClientId=self._client,
            )
            return cube
    def _discover_joints(self):
        """Find joint indices for arm DOF and gripper."""
        arm, gripper = [], []
        for i in range(p.getNumJoints(self._robot)):
            info  = p.getJointInfo(self._robot, i)
            jtype = info[2]
            jname = info[1].decode().lower()
            if jtype == p.JOINT_FIXED:
                continue
            if "gripper" in jname:
                gripper.append(i)
            else:
                arm.append(i)

        self._arm_joints   = arm       # [1, 2, 3, 4]
        self._finger_joints = gripper   # single gripper DOF
        self._ee_link      = 7        # wrist_roll link = EE

        logger.debug(
            f"Arm joints: {self._arm_joints}  "
            f"Finger joint: {self._finger_joints}  "
            f"EE link: {self._ee_link}"
        )

    def _reset_to_home(self):
        for i, angle in enumerate(HOME_ANGLES):
            p.resetJointState(
                self._robot, self._arm_joints[i], angle,
                physicsClientId=self._client,
            )
        for fi in self._finger_joints:            # both gripper joints
            p.resetJointState(
                self._robot, fi, GRIPPER_OPEN,
                physicsClientId=self._client,
            )
        # Let sim settle
        for _ in range(120):
            p.stepSimulation(physicsClientId=self._client)
        ee = p.getLinkState(self._robot, self._ee_link, physicsClientId=self._client)
        print("EE position at home:", ee[0])
        print("EE link index:", self._ee_link)
    def _read_joint_positions(self) -> dict[str, float]:
        """
        Read current joint positions and convert to normalised values
        that match what OmxFollower would return:
          arm joints  → RANGE_M100_100  (radians scaled to [-100, 100])
          gripper     → RANGE_0_100     (metres scaled to [0, 100])
        """
        positions = {}
        for i, name in enumerate(MOTOR_NAMES[:-1]):   # arm joints
            raw = p.getJointState(
                self._robot, self._arm_joints[i],
                physicsClientId=self._client,
            )[0]
            positions[name] = (raw / math.pi) * 100.0

        # Gripper
        raw_g = p.getJointState(
            self._robot, self._finger_joints[0],
            physicsClientId=self._client,
        )[0]
        gripper_val = (raw_g / GRIPPER_OPEN) * 100.0
        positions["gripper"] = (raw_g / GRIPPER_OPEN) * 100.0

        return positions

    def _write_joint_positions(self, goal_pos: dict[str, float]):
        """
        Accept normalised motor values (same range as OmxFollower sends)
        and convert back to radians/metres for PyBullet.
        """
        for i, name in enumerate(MOTOR_NAMES[:-1]):   # arm joints
            if name not in goal_pos:
                continue
            # RANGE_M100_100 → radians
            target = (goal_pos[name] / 100.0) * math.pi
            p.setJointMotorControl2(
                self._robot, self._arm_joints[i],
                p.POSITION_CONTROL,
                targetPosition=target,
                maxVelocity=1.0,
                force=50.0,
                physicsClientId=self._client,
            )

        if "gripper" in goal_pos:
            target_g = (goal_pos["gripper"] / 100.0) * GRIPPER_OPEN
            target_g = max(0.0, min(GRIPPER_OPEN, target_g))  # clamp to [0, 0.019]
            for fi in self._finger_joints:
                p.setJointMotorControl2(
                    self._robot, fi,
                    p.POSITION_CONTROL,
                    targetPosition=target_g,   # same target for both fingers
                    maxVelocity=0.5,
                    force=20.0,
                    physicsClientId=self._client,
                )

    def _render_camera(self, cam_key: str) -> np.ndarray:
        """
        Render one camera frame.
        Camera placement is determined by cam_key name convention:
          'top' or 'laptop' → overhead view
          'wrist'           → wrist-mounted view
        """
        cfg = self.config.cameras[cam_key]
        w, h = cfg.width, cfg.height

        if cam_key == "camera2":
            ee    = p.getLinkState(
                self._robot, self._ee_link,
                physicsClientId=self._client,
            )
            ee_pos = np.array(ee[0])
            ee_mat = np.array(
                p.getMatrixFromQuaternion(ee[1])
            ).reshape(3, 3)
            eye     = ee_pos + ee_mat @ np.array([0, 0, -0.08])
            target  = ee_pos + ee_mat @ np.array([0, 0,  0.08])
            up      = ee_mat[:, 1].tolist()
        else:
            # Overhead / laptop camera
            eye    = [0.0, -0.5, 1.2]   # front-left above table
            target = [0.6,  0.0, 0.65]  # looking at workspace
            up     = [0, 0, 1]

        view = p.computeViewMatrix(
            eye, target, up, physicsClientId=self._client
        )
        proj = p.computeProjectionMatrixFOV(
            fov=60, aspect=w / h,
            nearVal=0.01, farVal=10.0,
            physicsClientId=self._client,
        )
        # renderer = (
        #     p.ER_BULLET_HARDWARE_OPENGL
        #     if self.config.gui
        #     else p.ER_TINY_RENDERER
        # )
        renderer = p.ER_TINY_RENDERER  # more consistent rendering in headless mode
        _, _, rgb, _, _ = p.getCameraImage(
            w, h,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=renderer,
            physicsClientId=self._client,
        )
        return np.array(rgb, dtype=np.uint8)[:, :, :3]

    # ── Expose PyBullet handles for ScriptedOMXTeleop ────────────

    @property
    def pybullet_client(self) -> int:
        return self._client

    @property
    def pybullet_robot_id(self) -> int:
        return self._robot

    @property
    def ee_link(self) -> int:
        return self._ee_link
    
    @property
    def cameras(self) -> dict:
        """
        Mirrors OmxFollower.cameras structure.
        Returns empty dict since sim renders cameras internally via PyBullet.
        record.py uses len(robot.cameras) to set image writer threads.
        """
        return self.config.cameras
    
    @property
    def episode_done(self) -> bool:
        """True when teleop has completed one full pick-and-place."""
        if sim_state.teleop is not None:
            return sim_state.teleop.episode_complete
        return False