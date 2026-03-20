#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import random
from functools import cached_property
from typing import Any
from lerobot.robots.sim_omx import sim_state  # shared state for SimOMXRobot and ScriptedOMXTeleop
import numpy as np
import pybullet as p

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_sim_omx_teleop import ScriptedOMXTeleopConfig

logger = logging.getLogger(__name__)

# Must match SimOMXRobot MOTOR_NAMES exactly
MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "gripper",
]

GRIPPER_OPEN = 0.019   # metres — OMX gripper fully open
TABLE_Z      = 0.626
CUBE_X_RANGE   = (0.15, 0.25)   # within OMX reach
CUBE_Y_RANGE   = (-0.10, 0.10)  # centered
CUBE_POSITIONS = [
    (0.63, 0.0),
    (0.63, 0.08),
    (0.63, -0.08),
    (0.70, 0.0),
    (0.70, 0.08),
]
class ScriptedOMXTeleop(Teleoperator):
    """
    Scripted pick-and-place teleoperator for SimOMXRobot.
    Replaces a physical leader arm — computes IK waypoints and
    returns one joint target dict per get_action() call.
    Must be initialised AFTER robot.connect() so PyBullet
    client and robot ID are available for IK.
    """

    config_class = ScriptedOMXTeleopConfig
    name = "scripted_omx"

    def __init__(
        self,
        config: ScriptedOMXTeleopConfig,
    ):
        super().__init__(config)
        self.config      = config
        self._connected  = False
        self._waypoints  = []
        self._step       = 0
        self._cube       = None
        self._client     = None
        self._robot      = None
        self._ee_link    = None
        self._arm_joints = []
        self._constraint_id = None
        self._position_idx = 0
        
    # ── Abstract properties ───────────────────────────────────────

    @property
    def action_features(self) -> dict[str, type]:
        # Mirrors OmxLeader.action_features key format exactly
        return {f"{motor}.pos": float for motor in MOTOR_NAMES}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}   # no haptic feedback in sim

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True   # sim needs no calibration
    
    @property
    def episode_complete(self) -> bool:
        """True when all waypoints have been executed once."""
        return self._step >= len(self._waypoints)

    # ── Lifecycle ─────────────────────────────────────────────────

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        from lerobot.robots.sim_omx import sim_state
        sim_state.teleop = self
        if sim_state.pybullet_client == -1:
            raise RuntimeError(
                "SimOMXRobot.connect() must be called before ScriptedOMXTeleop.connect()."
            )

        self._client     = sim_state.pybullet_client
        self._robot      = sim_state.pybullet_robot_id
        self._ee_link    = sim_state.ee_link
        self._arm_joints = sim_state.arm_joints

        cx, cy = CUBE_POSITIONS[self._position_idx % len(CUBE_POSITIONS)]
        self._position_idx += 1
        self._cube      = self._spawn_cube(cx, cy)
        self._waypoints = self._build_waypoints(cx, cy)
        self._step      = 0
        self._connected = True
        logger.info(f"{self} connected. Cube at ({cx:.3f}, {cy:.3f})")
    def calibrate(self) -> None:
        pass   # no-op

    def configure(self) -> None:
        pass   # no-op

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass   # no-op — mirrors OmxLeader.send_feedback signature

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        if not self._waypoints:
            return {f"{m}.pos": 0.0 for m in MOTOR_NAMES}

        if self._step >= len(self._waypoints):
            return {
                f"{name}.pos": float(self._waypoints[-1][i])
                for i, name in enumerate(MOTOR_NAMES)
            }

        targets    = self._waypoints[self._step]
        self._step += 1

        return {
            f"{name}.pos": float(targets[i])
            for i, name in enumerate(MOTOR_NAMES)
        }
    @check_if_not_connected
    def disconnect(self) -> None:
        if self._cube is not None:
            try:
                p.removeBody(self._cube, physicsClientId=self._client)
            except Exception:
                pass
            self._cube = None
        p.disconnect(self._client)
        self._connected = False
        logger.info(f"{self} disconnected.")

    # ── Episode management ────────────────────────────────────────

    def reset_episode(self) -> None:
        self._release_cube() if hasattr(self, '_release_cube') else None
        if self._cube is not None:
            try:
                p.removeBody(self._cube, physicsClientId=self._client)
            except Exception:
                pass

        cx, cy = CUBE_POSITIONS[self._position_idx % len(CUBE_POSITIONS)]
        self._position_idx += 1
        self._cube      = self._spawn_cube(cx, cy)
        self._waypoints = self._build_waypoints(cx, cy)
        self._step      = 0
        logger.info(f"Episode reset. Cube at ({cx:.3f}, {cy:.3f})")

    # ── PyBullet helpers ──────────────────────────────────────────

    def _spawn_cube(self, x: float, y: float, half: float = 0.025) -> int:
        col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[half] * 3,
            physicsClientId=self._client,
        )
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[half] * 3,
            rgbaColor=[0.85, 0.12, 0.12, 1],
            physicsClientId=self._client,
        )
        cube = p.createMultiBody(
            baseMass=0.08,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[x, y, 0.670],
            physicsClientId=self._client,
        )

        p.changeDynamics(
        cube, -1,
        lateralFriction=5.0,
        spinningFriction=1.0,
        rollingFriction=1.0,
        restitution=0.0,
        contactProcessingThreshold=0.0,   # ← add this
        physicsClientId=self._client,
        )
        return cube

    def _compute_ik(self, pos: list, orn) -> list[float]:
        result = p.calculateInverseKinematics(
            self._robot, self._ee_link, pos, orn,
        lowerLimits=[-3.142, -1.5, -1.5, -1.7],
        upperLimits=[ 3.142,  1.5,  1.4,  1.97],
        jointRanges=[ 6.284,  3.0,  2.9,  3.67],
        restPoses=   [ 0.0,  -1.0,  1.2,  0.5],
            maxNumIterations=500,
            residualThreshold=0.001,
            physicsClientId=self._client,
        )
        return list(result[:4])   # 4 arm joints only

    def _joints_to_normalised(
        self, arm_angles: list[float], gripper_open: bool
    ) -> np.ndarray:
        """
        Convert raw radian angles + gripper state to normalised range:
          arm joints  → RANGE_M100_100  (radians → [-100, 100])
          gripper     → RANGE_0_100     (100=open, 0=closed)
        Returns (5,) array matching MOTOR_NAMES order.
        """
        normalised_arm = [(a / math.pi) * 100.0 for a in arm_angles]
        gripper_val    = 100.0 if gripper_open else 0.0
        return np.array(normalised_arm + [gripper_val], dtype=np.float32)

    def _build_waypoints(
        self, cube_x: float, cube_y: float
    ) -> list[np.ndarray]:
        """
        Build interpolated waypoints for a full pick-and-place.
        Each waypoint is a (5,) normalised array matching MOTOR_NAMES.
        """
        orn     = p.getQuaternionFromEuler([math.pi/2, 0, math.pi/2])
        cube_z  =  0.670
        hover_z = cube_z + 0.1
        place_y = cube_y + 0.20
        n       = self.config.n_interp


        # (ee_position, gripper_open)
        key_poses = [
            ([cube_x, cube_y,  hover_z],        True),   # hover
            ([cube_x, cube_y,  cube_z - 0.01],  True),   # descend lower
            ([cube_x, cube_y,  cube_z - 0.01],  False),  # close gripper
            ([cube_x, cube_y,  hover_z],        False),  # lift
            ([cube_x, place_y, hover_z],        False),  # transit
            ([cube_x, place_y, cube_z - 0.01],  False),  # lower
            ([cube_x, place_y, cube_z - 0.01],  True),   # release
            ([cube_x, place_y, hover_z],        True),   # retract
        ]

        waypoints = []
        prev      = None

        for pos, gripper_open in key_poses:
            arm_angles = self._compute_ik(pos, orn)
            target     = self._joints_to_normalised(arm_angles, gripper_open)

            if prev is None:
                for _ in range(n):
                    waypoints.append(target.copy())
            else:
                for i in range(1, n + 1):
                    t = i / n
                    waypoints.append(
                        (prev * (1 - t) + target * t).astype(np.float32)
                    )
            prev = target

        return waypoints
    
    