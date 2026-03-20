#!/usr/bin/env python

import random
from pathlib import Path

from lerobot.robots.sim_omx import SimOMXRobot, SimOMXRobotConfig
from lerobot.teleoperators.sim_omx_teleop import (
    ScriptedOMXTeleop,
    ScriptedOMXTeleopConfig,
)
from lerobot.record import record, RecordConfig, DatasetRecordConfig

# ── Config ────────────────────────────────────────────────────────
robot_cfg = SimOMXRobotConfig(
    urdf_path="/home/priya/lerobot/open_manipulator-main/open_manipulator_description/urdf/open_manipulator_x/open_manipulator_x.urdf",
    gui=False,
)

dataset_cfg = DatasetRecordConfig(
    repo_id="priya/lerobot/sim-omx-pick-place",
    single_task="Pick up the red cube and place it to the right.",
    fps=10,
    num_episodes=10,
    episode_time_s=20,
    reset_time_s=5,
    video=True,
    push_to_hub=False,
)

# ── Robot first — teleop needs pybullet handles ───────────────────
robot = SimOMXRobot(robot_cfg)
robot.connect()

teleop_cfg = ScriptedOMXTeleopConfig()
teleop = ScriptedOMXTeleop(
    config=teleop_cfg,
    pybullet_client=robot.pybullet_client,
    pybullet_robot_id=robot.pybullet_robot_id,
    ee_link=robot.ee_link,
    arm_joints=robot._arm_joints,
)

# ── Record ────────────────────────────────────────────────────────
cfg = RecordConfig(
    robot=robot_cfg,
    teleop=teleop_cfg,
    dataset=dataset_cfg,
)

dataset = record(cfg)