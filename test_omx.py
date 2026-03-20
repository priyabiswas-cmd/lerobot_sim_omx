# test_sim_omx.py  (repo root)
import numpy as np
from lerobot.robots.sim_omx import SimOMXRobot, SimOMXRobotConfig

def test_robot():
    config = SimOMXRobotConfig(
        urdf_path="open_manipulator-main/open_manipulator_description/urdf/open_manipulator_x/open_manipulator_x.urdf",  # your actual path
        gui=False,
        cameras={},   # empty for now — test motors first
    )

    robot = SimOMXRobot(config)

    # 1. Feature dicts
    print("observation_features:", robot.observation_features)
    print("action_features:     ", robot.action_features)

    expected_motor_keys = {
        "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
        "wrist_flex.pos",   "gripper.pos",
    }
    print("actual action_features:", robot.action_features)
    print("actual keys:", set(robot.action_features.keys()))
    print("expected keys:", expected_motor_keys)
    assert set(robot.action_features.keys()) == expected_motor_keys
    print("✓ feature dicts correct")

    # 2. Connect
    robot.connect()
    assert robot.is_connected
    print("✓ connected")
    print(f"  arm_joints    : {robot._arm_joints}")
    print(f"  finger_joints : {robot._finger_joints}")
    print(f"  ee_link       : {robot._ee_link}")

    # 3. get_observation
    obs = robot.get_observation()
    print("observation:", obs)
    print("gripper.pos value:", obs["gripper.pos"])
    for name in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]:
        val = obs[f"{name}.pos"]
        assert isinstance(val, float), f"{name}.pos should be float"
        assert -100.0 <= val <= 100.0, f"{name}.pos={val} out of range"

    assert -0.1 <= obs["gripper.pos"] <= 100.1
    print("✓ get_observation correct")

    # 4. send_action
    action = {
        "shoulder_pan.pos":   0.0,
        "shoulder_lift.pos": -50.0,
        "elbow_flex.pos":     15.0,
        "wrist_flex.pos":     40.0,
        "gripper.pos":       100.0,
    }
    returned = robot.send_action(action)
    assert set(returned.keys()) == expected_motor_keys
    print("✓ send_action correct")

    # 5. Multiple steps
    for i in range(30):
        obs = robot.get_observation()
        robot.send_action(action)
    print("✓ 30 steps completed")

    # 6. Disconnect
    robot.disconnect()
    assert not robot.is_connected
    print("✓ disconnected")

    print("\n✓ SimOMXRobot ready")

if __name__ == "__main__":
    test_robot()