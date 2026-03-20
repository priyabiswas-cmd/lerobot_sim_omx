# Shared state between SimOMXRobot and ScriptedOMXTeleop
pybullet_client: int = -1
pybullet_robot_id: int = -1
ee_link: int = -1
arm_joints: list = []
teleop = None