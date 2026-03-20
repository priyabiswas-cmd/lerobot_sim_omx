from dataclasses import dataclass, field
from pathlib import Path
from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig



@dataclass
class SimCameraConfig:
    """Minimal camera config for PyBullet rendered cameras."""
    height: int = 480
    width:  int = 640
    fps:    int = 10

@RobotConfig.register_subclass("sim_omx")
@dataclass
class SimOMXRobotConfig(RobotConfig):
    urdf_path: str = ""
    gui: bool = True
    cameras: dict = field(default_factory=lambda: {
    "camera1": SimCameraConfig(height=240, width=320),
    "camera2": SimCameraConfig(height=240, width=320),
})
    use_degrees: bool = False
    sim_hz: int = 240
    control_hz: int = 30
    id: str | None = "sim_omx"
    calibration_dir: Path | None = None