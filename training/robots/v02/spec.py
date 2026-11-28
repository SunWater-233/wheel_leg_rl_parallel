"""Single source of truth for Wheel Leg V0.2 physical, kinematic, and control-contract constants."""

from configs.schema.constants import TRAINING_DIR


MODEL_PATH = TRAINING_DIR / "assets/wheel_leg_xml_V0.2/wheel_leg_ground.xml"
LABEL = "Wheel Leg V0.2"
PHYSICS_DT = 0.002
DECIMATION = 10
HOME_HEIGHT = 0.2370327302
MIN_HEIGHT = 0.12
WHEEL_RADIUS = 0.058
HALF_TRACK = 0.21865
WHEEL_TARGET_LIMIT = 2.9224 / WHEEL_RADIUS
STANCE_SPAWN_GAP = 0.002

LEG_KP = 100.0
LEG_KV = 4.0
LEG_TORQUE_LIMIT_DEFAULT = 12.0
LEG_ARMATURE = 0.01
LEG_ACTION_SCALE = 0.20
# Static hip torque needed at neutral stance, divided by LEG_KP into PD target offset (rad).
LEG_BIAS_TORQUE = (-4.25, 4.30, 4.31, -4.32)
LEG_BIAS_RAD = tuple(torque / LEG_KP for torque in LEG_BIAS_TORQUE)

WHEEL_KV = 0.6
WHEEL_TORQUE_LIMIT_DEFAULT = 5.0
SPRING_FORCE_NOMINAL = 200.0
EXPECTED_EQUALITIES = 12

LEG_JOINTS = ("hip_L", "hip_R", "hipknee_L", "hipknee_R")
WHEEL_JOINTS = ("wheel_L", "wheel_R")
SPRING_JOINTS = ("gas_slide_L", "gas_slide_R")
WHEEL_BODIES = WHEEL_JOINTS
BODY_CONTACTS = ("base_link", "gimbal_yaw", "gimbal_pitch")
LEG_SIGNS = (1.0, -1.0, 1.0, -1.0)
WHEEL_SIGNS = (1.0, -1.0)

# Solved on the CAD assembly branch with both axles under the center of mass
# and the passive knees 0.20 rad inside their travel limits.
HOME_JOINTS = {
    "hip_L": -0.585,
    "hip_R": 0.585,
    "hipknee_L": -0.8201266288,
    "hipknee_R": 0.8201266397,
    "knee_L": 1.4566159,
    "knee_R": 1.8566159,
    "gas_up_L": -0.0011092105,
    "gas_up_R": 0.0011092105,
    "gas_slide_L": -0.0100408686,
    "gas_slide_R": -0.0100408686,
    "drive_l1_L": 0.2000000000,
    "drive_l1_R": -0.2000000043,
    "l1_node_L": -0.1648733712,
    "l1_node_R": 0.1648733647,
    "node_l2_L": 0.2000000000,
    "node_l2_R": -0.2000000001,
}
