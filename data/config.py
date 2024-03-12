import enum
from pathlib import Path


LOCAL_ROOT_DIR = Path("~/mlsearch").expanduser()
EXPERIMENT_ROOT = LOCAL_ROOT_DIR / "experiments"
DATA_ROOT = LOCAL_ROOT_DIR / "data"
TRAIN_DATA_ROOT = DATA_ROOT / "train"
TEST_DATA_ROOT = DATA_ROOT / "test"

# AV2 assumes 11sec history at 10hz
AV2_MAX_TIME = 110

# Constant scaling factors to limit the magnitude of position and velocity values
# so that using fp16 is feasible.
POS_SCALE = 100.0
VEL_SCALE = 25.0

ROADGRAPH_SEARCH_RADIUS = 100.0


"""
Enum for configured dimensions of scenario tensors.
"""
class Dim(enum.IntEnum):
    # Max agents
    A = 64
    # Number of time steps
    T = 55
    # Agent state size
    S = 7
    # Max agent interactions
    Ai = 16
    # Number of roadgraph features
    R = 1024
    # Dimension of roadgraph features
    Rd = 8
    # Dimension of the controls that can be applied to the vehicle
    C = 2
    # Discretization of each control dimension
    Cd = 17
