import numpy as np
import torch

from av2.datasets.motion_forecasting.data_schema import Track

from data.dimensions import Dim


# AV2 dataset was collected with a 2019 Ford Fusion, which has a wheel base of
# around 112 inches. This corresponds to a ~1.4m distance from the rear wheels
# to the measured pose, assuming that the pose was measured at the wheel base
# center.
WHEEL_BASE = 2.8
MAX_CURV = 0.1
MAX_STEER = np.arctan(MAX_CURV * WHEEL_BASE)
MAX_ACCEL = 10.0


def compute_from_track(track: Track) -> torch.Tensor:
    assert track.track_id == "AV"
    assert len(track.object_states) == 110

    x = []
    y = []
    yaw = []
    for state in track.object_states:
        x.append(state.position[0])
        y.append(state.position[1])
        yaw.append(state.heading)

    gvx = np.gradient(x, 0.1)
    gvy = np.gradient(y, 0.1)

    vx = []
    vy = []
    for vx0, vy0, yaw0 in zip(gvx, gvy, yaw):
        s = np.sin(yaw0)
        c = np.cos(yaw0)
        vx.append(c * vx0 + s * vy0)
        vy.append(-s * vx0 + c * vy0)

    accel = np.gradient(vx, 0.1)
    curv = np.gradient(yaw, 0.1)
    steer = np.arctan(WHEEL_BASE * curv)

    accel_sub = []
    steer_sub = []
    for chunk in np.array_split(accel, Dim.T - 1):
        norm = chunk.mean() / MAX_ACCEL
        norm = (norm + 1) / 2
        accel_sub.append(min(max(norm, 0.0), 1.0))
    for chunk in np.array_split(steer, Dim.T - 1):
        norm = chunk.mean() / MAX_STEER
        norm = (norm + 1) / 2
        steer_sub.append(min(max(norm, 0.0), 1.0))
    return torch.Tensor(list(zip(accel_sub, steer_sub)))


def discretize(controls: torch.Tensor) -> torch.Tensor:
    """
    controls[..., C]
    """
    # Controls should be normalized to the range [0,1]
    B = (Dim.Cd - 1) // 2
    accel_bucket = _bucketize(controls[..., 0], B)
    steer_bucket = _bucketize(controls[..., 1], B)
    return (Dim.Cd * accel_bucket + steer_bucket).long()


def undiscretize(discrete: torch.Tensor) -> torch.Tensor:
    # Extract discrete acceleration and steering values
    accel_bucket = discrete // Dim.Cd
    steer_bucket = discrete % Dim.Cd

    # Reverse the scaling and rounding applied in _bucketize
    B = (Dim.Cd - 1) // 2
    accel_continuous = _debucketize(accel_bucket, B)
    steer_continuous = _debucketize(steer_bucket, B)

    # Combine the continuous values back into the original tensor format
    return torch.stack([accel_continuous, steer_continuous], dim=-1)


def _bucketize(x: torch.Tensor, buckets: int):
    scaled = 2 * (x - 0.5)
    sign = torch.sign(scaled)
    sign[sign == 0] = 1
    scaled = buckets * torch.sqrt(sign * scaled)
    return sign * torch.round(scaled) + buckets


def _debucketize(x: torch.Tensor, buckets: int):
    scaled = x - buckets
    sign = torch.sign(scaled)
    sign[sign == 0] = 1
    scaled = sign * (scaled / buckets)**2
    scaled = scaled / 2 + 0.5
    return scaled