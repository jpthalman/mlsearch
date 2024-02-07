import numpy as np
import torch

from data.dimensions import Dim

ACCEL_BOUNDS = (-5.0, 5.0)


def normalize_accel(acc: float) -> float:
    BOUNDS = (-5.0, 5.0)
    norm = (acc - BOUNDS[0]) / (BOUNDS[1] - BOUNDS[0])
    return min(max(norm, 0.0), 1.0)


def normalize_steering(rate: float) -> float:
    BOUNDS = (-1.5, 1.5)
    norm = (rate - BOUNDS[0]) / (BOUNDS[1] - BOUNDS[0])
    return min(max(norm, 0.0), 1.0)


def discretize_controls(controls: torch.Tensor) -> torch.Tensor:
    """
    controls[..., C]
    """
    # Controls should be normalized to the range [0,1]
    dx = ((Dim.Cd - 1) * controls[..., 0]).round().long()
    dy = ((Dim.Cd - 1) * controls[..., 1]).round().long()
    idx = Dim.Cd * dx + dy

    # Computed such that at the discretized index for the original control, the
    # softmax probability of that index will be CONFIDENCE and the remaining
    # probabilities will be equal to each other.
    CONFIDENCE = 0.7
    SCALE = np.log((1 - CONFIDENCE) / (CONFIDENCE * (Dim.Cd**2 - 1)))
    logits = SCALE * torch.ones(
        [*controls.shape[:-1], Dim.Cd**2],
        dtype=controls.dtype,
        device=controls.device,
    )
    logits.scatter_(-1, idx.unsqueeze(-1), 0.0)
    return logits.softmax(dim=-1)
