import torch

from data.config import Dim


MAX_CURV = 0.1
MAX_ACCEL = 10.0


def integrate(state: torch.Tensor, controls_norm: torch.Tensor) -> torch.Tensor:
    """
    state[B, S]
    controls[B, C]
    """
    x0 = state[:, 0]
    y0 = state[:, 1]
    c0 = state[:, 2]
    s0 = state[:, 3]
    v0 = state[:, 4]

    controls = denormalize(controls_norm)
    accel = controls[:, 0]
    curv = controls[:, 1]

    v1 = (v0 + accel).relu()
    arclen = 0.5 * (v0 + v1)
    x1, y1 = _advance(x0, y0, c0, s0, curv, arclen)

    dyaw = curv * arclen
    c = dyaw.cos()
    s = dyaw.sin()
    c1_n = c * c0 - s * s0
    s1_n = s * c0 + c * s0
    norm = torch.sqrt(c1_n**2 + s1_n**2)
    c1 = c1_n / norm
    s1 = s1_n / norm

    return torch.cat(
        [
            x1.unsqueeze(1),
            y1.unsqueeze(1),
            c1.unsqueeze(1),
            s1.unsqueeze(1),
            v1.unsqueeze(1),
            state[:, 5:],
        ],
        dim=1,
    )


def relative_positions(
    state: torch.Tensor,
    next_state: torch.Tensor,
) -> torch.Tensor:
    """
    state[B, S]
    next_state[B, S]
    """
    x0 = state[:, 0]
    y0 = state[:, 1]
    c0 = state[:, 2]
    s0 = state[:, 3]
    x1 = next_state[:, 0]
    y1 = next_state[:, 1]
    rx, ry = _inv_transform(x0, y0, c0, s0, x1, y1)
    return torch.cat([rx.unsqueeze(1), ry.unsqueeze(1)], dim=1)


def denormalize(controls: torch.Tensor) -> torch.Tensor:
    """
    controls[B, C]
    """
    accel = (2 * controls[:, 0] - 1) * MAX_ACCEL
    curv = (2 * controls[:, 1] - 1) * MAX_CURV
    return torch.cat([accel.unsqueeze(1), curv.unsqueeze(1)], dim=1)


def normalize(controls: torch.Tensor) -> torch.Tensor:
    """
    controls[B, C]
    """
    accel = 0.5 * (controls[:, 0] / MAX_ACCEL + 1)
    accel = accel.clamp(0.0, 1.0)
    curv = 0.5 * (controls[:, 1] / MAX_CURV + 1)
    curv = curv.clamp(0.0, 1.0)
    return torch.cat([accel.unsqueeze(1), curv.unsqueeze(1)], dim=1)


def _transform(x, y, cyaw, syaw, rx, ry):
    dx = cyaw * rx - syaw * ry
    dy = syaw * rx + cyaw * ry
    return x + dx, y + dy


def _inv_transform(x, y, cyaw, syaw, x1, y1):
    dx = x1 - x
    dy = y1 - y
    rx = cyaw * dx + syaw * dy
    ry = -syaw * dx + cyaw * dy
    return rx, ry


class _Sinc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(
            x == 0,
            torch.tensor(1.0, device=x.device, dtype=x.dtype),
            x.sin() / x,
        )

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * torch.where(
            x == 0,
            torch.tensor(0.0, device=x.device, dtype=x.dtype),
            (x * x.cos() - x.sin()) / x**2,
        )


class _Cosc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(
            x == 0,
            torch.tensor(0.0, device=x.device, dtype=x.dtype),
            (1 - x.cos()) / x,
        )

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * torch.where(
            x == 0,
            torch.tensor(1.0, device=x.device, dtype=x.dtype),
            (x * torch.sin(x) - (1 - torch.cos(x))) / x**2,
        )


def _advance(x, y, cyaw, syaw, curv, arclen):
    dyaw = curv * arclen
    rx = _Sinc.apply(dyaw) * arclen
    ry = _Cosc.apply(dyaw) * arclen
    return _transform(x, y, cyaw, syaw, rx, ry)
