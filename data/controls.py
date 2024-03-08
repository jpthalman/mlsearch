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


def compute_controls(state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
    """
    state[B, S]
    next_state[B, S]
    """
    rel_gt = relative_positions(state, next_state)
    curv, arclen = _SegmentCurvAndArclen.apply(rel_gt[:, 0], rel_gt[:, 1])
    v0 = state[:, 4]
    v1 = (2 * arclen - v0).relu()
    accel = v1 - v0
    controls = torch.cat([accel.unsqueeze(1), curv.unsqueeze(1)], dim=1)
    return normalize(controls)


def relative_positions(state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
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


def compute_all_controls() -> torch.Tensor:
    B = (Dim.Cd - 1) // 2
    accel_discrete, steer_discrete = torch.meshgrid(
        torch.arange(Dim.Cd),
        torch.arange(Dim.Cd),
        indexing='ij',
    )
    accel = _debucketize(accel_discrete, B)
    steer = _debucketize(steer_discrete, B)
    return torch.stack([accel, steer], dim=-1)


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


class _SegmentCurvAndArclen(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        mask = x.abs() < 0.2
        z = torch.zeros_like(x)
        theta = 2 * torch.atan2(y, x)
        arclen = torch.where(mask, z, x / _Sinc.apply(theta / torch.pi))
        curv = torch.where(mask, z, theta / arclen)
        ctx.save_for_backward(x, y, theta, curv, arclen, mask)
        return curv, arclen

        if x.abs() < 1e-5:
            z = torch.zeros_like(x)
            ctx.save_for_backward(z, z, z, z, z)
            return z, z

        theta = 2 * torch.atan2(y, x)
        arclen = x / _Sinc.apply(theta)
        curv = theta / arclen
        ctx.save_for_backward(x, y, theta, curv, arclen)
        return curv, arclen

    @staticmethod
    def backward(ctx, grad_curv, grad_arclen):
        x, y, theta, curv, arclen, mask = ctx.saved_tensors
        s = torch.sin(theta)
        c = torch.cos(theta)
        D = x**2 + y**2

        dtheta_dx = -2 * y / D
        dtheta_dy = 2 * x / D
        darclen_dx = (s * (dtheta_dx - theta) - theta*x*c*dtheta_dx) / s**2
        darclen_dy = (x*s*dtheta_dy - theta*x*c*dtheta_dy) / s**2
        dcurv_dx = (x*c*dtheta_dx - s) / x**2
        dcurv_dy = dtheta_dy*c/x

        out_grad_curv = torch.where(mask, torch.zeros_like(grad_curv), grad_curv * (dcurv_dx + dcurv_dy) / 2)
        out_grad_arclen = torch.where(mask, torch.zeros_like(grad_arclen), grad_arclen * (darclen_dx + darclen_dy) / 2)
        return out_grad_curv, out_grad_arclen

        x, y, theta, curv, arclen = ctx.saved_tensors
        if x == 0:
            return grad_curv, grad_arclen

        s = torch.sin(theta)
        c = torch.cos(theta)
        D = x**2 + y**2

        dtheta_dx = -2 * y / D
        dtheta_dy = 2 * x / D
        darclen_dx = (s * (dtheta_dx - theta) - theta*x*c*dtheta_dx) / s**2
        darclen_dy = (x*s*dtheta_dy - theta*x*c*dtheta_dy) / s**2
        dcurv_dx = (x*c*dtheta_dx - s) / x**2
        dcurv_dy = dtheta_dy*c/x

        out_grad_curv = grad_curv + (dcurv_dx + dcurv_dy) / 2
        out_grad_arclen = grad_arclen + (darclen_dx + darclen_dy) / 2
        return out_grad_curv, out_grad_arclen


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
