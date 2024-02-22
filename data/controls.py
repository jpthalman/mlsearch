import numpy as np
import torch

from av2.datasets.motion_forecasting.data_schema import Track

from data.config import Dim
from data import config


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

    ground_truth = torch.tensor(
        [(e.position[0], e.position[1], e.heading) for e in track.object_states],
        dtype=torch.float64,
    )
    ground_truth = _preprocess_track_data(ground_truth)
    out = _downsample_path(ground_truth)

    curv = out["path"][:, 4]
    steer = np.arctan(WHEEL_BASE * curv)
    vel = out["path"][:, 6]
    accel = torch.gradient(vel, spacing=1.0)[0]

    steer_unit = torch.clamp(0.5 * (steer / MAX_STEER + 1), 0.0, 1.0)
    accel_unit = torch.clamp(0.5 * (accel / MAX_ACCEL + 1), 0.0, 1.0)
    out["controls"] = torch.stack([accel_unit, steer_unit], dim=-1)
    return out


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


def debug_visualization(track: Track) -> None:
    import matplotlib.pyplot as plt

    fig, (plt0, plt1) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot the ground truth path
    for a, b in zip(track.object_states[:-1], track.object_states[1:]):
        x = [a.position[0], b.position[0]]
        y = [a.position[1], b.position[1]]
        plt0.plot(x, y, "k.")

    # Plot the ground truth velocity
    t = []
    v = []
    for state in track.object_states:
        t.append(0.1 * state.timestep)
        v.append((state.velocity[0]**2 + state.velocity[1]**2)**0.5)
    plt1.plot(t, v, "k.")

    # Compute the controls
    out = compute_from_track(track)
    print(f"Lat Error: {out['lat_error']}, Long Error: {out['long_error']}")

    # Plot the reintegrated state
    path = _interpolate_path(out["path"])
    vel = _interpolate_velocity(out["path"])
    plt0.plot(path[:, 0], path[:, 1], "bo-")
    plt1.plot(0.1 * np.arange(vel.shape[0]) + 1, vel, "bo-")
    plt.savefig("tmp.png")


def _preprocess_track_data(pos):
    vx = torch.gradient(pos[:, 0], spacing=0.1)[0]
    vy = torch.gradient(pos[:, 1], spacing=0.1)[0]
    vel = list(torch.sqrt(vx**2 + vy**2))

    x = [pos[0, 0]]
    y = [pos[0, 1]]
    for i in range(1, pos.shape[0]):
        dx = pos[i, 0] - x[-1]
        dy = pos[i, 1] - y[-1]
        d = torch.sqrt(dx**2 + dy**2)
        if d < 0.2:
            x.append(x[-1])
            y.append(y[-1])
        else:
            x.append(pos[i, 0])
            y.append(pos[i, 1])

    vx = torch.gradient(torch.stack(x), spacing=0.1)[0]
    vy = torch.gradient(torch.stack(y), spacing=0.1)[0]

    yaw = []
    prepend = 0
    for i in range(len(vx)):
        if vx[i] == 0 and vy[i] == 0:
            if not yaw:
                prepend += 1
            else:
                yaw.append(yaw[-1])
            continue
        yaw.append(torch.arctan2(vy[i], vx[i]))
    while prepend > 0:
        yaw.insert(0, pos[0, 2] if not yaw else yaw[0])
        prepend -= 1

    x.append(x[-1] + 0.1 * vx[-1])
    y.append(y[-1] + 0.1 * vy[-1])
    yaw.append(yaw[-1])
    vel.append(vel[-1])

    data = torch.stack(
        [
            torch.stack(x),
            torch.stack(y),
            torch.stack(yaw),
            torch.stack(vel),
        ],
        dim=-1,
    )
    return data[10:-10, :]


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


def _compute_path(pos, initial_yaw, initial_vel):
    cyaw = [torch.cos(initial_yaw)]
    syaw = [torch.sin(initial_yaw)]
    vel = [initial_vel.clone()]
    curv = []
    arclen = []

    for i in range(pos.shape[0] - 1):
        rx, ry = _inv_transform(
            pos[i, 0],
            pos[i, 1],
            cyaw[-1],
            syaw[-1],
            pos[i + 1, 0],
            pos[i + 1, 1],
        )
        kappa, dl = _SegmentCurvAndArclen.apply(rx, ry)
        dyaw = kappa * dl
        c = torch.cos(dyaw)
        s = torch.sin(dyaw)
        next_cyaw = c * cyaw[-1] - s * syaw[-1]
        next_syaw = s * cyaw[-1] + c * syaw[-1]
        norm = torch.sqrt(next_cyaw**2 + next_syaw**2)
        cyaw.append(next_cyaw / norm)
        syaw.append(next_syaw / norm)
        curv.append(kappa)
        arclen.append(dl)
        vel.append(torch.nn.functional.relu(2 * dl - vel[-1]))

    curv.append(curv[-1])
    arclen.append(0*arclen[-1])
    out = torch.cat(
        [
            pos,
            torch.stack(cyaw).unsqueeze(1),
            torch.stack(syaw).unsqueeze(1),
            torch.stack(curv).unsqueeze(1),
            torch.stack(arclen).unsqueeze(1),
            torch.stack(vel).unsqueeze(1),
        ],
        dim=1,
    )
    return out


def _interpolate_path(path):
    x_interp = [path[0, 0]]
    y_interp = [path[0, 1]]
    for i in range(path.shape[0] - 1):
        x, y, cyaw, syaw, curv, arclen, _ = path[i, :]
        for frac in torch.arange(0.1, 1.1, 0.1):
            xi, yi = _advance(x, y, cyaw, syaw, curv, frac * arclen)
            x_interp.append(xi)
            y_interp.append(yi)
    return torch.cat(
        [
            torch.stack(x_interp).unsqueeze(1),
            torch.stack(y_interp).unsqueeze(1),
        ],
        dim=1,
    )


def _interpolate_velocity(path):
    vel_interp = [path[0, 6]]
    for i in range(path.shape[0] - 1):
        v0 = path[i, 6]
        v1 = path[i + 1, 6]
        accel = (v1 - v0) / 1.0
        for dt in torch.arange(0.1, 1.1, 0.1):
            vel_interp.append(v0 + accel * dt)
    return torch.stack(vel_interp)


def _downsample_path(ground_truth):
    MAX_ITER = 100
    MIN_POS_ERROR = 0.5**2
    MIN_VEL_ERROR = 0.3**2

    initial_yaw = ground_truth[:10, 2].mean()
    initial_yaw.requires_grad = True
    initial_yaw.retain_grad()

    initial_vel_sqr = ground_truth[:10, 3].mean()**2
    initial_vel_sqr.requires_grad = True
    initial_vel_sqr.retain_grad()

    downsampled = ground_truth[::10, :2].clone()
    offset = ground_truth[:1, :2].clone()
    ground_truth[:, :2] -= offset
    downsampled[:, :2] -= offset

    optimizer = torch.optim.Adam([initial_yaw], lr=1e-2)
    for i in range(MAX_ITER):
        path = _compute_path(downsampled, initial_yaw, initial_vel_sqr.sqrt())
        err = (_interpolate_path(path) - ground_truth[:, :2])**2
        if err.max() < MIN_POS_ERROR:
            break
        err.mean().backward()
        optimizer.step()
        optimizer.zero_grad()

    lat_error = err.detach().max().item()
    initial_yaw.requires_grad = False

    optimizer = torch.optim.Adam([initial_vel_sqr], lr=1e-1)
    for i in range(MAX_ITER):
        initial_vel = initial_vel_sqr.relu().sqrt()
        path = _compute_path(downsampled, initial_yaw, initial_vel)
        err = (_interpolate_velocity(path) - ground_truth[:, 3])**2
        if err.max() < MIN_VEL_ERROR:
            break
        err.mean().backward()
        optimizer.step()
        optimizer.zero_grad()

    long_error = err.detach().max().item()
    path = _compute_path(downsampled, initial_yaw, initial_vel_sqr.relu().sqrt())
    path[:, :2] += offset
    return dict(
        lat_error=lat_error,
        long_error=long_error,
        path=path.detach(),
    )
