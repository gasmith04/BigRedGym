"""Pure-torch quaternion utilities in scalar-last [x, y, z, w] order."""

import torch


def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Normalize along the last dimension."""
    return x / x.norm(p=2, dim=-1).clamp(min=eps).unsqueeze(-1)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q.  q is [x, y, z, w]."""
    xyz = q[..., :3]
    w = q[..., 3:]
    t = torch.cross(xyz, v, dim=-1) * 2.0
    return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by the inverse of quaternion q.  q is [x, y, z, w]."""
    q_w = q[..., 3:]  # [..., 1]
    q_vec = q[..., :3]  # [..., 3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * (q_vec * v).sum(dim=-1, keepdim=True) * 2.0
    return a - b + c


def quat_from_euler_xyz(
    roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor
) -> torch.Tensor:
    """Euler angles (XYZ extrinsic) to quaternion [x, y, z, w]."""
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return torch.stack([qx, qy, qz, qw], dim=-1)


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Rotation matrix [..., 3, 3] of unit quaternion q = [x, y, z, w]."""
    x, y, z, w = q.unbind(-1)
    rows = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)


def quat_from_matrix(m: torch.Tensor) -> torch.Tensor:
    """Unit quaternion [x, y, z, w] of rotation matrix m [..., 3, 3].

    Each candidate divides by its own largest-magnitude component, so the
    result stays accurate for rotations near 180 degrees.
    """
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    # Four times the magnitude of w, x, y, z respectively.
    four_abs = 2.0 * torch.sqrt(
        torch.stack(
            [
                1 + m00 + m11 + m22,
                1 + m00 - m11 - m22,
                1 - m00 + m11 - m22,
                1 - m00 - m11 + m22,
            ],
            dim=-1,
        ).clamp(min=0.0)
    )
    candidates = torch.stack(
        [
            torch.stack([m21 - m12, m02 - m20, m10 - m01, four_abs[..., 0] ** 2 / 4]),
            torch.stack([four_abs[..., 1] ** 2 / 4, m01 + m10, m02 + m20, m21 - m12]),
            torch.stack([m01 + m10, four_abs[..., 2] ** 2 / 4, m12 + m21, m02 - m20]),
            torch.stack([m02 + m20, m12 + m21, four_abs[..., 3] ** 2 / 4, m10 - m01]),
        ]
    )  # [candidate, xyzw, ...]
    best = four_abs.argmax(dim=-1)
    chosen = torch.gather(
        candidates.movedim((0, 1), (-2, -1)),
        -2,
        best[..., None, None].expand(*best.shape, 1, 4),
    ).squeeze(-2)
    scale = torch.gather(four_abs, -1, best[..., None])
    return normalize(chosen / scale)


def get_axis_params(value, axis_idx, x_val=0.0, dtype=float, n_dims=3):
    """Returns a list with `value` at `axis_idx`, `x_val` elsewhere."""
    params = [x_val] * n_dims
    params[axis_idx] = value
    return params


def to_torch(x, dtype=torch.float, device="cpu", requires_grad=False):
    """Convert array-like to torch tensor."""
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)
