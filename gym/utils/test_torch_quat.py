import math

import torch

from gym.utils.torch_quat import (
    normalize,
    quat_apply,
    quat_from_matrix,
    quat_to_matrix,
)


def _same_rotation(a, b):
    return torch.minimum((a - b).abs().amax(-1), (a + b).abs().amax(-1))


def test_matrix_round_trip_covers_half_turns():
    generator = torch.Generator().manual_seed(0)
    random = normalize(torch.randn(256, 4, generator=generator, dtype=torch.float64))
    s = math.sqrt(0.5)
    # Half turns about each axis and a diagonal, where w is zero.
    half_turns = torch.tensor(
        [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [s, s, 0, 0]],
        dtype=torch.float64,
    )
    quats = torch.cat([random, half_turns, torch.tensor([[0, 0, 0, 1.0]]).double()])

    matrices = quat_to_matrix(quats)

    assert _same_rotation(quat_from_matrix(matrices), quats).max() < 1e-12
    eye = torch.eye(3, dtype=torch.float64)
    torch.testing.assert_close(matrices @ matrices.mT, eye.expand_as(matrices))


def test_matrix_rotates_like_quat_apply():
    generator = torch.Generator().manual_seed(1)
    quats = normalize(torch.randn(32, 4, generator=generator))
    vectors = torch.randn(32, 3, generator=generator)

    rotated = (quat_to_matrix(quats) @ vectors[..., None]).squeeze(-1)

    torch.testing.assert_close(rotated, quat_apply(quats, vectors))
