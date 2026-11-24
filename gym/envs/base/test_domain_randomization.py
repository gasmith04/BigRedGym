from types import SimpleNamespace

import pytest
import torch

from gym.envs.base.domain_randomization import (
    DomainRandomizer,
    combine_inertials,
    contact_friction_range,
    payload_box_inertia,
    payload_com_bounds,
    scale_range,
)
from gym.utils.torch_quat import normalize, quat_to_matrix

# Payloads attach to the asset's root link, found from the URDF.
GO2_URDF = "{GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf"


def _cfg(
    contact_friction=(0.5, 1.0),
    episode_ranges=None,
    link_mass=None,
    payload=None,
    seed=7,
):
    dr = SimpleNamespace(
        startup=SimpleNamespace(
            contact_friction_range=contact_friction,
            link_mass_scale_range=link_mass,
        ),
        episode=SimpleNamespace(scale_ranges=episode_ranges or {}),
    )
    if payload is not None:
        dr.payload = payload
    return SimpleNamespace(
        seed=seed, domain_randomization=dr, asset=SimpleNamespace(file=GO2_URDF)
    )


def _payload(mass_range=(0.0, 5.0), **overrides):
    values = dict(
        mass_range=mass_range,
        mount_pos=[0.0, 0.0, 0.1],
        com_offset_range=[[-0.05, 0.05], [-0.02, 0.02], [0.0, 0.0]],
        size=[0.2, 0.1, 0.08],
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class _RecordingBackend:
    # The root link "base" sits second, so root lookup cannot assume index 0.
    body_names = ["FL_foot", "base", "FL_thigh"]
    nominal_mass = torch.tensor([[0.1, 6.0, 1.0]])
    nominal_com = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.02, 0.0, -0.01], [0.0, 0.0, -0.1]]]
    )
    nominal_inertia = torch.tensor([[[1e-4] * 3, [0.02, 0.09, 0.1], [1e-3] * 3]])
    # Base principal axes yawed by 90 degrees about z.
    nominal_quat = torch.tensor(
        [[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.5**0.5, 0.5**0.5], [0.0, 0.0, 0.0, 1.0]]]
    )

    def __init__(self, num_envs):
        self.contact_friction = torch.ones(num_envs)
        self.link_mass = self.nominal_mass.repeat(num_envs, 1)
        self.link_com = self.nominal_com.repeat(num_envs, 1, 1)
        self.link_inertia = self.nominal_inertia.repeat(num_envs, 1, 1)
        self.link_inertia_quat = self.nominal_quat.repeat(num_envs, 1, 1)
        self.inertial_updates = []
        self.updates = []

    def set_contact_friction(self, env_ids, coefficients):
        env_ids = env_ids.clone()
        coefficients = coefficients.clone()
        self.updates.append((env_ids, coefficients))
        self.contact_friction[env_ids] = coefficients

    def set_link_inertials(self, env_ids, masses, coms, inertias, quats):
        values = tuple(v.clone() for v in (env_ids, masses, coms, inertias, quats))
        self.inertial_updates.append(values)
        ids = values[0]
        self.link_mass[ids] = values[1]
        self.link_com[ids] = values[2]
        self.link_inertia[ids] = values[3]
        self.link_inertia_quat[ids] = values[4]


@pytest.mark.parametrize(
    "values",
    [
        (-0.1, 1.0),
        (1.0, 0.5),
    ],
)
def test_contact_friction_range_rejects_invalid_bounds(values):
    with pytest.raises(ValueError, match="0 <= low <= high"):
        contact_friction_range(_cfg(values))


@pytest.mark.parametrize("values", [(0.0, 1.0), (0.9, 0.95), (1.05, 1.1)])
def test_scale_range_must_be_positive_and_contain_nominal(values):
    cfg = _cfg(episode_ranges={"p_gains": values})
    with pytest.raises(ValueError, match="0 < low <= 1 <= high"):
        scale_range(cfg, "p_gains")


def test_disabled_randomization_needs_no_seed_or_backend_support():
    cfg = _cfg(contact_friction=None, seed=-1)
    backend = _RecordingBackend(3)
    randomizer = DomainRandomizer(cfg, backend, "cpu")

    randomizer.bind_link_inertials()
    randomizer.randomize_startup(torch.arange(3))
    randomizer.randomize_episode(torch.ones(3, dtype=torch.bool))
    randomizer.randomize_payload(torch.ones(3, dtype=torch.bool))

    assert backend.updates == []
    assert backend.inertial_updates == []
    assert randomizer.payload_mass is None


def test_missing_domain_randomization_config_fails():
    with pytest.raises(AttributeError, match="domain_randomization"):
        DomainRandomizer(SimpleNamespace(seed=-1), _RecordingBackend(3), "cpu")


def test_flat_domain_randomization_config_fails():
    cfg = SimpleNamespace(
        seed=7,
        domain_randomization=SimpleNamespace(contact_friction_range=(0.5, 1.0)),
    )

    with pytest.raises(AttributeError, match="startup"):
        DomainRandomizer(cfg, _RecordingBackend(3), "cpu")


def test_episode_target_with_no_range_fails_at_construction():
    with pytest.raises(ValueError, match="omit it to disable it"):
        DomainRandomizer(
            _cfg(contact_friction=None, episode_ranges={"p_gains": None}),
            _RecordingBackend(3),
            "cpu",
        )


def test_sampler_is_seeded_and_isolated_from_task_rng():
    first_backend = _RecordingBackend(4)
    second_backend = _RecordingBackend(4)
    first = DomainRandomizer(_cfg(seed=123), first_backend, "cpu")
    second = DomainRandomizer(_cfg(seed=123), second_backend, "cpu")

    first.randomize_startup(torch.tensor([1, 3]))
    torch.rand(20)  # task-level RNG use must not perturb the DR stream
    second.randomize_startup(torch.tensor([1, 3]))

    first_ids, first_values = first_backend.updates[0]
    second_ids, second_values = second_backend.updates[0]
    assert torch.equal(first_ids, torch.tensor([1, 3]))
    assert torch.equal(first_ids, second_ids)
    assert torch.equal(first_values, second_values)
    assert first_values.device == torch.device("cpu")
    assert ((0.5 <= first_values) & (first_values <= 1.0)).all()


def test_episode_reset_does_not_change_startup_friction():
    backend = _RecordingBackend(4)
    randomizer = DomainRandomizer(_cfg(seed=5), backend, "cpu")
    randomizer.randomize_startup(torch.arange(4))
    before = backend.contact_friction.clone()

    randomizer.randomize_episode(torch.tensor([False, True, False, True]))

    assert torch.equal(backend.contact_friction, before)
    assert len(backend.updates) == 1


def test_pd_scales_rebuild_selected_gains_from_nominal():
    backend = _RecordingBackend(4)
    cfg = _cfg(
        contact_friction=None,
        episode_ranges={"p_gains": (0.8, 1.2), "d_gains": (0.7, 1.3)},
        seed=23,
    )
    randomizer = DomainRandomizer(cfg, backend, "cpu")
    p_gains = torch.tensor([[10.0, 20.0]]).repeat(4, 1)
    d_gains = torch.tensor([[1.0, 2.0]]).repeat(4, 1)
    nominal_p = p_gains.clone()
    nominal_d = d_gains.clone()
    randomizer.bind_episode_targets({"p_gains": p_gains, "d_gains": d_gains})

    randomizer.randomize_episode(torch.ones(4, dtype=torch.bool))
    before_p = p_gains.clone()
    before_d = d_gains.clone()
    randomizer.randomize_episode(torch.tensor([False, True, False, True]))

    torch.testing.assert_close(p_gains, nominal_p * randomizer.episode_scale("p_gains"))
    torch.testing.assert_close(d_gains, nominal_d * randomizer.episode_scale("d_gains"))
    torch.testing.assert_close(p_gains[[0, 2]], before_p[[0, 2]])
    torch.testing.assert_close(d_gains[[0, 2]], before_d[[0, 2]])
    assert not torch.equal(p_gains[[1, 3]], before_p[[1, 3]])
    assert not torch.equal(d_gains[[1, 3]], before_d[[1, 3]])


def test_all_false_mask_leaves_episode_targets_and_scales_unchanged():
    backend = _RecordingBackend(4)
    randomizer = DomainRandomizer(
        _cfg(
            contact_friction=None,
            episode_ranges={"p_gains": (0.8, 1.2)},
        ),
        backend,
        "cpu",
    )
    p_gains = torch.tensor([[10.0, 20.0]]).repeat(4, 1)
    randomizer.bind_episode_targets({"p_gains": p_gains})
    before_target = p_gains.clone()
    before_scale = randomizer.episode_scale("p_gains").clone()

    randomizer.randomize_episode(torch.zeros(4, dtype=torch.bool))

    torch.testing.assert_close(p_gains, before_target)
    torch.testing.assert_close(randomizer.episode_scale("p_gains"), before_scale)


def test_pd_axes_have_independent_stable_random_streams():
    backend = _RecordingBackend(2)
    both = DomainRandomizer(
        _cfg(
            contact_friction=None,
            episode_ranges={"p_gains": (0.8, 1.2), "d_gains": (0.7, 1.3)},
        ),
        backend,
        "cpu",
    )
    damping_only = DomainRandomizer(
        _cfg(contact_friction=None, episode_ranges={"d_gains": (0.7, 1.3)}),
        backend,
        "cpu",
    )
    both.bind_episode_targets(
        {"p_gains": torch.ones(2, 3), "d_gains": torch.ones(2, 3)}
    )
    damping_only.bind_episode_targets(
        {"p_gains": torch.ones(2, 3), "d_gains": torch.ones(2, 3)}
    )

    reset_mask = torch.ones(2, dtype=torch.bool)
    both.randomize_episode(reset_mask)
    damping_only.randomize_episode(reset_mask)

    torch.testing.assert_close(
        both.episode_scale("d_gains"), damping_only.episode_scale("d_gains")
    )


def test_episode_randomizer_owns_nominal_only_for_configured_targets():
    backend = _RecordingBackend(2)
    randomizer = DomainRandomizer(
        _cfg(
            contact_friction=None,
            episode_ranges={"p_gains": (0.8, 1.2)},
        ),
        backend,
        "cpu",
    )
    p_gains = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    d_gains = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    nominal_p = p_gains.clone()
    randomizer.bind_episode_targets({"p_gains": p_gains, "d_gains": d_gains})

    assert randomizer.episode_scale("p_gains") is not None
    assert randomizer.episode_scale("d_gains") is None
    p_gains.add_(1000.0)
    randomizer.set_episode_scale(
        "p_gains",
        torch.ones(2, dtype=torch.bool),
        torch.ones_like(p_gains),
    )

    torch.testing.assert_close(p_gains, nominal_p)
    torch.testing.assert_close(d_gains, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_link_mass_scales_each_body_once_at_startup():
    backend = _RecordingBackend(4)
    randomizer = DomainRandomizer(
        _cfg(contact_friction=None, link_mass=(0.8, 1.2), seed=31),
        backend,
        "cpu",
    )
    randomizer.bind_link_inertials()
    randomizer.randomize_startup(torch.arange(4))
    before = backend.link_mass.clone()

    randomizer.randomize_episode(torch.tensor([False, True, False, True]))

    scale = randomizer.link_mass_scale
    torch.testing.assert_close(backend.link_mass, backend.nominal_mass * scale)
    torch.testing.assert_close(
        backend.link_inertia, backend.nominal_inertia * scale.unsqueeze(-1)
    )
    # Scaling keeps COMs and principal axes.
    torch.testing.assert_close(backend.link_com, backend.nominal_com.expand(4, -1, -1))
    torch.testing.assert_close(
        backend.link_inertia_quat, backend.nominal_quat.expand(4, -1, -1)
    )
    torch.testing.assert_close(backend.link_mass, before)
    assert len(backend.inertial_updates) == 1
    assert torch.unique(scale[0]).numel() > 1


def _payload_randomizer(num_envs=4, seed=7, link_mass=None, **payload_overrides):
    backend = _RecordingBackend(num_envs)
    cfg = _cfg(
        contact_friction=None,
        link_mass=link_mass,
        payload=_payload(**payload_overrides),
        seed=seed,
    )
    randomizer = DomainRandomizer(cfg, backend, "cpu")
    randomizer.bind_link_inertials()
    return randomizer, backend


def test_payload_samples_selected_envs_within_bounds_in_one_call():
    randomizer, backend = _payload_randomizer()
    mask = torch.tensor([False, True, False, True])

    randomizer.randomize_payload(mask)

    assert len(backend.inertial_updates) == 1
    assert torch.equal(backend.inertial_updates[0][0], torch.tensor([1, 3]))
    masses = randomizer.payload_mass
    coms = randomizer.payload_com
    assert ((0.0 <= masses[mask]) & (masses[mask] <= 5.0)).all()
    assert not torch.equal(masses[1], masses[3])
    low = torch.tensor([-0.05, -0.02, 0.1])
    high = torch.tensor([0.05, 0.02, 0.1])
    assert ((low - 1e-7 <= coms[mask]) & (coms[mask] <= high + 1e-7)).all()
    # Unselected environments keep no payload at the mount.
    torch.testing.assert_close(masses[~mask], torch.zeros(2))
    torch.testing.assert_close(coms[~mask], torch.tensor([[0.0, 0.0, 0.1]] * 2))


def test_payload_all_false_mask_skips_backend():
    randomizer, backend = _payload_randomizer()

    randomizer.randomize_payload(torch.zeros(4, dtype=torch.bool))

    assert backend.inertial_updates == []


def test_payload_stream_is_seeded_and_independent_of_other_resets():
    first, _ = _payload_randomizer(seed=123)
    second, _ = _payload_randomizer(seed=123)

    first.randomize_payload(torch.tensor([True, False, False, False]))
    torch.rand(20)  # task-level RNG use must not perturb the DR stream
    second.randomize_payload(torch.tensor([True, True, True, True]))

    assert torch.equal(first.payload_mass[0], second.payload_mass[0])
    assert torch.equal(first.payload_com[0], second.payload_com[0])


def test_payload_joins_the_scaled_root_link_only():
    randomizer, backend = _payload_randomizer(link_mass=(0.8, 1.2))
    randomizer.randomize_startup(torch.arange(4))
    randomizer.set_payload(
        torch.tensor([1, 2]),
        torch.tensor([3.0, 0.0]),
        torch.tensor([[0.03, -0.01, 0.12], [0.0, 0.0, 0.1]]),
    )

    scale = randomizer.link_mass_scale
    scaled_mass = backend.nominal_mass * scale
    root = backend.body_names.index("base")
    others = [i for i in range(3) if i != root]
    torch.testing.assert_close(backend.link_mass[:, others], scaled_mass[:, others])
    torch.testing.assert_close(
        backend.link_mass[:, root], scaled_mass[:, root] + torch.tensor([0, 3.0, 0, 0])
    )
    # A massless payload leaves the scaled root link exactly as it was.
    for env_id in (0, 2, 3):
        torch.testing.assert_close(
            backend.link_com[env_id, root], backend.nominal_com[0, root]
        )
        torch.testing.assert_close(
            backend.link_inertia_quat[env_id, root], backend.nominal_quat[0, root]
        )
    expected_com = (
        scaled_mass[1, root] * backend.nominal_com[0, root]
        + 3.0 * torch.tensor([0.03, -0.01, 0.12])
    ) / (scaled_mass[1, root] + 3.0)
    torch.testing.assert_close(backend.link_com[1, root], expected_com)


def test_set_payload_validates_inputs_and_enablement():
    randomizer, backend = _payload_randomizer()
    with pytest.raises(ValueError, match="one payload mass"):
        randomizer.set_payload(torch.tensor([0, 1]), torch.ones(1), torch.zeros(2, 3))
    with pytest.raises(ValueError, match="one payload mass"):
        randomizer.set_payload(torch.tensor([0]), torch.ones(1), torch.zeros(1, 2))
    with pytest.raises(ValueError, match="non-negative"):
        randomizer.set_payload(torch.tensor([0]), -torch.ones(1), torch.zeros(1, 3))
    # Empty updates are no-ops, matching the other DR setters.
    randomizer.set_payload(
        torch.tensor([], dtype=torch.long), torch.zeros(0), torch.zeros(0, 3)
    )
    assert backend.inertial_updates == []

    unbound = DomainRandomizer(_cfg(payload=_payload()), _RecordingBackend(2), "cpu")
    with pytest.raises(RuntimeError, match="bind_link_inertials"):
        unbound.set_payload(torch.tensor([0]), torch.ones(1), torch.zeros(1, 3))
    disabled = DomainRandomizer(_cfg(), _RecordingBackend(2), "cpu")
    with pytest.raises(RuntimeError, match="payload was not enabled"):
        disabled.set_payload(torch.tensor([0]), torch.ones(1), torch.zeros(1, 3))


def _inertia_tensor(inertia, quat):
    rotation = quat_to_matrix(quat.double())
    return rotation @ torch.diag_embed(inertia.double()) @ rotation.mT


def test_combine_inertials_matches_parallel_axis_theorem():
    generator = torch.Generator().manual_seed(0)
    n = 64
    mass_a = 1.0 + 9.0 * torch.rand(n, generator=generator, dtype=torch.float64)
    mass_b = 5.0 * torch.rand(n, generator=generator, dtype=torch.float64)
    com_a = 0.05 * torch.randn(n, 3, generator=generator, dtype=torch.float64)
    com_b = 0.1 * torch.randn(n, 3, generator=generator, dtype=torch.float64)
    inertia_a = 0.01 + 0.1 * torch.rand(n, 3, generator=generator, dtype=torch.float64)
    quat_a = normalize(torch.randn(n, 4, generator=generator, dtype=torch.float64))
    inertia_b = payload_box_inertia(mass_b, [0.2, 0.12, 0.08])

    mass, com, inertia, quat = combine_inertials(
        mass_a, com_a, inertia_a, quat_a, mass_b, com_b, inertia_b
    )

    torch.testing.assert_close(mass, mass_a + mass_b)
    torch.testing.assert_close(
        com, (mass_a[:, None] * com_a + mass_b[:, None] * com_b) / mass[:, None]
    )

    def about(point, body_mass, body_com, tensor):
        d = body_com - point
        eye = torch.eye(3, dtype=torch.float64)
        shift = (d * d).sum(-1)[:, None, None] * eye - d[:, :, None] * d[:, None, :]
        return tensor + body_mass[:, None, None] * shift

    expected = about(com, mass_a, com_a, _inertia_tensor(inertia_a, quat_a)) + about(
        com, mass_b, com_b, torch.diag_embed(inertia_b)
    )
    torch.testing.assert_close(_inertia_tensor(inertia, quat), expected)
    torch.testing.assert_close(quat.norm(dim=-1), torch.ones(n, dtype=torch.float64))
    assert (inertia > 0).all()


def test_combine_inertials_keeps_body_unchanged_without_payload():
    mass_a = torch.tensor([6.9])
    com_a = torch.tensor([[0.021, 0.0, -0.005]])
    inertia_a = torch.tensor([[0.0245, 0.0981, 0.107]])
    quat_a = torch.tensor([[0.0016, 0.0090, -0.0008, 0.99996]])
    result = combine_inertials(
        mass_a,
        com_a,
        inertia_a,
        quat_a,
        torch.zeros(1),
        torch.ones(1, 3),
        torch.zeros(1, 3),
    )
    for actual, expected in zip(result, (mass_a, com_a, inertia_a, quat_a)):
        assert torch.equal(actual, expected)


def test_payload_requires_seed():
    with pytest.raises(ValueError, match="non-negative"):
        _payload_randomizer(seed=-1)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"size": [0.2, 0.0, 0.08]}, "size must be positive"),
        ({"mount_pos": [0.0, 0.1]}, "three entries"),
        ({"com_offset_range": [[0.0, 0.0], [0.0, 0.0]]}, r"\[\[x\], \[y\], \[z\]\]"),
        ({"com_offset_range": [[0.1, -0.1], [0.0, 0.0], [0.0, 0.0]]}, "low <= high"),
    ],
)
def test_payload_geometry_is_validated_at_construction(overrides, match):
    with pytest.raises(ValueError, match=match):
        _payload_randomizer(**overrides)


def test_payload_com_bounds_add_offsets_to_mount():
    cfg = _cfg(payload=_payload())
    low, high = payload_com_bounds(cfg)
    assert low == pytest.approx([-0.05, -0.02, 0.1])
    assert high == pytest.approx([0.05, 0.02, 0.1])
