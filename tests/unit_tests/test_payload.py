"""Payload DR: exact folding into the root link, routing, and task wiring.

The payload is a contact-free box rigidly attached to the root link, so the
DomainRandomizer folds its mass, COM, and inertia into that link and every
backend applies the result through ``set_link_inertials``. The reference for
"exact" is MuJoCo itself: a model with the box as a separate welded body.
"""

import mujoco
import numpy as np
import pytest
import torch

from gym import GYM_ROOT_DIR
from gym.envs.base.domain_randomization import DomainRandomizer, payload_box_inertia
from tests.unit_tests.conftest import vsim_guard

ROOT_LINK = "base"


def test_payload_box_inertia_matches_solid_box():
    inertia = payload_box_inertia(torch.tensor([12.0, 0.0]), [1.0, 2.0, 3.0])
    torch.testing.assert_close(
        inertia, torch.tensor([[13.0, 10.0, 5.0], [0.0, 0.0, 0.0]])
    )


def _go2payload_payload_cfg():
    """Go2Payload with only the payload enabled, so loads are the sole difference."""
    from gym.envs.go2.go2payload_config import Go2PayloadCfg

    cfg = Go2PayloadCfg()
    cfg.seed = 3
    cfg.domain_randomization.startup.link_mass_scale_range = None
    cfg.domain_randomization.startup.contact_friction_range = None
    return cfg


def _setup(backend, num_envs, device):
    cfg = _go2payload_payload_cfg()
    backend.setup(cfg, num_envs=num_envs, device=device, task=None)
    randomizer = DomainRandomizer(cfg, backend, device)
    randomizer.bind_link_inertials()
    return randomizer


def _model_inertials(model, body):
    return [
        np.array(getattr(model, name)[body])
        for name in ("body_mass", "body_ipos", "body_inertia", "body_iquat")
    ]


def test_payload_visual_is_massless_and_adds_no_bodies():
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
    from gym.envs.go2.go2_config import Go2Cfg
    from gym.envs.go2.go2payload_config import Go2PayloadCfg

    cfg = Go2PayloadCfg()
    payload_cfg = cfg.domain_randomization.payload
    with_box = MuJocoCPUBackend()._load_model(cfg, discard_visual=False)
    headless = MuJocoCPUBackend()._load_model(cfg, discard_visual=True)
    plain = MuJocoCPUBackend()._load_model(Go2Cfg(), discard_visual=False)

    for model in (with_box, headless):
        assert model.nbody == plain.nbody
        for name in ("body_mass", "body_ipos", "body_inertia", "body_iquat"):
            np.testing.assert_array_equal(getattr(model, name), getattr(plain, name))

    root = mujoco.mj_name2id(with_box, mujoco.mjtObj.mjOBJ_BODY, ROOT_LINK)
    boxes = [
        geom
        for geom in np.flatnonzero(with_box.geom_bodyid == root)
        if with_box.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX
        and with_box.geom_contype[geom] == 0
        and with_box.geom_conaffinity[geom] == 0
    ]
    assert len(boxes) == 1
    np.testing.assert_allclose(
        with_box.geom_size[boxes[0]], 0.5 * np.array(payload_cfg.size)
    )
    np.testing.assert_allclose(with_box.geom_pos[boxes[0]], payload_cfg.mount_pos)
    assert with_box.ngeom == plain.ngeom + 1
    assert headless.ngeom <= plain.ngeom


def _welded_payload_model(cfg, mass, com):
    """Reference model: the payload as a separate body welded to the root link."""
    from gym.envs.base.mujoco_backend_base import MuJocoBackendBase

    spec = MuJocoBackendBase._load_urdf_spec(
        cfg.asset.file.format(GYM_ROOT_DIR=GYM_ROOT_DIR)
    )
    spec.compiler.balanceinertia = True
    spec.compiler.fusestatic = False
    root = spec.worldbody.first_body()
    assert root.name == ROOT_LINK
    root.add_freejoint()
    body = root.add_body()
    body.explicitinertial = True
    body.mass = mass
    body.ipos = com
    size = cfg.domain_randomization.payload.size
    body.inertia = payload_box_inertia(torch.tensor(mass), size).tolist()
    return spec.compile()


def _dynamics(model, qpos, qvel):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    mass_matrix = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, mass_matrix)
    return mass_matrix, data.qfrc_bias.copy()


def test_folded_payload_matches_welded_body_dynamics():
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend

    backend = MuJocoCPUBackend()
    randomizer = _setup(backend, num_envs=2, device="cpu")
    try:
        payloads = [(5.0, [0.03, -0.01, 0.11]), (2.0, [-0.05, 0.02, 0.09])]
        randomizer.set_payload(
            torch.arange(2),
            torch.tensor([mass for mass, _ in payloads]),
            torch.tensor([com for _, com in payloads]),
        )
        rng = np.random.default_rng(0)
        for env_id, (mass, com) in enumerate(payloads):
            folded = backend._models[env_id]
            welded = _welded_payload_model(_go2payload_payload_cfg(), mass, com)
            assert folded.nv == welded.nv
            welded.dof_armature[:] = folded.dof_armature
            for _ in range(3):
                qpos = folded.qpos0.copy()
                quat = rng.normal(size=4)
                qpos[3:7] = quat / np.linalg.norm(quat)
                qpos[7:] += rng.uniform(-0.3, 0.3, size=folded.nq - 7)
                qvel = rng.normal(size=folded.nv)
                folded_m, folded_bias = _dynamics(folded, qpos, qvel)
                welded_m, welded_bias = _dynamics(welded, qpos, qvel)
                # Mass properties pass through float32 tensors.
                np.testing.assert_allclose(folded_m, welded_m, rtol=1e-5, atol=1e-6)
                np.testing.assert_allclose(
                    folded_bias, welded_bias, rtol=1e-5, atol=1e-5
                )
    finally:
        backend.close()


def test_mujoco_cpu_payload_changes_only_selected_models():
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend

    backend = MuJocoCPUBackend()
    randomizer = _setup(backend, num_envs=3, device="cpu")
    try:
        root = backend._canonical_to_native_body_np[backend.body_names.index(ROOT_LINK)]
        nominal = [model.body_mass.copy() for model in backend._models]
        nominal_inertials = _model_inertials(backend._models[0], slice(None))
        robot_mass = float(backend._models[0].body_subtreemass[root])
        root_states = backend.root_states.clone()

        masses = torch.tensor([2.0, 5.0])
        coms = torch.tensor([[0.04, -0.01, 0.09], [-0.05, 0.02, 0.11]])
        randomizer.set_payload(torch.tensor([1, 2]), masses, coms)

        for actual, expected in zip(
            _model_inertials(backend._models[0], slice(None)), nominal_inertials
        ):
            np.testing.assert_array_equal(actual, expected)
        for env_id, mass in ((1, 2.0), (2, 5.0)):
            model = backend._models[env_id]
            others = np.arange(model.nbody) != root
            # Updated environments round-trip every link through float32.
            np.testing.assert_allclose(
                model.body_mass[others], nominal[env_id][others], rtol=1e-6
            )
            assert model.body_mass[root] == pytest.approx(nominal[env_id][root] + mass)
            assert model.body_subtreemass[root] == pytest.approx(robot_mass + mass)
        torch.testing.assert_close(
            backend.link_mass.sum(dim=1),
            robot_mass + torch.tensor([0.0, 2.0, 5.0]),
        )
        torch.testing.assert_close(randomizer.payload_mass, torch.tensor([0, 2.0, 5.0]))
        # Recomputing model constants must not move the robot.
        torch.testing.assert_close(backend.root_states, root_states)
    finally:
        backend.close()


def _drop_and_settle(backend, steps=600):
    """Drop the unpowered robot from rest and let it settle on the ground."""
    num_envs = backend.root_states.shape[0]
    backend.root_states[:, :3] = torch.tensor([0.0, 0.0, 0.35])
    backend.root_states[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    backend.root_states[:, 7:] = 0.0
    backend.dof_pos.zero_()
    backend.dof_vel.zero_()
    backend.reset_state(torch.ones(num_envs, dtype=torch.bool, device=backend.device))
    zeros = torch.zeros(num_envs, backend.num_dof, device=backend.device)
    for _ in range(steps):
        backend.step(zeros)


def _assert_payload_weight_reaches_ground(backend, randomizer):
    robot = backend.link_mass.sum(dim=1).cpu()
    randomizer.set_payload(
        torch.tensor([1], device=backend.device),
        torch.tensor([5.0], device=backend.device),
        torch.tensor([[0.03, 0.0, 0.1]], device=backend.device),
    )
    expected = robot + torch.tensor([0.0, 5.0])
    torch.testing.assert_close(backend.link_mass.sum(dim=1).cpu(), expected)

    _drop_and_settle(backend)

    supported = backend.contact_forces[..., 2].sum(dim=1).cpu() / 9.81
    torch.testing.assert_close(supported, expected, rtol=1e-3, atol=1e-3)


def test_mujoco_cpu_payload_weight_reaches_ground():
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend

    backend = MuJocoCPUBackend()
    randomizer = _setup(backend, num_envs=2, device="cpu")
    try:
        _assert_payload_weight_reaches_ground(backend, randomizer)
    finally:
        backend.close()


@pytest.mark.warp
def test_mujoco_warp_payload_is_per_world():
    if not torch.cuda.is_available():
        pytest.fail("Warp tests requested but CUDA is not available", pytrace=False)
    from gym.envs.base.mujoco_warp_backend import MuJocoWarpBackend

    backend = MuJocoWarpBackend()
    randomizer = _setup(backend, num_envs=2, device="cuda:0")
    try:
        for name in (
            "body_mass",
            "body_ipos",
            "body_inertia",
            "body_iquat",
            "body_subtreemass",
        ):
            assert getattr(backend._m, name).shape[0] == 2, name
        _assert_payload_weight_reaches_ground(backend, randomizer)
        root = backend._canonical_to_native_body_np[backend.body_names.index(ROOT_LINK)]
        world_mass = backend._body_mass_native_t[:, root].cpu()
        assert world_mass[1] == pytest.approx(float(world_mass[0]) + 5.0)
    finally:
        backend.close()


@pytest.mark.vsim
def test_vsim_applies_folded_payload_per_environment():
    vsim_guard()
    from gym.envs.base.vsim_backend import VSimBackend

    backend = VSimBackend()
    randomizer = _setup(backend, num_envs=2, device="cuda:0")
    try:
        # Mass properties need one environment per set.
        assert backend._grp.get_num_environments() == [1, 1]
        _assert_payload_weight_reaches_ground(backend, randomizer)

        # The engine holds exactly the canonical values the backend reports.
        bodies = backend._canonical_to_native_body
        mass, inertia, body_to_model = (
            value[:, bodies] for value in backend._read_link_inertials()
        )
        torch.testing.assert_close(mass, backend.link_mass)
        torch.testing.assert_close(inertia, backend.link_inertia)
        torch.testing.assert_close(body_to_model[..., 0:4], backend.link_inertia_quat)
        torch.testing.assert_close(body_to_model[..., 4:7], backend.link_com)
        root = backend.body_names.index(ROOT_LINK)
        assert not torch.equal(backend.link_com[0, root], backend.link_com[1, root])
    finally:
        backend.close()


def _build_go2payload_task(num_envs=4):
    from gym.envs.go2.go2payload import Go2Payload
    from gym.envs.go2.go2payload_config import Go2PayloadCfg, Go2PayloadRunnerCfg
    from gym.utils.task_registry import select_backend, task_registry

    cfg = Go2PayloadCfg()
    runner_cfg = Go2PayloadRunnerCfg()
    cfg.env.num_envs = num_envs
    cfg.seed = 5
    cfg.push_robots.toggle = False
    task_registry.convert_frequencies_to_params(cfg, runner_cfg)
    backend = select_backend(cfg, "cpu", "mujoco")
    return Go2Payload(cfg, "cpu", True, backend)


def test_go2payload_task_resamples_payload_only_for_reset_envs():
    from gym.envs.base.domain_randomization import payload_com_bounds

    env = _build_go2payload_task()
    try:
        randomizer = env.domain_randomizer
        payload_cfg = env.cfg.domain_randomization.payload
        low, high = (torch.tensor(v) for v in payload_com_bounds(env.cfg))

        # The initial reset samples every environment.
        initial_mass = randomizer.payload_mass.clone()
        initial_com = randomizer.payload_com.clone()
        assert (
            (initial_mass >= payload_cfg.mass_range[0])
            & (initial_mass <= payload_cfg.mass_range[1])
        ).all()
        assert ((initial_com >= low - 1e-6) & (initial_com <= high + 1e-6)).all()
        assert initial_mass.unique().numel() == env.num_envs
        initial_link_mass = env._backend.link_mass.clone()

        mask = torch.tensor([False, True, False, True])
        env._reset_idx(mask)

        changed = randomizer.payload_mass != initial_mass
        assert torch.equal(changed, mask)
        torch.testing.assert_close(randomizer.payload_com[~mask], initial_com[~mask])
        torch.testing.assert_close(
            env._backend.link_mass[~mask], initial_link_mass[~mask]
        )
        # Link-mass scales are startup-only; only the payload term moves.
        torch.testing.assert_close(
            env._backend.link_mass.sum(dim=1) - initial_link_mass.sum(dim=1),
            randomizer.payload_mass - initial_mass,
        )

        env.step()
        assert torch.isfinite(env.root_states).all()
    finally:
        env._backend.close()


def test_go2payload_payload_observation_is_critic_only_and_tracks_resets():
    from gym.envs.go2.go2payload_config import Go2PayloadRunnerCfg

    runner_cfg = Go2PayloadRunnerCfg()
    assert "payload_obs" in runner_cfg.critic.obs
    assert "payload_obs" not in runner_cfg.actor.obs

    env = _build_go2payload_task()
    try:
        randomizer = env.domain_randomizer
        mount = torch.tensor(env.cfg.domain_randomization.payload.mount_pos)

        def assert_matches_randomizer():
            torch.testing.assert_close(env.payload_obs[:, 0], randomizer.payload_mass)
            torch.testing.assert_close(
                env.payload_obs[:, 1:], randomizer.payload_com - mount
            )

        assert_matches_randomizer()
        env._reset_idx(torch.tensor([True, False, True, False]))
        assert_matches_randomizer()

        # Scaling maps the configured ranges into [-1, 1].
        scaled = env.get_state("payload_obs")
        assert ((scaled >= -1.0 - 1e-6) & (scaled <= 1.0 + 1e-6)).all()
        assert (scaled[:, 0] >= 0.0).all()
        critic = env.get_states(runner_cfg.critic.obs)
        torch.testing.assert_close(critic[:, -4:], scaled)
    finally:
        env._backend.close()


def test_go2payload_payload_observation_is_zero_when_disabled():
    from gym.envs.base.domain_randomization import apply_domain_randomization_override
    from gym.envs.go2.go2payload import Go2Payload
    from gym.envs.go2.go2payload_config import Go2PayloadCfg, Go2PayloadRunnerCfg
    from gym.utils.task_registry import select_backend, task_registry

    cfg = Go2PayloadCfg()
    runner_cfg = Go2PayloadRunnerCfg()
    cfg.env.num_envs = 2
    cfg.seed = 5
    apply_domain_randomization_override(cfg, "off")
    task_registry.convert_frequencies_to_params(cfg, runner_cfg)
    env = Go2Payload(cfg, "cpu", True, select_backend(cfg, "cpu", "mujoco"))
    try:
        assert env.domain_randomizer.payload_mass is None
        assert env._backend._models is None  # no per-environment mass properties
        torch.testing.assert_close(env.payload_obs, torch.zeros(2, 4))
        env._reset_idx(torch.ones(2, dtype=torch.bool))
        torch.testing.assert_close(env.payload_obs, torch.zeros(2, 4))
    finally:
        env._backend.close()
