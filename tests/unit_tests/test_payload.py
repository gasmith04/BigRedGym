"""Welded payload body: model construction and canonical-layout isolation."""

import mujoco
import numpy as np
import pytest
import torch

from gym.envs.base.domain_randomization import payload_box_inertia
from gym.envs.base.mujoco_backend_base import PAYLOAD_BODY_NAME


def _payload_id(model):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PAYLOAD_BODY_NAME)


def test_payload_box_inertia_matches_solid_box():
    inertia = payload_box_inertia(torch.tensor([12.0, 0.0]), [1.0, 2.0, 3.0])
    torch.testing.assert_close(
        inertia, torch.tensor([[13.0, 10.0, 5.0], [0.0, 0.0, 0.0]])
    )


def test_go2trot_payload_body_is_welded_visual_box():
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
    from gym.envs.go2.go2trot_config import Go2TrotCfg

    cfg = Go2TrotCfg()
    payload_cfg = cfg.domain_randomization.payload
    model = MuJocoCPUBackend()._load_model(cfg, discard_visual=False)
    body = _payload_id(model)

    assert body >= 0
    parent = model.body_parentid[body]
    assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent) == "base"
    # No joint: the payload moves rigidly with the base.
    assert model.body_jntnum[body] == 0
    assert model.body_weldid[body] == model.body_weldid[parent]

    mass = payload_cfg.mass_range[0]
    assert model.body_mass[body] == pytest.approx(mass)
    np.testing.assert_allclose(model.body_ipos[body], payload_cfg.mount_pos)
    np.testing.assert_allclose(
        model.body_inertia[body],
        payload_box_inertia(torch.tensor(mass), payload_cfg.size).numpy(),
    )

    geoms = np.flatnonzero(model.geom_bodyid == body)
    assert len(geoms) == 1
    geom = geoms[0]
    assert model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX
    np.testing.assert_allclose(model.geom_size[geom], 0.5 * np.array(payload_cfg.size))
    np.testing.assert_allclose(model.geom_pos[geom], payload_cfg.mount_pos)
    assert model.geom_contype[geom] == 0
    assert model.geom_conaffinity[geom] == 0


def test_payload_stays_out_of_canonical_layout():
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
    from gym.envs.go2.go2_config import Go2Cfg
    from gym.envs.go2.go2trot_config import Go2TrotCfg

    with_payload = MuJocoCPUBackend()
    with_payload.setup(Go2TrotCfg(), num_envs=1, device="cpu", task=None)
    without_payload = MuJocoCPUBackend()
    without_payload.setup(Go2Cfg(), num_envs=1, device="cpu", task=None)
    try:
        assert with_payload._payload_body_id == _payload_id(with_payload._mjm)
        assert without_payload._payload_body_id is None
        assert _payload_id(without_payload._mjm) == -1
        assert with_payload._mjm.nbody == without_payload._mjm.nbody + 1

        assert PAYLOAD_BODY_NAME not in with_payload.body_names
        assert with_payload.body_names == without_payload.body_names
        assert with_payload.link_mass.shape == without_payload.link_mass.shape
        torch.testing.assert_close(
            with_payload.termination_contact_indices,
            without_payload.termination_contact_indices,
        )
        torch.testing.assert_close(
            with_payload.penalised_contact_indices,
            without_payload.penalised_contact_indices,
        )
    finally:
        with_payload.close()
        without_payload.close()


def _go2trot_payload_cfg():
    """Go2Trot with only the payload enabled, so loads are the sole difference."""
    from gym.envs.go2.go2trot_config import Go2TrotCfg

    cfg = Go2TrotCfg()
    cfg.seed = 3
    cfg.domain_randomization.startup.link_mass_scale_range = None
    cfg.domain_randomization.startup.contact_friction_range = None
    return cfg


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


def _assert_payload_weight_reaches_ground(backend):
    ids = torch.tensor([1], device=backend.device)
    backend.set_payload(
        ids,
        torch.tensor([5.0], device=backend.device),
        torch.tensor([[0.03, 0.0, 0.1]], device=backend.device),
    )
    _drop_and_settle(backend)

    supported = backend.contact_forces[..., 2].sum(dim=1).cpu() / 9.81
    robot = backend.link_mass.sum(dim=1).cpu()
    torch.testing.assert_close(
        supported, robot + torch.tensor([0.0, 5.0]), rtol=1e-3, atol=1e-3
    )


def test_mujoco_cpu_set_payload_updates_selected_models_only():
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend

    backend = MuJocoCPUBackend()
    backend.setup(_go2trot_payload_cfg(), num_envs=3, device="cpu", task=None)
    try:
        body = backend._payload_body_id
        root = backend._mjm.body_parentid[body]
        robot_mass = float(backend._models[0].body_subtreemass[root])
        size = backend._payload_size
        nominal_com = backend.payload_com[0].clone()
        root_states = backend.root_states.clone()

        masses = torch.tensor([2.0, 5.0])
        coms = torch.tensor([[0.04, -0.01, 0.09], [-0.05, 0.02, 0.11]])
        backend.set_payload(torch.tensor([1, 2]), masses, coms)

        torch.testing.assert_close(backend.payload_mass, torch.tensor([0.0, 2.0, 5.0]))
        torch.testing.assert_close(backend.payload_com[0], nominal_com)
        torch.testing.assert_close(backend.payload_com[1:], coms)
        # Recomputing model constants must not move the robot.
        torch.testing.assert_close(backend.root_states, root_states)

        expected_inertia = payload_box_inertia(
            torch.cat([torch.zeros(1), masses]), size
        ).numpy()
        expected_com = torch.cat([nominal_com[None], coms]).numpy()
        for env_id, mass in enumerate([0.0, 2.0, 5.0]):
            model = backend._models[env_id]
            assert model.body_mass[body] == pytest.approx(mass)
            np.testing.assert_allclose(
                model.body_ipos[body], expected_com[env_id], rtol=1e-6
            )
            np.testing.assert_allclose(
                model.body_inertia[body], expected_inertia[env_id], rtol=1e-6
            )
            assert model.body_subtreemass[root] == pytest.approx(robot_mass + mass)
    finally:
        backend.close()


def test_mujoco_cpu_payload_weight_reaches_ground():
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend

    backend = MuJocoCPUBackend()
    backend.setup(_go2trot_payload_cfg(), num_envs=2, device="cpu", task=None)
    try:
        _assert_payload_weight_reaches_ground(backend)
    finally:
        backend.close()


def test_set_payload_validates_inputs_and_enablement():
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
    from gym.envs.go2.go2_config import Go2Cfg

    enabled = MuJocoCPUBackend()
    enabled.setup(_go2trot_payload_cfg(), num_envs=2, device="cpu", task=None)
    disabled = MuJocoCPUBackend()
    disabled.setup(Go2Cfg(), num_envs=2, device="cpu", task=None)
    try:
        with pytest.raises(ValueError, match="one payload mass"):
            enabled.set_payload(
                torch.tensor([0, 1]), torch.tensor([1.0]), torch.zeros(2, 3)
            )
        with pytest.raises(ValueError, match="one payload mass"):
            enabled.set_payload(
                torch.tensor([0]), torch.tensor([1.0]), torch.zeros(1, 2)
            )
        with pytest.raises(ValueError, match="non-negative"):
            enabled.set_payload(
                torch.tensor([0]), torch.tensor([-1.0]), torch.zeros(1, 3)
            )

        torch.testing.assert_close(disabled.payload_mass, torch.zeros(2))
        with pytest.raises(RuntimeError, match="payload was not enabled"):
            disabled.set_payload(
                torch.tensor([0]), torch.tensor([1.0]), torch.zeros(1, 3)
            )
        # Empty updates are no-ops, matching the other DR setters.
        disabled.set_payload(
            torch.tensor([], dtype=torch.long), torch.zeros(0), torch.zeros(0, 3)
        )
    finally:
        enabled.close()
        disabled.close()


@pytest.mark.warp
def test_mujoco_warp_payload_is_per_world():
    if not torch.cuda.is_available():
        pytest.fail("Warp tests requested but CUDA is not available", pytrace=False)
    from gym.envs.base.mujoco_warp_backend import MuJocoWarpBackend

    backend = MuJocoWarpBackend()
    backend.setup(_go2trot_payload_cfg(), num_envs=2, device="cuda:0", task=None)
    try:
        for name in ("body_mass", "body_inertia", "body_ipos", "body_subtreemass"):
            assert getattr(backend._m, name).shape[0] == 2, name
        _assert_payload_weight_reaches_ground(backend)
        body = backend._payload_body_id
        torch.testing.assert_close(
            backend._body_mass_native_t[:, body].cpu(), torch.tensor([0.0, 5.0])
        )
    finally:
        backend.close()
