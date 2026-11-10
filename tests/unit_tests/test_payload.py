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
