"""The course-deploy custom implementation must match the go2payload policy.

The file is self-contained (it is run inside the course deploy tree, whose
Go2TrotCfg differs), so these tests pin every value to the training config.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from go2_deploy.deploy_config import DeployConfig
from go2_deploy.policy_bundle import export_bundle, load_bundle
from go2_deploy.utility.import_custom_impl import load_custom_impl
from gym.envs.go2.go2payload_config import Go2PayloadCfg, Go2PayloadRunnerCfg
from learning.modules.actor import Actor

STAFF_COMMAND_LIMITS = {"lin_vel_x": 1.0, "lin_vel_y": 0.5, "yaw_vel": 1.5}


@pytest.fixture(scope="module")
def impl():
    return load_custom_impl("go2payload")


def test_observation_layout_matches_trained_actor(impl):
    cfg = impl
    assert cfg.obs_vector == Go2PayloadRunnerCfg.actor.obs
    standard = {
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "commands": 3,
        "dof_pos_obs": 12,
        "dof_vel": 12,
        "dof_pos_target": 12,
    }
    sizes = standard | cfg.custom_obs_sizes
    assert sum(sizes[name] for name in cfg.obs_vector) == 48


def test_scaling_matches_training_for_every_observation_and_the_action(impl):
    scaling = impl.DeployScaling
    for name in impl.obs_vector + ["dof_pos_target"]:
        trained = getattr(Go2PayloadCfg.scaling, name, 1.0)
        assert getattr(scaling, name) == trained, name


def test_listed_training_settings_match_the_task(impl):
    cfg = DeployConfig(impl)
    control = Go2PayloadCfg.control
    assert control.ctrl_frequency == 50
    assert set(control.stiffness.values()) == {20.0}
    assert set(control.damping.values()) == {0.5}
    assert getattr(control, "action_ema_alpha", 1.0) == 1.0
    assert Go2PayloadRunnerCfg.runner.max_iterations == 2000
    angles = Go2PayloadCfg.init_state.default_joint_angles
    expected = [
        angle
        for leg in ("FL", "FR", "RL", "RR")
        for angle in (
            angles["hip_joint"],
            angles[f"{leg}_thigh_joint"],
            angles["calf_joint"],
        )
    ]
    torch.testing.assert_close(cfg.default_dof_pos, torch.tensor(expected))
    ranges = Go2PayloadCfg.commands.ranges
    limits = STAFF_COMMAND_LIMITS
    assert limits["lin_vel_x"] <= min(-min(ranges.lin_vel_x), max(ranges.lin_vel_x))
    assert limits["lin_vel_y"] <= ranges.lin_vel_y
    assert limits["yaw_vel"] <= ranges.yaw_vel
    low, high = control.gait_freq
    assert low <= impl.get_custom_buffers_dict()["phase_frequency"] <= high


def test_phase_buffers_and_observation_pieces(impl):
    controller = SimpleNamespace(**impl.get_custom_buffers_dict())
    pieces = impl.get_custom_obs_pieces()
    assert set(pieces) == set(impl.custom_obs_sizes)

    # A quarter of a 2 Hz cycle is 0.125 s.
    impl.update_custom_buffers(controller, 10.125)
    assert controller.phase == pytest.approx(math.pi / 2)
    torch.testing.assert_close(
        pieces["phase_obs"](controller, None),
        torch.tensor([1.0, 0.0]),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        pieces["phase_frequency"](controller, None), torch.tensor([2.0])
    )
    for name, piece in pieces.items():
        assert piece(controller, None).shape == (impl.custom_obs_sizes[name],)


def test_stock_bundle_preserves_payload_observations_and_supported_settings(
    tmp_path, impl
):
    cfg = DeployConfig(impl)
    # Staff set these values on their side. The stock bundle format stores
    # gains and joystick limits; staff must also set rate and smoothing there.
    cfg.kp, cfg.kd = 20.0, 0.5
    cfg.command_limits = dict(STAFF_COMMAND_LIMITS)
    actor = Actor(48, 12, [16], normalize_obs=False).eval()
    export_bundle(actor, cfg, tmp_path)
    policy, reloaded = load_bundle(tmp_path)

    assert (reloaded.kp, reloaded.kd) == (20.0, 0.5)
    assert reloaded.command_limits == cfg.command_limits
    assert reloaded.obs_vector == cfg.obs_vector
    assert reloaded.obs_dim() == 48
    assert reloaded.DeployScaling.dof_pos_target == pytest.approx(4 * [0.2, 0.5, 0.5])
    torch.testing.assert_close(reloaded.default_dof_pos, cfg.default_dof_pos)
    obs = torch.linspace(-1, 1, 48)
    torch.testing.assert_close(policy(obs), actor(obs))
