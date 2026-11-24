from types import SimpleNamespace

import pytest
import torch

from gym.envs.go2.go2payload import Go2Payload


def test_contact_strength_counts_payload_as_body_weight():
    task = Go2Payload.__new__(Go2Payload)
    task.feet_indices = torch.arange(4)
    task.contact_forces = torch.zeros(2, 4, 3)
    task.contact_forces[:, :, 2] = 98.1
    # Same 20 kg total load either way, whether robot or payload.
    task._backend = SimpleNamespace(
        link_mass=torch.tensor([[10.0, 10.0], [5.0, 10.0]]),
        payload_mass=torch.tensor([0.0, 5.0]),
    )

    strength = task._foot_contact_strength()

    torch.testing.assert_close(strength[0], strength[1])
    torch.testing.assert_close(strength[1], torch.full((4,), 0.5))


def test_lin_vel_tracking_charges_the_same_shortfall_at_any_speed():
    task = Go2Payload.__new__(Go2Payload)
    task.cfg = SimpleNamespace(reward_settings=SimpleNamespace(tracking_sigma=0.25))
    task.commands = torch.tensor([[0.5, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    task.base_lin_vel = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [1.0, 0.0, 0.0]])

    reward = task._reward_tracking_lin_vel()

    torch.testing.assert_close(reward[0], reward[1])
    assert reward[1] == pytest.approx(torch.exp(torch.tensor(-1.0)).item() - 0.1)
    assert reward[2] == pytest.approx(1.0)


def _stride_task(forward_speed, frequency, limit=0.6):
    task = Go2Payload.__new__(Go2Payload)
    task.cfg = SimpleNamespace(control=SimpleNamespace(stride_amplitude_max=limit))
    task.commands = torch.tensor([[forward_speed, 0.0, 0.0]])
    task.phase_frequency = torch.tensor([[frequency]])
    task.base_height_target = torch.tensor([[0.30]])
    task.gait_reference = torch.zeros(1, 12)
    task._gait_phase_offsets = torch.tensor([0.0, torch.pi, torch.pi, 0.0])
    return task


def _foot_x(thigh, calf, link=0.213):
    """Unitree leg forward kinematics: foot x relative to the thigh joint."""
    return -link * torch.sin(thigh) - link * torch.sin(thigh + calf)


def test_stride_moves_stance_foot_back_by_the_distance_the_body_travels():
    speed, frequency = 0.5, 2.0
    task = _stride_task(speed, frequency)
    task.default_dof_pos = torch.tensor([[0.0, 0.96, -1.36] * 4])
    # The reference stands at the commanded 0.30 m pose, not the default.
    pose = task.default_dof_pos + task._height_posture_offset()
    thigh = {}
    for name, phase in (("touchdown", 0.0), ("liftoff", torch.pi)):
        task.phase = torch.tensor([[phase]])
        # Front-left leg has no phase offset; calf is unchanged by the sweep.
        thigh[name] = pose[0, 1] + task._stride_offset()[0, 1]
    calf = pose[0, 2]
    travel = _foot_x(thigh["touchdown"], calf) - _foot_x(thigh["liftoff"], calf)

    # Half a cycle of stance at 0.5 m/s and 2 Hz.
    assert travel.item() == pytest.approx(speed / (2 * frequency), rel=0.05)


def test_stride_is_zero_when_stopped_reverses_and_saturates():
    task = _stride_task(0.0, 2.0)
    task.phase = torch.zeros(1, 1)
    assert not task._stride_offset().any()

    forward, reverse = _stride_task(1.0, 2.0), _stride_task(-1.0, 2.0)
    forward.phase = reverse.phase = torch.zeros(1, 1)
    torch.testing.assert_close(forward._stride_offset(), -reverse._stride_offset())

    fast = _stride_task(5.0, 1.0)
    fast.phase = torch.zeros(1, 1)
    assert fast._stride_offset()[0, 1::3].abs().max() == pytest.approx(0.6)


def test_configs_saved_before_the_stride_keep_stepping_in_place():
    task = _stride_task(1.0, 2.0)
    task.cfg.control = SimpleNamespace()
    task.phase = torch.zeros(1, 1)
    assert not task._stride_offset().any()


def test_policy_runs_at_50hz_over_200hz_physics():
    from gym.envs.go2.go2payload_config import Go2PayloadCfg, Go2PayloadRunnerCfg
    from gym.utils.task_registry import task_registry

    cfg = Go2PayloadCfg()
    task_registry.convert_frequencies_to_params(cfg, Go2PayloadRunnerCfg())

    assert cfg.control.decimation == 4
    assert cfg.control.ctrl_dt == pytest.approx(0.02)
    assert cfg.sim_dt == pytest.approx(0.005)


def test_every_sensed_actor_observation_is_noisy():
    from gym.envs.go2.go2payload_config import Go2PayloadRunnerCfg

    actor = Go2PayloadRunnerCfg.actor
    # Commands, phase and the previous action are exact on hardware too.
    sensed = set(actor.obs) - {
        "commands",
        "dof_pos_target",
        "phase_obs",
        "phase_frequency",
    }
    for name in sensed:
        assert getattr(actor.noise, name, 0.0) > 0.0, name
