import pytest


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
