"""Custom deploy implementation for the go2payload task (mjh449, gls246, rry9).

Self-contained on purpose: every value is written out instead of imported from
a training config, because the course deploy tree's Go2TrotCfg uses different
action scaling (0.2/0.3/0.3) than this policy was trained with (0.2/0.5/0.5).
tests/unit_tests/test_custom_implementation.py checks these against
Go2PayloadCfg in the training repo.
"""

import math

import torch

from go2_deploy.custom_implementation.custom_config import CustomConfig


class Go2PayloadDeployConfig(CustomConfig):
    task_name = "go2payload"

    # 48 values; matches Go2PayloadRunnerCfg.actor.obs. The actor observes no
    # body-height target and no payload information.
    obs_vector = [
        "base_ang_vel",
        "projected_gravity",
        "commands",
        "dof_pos_obs",
        "dof_vel",
        "dof_pos_target",
        "phase_obs",
        "phase_frequency",
    ]

    # Length of observation vector piece
    custom_obs_sizes = {
        "phase_obs": 2,
        "phase_frequency": 1,
    }

    # Observations are divided by these; actions are multiplied by
    # dof_pos_target. Canonical FL, FR, RL, RR order; hip, thigh, calf per leg.
    class DeployScaling:
        base_ang_vel = 0.3
        projected_gravity = 1.0
        commands = [3, 1, 3]
        dof_pos_obs = 4 * [1.0472, 2.53075, 0.94247]
        dof_vel = 4 * [2.0, 2.0, 4.0]
        dof_pos_target = 4 * [0.2, 0.5, 0.5]
        phase_obs = 1.0
        phase_frequency = 1.0

    # Nominal posture (rad); policy actions are offsets from it.
    default_dof_pos = 4 * [0.0, 0.96, -1.36]

    def get_custom_buffers_dict(self):
        return {"phase_frequency": 2.0, "phase": 0.0}

    def update_custom_buffers(self, main_controller, t):
        main_controller.phase = (
            2 * math.pi * main_controller.phase_frequency * t % (2 * math.pi)
        )

    def get_custom_obs_pieces(self):
        return {
            "phase_obs": self._get_obs_phase_obs,
            "phase_frequency": self._get_obs_phase_frequency,
        }

    @staticmethod
    def _get_obs_phase_obs(main_controller, lowstate_msg):
        phase = main_controller.phase
        return torch.tensor([math.sin(phase), math.cos(phase)])

    @staticmethod
    def _get_obs_phase_frequency(main_controller, lowstate_msg):
        return torch.tensor([main_controller.phase_frequency])
