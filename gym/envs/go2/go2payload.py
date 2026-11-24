import torch

from gym.envs.base.domain_randomization import payload_mass_range
from gym.envs.go2.go2trot import Go2Trot
from gym.utils.sampling import masked_update, torch_rand_float


class Go2Payload(Go2Trot):
    """Go2Trot carrying a welded payload resampled every episode."""

    def _init_buffers(self):
        self.base_height_target = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device
        )
        super()._init_buffers()
        # Privileged critic observation: [mass, COM offset from mount_pos].
        # Stays zero when the payload is disabled.
        self.payload_obs = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device
        )
        self._payload_enabled = payload_mass_range(self.cfg) is not None
        # Configs saved before the payload existed have no payload block.
        payload_cfg = getattr(self.cfg.domain_randomization, "payload", None)
        self._payload_mount_pos = torch.tensor(
            payload_cfg.mount_pos if payload_cfg else [0.0, 0.0, 0.0],
            dtype=torch.float,
            device=self.device,
        )

    def _reset_idx(self, reset_mask):
        super()._reset_idx(reset_mask)
        self._update_payload_observation()

    def _resample_commands(self, command_mask):
        super()._resample_commands(command_mask)
        height = torch_rand_float(
            *self.command_ranges["base_height"],
            shape=self.base_height_target.shape,
            device=self.device,
        )
        masked_update(self.base_height_target, height, command_mask)
        masked_update(self.commands, self.commands * self._speed_scale(), command_mask)

    def _speed_scale(self):
        """Limit speed for tall poses. The trained 0.30 m pose uses full speed."""
        h0 = 0.33
        h_max = self.command_ranges["base_height"][1]
        if h_max <= h0:
            return torch.ones_like(self.base_height_target)
        frac = ((self.base_height_target - h0) / (h_max - h0)).clamp(0.0, 1.0)
        return 1.0 - 0.6 * frac

    def _update_gait_reference(self):
        super()._update_gait_reference()
        self.gait_reference += self._height_posture_offset()
        default = self.default_dof_pos.index_select(1, self.actuated_dof_indices)
        lower = self._actuated_dof_pos_limits[:, 0] + 0.05 - default
        upper = self._actuated_dof_pos_limits[:, 1] - 0.05 - default
        self.gait_reference.clamp_(min=lower, max=upper)

    def _height_posture_offset(self):
        """Two-link leg IK used by the reward's height-adjusted gait reference."""
        leg_length, x0, d0, h0 = 0.213, -0.0915, 0.3183, 0.33
        depth = (d0 + self.base_height_target - h0).clamp(0.12, 0.40)
        radius = torch.sqrt(x0**2 + depth.square())
        half_knee = torch.acos((radius / (2 * leg_length)).clamp(max=1.0))
        thigh = torch.atan2(torch.full_like(depth, -x0), depth) + half_knee
        calf = -2.0 * half_knee
        offset = torch.zeros_like(self.gait_reference)
        offset[:, 1::3] = thigh - self.default_dof_pos[:, 1:2]
        offset[:, 2::3] = calf - self.default_dof_pos[:, 2:3]
        return offset

    def _reward_orientation(self):
        error = self.projected_gravity[:, :2].square() / 0.1
        return torch.exp(-error).sum(dim=1)

    def _reward_hip_splay(self):
        hips = [0, 3, 6, 9]
        return (
            -(self.dof_pos[:, hips] - self.default_dof_pos[:, hips]).square().sum(dim=1)
        )

    def _reward_tracking_base_height(self):
        error = (self.base_height - self.base_height_target).flatten()
        error = error / self.scales["base_height"]
        return self._sqrdexp(error, sigma=0.03) - 0.5 * error.abs()

    def _update_payload_observation(self):
        """Payloads only change at reset, so refresh the observation there."""
        if not self._payload_enabled:
            return
        self.payload_obs[:, 0] = self._backend.payload_mass
        self.payload_obs[:, 1:] = self._backend.payload_com - self._payload_mount_pos

    def _foot_contact_strength(self):
        """Map upward foot load smoothly from zero to robot-plus-payload weight."""
        total_mass = self._backend.link_mass.sum(dim=1, keepdim=True)
        total_mass = total_mass + self._backend.payload_mass.unsqueeze(1)
        load = torch.clamp(
            self.contact_forces[:, self.feet_indices, 2] / (9.81 * total_mass),
            min=0.0,
            max=1.0,
        )
        return load.square() * (3.0 - 2.0 * load)
