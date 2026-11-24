from gym.envs.go2.go2trot_config import Go2TrotCfg, Go2TrotRunnerCfg


class Go2PayloadCfg(Go2TrotCfg):
    class control(Go2TrotCfg.control):
        # Policy stays at 50 Hz; PD and contacts run 4 substeps per action so
        # the policy can't exploit coarse 20 ms physics the robot doesn't have.
        desired_sim_frequency = 200

    class commands(Go2TrotCfg.commands):
        class ranges(Go2TrotCfg.commands.ranges):
            lin_vel_x = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.75, 2.5]
            # Fixed height: the actor doesn't observe it, matching deploy.
            base_height = [0.30, 0.30]

    class domain_randomization(Go2TrotCfg.domain_randomization):
        class payload(Go2TrotCfg.domain_randomization.payload):
            mass_range = [0.0, 5.0]
            # Base collision box top is z=0.057; box center sits half its
            # height above that. Re-measure against the hardware mount.
            mount_pos = [0.0, 0.0, 0.097]
            com_offset_range = [[-0.05, 0.05], [-0.02, 0.02], [-0.01, 0.02]]
            size = [0.2, 0.12, 0.08]

    class scaling(Go2TrotCfg.scaling):
        dof_pos_target = 4 * [0.2, 0.5, 0.5]
        base_height_target = 0.3
        # [mass kg, COM offset x, y, z m], matching domain_randomization.payload.
        payload_obs = [5.0, 0.05, 0.02, 0.02]


class Go2PayloadRunnerCfg(Go2TrotRunnerCfg):
    class actor(Go2TrotRunnerCfg.actor):
        # Matches the deploy obs_vector: no height target, no payload sensor.
        obs = [
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel",
            "dof_pos_target",
            "phase_obs",
            "phase_frequency",
        ]

        class noise(Go2TrotRunnerCfg.actor.noise):
            # Std in physical units, keyed by actor obs name; the runner always
            # applies it (add_noise is unused) and divides by obs scaling.
            # Std of Unitree rl_gym's uniform Go2 noise (its scale / sqrt(3)).
            base_ang_vel = 0.115  # rad/s
            projected_gravity = 0.029
            dof_pos_obs = 0.006  # rad
            dof_vel = 0.87  # rad/s

    class critic(Go2TrotRunnerCfg.critic):
        obs = [
            "base_height",
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel",
            "dof_pos_target",
            "phase_obs",
            "phase_frequency",
            "payload_obs",
        ]

        class reward(Go2TrotRunnerCfg.critic.reward):
            class weights(Go2TrotRunnerCfg.critic.reward.weights):
                tracking_lin_vel = 6.0
                orientation = 2.0
                min_base_height = 0.0
                tracking_base_height = 4.0
                hip_splay = 0.5

    class runner(Go2TrotRunnerCfg.runner):
        experiment_name = "go2payload"
        max_iterations = 2000
