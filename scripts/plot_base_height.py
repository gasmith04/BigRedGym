import matplotlib.pyplot as plt
import torch

from play import get_play_args, setup


def target_at(t, low, high):
    # 2 s at mid height, then low/high/low/high for 3 s each
    if t < 2.0:
        return 0.5 * (low + high)
    return low if int((t - 2.0) // 3.0) % 2 == 0 else high


if __name__ == "__main__":
    args = get_play_args()
    with torch.no_grad():
        env, runner = setup(args)
        low, high = env.command_ranges["base_height"]
        times, targets, heights = [], [], []
        for i in range(int(14.0 / env.dt)):
            t = i * env.dt
            env.commands[:] = 0.0
            env.base_height_target[:] = target_at(t, low, high)
            runner.set_actions(
                runner.actor_cfg["actions"],
                runner.get_inference_actions(),
                runner.actor_cfg["disable_actions"],
            )
            env.step()
            times.append(t)
            targets.append(env.base_height_target[0, 0].item())
            heights.append(env.base_height[:, 0].clone())
        heights = torch.stack(heights)  # [time, env]
        mean, std = heights.mean(1), heights.std(1)
        plt.plot(times, targets, "k--", label="target")
        plt.plot(times, mean, label="actual (mean)")
        plt.fill_between(times, mean - std, mean + std, alpha=0.3, label="±1 std")
        plt.xlabel("time [s]")
        plt.ylabel("base height [m]")
        plt.legend()
        plt.savefig("base_height_tracking.pdf")
