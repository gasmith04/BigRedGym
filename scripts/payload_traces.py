"""Record Go2Trot speed and height traces at fixed payload masses.

Works with checkpoints trained before the payload existed: with
--original_cfg the run's own config is loaded and a payload is
attached to it. Only the actor is used, so critic size does not matter.

Example:
    uv run scripts/payload_traces.py --load_run Oct17_01-22-19_ --original_cfg \
        --masses 0 2.5 5 --output logs/figures/baseline_payload
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from gym.envs.base.domain_randomization import apply_domain_randomization_override
from gym.utils.helpers import get_load_path, set_seed
from gym.utils.task_registry import task_registry

# Ordinal blue ramp, light to dark (reference dataviz palette, steps 250-700).
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]


def attach_payload(env_cfg):
    """Give any Go2Trot config the Go2Payload payload block, with a fixed COM."""
    from gym.envs.go2.go2payload_config import Go2PayloadCfg

    current = Go2PayloadCfg.domain_randomization.payload
    env_cfg.domain_randomization.payload = type(
        "payload",
        (),
        {
            # Any enabled range builds the body; masses are set per env below.
            "mass_range": [0.0, max(current.mass_range)],
            "mount_pos": list(current.mount_pos),
            "com_offset_range": [[0.0, 0.0]] * 3,
            "size": list(current.size),
        },
    )


@torch.no_grad()
def record(env, runner, masses, args):
    replicas = args.replicas
    n = env.num_envs
    ids = torch.arange(n, device=env.device)
    mass = torch.tensor(masses, device=env.device).repeat_interleave(replicas)
    mount = torch.tensor(
        env.cfg.domain_randomization.payload.mount_pos, device=env.device
    )
    env.domain_randomizer.set_payload(ids, mass, mount.expand(n, 3))

    command = torch.tensor([args.vx, 0.0, 0.0], device=env.device)
    steps = round(args.seconds / env.dt)
    step_index = None if args.step_time is None else round(args.step_time / env.dt)
    alive = torch.ones(n, dtype=torch.bool, device=env.device)
    speed, height = [], []
    for i in range(steps):
        if i == step_index:
            env.domain_randomizer.set_payload(
                ids, torch.full_like(mass, args.step_mass), mount.expand(n, 3)
            )
        # Pin commands each step so no sampler can change them.
        env.commands[:] = command
        if hasattr(env, "base_height_target"):
            env.base_height_target[:] = args.height
        env.phase_frequency[:] = args.gait_hz
        runner.set_actions(runner.actor_cfg["actions"], runner.get_inference_actions())
        env.step()
        alive &= ~env.terminated
        # Fallen robots are masked out rather than averaged in.
        speed.append(torch.where(alive, env.base_lin_vel[:, 0], torch.nan).clone())
        height.append(torch.where(alive, env.base_height[:, 0], torch.nan).clone())
    time = np.arange(1, steps + 1) * env.dt
    shape = (steps, len(masses), replicas)
    return {
        "time": time,
        "masses": np.array(masses),
        "speed": torch.stack(speed).cpu().numpy().reshape(shape),
        "height": torch.stack(height).cpu().numpy().reshape(shape),
        "survived": alive.cpu().numpy().reshape(len(masses), replicas),
    }


def summarize(data, args):
    measured = data["time"] > args.warmup
    print(f"command: {args.vx} m/s at {args.height} m, after {args.warmup} s warm-up")
    print("mass_kg  speed_m_s  speed_rmse  height_m  height_error_cm  survived")
    for j, m in enumerate(data["masses"]):
        v = data["speed"][measured, j]
        h = data["height"][measured, j]
        rmse = np.sqrt(np.nanmean((v - args.vx) ** 2))
        print(
            f"{m:7.1f}  {np.nanmean(v):9.3f}  {rmse:10.3f}"
            f"  {np.nanmean(h):8.3f}  {100 * np.nanmean(args.height - h):15.1f}"
            f"  {int(data['survived'][j].sum())}/{args.replicas}"
        )


def plot(data, args, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    masses = data["masses"]
    picks = np.linspace(0, len(RAMP) - 1, len(masses)).round().astype(int)
    colors = [RAMP[i] for i in picks]
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.0), sharex=True)
    panels = (
        (axes[0], "speed", args.vx, "Forward speed (m/s)"),
        (axes[1], "height", args.height, "Body height (m)"),
    )
    for ax, key, target, label in panels:
        ax.axhline(target, color="#8a8a86", lw=1, ls="--", zorder=1)
        for j, m in enumerate(masses):
            mean = np.nanmean(data[key][:, j], axis=1)
            ax.plot(data["time"], mean, color=colors[j], lw=2, label=f"{m:g} kg")
        ax.set_ylabel(label)
        ax.grid(axis="y", color="#e6e5e1", lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.axvspan(0, args.warmup, color="#f0efec", zorder=0)
    if args.step_time is not None:
        for ax in axes:
            ax.axvline(args.step_time, color="#5f5e5a", lw=1, zorder=1)
        axes[0].annotate(
            f"payload → {args.step_mass:g} kg",
            (args.step_time, 1),
            xycoords=("data", "axes fraction"),
            xytext=(4, -10),
            textcoords="offset points",
            fontsize=8,
            color="#5f5e5a",
        )
    # Shared limits so figures from different policies compare directly.
    axes[0].set_ylim(-0.1, max(1.1, 1.1 * args.vx))
    axes[1].set_ylim(0.18, 0.40)
    # The shaded warm-up band is explained in the caption, not labeled here.
    axes[1].set_xlabel("Time (s)")
    if args.title:
        axes[0].set_title(args.title, loc="left", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.legend(
        handles,
        labels,
        title="Payload",
        frameon=False,
        fontsize=8,
        title_fontsize=8,
        loc="upper right",
        ncol=len(masses),
        bbox_to_anchor=(1, 1),
    )
    fig.savefig(path, dpi=300)
    print(f"figure: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=("go2trot", "go2payload"), default="go2payload"
    )
    parser.add_argument("--experiment_name", default=None, help="defaults to --task")
    parser.add_argument("--load_run", required=True)
    parser.add_argument("--checkpoint", type=int, default=-1)
    parser.add_argument("--original_cfg", action="store_true")
    parser.add_argument("--masses", type=float, nargs="+", default=[0.0, 2.5, 5.0])
    parser.add_argument("--vx", type=float, default=1.0)
    parser.add_argument("--height", type=float, default=0.30)
    parser.add_argument("--gait_hz", type=float, default=2.0)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument(
        "--step_time",
        type=float,
        default=None,
        help="optionally change every payload at this time (s)",
    )
    parser.add_argument("--step_mass", type=float, default=5.0)
    parser.add_argument("--title", default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="path prefix; writes <output>.npz and <output>.png",
    )
    args = parser.parse_args()
    args.experiment_name = args.experiment_name or args.task
    import gym.envs  # noqa: F401

    env_cfg, train_cfg = copy.deepcopy(
        task_registry.get_cfgs(
            args.task,
            original_cfg=args.original_cfg,
            experiment_name=args.experiment_name,
            load_run=args.load_run,
        )
    )
    env_cfg.env.num_envs = len(args.masses) * args.replicas
    env_cfg.env.episode_length_s = args.seconds + 1
    env_cfg.commands.resampling_time = args.seconds + 1
    env_cfg.push_robots.toggle = False
    env_cfg.init_state.root_vel_range = [[0.0, 0.0]] * 6
    attach_payload(env_cfg)
    apply_domain_randomization_override(env_cfg, "payload-only")
    env_cfg.seed = args.seed
    train_cfg.runner.device = "cpu"
    train_cfg.runner.resume = False
    train_cfg.logging.enable_local_saving = False
    train_cfg.log_dir = None
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    set_seed(args.seed)

    path = get_load_path(
        name=args.experiment_name, load_run=args.load_run, checkpoint=args.checkpoint
    )
    env = task_registry.make_env(args.task, env_cfg, "cpu", True, "mujoco")
    try:
        runner = task_registry.make_alg_runner(env, train_cfg)
        state = torch.load(path, weights_only=True, map_location="cpu")
        runner.alg.actor.load_state_dict(state["actor_state_dict"])
        runner.switch_to_eval()
        print(f"actor: {path}")
        data = record(env, runner, args.masses, args)
    finally:
        env._backend.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output.with_suffix(".npz"), **data, checkpoint=str(path))
    summarize(data, args)
    plot(data, args, args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
