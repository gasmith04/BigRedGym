"""Go2Payload trajectories must survive changes to how the payload is applied.

``data/go2payload_trajectory_reference.npz`` was recorded with the original
MuJoCo payload: a contact-free box welded to the base as a separate body.
Folding such a body into the base link's mass, COM, and inertia is physically
exact, so a re-implementation must reproduce each MuJoCo backend's reference
to within numerical noise. Tolerances sit well above float32 rounding of the
mass properties and Warp's run-to-run nondeterminism, and well below the effect
of a 1 mm payload COM error.

The scenario drops the robot from 0.8 m in a crouched, tilted posture while PD
drives the legs to a standing posture: about 0.3 s of flight, then landing. The
task's configured DR (contact friction, link-mass scale, payload) stays on, so
a payload on a scaled base is covered. DR draws come from per-device torch
generators, so CPU and CUDA runs carry different payloads.

VSim is compared with the Warp reference, which shares its CUDA DR streams,
during flight only. After touchdown the engines' contact responses diverge by
more than the payload's own effect, even without a payload. Flight agreement
catches gross errors (missing payload, mass-only lumping) but not inertia
details; exact VSim mass-property readback must cover those. Root linear
velocity is excluded for VSim because it reports velocity at the link COM,
not the link origin.

Regenerate only for an intended physics change, never to make this pass::

    uv run --frozen python -m tests.unit_tests.test_payload_trajectory_regression \
        --record cpu warp
"""

import argparse
import math
import os
import subprocess

import numpy as np
import pytest
import torch

REFERENCE_PATH = os.path.join(
    os.path.dirname(__file__), "data", "go2payload_trajectory_reference.npz"
)
BACKENDS = {
    "cpu": ("cpu", "mujoco"),
    "warp": ("cuda:0", "mujoco"),
    "vsim": ("cuda:0", "vsim"),
}

NUM_ENVS = 4
SEED = 11
NUM_STEPS = 180  # 0.9 s at the task's 200 Hz physics rate
RECORD_EVERY = 8
KP, KD = 25.0, 0.6
TARGET_POSTURE = {"hip": 0.0, "thigh": 0.66, "calf": -1.36}

# Tilt and crouch load the payload COM and inertia during flight and landing.
# Angular velocity starts at zero, so the reset's root linear velocity means
# the same thing to every engine.
INITIAL_HEIGHT = 0.8
INITIAL_ROLL_PITCH = [(0.15, 0.0), (-0.10, 0.10), (0.05, -0.12), (0.0, 0.08)]
INITIAL_LIN_VEL = [
    (0.4, 0.0, 0.0),
    (0.0, 0.3, 0.0),
    (-0.2, 0.2, 0.0),
    (0.3, -0.3, 0.0),
]
INITIAL_POSTURE_OFFSET = {"hip": 0.1, "thigh": 0.3, "calf": -0.4}

CHANNELS = (
    "root_pos",
    "root_quat",
    "root_lin_vel",
    "root_ang_vel",
    "dof_pos",
    "dof_vel",
    "foot_force_z",
)
# Max absolute error per channel; quaternions compare sign-aligned components.
MUJOCO_TOLERANCES = {
    "root_pos": 1e-5,
    "root_quat": 2e-5,
    "root_lin_vel": 3e-4,
    "root_ang_vel": 3e-3,
    "dof_pos": 1e-4,
    "dof_vel": 1e-2,
    "foot_force_z": 0.5,
}
# About three times the VSim-vs-Warp flight gap measured without a payload.
VSIM_FLIGHT_TOLERANCES = {
    "root_pos": 5e-4,
    "root_quat": 1.5e-3,
    "root_ang_vel": 5e-2,
    "dof_pos": 1e-2,
}


def _quat_from_roll_pitch(roll, pitch):
    """Scalar-last quaternion: roll about x, then pitch about y."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    return [sr * cp, cr * sp, -sr * sp, cr * cp]


def _build_task(device, backend_name, payload=True):
    from gym.envs.go2.go2payload import Go2Payload
    from gym.envs.go2.go2payload_config import Go2PayloadCfg, Go2PayloadRunnerCfg
    from gym.utils.helpers import set_seed
    from gym.utils.task_registry import select_backend, task_registry

    cfg = Go2PayloadCfg()
    cfg.seed = SEED
    cfg.env.num_envs = NUM_ENVS
    cfg.push_robots.toggle = False
    if not payload:
        cfg.domain_randomization.payload.mass_range = None
    task_registry.convert_frequencies_to_params(cfg, Go2PayloadRunnerCfg())
    set_seed(cfg.seed)
    backend = select_backend(cfg, device, backend_name)
    return Go2Payload(cfg, device, True, backend)


def _dof_values(env, values_by_joint):
    return torch.tensor(
        [
            next(v for key, v in values_by_joint.items() if f"_{key}_" in name)
            for name in env.dof_names
        ],
        device=env.device,
    )


def _write_initial_state(env):
    backend = env._backend
    device = env.device
    root = backend.root_states
    root[:, :3] = env.env_origins + torch.tensor(
        [0.0, 0.0, INITIAL_HEIGHT], device=device
    )
    root[:, 3:7] = torch.tensor(
        [_quat_from_roll_pitch(*rp) for rp in INITIAL_ROLL_PITCH], device=device
    )
    root[:, 7:10] = torch.tensor(INITIAL_LIN_VEL, device=device)
    root[:, 10:13] = 0.0
    target = _dof_values(env, TARGET_POSTURE)
    # Mirror the hip offset on the right legs so the crouch is asymmetric.
    hip_sign = torch.tensor(
        [
            -1.0 if name.startswith(("FR", "RR")) and "_hip_" in name else 1.0
            for name in env.dof_names
        ],
        device=device,
    )
    backend.dof_pos[:] = target + hip_sign * _dof_values(env, INITIAL_POSTURE_OFFSET)
    backend.dof_vel.zero_()
    backend.reset_state(torch.ones(NUM_ENVS, dtype=torch.bool, device=device))
    return target


def _sample(env):
    backend = env._backend
    root = backend.root_states
    return {
        "root_pos": root[:, 0:3] - env.env_origins,
        "root_quat": root[:, 3:7],
        "root_lin_vel": root[:, 7:10],
        "root_ang_vel": root[:, 10:13],
        "dof_pos": backend.dof_pos,
        "dof_vel": backend.dof_vel,
        "foot_force_z": backend.contact_forces[:, env.feet_indices, 2],
    }


def simulate(name, payload=True):
    """Run the scenario on one backend and return the sampled trajectory."""
    device, backend_name = BACKENDS[name]
    env = _build_task(device, backend_name, payload=payload)
    try:
        backend = env._backend
        target = _write_initial_state(env)
        limits = env.torque_limits
        samples = {channel: [] for channel in CHANNELS}

        def record():
            for channel, value in _sample(env).items():
                samples[channel].append(value.detach().double().cpu().numpy().copy())

        record()
        for step in range(1, NUM_STEPS + 1):
            torques = KP * (target - backend.dof_pos) - KD * backend.dof_vel
            backend.step(torch.clamp(torques, -limits, limits))
            if step % RECORD_EVERY == 0:
                record()
        result = {channel: np.stack(values) for channel, values in samples.items()}
        # Task-level payload readout: [mass, COM offset from mount_pos].
        result["payload_obs"] = env.payload_obs.detach().double().cpu().numpy()
        result["time"] = np.arange(0, NUM_STEPS + 1, RECORD_EVERY) * env.cfg.sim_dt
        return result
    finally:
        env._backend.close()


def max_errors(actual, reference, channels, samples=slice(None)):
    """Return {channel: (max abs error, sample index, env index)}."""
    errors = {}
    for channel in channels:
        a, r = actual[channel][samples], reference[channel][samples]
        diff = np.abs(a - r)
        if channel == "root_quat":
            diff = np.minimum(diff, np.abs(a + r))
        per_sample_env = diff.reshape(diff.shape[0], diff.shape[1], -1).max(-1)
        index = np.unravel_index(per_sample_env.argmax(), per_sample_env.shape)
        errors[channel] = (float(per_sample_env[index]), *map(int, index))
    return errors


def assert_matches(actual, reference, tolerances, samples=slice(None)):
    errors = max_errors(actual, reference, tolerances, samples)
    failures = [
        f"{channel}: {error:.2e} > {tolerances[channel]:.0e} "
        f"(t={reference['time'][samples][sample]:.2f} s, env {env})"
        for channel, (error, sample, env) in errors.items()
        if not error <= tolerances[channel]
    ]
    assert not failures, "trajectory deviates from reference:\n" + "\n".join(failures)


def assert_same_payloads(actual, reference):
    np.testing.assert_allclose(
        actual["payload_obs"],
        reference["payload_obs"],
        atol=1e-6,
        err_msg="payload draws changed; the trajectories are not comparable",
    )


def load_reference(name):
    if not os.path.exists(REFERENCE_PATH):
        pytest.fail(f"missing trajectory reference {REFERENCE_PATH}")
    prefix = f"{name}/"
    with np.load(REFERENCE_PATH) as data:
        return {
            key[len(prefix) :]: data[key]
            for key in data.files
            if key.startswith(prefix)
        }


def flight_samples(reference):
    """Samples before touchdown, minus one as margin for VSim's earlier contact."""
    touching = reference["foot_force_z"].max(axis=(1, 2)) > 1.0
    first_contact = int(touching.argmax())
    assert first_contact > 2, "scenario no longer has a flight phase"
    return slice(0, first_contact - 1)


def test_mujoco_cpu_reproduces_reference():
    reference = load_reference("cpu")
    actual = simulate("cpu")
    assert_same_payloads(actual, reference)
    assert_matches(actual, reference, MUJOCO_TOLERANCES)


@pytest.mark.warp
def test_mujoco_warp_reproduces_reference():
    if not torch.cuda.is_available():
        pytest.skip("MuJoCo Warp requires CUDA")
    reference = load_reference("warp")
    actual = simulate("warp")
    assert_same_payloads(actual, reference)
    assert_matches(actual, reference, MUJOCO_TOLERANCES)


@pytest.mark.vsim
def test_vsim_flight_tracks_warp_reference():
    from tests.unit_tests.conftest import vsim_guard

    vsim_guard()
    reference = load_reference("warp")
    actual = simulate("vsim")
    assert_same_payloads(actual, reference)
    assert_matches(actual, reference, VSIM_FLIGHT_TOLERANCES, flight_samples(reference))
    for channel in CHANNELS:
        assert np.isfinite(actual[channel]).all(), channel
    assert (actual["root_pos"][:, :, 2] > 0.1).all(), "base fell through the ground"


def record_reference(names):
    data = {}
    if os.path.exists(REFERENCE_PATH):
        with np.load(REFERENCE_PATH) as existing:
            data.update(existing)
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    for name in names:
        for key, value in simulate(name).items():
            data[f"{name}/{key}"] = value
        data[f"{name}/recorded_at_commit"] = np.array(commit)
    os.makedirs(os.path.dirname(REFERENCE_PATH), exist_ok=True)
    np.savez_compressed(REFERENCE_PATH, **data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", nargs="+", choices=("cpu", "warp"), required=True)
    record_reference(parser.parse_args().record)
