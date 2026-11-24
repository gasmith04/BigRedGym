"""Keyboard teleop for the MuJoCo passive viewer.

Wires into mujoco.viewer.launch_passive's key_callback, set on the backend
before the first render() call.  MuJoCo delivers discrete key EVENTS, so a
press maps straight to one command increment (no edge detection needed —
contrast VsimKeyboardInterface, which polls).

The callback runs on the viewer's UI thread, concurrently with the loop that
steps the simulation. A reset from that thread rewrites MjModel/MjData mid
mj_step and segfaults, so presses are queued here and applied by the stepping
loop between steps via apply_pending().

Bindings and command logic are shared with the vsim interface — see
teleop_bindings.py for the key layout and why it is IJKL/NM.
"""

import queue

from gym.utils.interfaces.teleop_bindings import KEY_TO_ACTION, TeleopCommands

# GLFW uses ASCII codes for letters and separate codes for special keys.
# Hardcoded to avoid a glfw dependency.
SPECIAL_KEYCODES = {"UP": 265, "DOWN": 264}
KEYCODE_TO_ACTION = {
    ord(key) if len(key) == 1 else SPECIAL_KEYCODES[key]: action
    for key, action in KEY_TO_ACTION.items()
}


class MujocoKeyboardInterface:
    def __init__(self, env):
        self.env = env
        self.commands = TeleopCommands(env)
        self._pending = queue.SimpleQueue()

        env._backend._viewer_key_callback = self._on_key
        self.commands.print_help("MuJoCo viewer")

    def _on_key(self, keycode: int) -> None:
        action = KEYCODE_TO_ACTION.get(keycode)
        if action is not None:
            self._pending.put(action)

    def apply_pending(self) -> None:
        """Apply queued presses; call from the thread that steps the env."""
        while True:
            try:
                action = self._pending.get_nowait()
            except queue.Empty:
                return
            self.commands.apply(action)
