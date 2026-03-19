# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

"""
Sim2Sim validation script for standing task.

This script loads a trained stand policy and runs real-time simulation in MuJoCo
to validate the trained policy performance.
"""

import argparse
import os
import sys

import mujoco
import mujoco_viewer
import numpy as np
import torch
from pynput import keyboard
import time


class SimToSimStandCfg:
    """Configuration class for stand sim2sim parameters.
    
    Must be kept consistent with the stand training configuration.
    """

    class sim:
        sim_duration = 100.0
        num_action = 20
        num_obs_per_step = 75  # Stand task observation dimension
        actor_obs_history_length = 10
        dt = 0.005
        decimation = 4
        clip_observations = 100.0
        clip_actions = 100.0
        action_scale = 0.25



class MujocoStandRunner:
    """
    Sim2Sim runner for standing task.
    
    Loads a policy and runs real-time humanoid standing simulation in MuJoCo.
    """

    def __init__(self, cfg: SimToSimStandCfg, policy_path, model_path):
        self.cfg = cfg
        network_path = policy_path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.cfg.sim.dt

        self.policy = torch.jit.load(network_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer._render_every_frame = False
        self.init_variables()

    def init_variables(self) -> None:
        """Initialize simulation variables and joint index mappings."""
        self.dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.dof_pos = np.zeros(self.cfg.sim.num_action)
        self.dof_vel = np.zeros(self.cfg.sim.num_action)
        self.action = np.zeros(self.cfg.sim.num_action)
        
        # Default joint positions for standing
        self.default_dof_pos = np.array(
            [0, -0.5, 0, 1.0, -0.5, 0, 0, -0.5, 0, 1.0, -0.5, 0, 0, 0.1, 0.0, -0.3, 0, -0.1, 0.0, -0.3]
        )
        
        self.episode_length_buf = 0
        
        # Stand task uses fixed gait parameters (not changing)
        self.gait_phase = np.zeros(2)
        self.gait_cycle = 1.0  # Dummy value
        self.phase_ratio = np.array([0.5, 0.5])  # Both feet on ground
        self.phase_offset = np.array([0.0, 0.0])  # No offset

        # Joint index mapping from MuJoCo to Isaac Lab order
        self.mujoco_to_isaac_idx = [
            0,  # hip_roll_l_joint
            6,  # hip_roll_r_joint
            12,  # shoulder_pitch_l_joint
            16,  # shoulder_pitch_r_joint
            1,  # hip_pitch_l_joint
            7,  # hip_pitch_r_joint
            13,  # shoulder_roll_l_joint
            17,  # shoulder_roll_r_joint
            2,  # hip_yaw_l_joint
            8,  # hip_yaw_r_joint
            14,  # shoulder_yaw_l_joint
            18,  # shoulder_yaw_r_joint
            3,  # knee_pitch_l_joint
            9,  # knee_pitch_r_joint
            15,  # elbow_pitch_l_joint
            19,  # elbow_pitch_r_joint
            4,  # ankle_pitch_l_joint
            10,  # ankle_pitch_r_joint
            5,  # ankle_roll_l_joint
            11,  # ankle_roll_r_joint
        ]
        
        # Joint index mapping from Isaac Lab to MuJoCo order
        self.isaac_to_mujoco_idx = [
            0,  # hip_roll_l_joint
            4,  # hip_pitch_l_joint
            8,  # hip_yaw_l_joint
            12,  # knee_pitch_l_joint
            16,  # ankle_pitch_l_joint
            18,  # ankle_roll_l_joint
            1,  # hip_roll_r_joint
            5,  # hip_pitch_r_joint
            9,  # hip_yaw_r_joint
            13,  # knee_pitch_r_joint
            17,  # ankle_pitch_r_joint
            19,  # ankle_roll_r_joint
            2,  # shoulder_pitch_l_joint
            6,  # shoulder_roll_l_joint
            10,  # shoulder_yaw_l_joint
            14,  # elbow_pitch_l_joint
            3,  # shoulder_pitch_r_joint
            7,  # shoulder_roll_r_joint
            11,  # shoulder_yaw_r_joint
            15,  # elbow_pitch_r_joint
        ]
        
        # Initial command - stand still (zero velocity)
        self.command_vel = np.array([0.0, 0.0, 0.0])
        
        # Observation history buffer
        # Stand task: 54 obs per step * 10 history = 540
        self.obs_history = np.zeros(
            (self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length,), dtype=np.float32
        )

    def get_obs(self) -> np.ndarray:
        """
        Compute current observation vector from MuJoCo sensors and internal state.
        
        Stand task observation:
        - 3: angular velocity
        - 3: projected gravity
        - 3: command velocity
        - 20: joint positions (relative to default)
        - 20: joint velocities
        - 20: previous actions
        - 2: sin(gait_phase)
        - 2: cos(gait_phase)
        - 2: phase_ratio
        Total: 55 per step, but history makes it 54*10=540

        Returns:
            np.ndarray: Normalized and clipped observation history.
        """
        self.dof_pos = self.data.sensordata[0:20]
        self.dof_vel = self.data.sensordata[20:40]

        # Stand task: simplified observation (same structure but fixed gait)
        obs = np.concatenate(
            [
                self.data.sensor("angular-velocity").data.astype(np.double),  # 3
                self.quat_rotate_inverse(
                    self.data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double), np.array([0, 0, -1])
                ),  # 3 - projected gravity
                self.command_vel,  # 3 - command (standing = 0)
                (self.dof_pos - self.default_dof_pos)[self.mujoco_to_isaac_idx],  # 20
                self.dof_vel[self.mujoco_to_isaac_idx],  # 20
                np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions),  # 20
                np.sin(2 * np.pi * self.gait_phase),  # 2
                np.cos(2 * np.pi * self.gait_phase),  # 2
                self.phase_ratio,  # 2
            ],
            axis=0,
        ).astype(np.float32)

        # Update observation history
        self.obs_history = np.roll(self.obs_history, shift=-self.cfg.sim.num_obs_per_step)
        self.obs_history[-self.cfg.sim.num_obs_per_step:] = obs.copy()

        return np.clip(self.obs_history, -self.cfg.sim.clip_observations, self.cfg.sim.clip_observations)

    def position_control(self) -> np.ndarray:
        """
        Apply position control using scaled action.

        Returns:
            np.ndarray: Target joint positions in MuJoCo order.
        """
        actions_scaled = self.action * self.cfg.sim.action_scale
        return actions_scaled[self.isaac_to_mujoco_idx] + self.default_dof_pos

    def run(self) -> None:
        """
        Run the simulation loop with keyboard-controlled commands.
        """
        self.setup_keyboard_listener()
        self.listener.start()

        print("[INFO] Starting stand simulation...")
        print("[INFO] Controls: NumPad 8/2=forward/back, 4/6=left/right, 7/9=turn")

        while self.data.time < self.cfg.sim.sim_duration:
            self.obs_history = self.get_obs()
            self.action[:] = self.policy(torch.tensor(self.obs_history, dtype=torch.float32)).detach().numpy()[:20]
            self.action = np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions)

            for sim_update in range(self.cfg.sim.decimation):
                step_start_time = time.time()

                self.data.ctrl = self.position_control()
                mujoco.mj_step(self.model, self.data)
                self.viewer.render()

                elapsed = time.time() - step_start_time
                sleep_time = self.cfg.sim.dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            self.episode_length_buf += 1
            # Stand task: gait parameters stay fixed (no walking)

        self.listener.stop()
        self.viewer.close()

    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Rotate a vector by the inverse of a quaternion.

        Args:
            q (np.ndarray): Quaternion (x, y, z, w) format.
            v (np.ndarray): Vector to rotate.

        Returns:
            np.ndarray: Rotated vector.
        """
        q_w = q[-1]
        q_vec = q[:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * q_w * 2.0
        c = q_vec * np.dot(q_vec, v) * 2.0

        return a - b + c

    def adjust_command_vel(self, idx: int, increment: float) -> None:
        """
        Adjust command velocity vector.

        Args:
            idx (int): Index of velocity component (0=x, 1=y, 2=yaw).
            increment (float): Value to increment.
        """
        self.command_vel[idx] += increment
        # Clamp velocity range for standing (smaller range than walking)
        self.command_vel[idx] = np.clip(self.command_vel[idx], -0.3, 0.3)

    def setup_keyboard_listener(self) -> None:
        """
        Set up keyboard event listener for user control input.
        """

        def on_press(key):
            try:
                if key.char == "8":  # NumPad 8 - forward
                    self.adjust_command_vel(0, 0.05)
                elif key.char == "2":  # NumPad 2 - backward
                    self.adjust_command_vel(0, -0.05)
                elif key.char == "4":  # NumPad 4 - left
                    self.adjust_command_vel(1, -0.05)
                elif key.char == "6":  # NumPad 6 - right
                    self.adjust_command_vel(1, 0.05)
                elif key.char == "7":  # NumPad 7 - turn left
                    self.adjust_command_vel(2, -0.05)
                elif key.char == "9":  # NumPad 9 - turn right
                    self.adjust_command_vel(2, 0.05)
                elif key.char == "5":  # NumPad 5 - stop
                    self.command_vel = np.array([0.0, 0.0, 0.0])
            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press)


if __name__ == "__main__":
    LEGGED_LAB_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    parser = argparse.ArgumentParser(description="Run sim2sim for standing task with MuJoCo controller.")
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Path to stand policy.pt. If not specified, uses default stand policy",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(LEGGED_LAB_ROOT_DIR, "legged_lab/assets/tienkung2_lite/mjcf/tienkung.xml"),
        help="Path to MuJoCo XML model",
    )
    parser.add_argument("--duration", type=float, default=100.0, help="Simulation duration in seconds")
    args = parser.parse_args()

    # Default stand policy path
    if args.policy is None:
        # Check for stand policy in Exported_policy folder
        default_policy = os.path.join(LEGGED_LAB_ROOT_DIR, "Exported_policy", "stand.pt")
        if os.path.isfile(default_policy):
            args.policy = default_policy
        else:
            # Try to find any stand-related policy
            print("[WARN] No stand policy found. Please specify --policy path.")
            print("[INFO] Example: python legged_lab/scripts/sim2sim_stand.py --policy Exported_policy/stand.pt")

    if args.policy is not None and not os.path.isfile(args.policy):
        print(f"[ERROR] Policy file not found: {args.policy}")
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"[ERROR] MuJoCo model file not found: {args.model}")
        sys.exit(1)

    print(f"[INFO] Loaded policy: {args.policy}")
    print(f"[INFO] Loaded model: {args.model}")

    sim_cfg = SimToSimStandCfg()
    sim_cfg.sim.sim_duration = args.duration

    # Stand task uses fixed gait parameters (no actual walking)
    # Phase ratio = 0.5 means both feet on ground (standing)

    runner = MujocoStandRunner(
        cfg=sim_cfg,
        policy_path=args.policy,
        model_path=args.model,
    )
    runner.run()
