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
Sim2Sim Standup Controller with ROS2 Policy Switching

This script runs a humanoid robot in MuJoCo simulation with the following behavior:
1. First 5 seconds: Robot performs in-place stepping (原地踏步)
2. After 5 seconds: Robot transitions to standing posture using synchronized gait parameters

Additionally, it supports switching between walk and stand policies via ROS2 topic:
- Subscribe to /robot_mode topic (std_msgs/String) to switch between "walk" and "stand" policies
- When mode is "walk": uses walk.pt policy
- When mode is "stand": uses only_stand.pt policy

The standing is achieved by modifying gait parameters:
- gait_air_ratio: Set to 0.0 (feet stay on ground)
- gait_phase: Synchronized to maintain stable standing

ROS2 Topic Usage:
    # Switch to stand mode (uses only_stand.pt)
    ros2 topic pub /robot_mode std_msgs/String "data: 'stand'" -1

    # Switch to walk mode (uses walk.pt)
    ros2 topic pub /robot_mode std_msgs/String "data: 'walk'" -1
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

# ROS2 imports for policy switching
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from rclpy.executors import SingleThreadedExecutor
    import threading
    ROS2_AVAILABLE = True
except ImportError:
    print("[WARN] rclpy not available. ROS2 policy switching will be disabled.")
    ROS2_AVAILABLE = False
    ROS2_NODE = None
    ROS2_EXECUTOR = None


class PolicySwitchSubscriber:
    """
    ROS2 subscriber for policy switching (walk/stand).
    
    This class subscribes to mode commands from ROS2 and stores them
    for use in switching between different policies.
    
    The subscriber runs in a separate thread to avoid blocking the simulation.
    
    Modes:
        - "walk": Use walk policy for movement
        - "stand": Use stand policy for stationary standing
    """
    
    def __init__(self, topic_name: str = "/robot_mode", domain_id: int = 0):
        """
        Initialize the policy switch subscriber.
        
        Args:
            topic_name: ROS2 topic name for mode commands (std_msgs/String)
            domain_id: ROS2 domain ID
        """
        if not ROS2_AVAILABLE:
            raise RuntimeError("rclpy is not available. Cannot create PolicySwitchSubscriber.")
        
        self.topic_name = topic_name
        
        # Initialize mode to "walk" by default
        self._mode = "walk"
        self._lock = threading.Lock()
        
        # Set ROS_DOMAIN_ID if not already set
        os.environ.setdefault('ROS_DOMAIN_ID', str(domain_id))
        
        # Initialize rclpy if not already initialized
        if not rclpy.ok():
            rclpy.init()
        
        # Create ROS2 node and subscriber
        self._node = rclpy.create_node('mujoco_policy_switch_subscriber')
        self._subscription = self._node.create_subscription(
            String,
            topic_name,
            self._mode_callback,
            10  # QoS profile depth
        )
        
        # Create executor and run in separate thread
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._running = True
        self._thread = threading.Thread(target=self._spin_thread, daemon=True)
        self._thread.start()
        
        print(f"[INFO] PolicySwitchSubscriber initialized on topic: {topic_name}")
        print("[INFO] Supported modes: 'walk' (walk policy), 'stand' (stand policy)")
    
    def _mode_callback(self, msg: String):
        """Callback function for mode switch messages."""
        with self._lock:
            mode = msg.data.strip().lower()
            if mode in ["walk", "stand"]:
                if mode != self._mode:
                    self._mode = mode
                    print(f"[INFO] Policy switched to: {self._mode}")
            else:
                print(f"[WARN] Unknown mode received: {msg.data}. Supported modes: 'walk', 'stand'")
    
    def _spin_thread(self):
        """Thread function to spin the ROS2 node."""
        while self._running and rclpy.ok():
            self._executor.spin_once(timeout_sec=0.01)
    
    def get_mode(self) -> str:
        """
        Get the current mode.
        
        Returns:
            str: Current mode ("walk" or "stand")
        """
        with self._lock:
            return self._mode
    
    def shutdown(self):
        """Shutdown the subscriber and cleanup resources."""
        self._running = False
        if hasattr(self, '_thread') and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        
        if hasattr(self, '_node') and self._node:
            self._node.destroy_node()
        
        print("[INFO] PolicySwitchSubscriber shutdown complete")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.shutdown()
        except Exception:
            pass


class SimToSimCfg:
    """Configuration class for sim2sim parameters.

    Must be kept consistent with the training configuration.
    """

    class sim:
        sim_duration = 100.0
        num_action = 20
        num_obs_per_step = 75
        actor_obs_history_length = 10
        dt = 0.005
        decimation = 4
        clip_observations = 100.0
        clip_actions = 100.0
        action_scale = 0.25

    class robot:
        # Walking gait parameters (default)
        gait_air_ratio_l: float = 0.38
        gait_air_ratio_r: float = 0.38
        gait_phase_offset_l: float = 0.38
        gait_phase_offset_r: float = 0.88
        gait_cycle: float = 0.85
        
    class standup:
        # 缩短站立延迟时间，让机器人更快过渡到站立状态
        # 原来是5.0秒，现在改为2.0秒
        standup_delay = 2.0  # Time in seconds before transitioning to standing
        # 添加站立过渡的平滑因子
        standup_transition_steps = 100  # Number of steps for smooth transition


class MujocoRunner:
    """
    Sim2Sim runner with standup functionality that loads a policy and a MuJoCo model
    to run real-time humanoid control simulation.

    The robot will perform in-place stepping for the first 5 seconds,
    then transition to standing posture.
    
    Additionally supports switching between walk and stand policies via ROS2 topic.

    Args:
        cfg (SimToSimCfg): Configuration object for simulation.
        walk_policy_path (str): Path to the TorchScript exported walk policy.
        stand_policy_path (str): Path to the TorchScript exported stand policy.
        model_path (str): Path to the MuJoCo XML model.
    """

    def __init__(self, cfg: SimToSimCfg, walk_policy_path, stand_policy_path, model_path):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.cfg.sim.dt

        # Load both policies
        self.walk_policy = torch.jit.load(walk_policy_path)
        self.stand_policy = torch.jit.load(stand_policy_path)
        
        # Set default policy to walk
        self.policy = self.walk_policy
        self.current_policy_name = "walk"
        
        # ROS2 policy switch subscriber
        self.policy_switch_subscriber = None
        
        # Smooth transition parameters
        self.is_transitioning = False
        self.transition_start_time = 0.0
        self.transition_duration = 1.0  # 1 second smooth transition
        self.target_policy = None
        self.previous_action = np.zeros(self.cfg.sim.num_action)
        
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer._render_every_frame = False
        self.init_variables()
        
    def set_policy_switch_subscriber(self, subscriber):
        """Set the ROS2 policy switch subscriber."""
        self.policy_switch_subscriber = subscriber
    
    def start_smooth_transition(self, target_mode: str):
        """
        Start a smooth transition to the target policy.
        
        Args:
            target_mode: "walk" or "stand"
        """
        if target_mode == "stand" and self.current_policy_name != "stand":
            print(f"\n[INFO] Starting smooth transition to stand policy at time {self.data.time:.2f}s")
            self.target_policy = self.stand_policy
            self.current_policy_name = "stand"
            self.is_transitioning = True
            self.transition_start_time = self.data.time
            # Set target gait parameters for standing
            self.target_phase_ratio = np.array([0.0, 0.0])
            self.target_phase_offset = np.array([0.0, 0.0])
            self.target_command_vel = np.array([0.0, 0.0, 0.0])
            self.target_is_standing = True
            print(f"[INFO] Smooth transition started (duration: {self.transition_duration}s)")
            
        elif target_mode == "walk" and self.current_policy_name != "walk":
            print(f"\n[INFO] Starting smooth transition to walk policy at time {self.data.time:.2f}s")
            self.target_policy = self.walk_policy
            self.current_policy_name = "walk"
            self.is_transitioning = True
            self.transition_start_time = self.data.time
            # Set target gait parameters for walking
            self.target_phase_ratio = np.array([self.cfg.robot.gait_air_ratio_l, self.cfg.robot.gait_air_ratio_r])
            self.target_phase_offset = np.array([self.cfg.robot.gait_phase_offset_l, self.cfg.robot.gait_phase_offset_r])
            self.target_command_vel = np.array([0.0, 0.0, 0.0])
            self.target_is_standing = False
            print(f"[INFO] Smooth transition started (duration: {self.transition_duration}s)")
    
    def update_smooth_transition(self):
        """Update the smooth transition progress."""
        if not self.is_transitioning:
            return
        
        # Calculate transition progress (0 to 1)
        elapsed = self.data.time - self.transition_start_time
        progress = min(elapsed / self.transition_duration, 1.0)
        
        # Use smooth easing function (ease-in-out)
        if progress < 0.5:
            eased_progress = 2 * progress * progress
        else:
            eased_progress = 1 - pow(-2 * progress + 2, 2) / 2
        
        # Interpolate gait parameters smoothly
        self.phase_ratio = self.phase_ratio + (self.target_phase_ratio - self.phase_ratio) * 0.1
        self.phase_offset = self.phase_offset + (self.target_phase_offset - self.phase_offset) * 0.1
        self.command_vel = self.command_vel + (self.target_command_vel - self.command_vel) * 0.1
        self.is_standing = self.target_is_standing
        
        # Check if transition is complete
        if progress >= 1.0:
            self.is_transitioning = False
            self.policy = self.target_policy
            self.phase_ratio = self.target_phase_ratio.copy()
            self.phase_offset = self.target_phase_offset.copy()
            self.command_vel = self.target_command_vel.copy()
            print(f"[INFO] Smooth transition completed at time {self.data.time:.2f}s")
    
    def switch_policy(self, mode: str):
        """
        Switch between walk and stand policies (instant, for backwards compatibility).
        
        Args:
            mode: "walk" or "stand"
        """
        if mode == "stand" and self.current_policy_name != "stand":
            print(f"\n[INFO] Switching to stand policy at time {self.data.time:.2f}s")
            self.policy = self.stand_policy
            self.current_policy_name = "stand"
            # Set gait parameters for standing
            self.is_standing = True
            self.phase_ratio = np.array([0.0, 0.0])
            self.gait_phase[0] = 0.0
            self.gait_phase[1] = 0.0
            self.phase_offset[0] = 0.0
            self.phase_offset[1] = 0.0
            self.command_vel = np.array([0.0, 0.0, 0.0])
            print(f"[INFO] Stand policy activated")
            
        elif mode == "walk" and self.current_policy_name != "walk":
            print(f"\n[INFO] Switching to walk policy at time {self.data.time:.2f}s")
            self.policy = self.walk_policy
            self.current_policy_name = "walk"
            self.is_standing = False
            # Restore walking gait parameters
            self.phase_ratio = np.array([self.cfg.robot.gait_air_ratio_l, self.cfg.robot.gait_air_ratio_r])
            self.phase_offset = np.array([self.cfg.robot.gait_phase_offset_l, self.cfg.robot.gait_phase_offset_r])
            print(f"[INFO] Walk policy activated")

    def init_variables(self) -> None:
        """Initialize simulation variables and joint index mappings."""
        self.dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.dof_pos = np.zeros(self.cfg.sim.num_action)
        self.dof_vel = np.zeros(self.cfg.sim.num_action)
        self.action = np.zeros(self.cfg.sim.num_action)
        self.default_dof_pos = np.array(
            [0, -0.5, 0, 1.0, -0.5, 0, 0, -0.5, 0, 1.0, -0.5, 0, 0, 0.1, 0.0, -0.3, 0, -0.1, 0.0, -0.3]
        )
        self.episode_length_buf = 0
        self.gait_phase = np.zeros(2)
        self.gait_cycle = self.cfg.robot.gait_cycle
        self.phase_ratio = np.array([self.cfg.robot.gait_air_ratio_l, self.cfg.robot.gait_air_ratio_r])
        self.phase_offset = np.array([self.cfg.robot.gait_phase_offset_l, self.cfg.robot.gait_phase_offset_r])
        
        # Standup state variables
        self.is_standing = False
        self.standup_start_time = self.cfg.standup.standup_delay
        
        # Original (walking) gait parameters for reference
        self.walking_gait_air_ratio_l = self.cfg.robot.gait_air_ratio_l
        self.walking_gait_air_ratio_r = self.cfg.robot.gait_air_ratio_r
        self.walking_gait_phase_offset_l = self.cfg.robot.gait_phase_offset_l
        self.walking_gait_phase_offset_r = self.cfg.robot.gait_phase_offset_r
        self.walking_gait_cycle = self.cfg.robot.gait_cycle

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
            11,  # ankle_roll_r_joint,
        ]
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
            15,  # elbow_pitch_r_joint,
        ]
        # Initial command vel
        self.command_vel = np.array([0.0, 0.0, 0.0])
        self.obs_history = np.zeros(
            (self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length,), dtype=np.float32
        )

    def get_obs(self) -> np.ndarray:
        """
        Compute current observation vector from MuJoCo sensors and internal state.

        Returns:
            np.ndarray: Normalized and clipped observation history.
        """
        # Print raw quaternion data for debugging
        orientation_data = self.data.sensor("orientation").data
        
        self.dof_pos = self.data.sensordata[0:20]
        self.dof_vel = self.data.sensordata[20:40]
        self.imu_increase = 1.0

        obs = np.concatenate(
            [
                self.imu_increase * self.data.sensor("angular-velocity").data.astype(np.double),  # 3
                self.quat_rotate_inverse(
                    self.imu_increase * self.data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double), np.array([0, 0, -1])
                ),  # 3
                self.command_vel,  # 3
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
        self.obs_history[-self.cfg.sim.num_obs_per_step :] = obs.copy()

        return np.clip(self.obs_history, -self.cfg.sim.clip_observations, self.cfg.sim.clip_observations)

    def position_control(self) -> np.ndarray:
        """
        Apply position control using scaled action.

        Returns:
            np.ndarray: Target joint positions in MuJoCo order.
        """
        actions_scaled = self.action * self.cfg.sim.action_scale
        return actions_scaled[self.isaac_to_mujoco_idx] + self.default_dof_pos

    def transition_to_standing(self) -> None:
        """
        Transition gait parameters to achieve standing posture.
        
        Standing is achieved by:
        1. Setting gait_air_ratio to 0.0 (feet stay on ground)
        2. Synchronizing gait phases (both legs have same phase)
        3. Setting phase_ratio to 0.0
        """
        if not self.is_standing:
            print(f"\n[INFO] Transitioning to standing at time {self.data.time:.2f}s")
            self.is_standing = True
            
            # Set gait parameters for standing
            # Air ratio 0 means feet stay on ground
            self.phase_ratio = np.array([0.0, 0.0])
            
            # Synchronize both legs to same phase for stable standing
            self.gait_phase[0] = 0.0
            self.gait_phase[1] = 0.0
            
            # Adjust phase offset to synchronize
            self.phase_offset[0] = 0.0
            self.phase_offset[1] = 0.0
            
            # Keep zero velocity command
            self.command_vel = np.array([0.0, 0.0, 0.0])
            
            print(f"[INFO] Standing mode activated - gait parameters synchronized")

    def run(self) -> None:
        """
        Run the simulation loop with keyboard-controlled commands and ROS2 policy switching.
        """
        self.setup_keyboard_listener()
        self.listener.start()

        while self.data.time < self.cfg.sim.sim_duration:
            # Update smooth transition if active
            self.update_smooth_transition()
            
            # Check for ROS2 policy switch command
            if self.policy_switch_subscriber is not None:
                target_mode = self.policy_switch_subscriber.get_mode()
                if target_mode != self.current_policy_name:
                    self.start_smooth_transition(target_mode)
            
            # Check if it's time to transition to standing (auto standup with smooth transition)
            if not self.is_standing and not self.is_transitioning and self.data.time >= self.standup_start_time:
                # Start smooth transition to stand policy
                self.start_smooth_transition("stand")
            
            # Check for manual standup trigger (press 's' key)
            if not self.is_standing:
                pass  # Will be handled by keyboard listener

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
            self.calculate_gait_para()

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

    def calculate_gait_para(self) -> None:
        """
        Update gait phase parameters based on simulation time and offset.
        
        If standing, both legs are synchronized with zero phase.
        """
        if self.is_standing:
            # During standing, keep phases synchronized at 0
            self.gait_phase[0] = 0.0
            self.gait_phase[1] = 0.0
        else:
            # Normal walking/trotting gait calculation
            t = self.episode_length_buf * self.dt / self.gait_cycle
            self.gait_phase[0] = (t + self.phase_offset[0]) % 1.0
            self.gait_phase[1] = (t + self.phase_offset[1]) % 1.0

    def adjust_command_vel(self, idx: int, increment: float) -> None:
        """
        Adjust command velocity vector.

        Args:
            idx (int): Index of velocity component (0=x, 1=y, 2=yaw).
            increment (float): Value to increment.
        """
        self.command_vel[idx] += increment
        self.command_vel[idx] = np.clip(self.command_vel[idx], -1.0, 1.0)  # vel clip

    def setup_keyboard_listener(self) -> None:
        """
        Set up keyboard event listener for user control input.
        """

        def on_press(key):
            try:
                if key.char == "8":  # NumPad 8      x += 0.2
                    self.adjust_command_vel(0, 0.2)
                elif key.char == "2":  # NumPad 2      x -= 0.2
                    self.adjust_command_vel(0, -0.2)
                elif key.char == "4":  # NumPad 4      y -= 0.2
                    self.adjust_command_vel(1, -0.2)
                elif key.char == "6":  # NumPad 6      y += 0.2
                    self.adjust_command_vel(1, 0.2)
                elif key.char == "7":  # NumPad 7      yaw += 0.2
                    self.adjust_command_vel(2, -0.2)
                elif key.char == "9":  # NumPad 9      yaw -= 0.2
                    self.adjust_command_vel(2, 0.2)
                elif key.char == "s" or key.char == "S":  # Manual standup trigger
                    if not self.is_standing:
                        self.transition_to_standing()
            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press)


if __name__ == "__main__":
    LEGGED_LAB_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    parser = argparse.ArgumentParser(description="Run sim2sim Mujoco controller with ROS2 policy switching.")
    parser.add_argument(
        "--task",
        type=str,
        default="walk",
        choices=["walk", "run"],
        help="Task type: 'walk' or 'run' to set gait parameters",
    )
    parser.add_argument(
        "--walk_policy",
        type=str,
        default=None,
        help="Path to walk policy.pt. If not specified, uses Exported_policy/walk.pt",
    )
    parser.add_argument(
        "--stand_policy",
        type=str,
        default=None,
        help="Path to stand policy.pt. If not specified, uses Exported_policy/only_stand.pt",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(LEGGED_LAB_ROOT_DIR, "legged_lab/assets/tienkung2_lite/mjcf/tienkung.xml"),
        help="Path to model.xml",
    )
    parser.add_argument("--duration", type=float, default=100.0, help="Simulation duration in seconds")
    parser.add_argument(
        "--standup-delay",
        type=float,
        default=5.0,
        help="Time in seconds before transitioning to standing (default: 5.0, set to 0 to disable auto standup)",
    )
    # ROS2 configuration
    parser.add_argument(
        "--mode_topic",
        type=str,
        default="/robot_mode",
        help="ROS2 topic name for mode switching (std_msgs/String)",
    )
    parser.add_argument(
        "--ros2_domain_id",
        type=int,
        default=0,
        help="ROS2 domain ID",
    )
    parser.add_argument(
        "--enable_ros2",
        action="store_true",
        help="Enable ROS2 policy switching (requires rclpy)",
    )
    args = parser.parse_args()

    # Set default policy paths
    if args.walk_policy is None:
        args.walk_policy = os.path.join(LEGGED_LAB_ROOT_DIR, "Exported_policy", "walk.pt")
    if args.stand_policy is None:
        args.stand_policy = os.path.join(LEGGED_LAB_ROOT_DIR, "Exported_policy", "stand_zero.pt")

    # Validate policy files
    if not os.path.isfile(args.walk_policy):
        print(f"[ERROR] Walk policy file not found: {args.walk_policy}")
        sys.exit(1)
    if not os.path.isfile(args.stand_policy):
        print(f"[ERROR] Stand policy file not found: {args.stand_policy}")
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"[ERROR] MuJoCo model file not found: {args.model}")
        sys.exit(1)

    print(f"[INFO] Loaded task preset: {args.task.upper()}")
    print(f"[INFO] Loaded walk policy: {args.walk_policy}")
    print(f"[INFO] Loaded stand policy: {args.stand_policy}")
    print(f"[INFO] Loaded model: {args.model}")
    print(f"[INFO] Standup delay: {args.standup_delay} seconds")
    print(f"[INFO] ROS2 enabled: {args.enable_ros2}")

    sim_cfg = SimToSimCfg()
    sim_cfg.sim.sim_duration = args.duration
    sim_cfg.standup.standup_delay = args.standup_delay

    # Set gait parameters according to task
    if args.task == "walk":
        sim_cfg.robot.gait_air_ratio_l = 0.38
        sim_cfg.robot.gait_air_ratio_r = 0.38
        sim_cfg.robot.gait_phase_offset_l = 0.38
        sim_cfg.robot.gait_phase_offset_r = 0.88
        sim_cfg.robot.gait_cycle = 0.85
    elif args.task == "run":
        sim_cfg.robot.gait_air_ratio_l = 0.6
        sim_cfg.robot.gait_air_ratio_r = 0.6
        sim_cfg.robot.gait_phase_offset_l = 0.6
        sim_cfg.robot.gait_phase_offset_r = 0.1
        sim_cfg.robot.gait_cycle = 0.5

    # Create the MujocoRunner with both policies
    runner = MujocoRunner(
        cfg=sim_cfg,
        walk_policy_path=args.walk_policy,
        stand_policy_path=args.stand_policy,
        model_path=args.model,
    )

    # Initialize ROS2 policy switch subscriber if enabled
    policy_switch_subscriber = None
    if args.enable_ros2 and ROS2_AVAILABLE:
        try:
            policy_switch_subscriber = PolicySwitchSubscriber(
                topic_name=args.mode_topic,
                domain_id=args.ros2_domain_id
            )
            runner.set_policy_switch_subscriber(policy_switch_subscriber)
            print(f"[INFO] ROS2 policy switching enabled!")
            print(f"[INFO] To switch to stand mode: ros2 topic pub {args.mode_topic} std_msgs/msg/String '{{data: stand}}' -1")
            print(f"[INFO] To switch to walk mode: ros2 topic pub {args.mode_topic} std_msgs/msg/String '{{data: walk}}' -1")
        except Exception as e:
            print(f"[ERROR] Failed to initialize ROS2 subscriber: {e}")
            print("[WARN] Continuing without ROS2 policy switching...")
    elif args.enable_ros2 and not ROS2_AVAILABLE:
        print("[WARN] ROS2 requested but rclpy not available. Installing rclpy may be required.")

    # Run the simulation
    runner.run()

    # Cleanup
    if policy_switch_subscriber is not None:
        try:
            policy_switch_subscriber.shutdown()
        except Exception as e:
            print(f"[WARN] Error shutting down policy switch subscriber: {e}")
    
    # Shutdown rclpy if it was initialized
    if ROS2_AVAILABLE and args.enable_ros2:
        try:
            if rclpy.ok():
                rclpy.shutdown()
                print("[INFO] rclpy shutdown complete")
        except Exception as e:
            print(f"[WARN] Error shutting down rclpy: {e}")
