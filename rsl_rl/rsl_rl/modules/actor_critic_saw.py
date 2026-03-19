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
StandAndWalk (SaW) Actor-Critic Module

该模块实现了论文中描述的StandAndWalk控制器，基于双层LSTM循环神经网络。
网络输入包括机器人状态和用户命令，输出为20个关节空间的PD设定点。

网络架构：
- 输入层：状态向量(关节速度、位置、躯干方向) + 命令向量[cx, cy, cyaw]
- LSTM层：(64, 64)双层LSTM
- 输出层：20个关节位置目标

SaW控制器运行在50Hz，而底层PD控制器运行在2kHz。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class ActorCriticSaW(nn.Module):
    """StandAndWalk Actor-Critic with dual-layer LSTM.
    
    论文描述的控制器架构：
    - 输入: 当前机器人状态 + 用户命令 cu=[cx, cy, cyaw]
    - 网络: (64, 64)双层LSTM循环神经网络
    - 输出: 20个关节空间的PD设定点
    
    站立对应于 cu = [0, 0, 0]
    """
    
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        # LSTM specific parameters for SaW controller
        lstm_hidden_dim=64,
        lstm_num_layers=2,
        init_noise_std=1.0,
        noise_std_type="scalar",
        **kwargs,
    ):
        """
        Initialize the StandAndWalk Actor-Critic.
        
        Args:
            num_actor_obs: Actor observation dimension (robot state + command)
            num_critic_obs: Critic observation dimension
            num_actions: Number of actions (20 joints)
            actor_hidden_dims: Hidden layer dimensions for actor MLP (after LSTM)
            critic_hidden_dims: Hidden layer dimensions for critic MLP (after LSTM)
            activation: Activation function name
            lstm_hidden_dim: LSTM hidden dimension (default: 64 as per paper)
            lstm_num_layers: Number of LSTM layers (default: 2 as per paper)
            init_noise_std: Initial action noise standard deviation
            noise_std_type: Type of noise std ('scalar' or 'log')
        """
        if kwargs:
            print(
                f"ActorCriticSaW.__init__ got unexpected arguments, which will be ignored: {list(kwargs.keys())}"
            )
        
        super().__init__()
        activation = resolve_nn_activation(activation)
        
        # Store dimensions
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        
        # Actor LSTM (64, 64) as per paper specification
        self.actor_lstm = nn.LSTM(
            input_size=num_actor_obs,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True
        )
        
        # Actor MLP layers (after LSTM)
        actor_layers = []
        actor_layers.append(nn.Linear(lstm_hidden_dim, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor_mlp = nn.Sequential(*actor_layers)
        
        # Critic LSTM (64, 64) as per paper specification
        self.critic_lstm = nn.LSTM(
            input_size=num_critic_obs,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True
        )
        
        # Critic MLP layers (after LSTM)
        critic_layers = []
        critic_layers.append(nn.Linear(lstm_hidden_dim, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic_mlp = nn.Sequential(*critic_layers)
        
        print(f"Actor LSTM: {lstm_num_layers}x{lstm_hidden_dim}")
        print(f"Actor MLP: {self.actor_mlp}")
        print(f"Critic LSTM: {lstm_num_layers}x{lstm_hidden_dim}")
        print(f"Critic MLP: {self.critic_mlp}")
        
        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type: {noise_std_type}. Should be 'scalar' or 'log'")
        
        # Initialize hidden states
        self.actor_hidden_states = None
        self.critic_hidden_states = None
        
        # Action distribution
        self.distribution = None
        Normal.set_default_validate_args(False)
        
        # Initialize LSTM weights
        self._init_lstm_weights()
    
    def _init_lstm_weights(self):
        """Initialize LSTM weights with orthogonal initialization."""
        for name, param in self.actor_lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
        
        for name, param in self.critic_lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
    
    def reset(self, dones=None):
        """Reset hidden states for environments that are done.
        
        Args:
            dones: Boolean tensor indicating which environments are done
        """
        if dones is None:
            # Reset all hidden states
            self.actor_hidden_states = None
            self.critic_hidden_states = None
        else:
            # Reset hidden states for done environments
            if self.actor_hidden_states is not None:
                if isinstance(self.actor_hidden_states, tuple):
                    for h in self.actor_hidden_states:
                        h[:, dones == 1, :] = 0.0
                else:
                    self.actor_hidden_states[:, dones == 1, :] = 0.0
            
            if self.critic_hidden_states is not None:
                if isinstance(self.critic_hidden_states, tuple):
                    for h in self.critic_hidden_states:
                        h[:, dones == 1, :] = 0.0
                else:
                    self.critic_hidden_states[:, dones == 1, :] = 0.0
    
    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean
    
    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def update_distribution(self, observations):
        """Update the action distribution based on observations.
        
        Args:
            observations: Input observations [batch_size, obs_dim]
        """
        # Handle hidden states format for LSTM
        # LSTM expects (h, c) tuple, but we may receive different formats from the storage
        actor_hx = self.actor_hidden_states
        
        # If hidden states is ((h, c), (h, c)) format (actor, critic), extract actor part
        if isinstance(actor_hx, tuple) and len(actor_hx) == 2:
            first = actor_hx[0]
            # Check if it's (h, c) tuple where h and c are tensors
            if isinstance(first, tuple) and len(first) == 2:
                # It's ((h, c), (h, c)) format - extract actor part
                actor_hx = (actor_hx[0], actor_hx[1]) if len(actor_hx) == 2 else actor_hx[0]
        
        # Process through actor LSTM
        lstm_out, self.actor_hidden_states = self.actor_lstm(
            observations.unsqueeze(1), 
            actor_hx
        )
        lstm_out = lstm_out.squeeze(1)  # [batch_size, lstm_hidden_dim]
        
        # Process through actor MLP
        mean = self.actor_mlp(lstm_out)
        
        # Compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown noise_std_type: {self.noise_std_type}")
        
        # Create distribution
        self.distribution = Normal(mean, std)
    
    def act(self, observations, masks=None, hidden_states=None):
        """Sample action from the policy distribution.
        
        Args:
            observations: Input observations [batch_size, obs_dim]
            masks: Environment done masks (not used in current implementation)
            hidden_states: Hidden states from rollout storage
            
        Returns:
            actions: Sampled actions [batch_size, num_actions]
        """
        # Use provided hidden states if available, otherwise use stored ones
        if hidden_states is not None:
            self.actor_hidden_states = hidden_states
        
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        """Get log probability of actions under current distribution.
        
        Args:
            actions: Actions to evaluate [batch_size, num_actions]
            
        Returns:
            log_probs: Log probabilities [batch_size]
        """
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, observations):
        """Get deterministic action for inference.
        
        Args:
            observations: Input observations [batch_size, obs_dim]
            
        Returns:
            actions: Mean actions [batch_size, num_actions]
        """
        # Process through actor LSTM
        lstm_out, self.actor_hidden_states = self.actor_lstm(
            observations.unsqueeze(1),
            self.actor_hidden_states
        )
        lstm_out = lstm_out.squeeze(1)
        
        # Process through actor MLP
        actions_mean = self.actor_mlp(lstm_out)
        return actions_mean
    
    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        """Evaluate value function.
        
        Args:
            critic_observations: Critic observations [batch_size, obs_dim]
            masks: Environment done masks (not used in current implementation)
            hidden_states: Hidden states from rollout storage
            
        Returns:
            values: State values [batch_size, 1]
        """
        # Use provided hidden states if available, otherwise use stored ones
        if hidden_states is not None:
            self.critic_hidden_states = hidden_states
        
        # Process through critic LSTM
        lstm_out, self.critic_hidden_states = self.critic_lstm(
            critic_observations.unsqueeze(1),
            self.critic_hidden_states
        )
        lstm_out = lstm_out.squeeze(1)
        
        # Process through critic MLP
        value = self.critic_mlp(lstm_out)
        return value
    
    def get_hidden_states(self):
        """Get current hidden states for storage.
        
        Returns:
            actor_hidden_states, critic_hidden_states
        """
        return self.actor_hidden_states, self.critic_hidden_states
    
    def set_hidden_states(self, actor_hs, critic_hs):
        """Set hidden states from storage.
        
        Args:
            actor_hs: Actor hidden states
            critic_hs: Critic hidden states
        """
        self.actor_hidden_states = actor_hs
        self.critic_hidden_states = critic_hs
    
    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the model.
        
        Args:
            state_dict: State dictionary of the model
            strict: Whether to strictly enforce key matching
            
        Returns:
            True if training is resumed
        """
        super().load_state_dict(state_dict, strict=strict)
        return True


class ActorCriticRecurrentSaW(ActorCriticSaW):
    """Alias for ActorCriticSaW to maintain backward compatibility."""
    pass
