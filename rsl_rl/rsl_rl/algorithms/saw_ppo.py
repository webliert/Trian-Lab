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

from __future__ import annotations

from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import StandAndWalkRolloutStorage
from rsl_rl.utils import string_to_callable


def _mirror_actions(actions):
    """Mirror actions for symmetry loss (local import to avoid circular dependency)."""
    if actions is None:
        return None
    
    # Clone the input to avoid modifying it
    result = actions.clone()
    action_dim = actions.shape[1]
    
    # 对于天工机器人动作 (20维)
    if action_dim == 20:
        # Swap left and right leg actions (indices 0-5 and 6-11)
        left_leg = result[:, 0:6].clone()
        right_leg = result[:, 6:12].clone()
        result[:, 0:6] = right_leg
        result[:, 6:12] = left_leg
        
        # Swap left and right arm actions (indices 12-15 and 16-19)
        left_arm = result[:, 12:16].clone()
        right_arm = result[:, 16:20].clone()
        result[:, 12:16] = right_arm
        result[:, 16:20] = left_arm
    elif action_dim == 12:
        # Only leg actions (indices 0-5 and 6-11)
        left_leg = result[:, 0:6].clone()
        right_leg = result[:, 6:12].clone()
        result[:, 0:6] = right_leg
        result[:, 6:12] = left_leg
    
    return result


class SAWPPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347).

    This is a modified version of PPO that fixes the tensor repeat issue
    when using recurrent networks (LSTM) with data augmentation.
    """

    policy: ActorCritic
    """The actor critic module."""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND components
        if rnd_cfg is not None:
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=rnd_cfg.get("learning_rate", 1e-3))
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # If function is a string then resolve it to a function
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if symmetry_cfg["use_data_augmentation"] and not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    "Data augmentation enabled but the function is not callable:"
                    f" {symmetry_cfg['data_augmentation_func']}"
                )
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        # Create optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        # Create rollout storage
        self.storage: StandAndWalkRolloutStorage = None  # type: ignore
        self.transition = StandAndWalkRolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape
    ):
        # create memory for RND as well :)
        if self.rnd:
            rnd_state_shape = [self.rnd.num_states]
        else:
            rnd_state_shape = None
        # create rollout storage
        self.storage = StandAndWalkRolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            rnd_state_shape,
            self.device,
        )

    def act(self, obs, critic_obs):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # compute the actions and values
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Obtain curiosity gates / observations from infos
            rnd_state = infos["observations"]["rnd_state"]
            # Compute the intrinsic rewards
            # note: rnd_state is the gated_state after normalization if normalization is used
            self.intrinsic_rewards, rnd_state = self.rnd.get_intrinsic_reward(rnd_state)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards
            # Record the curiosity gates
            self.transition.rnd_state = rnd_state.clone()

        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs):
        # compute value for the last step
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # generator for mini batches
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # iterate over batches
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
        ) in generator:

            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            # original batch size
            original_batch_size = obs_batch.shape[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                
                # Check if obs_batch is 3D (from recurrent network) or 2D (from feedforward network)
                obs_ndims = obs_batch.dim()
                
                if obs_ndims == 3:
                    # For recurrent networks: obs_batch is (time_steps, batch_size, obs_dim)
                    # Flatten to 2D for augmentation: (time_steps * batch_size, obs_dim)
                    time_steps = obs_batch.shape[0]
                    obs_dim = obs_batch.shape[2]
                    critic_obs_dim = critic_obs_batch.shape[2]
                    
                    # Flatten: (T*B, dim)
                    obs_batch_flat = obs_batch.reshape(-1, obs_dim)
                    critic_obs_batch_flat = critic_obs_batch.reshape(-1, critic_obs_dim)
                    actions_batch_flat = actions_batch.reshape(-1, actions_batch.shape[-1]) if actions_batch is not None else None
                    
                    # Apply augmentation (returns 2x batch_size for each timestep)
                    obs_batch_aug, actions_batch_aug = data_augmentation_func(
                        obs=obs_batch_flat, 
                        actions=actions_batch_flat, 
                        env=self.symmetry["_env"], 
                        obs_type="policy"
                    )
                    critic_obs_batch_aug, _ = data_augmentation_func(
                        obs=critic_obs_batch_flat, 
                        actions=None, 
                        env=self.symmetry["_env"], 
                        obs_type="critic"
                    )
                    
                    # Keep as 2D: obs_batch_aug is now (2 * T * B, obs_dim)
                    # Update batch sizes
                    flat_batch_size = obs_batch_flat.shape[0]  # T * B
                    aug_flat_batch_size = obs_batch_aug.shape[0]  # 2 * T * B
                    original_batch_size = flat_batch_size
                    num_aug = 2  # symmetry augmentation doubles the batch
                    
                    # Flatten all other tensors too
                    old_actions_log_prob_batch = old_actions_log_prob_batch.reshape(-1, 1).repeat(2, 1).squeeze(-1)
                    target_values_batch = target_values_batch.reshape(-1, 1).repeat(2, 1).squeeze(-1)
                    advantages_batch = advantages_batch.reshape(-1, 1).repeat(2, 1).squeeze(-1)
                    returns_batch = returns_batch.reshape(-1, 1).repeat(2, 1).squeeze(-1)
                    
                    obs_batch = obs_batch_aug
                    critic_obs_batch = critic_obs_batch_aug
                    actions_batch = actions_batch_aug
                    
                    # Adjust hidden states to match new batch size
                    # After augmentation, batch size changes, so we need to reinitialize hidden states
                    new_batch_size = obs_batch.shape[0]
                    # Handle both list and tuple cases for hidden states
                    if isinstance(hid_states_batch[0], (list, tuple)):
                        first_actor_hid = hid_states_batch[0][0]
                        first_critic_hid = hid_states_batch[1][0]
                    else:
                        first_actor_hid = hid_states_batch[0]
                        first_critic_hid = hid_states_batch[1]
                    num_layers = first_actor_hid.shape[0]
                    hidden_dim = first_actor_hid.shape[2]
                    
                    # Reinitialize hidden states with zeros for the new batch size
                    actor_hidden_states = torch.zeros(num_layers, new_batch_size, hidden_dim, device=first_actor_hid.device)
                    critic_hidden_states = torch.zeros(num_layers, new_batch_size, hidden_dim, device=first_critic_hid.device)
                    hid_states_batch = (actor_hidden_states, critic_hidden_states)
                    # Reset masks for the new batch (all ones = treat as continuous)
                    masks_batch = torch.ones(new_batch_size, 1, device=masks_batch.device)
                else:
                    # For non-recurrent networks: obs_batch is (batch_size, obs_dim)
                    obs_batch, actions_batch = data_augmentation_func(
                        obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
                    )
                    critic_obs_batch, _ = data_augmentation_func(
                        obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic"
                    )
                    # compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)
                    
                    # For 2D tensors from non-recurrent networks (original behavior)
                    old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                    target_values_batch = target_values_batch.repeat(num_aug, 1)
                    advantages_batch = advantages_batch.repeat(num_aug, 1)
                    returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            
            # Get current batch size
            current_batch_size = obs_batch.shape[0]
            
            # Handle symmetry augmentation case - need to resize hidden states
            if self.symmetry and self.symmetry["use_data_augmentation"] and obs_ndims == 3:
                # hid_states_batch was already reinitialized for augmented batch size
                # Just use it directly
                actor_hid_states = (hid_states_batch[0], hid_states_batch[0].clone())
                critic_hid_states = (hid_states_batch[1], hid_states_batch[1].clone())
            else:
                # For LSTM: hid_states_batch = (actor_h, critic_h)
                # where actor_h and critic_h are tensors of shape [num_layers, batch_size, hidden_dim]
                # We need to combine h and c into (h, c) tuple format for LSTM
                hid_0_size = hid_states_batch[0].shape[1] if hid_states_batch[0] is not None else 0
                
                # Check if we need to resize hidden states to match current batch
                if hid_0_size != current_batch_size:
                    # Resize hidden states to match current batch size
                    num_layers = hid_states_batch[0].shape[0]
                    hidden_dim = hid_states_batch[0].shape[2]
                    actor_h = torch.zeros(num_layers, current_batch_size, hidden_dim, device=hid_states_batch[0].device)
                    actor_c = torch.zeros(num_layers, current_batch_size, hidden_dim, device=hid_states_batch[0].device)
                    critic_h = torch.zeros(num_layers, current_batch_size, hidden_dim, device=hid_states_batch[0].device)
                    critic_c = torch.zeros(num_layers, current_batch_size, hidden_dim, device=hid_states_batch[0].device)
                    actor_hid_states = (actor_h, actor_c)
                    critic_hid_states = (critic_h, critic_c)
                else:
                    actor_hid_states = (hid_states_batch[0], hid_states_batch[0].clone())
                    critic_hid_states = (hid_states_batch[1], hid_states_batch[1].clone())
            
            # -- actor
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=actor_hid_states)
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=critic_hid_states)
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate
                    # Perform this adaptation only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Symmetry loss
            if self.symmetry and self.symmetry["use_mirror_loss"]:
                # Compute symmetry loss to encourage symmetric behavior
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                
                # Get the original actions (first part of batch before augmentation)
                action_mean_orig = mean_actions_batch[:original_batch_size]
                
                # Get mirrored version of the original actions
                actions_mean_symm_batch = _mirror_actions(action_mean_orig)
                
                # Compute symmetry loss: predicted actions for augmented obs should match mirrored original actions
                # When use_data_augmentation is True, mean_actions_batch has both original and augmented predictions
                if mean_actions_batch.shape[0] > original_batch_size:
                    # Augmentation was applied, compare augmented predictions with mirrored originals
                    augmented_predictions = mean_actions_batch[original_batch_size:]
                    if augmented_predictions.shape[0] == actions_mean_symm_batch.shape[0]:
                        symmetry_loss = torch.nn.functional.mse_loss(
                            augmented_predictions, 
                            actions_mean_symm_batch.detach()
                        )
                    else:
                        # Size mismatch, skip symmetry loss for this batch
                        symmetry_loss = torch.tensor(0.0, device=mean_actions_batch.device)
                else:
                    # No augmentation, use self-supervised symmetry loss on original actions
                    # Loss: actions should equal their mirrored version (symmetric policy)
                    symmetry_loss = torch.nn.functional.mse_loss(
                        action_mean_orig,
                        actions_mean_symm_batch.detach()
                    )
                
                # add the loss to the total loss
                loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss

            # Random Network Distillation loss
            if self.rnd:
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients
            # -- For PPO
            self.optimizer.zero_grad()
            loss.backward()
            # -- For RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients
            # -- For PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- Clear the storage
        self.storage.clear()
        
        # Reset policy hidden states to initial state for next rollout
        # This is needed because training with augmentation may change the hidden state size
        self.policy.reset()

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel
