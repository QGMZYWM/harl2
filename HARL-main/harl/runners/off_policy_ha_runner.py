"""Runner for off-policy HARL algorithms."""
import torch
import numpy as np
import torch.nn.functional as F
from harl.runners.off_policy_base_runner import OffPolicyBaseRunner


class OffPolicyHARunner(OffPolicyBaseRunner):
    """Runner for off-policy HA algorithms."""

    def train(self):
        """Train the model"""
        self.total_it += 1
        data = self.buffer.sample()
        (
            sp_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            sp_obs,  # (n_agents, batch_size, dim)
            sp_actions,  # (n_agents, batch_size, dim)
            sp_available_actions,  # (n_agents, batch_size, dim)
            sp_reward,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_done,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_valid_transition,  # (n_agents, batch_size, 1)
            sp_term,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_next_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            sp_next_obs,  # (n_agents, batch_size, dim)
            sp_next_available_actions,  # (n_agents, batch_size, dim)
            sp_gamma,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
        ) = data
        # Check if Transformer is enabled (needed for both critic and actor training)
        use_transformer = self.algo_args["model"].get("use_transformer", False)
        
        # train critic
        self.critic.turn_on_grad()
        if self.args["algo"] == "hasac":
            next_actions = []
            next_logp_actions = []
            for agent_id in range(self.num_agents):
                # Get actions with contrastive learning support
                result = self.actor[agent_id].get_actions_with_logprobs(
                    sp_next_obs[agent_id],
                    sp_next_available_actions[agent_id]
                    if sp_next_available_actions is not None
                    else None,
                    agent_id=agent_id
                )
                if len(result) == 3:  # Transformer-enhanced policy
                    next_action, next_logp_action, _ = result
                else:  # Standard policy
                    next_action, next_logp_action = result
                next_actions.append(next_action)
                next_logp_actions.append(next_logp_action)
            
            # Process observations through Transformer if enabled
            critic_share_obs = sp_share_obs
            critic_next_share_obs = sp_next_share_obs
            
            if use_transformer:
                # Get context embeddings from Transformer actors
                with torch.no_grad():
                    # Process current shared observations (85 dimensions -> 128 dimensions)
                    if hasattr(self.actor[0].actor, '_get_context_embedding'):
                        # Use the first agent's Transformer to process shared observation
                        # Convert shared observation to match individual observation format for Transformer
                        shared_obs_tensor = torch.from_numpy(sp_share_obs).float().to(self.device)
                        if shared_obs_tensor.dim() == 1:
                            shared_obs_tensor = shared_obs_tensor.unsqueeze(0)
                        
                        # Process through Transformer (truncate to match individual obs size if needed)
                        obs_dim = self.actor[0].actor.obs_dim
                        if shared_obs_tensor.size(-1) > obs_dim:
                            # Truncate shared observation to match individual observation size
                            truncated_obs = shared_obs_tensor[..., :obs_dim]
                        else:
                            truncated_obs = shared_obs_tensor
                        
                        context_emb, _ = self.actor[0].actor._get_context_embedding(
                            truncated_obs, None, 0
                        )
                        
                        if context_emb.dim() == 1:
                            context_emb = context_emb.unsqueeze(0)
                        critic_share_obs = context_emb.cpu().numpy()
                    
                    # Process next shared observations
                    if hasattr(self.actor[0].actor, '_get_context_embedding'):
                        next_shared_obs_tensor = torch.from_numpy(sp_next_share_obs).float().to(self.device)
                        if next_shared_obs_tensor.dim() == 1:
                            next_shared_obs_tensor = next_shared_obs_tensor.unsqueeze(0)
                        
                        # Process through Transformer (truncate to match individual obs size if needed)
                        if next_shared_obs_tensor.size(-1) > obs_dim:
                            # Truncate shared observation to match individual observation size
                            truncated_next_obs = next_shared_obs_tensor[..., :obs_dim]
                        else:
                            truncated_next_obs = next_shared_obs_tensor
                        
                        next_context_emb, _ = self.actor[0].actor._get_context_embedding(
                            truncated_next_obs, None, 0
                        )
                        
                        if next_context_emb.dim() == 1:
                            next_context_emb = next_context_emb.unsqueeze(0)
                        critic_next_share_obs = next_context_emb.cpu().numpy()

            
            self.critic.train(
                critic_share_obs,
                sp_actions,
                sp_reward,
                sp_done,
                sp_valid_transition,
                sp_term,
                critic_next_share_obs,
                next_actions,
                next_logp_actions,
                sp_gamma,
                self.value_normalizer,
            )
        else:
            next_actions = []
            for agent_id in range(self.num_agents):
                next_actions.append(
                    self.actor[agent_id].get_target_actions(sp_next_obs[agent_id])
                )
            self.critic.train(
                sp_share_obs,
                sp_actions,
                sp_reward,
                sp_done,
                sp_term,
                sp_next_share_obs,
                next_actions,
                sp_gamma,
            )
        self.critic.turn_off_grad()
        sp_valid_transition = torch.tensor(sp_valid_transition, device=self.device)
        if self.total_it % self.policy_freq == 0:
            # train actors
            if self.args["algo"] == "hasac":
                actions = []
                logp_actions = []
                contrastive_infos = []
                with torch.no_grad():
                    for agent_id in range(self.num_agents):
                        result = self.actor[agent_id].get_actions_with_logprobs(
                            sp_obs[agent_id],
                            sp_available_actions[agent_id]
                            if sp_available_actions is not None
                            else None,
                            agent_id=agent_id
                        )
                        if len(result) == 3:  # Transformer-enhanced policy
                            action, logp_action, contrastive_info = result
                            contrastive_infos.append(contrastive_info)
                        else:  # Standard policy
                            action, logp_action = result
                            contrastive_infos.append(None)
                        actions.append(action)
                        logp_actions.append(logp_action)
                # actions shape: (n_agents, batch_size, dim)
                # logp_actions shape: (n_agents, batch_size, 1)
                if self.fixed_order:
                    agent_order = list(range(self.num_agents))
                else:
                    agent_order = list(np.random.permutation(self.num_agents))
                for agent_id in agent_order:
                    self.actor[agent_id].turn_on_grad()
                    # train this agent
                    result = self.actor[agent_id].get_actions_with_logprobs(
                        sp_obs[agent_id],
                        sp_available_actions[agent_id]
                        if sp_available_actions is not None
                        else None,
                        agent_id=agent_id
                    )
                    if len(result) == 3:  # Transformer-enhanced policy
                        actions[agent_id], logp_actions[agent_id], contrastive_info = result
                        contrastive_infos[agent_id] = contrastive_info
                    else:  # Standard policy
                        actions[agent_id], logp_actions[agent_id] = result
                    if self.state_type == "EP":
                        logp_action = logp_actions[agent_id]
                        actions_t = torch.cat(actions, dim=-1)
                    elif self.state_type == "FP":
                        logp_action = torch.tile(
                            logp_actions[agent_id], (self.num_agents, 1)
                        )
                        actions_t = torch.tile(
                            torch.cat(actions, dim=-1), (self.num_agents, 1)
                        )
                    
                    # Get context embeddings for critic if Transformer is enabled
                    critic_obs_for_values = sp_share_obs
                    if use_transformer:
                        with torch.no_grad():
                            # Process shared observation through Transformer
                            if hasattr(self.actor[agent_id].actor, '_get_context_embedding'):
                                shared_obs_tensor = torch.from_numpy(sp_share_obs).float().to(self.device)
                                if shared_obs_tensor.dim() == 1:
                                    shared_obs_tensor = shared_obs_tensor.unsqueeze(0)
                                
                                # Process through Transformer (truncate to match individual obs size if needed)
                                obs_dim = self.actor[agent_id].actor.obs_dim
                                if shared_obs_tensor.size(-1) > obs_dim:
                                    # Truncate shared observation to match individual observation size
                                    truncated_obs = shared_obs_tensor[..., :obs_dim]
                                else:
                                    truncated_obs = shared_obs_tensor
                                
                                context_emb, _ = self.actor[agent_id].actor._get_context_embedding(
                                    truncated_obs, None, agent_id
                                )
                                
                                if context_emb.dim() == 1:
                                    context_emb = context_emb.unsqueeze(0)
                                critic_obs_for_values = context_emb.cpu().numpy()
                    
                    value_pred = self.critic.get_values(critic_obs_for_values, actions_t)
                    if self.algo_args["algo"]["use_policy_active_masks"]:
                        if self.state_type == "EP":
                            actor_loss = (
                                -torch.sum(
                                    (value_pred - self.alpha[agent_id] * logp_action)
                                    * sp_valid_transition[agent_id]
                                )
                                / sp_valid_transition[agent_id].sum()
                            )
                        elif self.state_type == "FP":
                            valid_transition = torch.tile(
                                sp_valid_transition[agent_id], (self.num_agents, 1)
                            )
                            actor_loss = (
                                -torch.sum(
                                    (value_pred - self.alpha[agent_id] * logp_action)
                                    * valid_transition
                                )
                                / valid_transition.sum()
                            )
                    else:
                        actor_loss = -torch.mean(
                            value_pred - self.alpha[agent_id] * logp_action
                        )
                    
                    # Add contrastive learning loss if using Transformer
                    contrastive_loss = torch.tensor(0.0, device=self.device)
                    if contrastive_infos[agent_id] is not None:
                        contrastive_loss = self.actor[agent_id].compute_contrastive_loss(
                            contrastive_infos[agent_id]
                        )
                        # Get lambda_cl from args, default to 0.1
                        lambda_cl = self.algo_args.get("lambda_cl", 0.1)
                        actor_loss = actor_loss + lambda_cl * contrastive_loss
                    
                    self.actor[agent_id].actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor[agent_id].actor_optimizer.step()
                    self.actor[agent_id].turn_off_grad()
                    # train this agent's alpha
                    if self.algo_args["algo"]["auto_alpha"]:
                        log_prob = (
                            logp_actions[agent_id].detach()
                            + self.target_entropy[agent_id]
                        )
                        alpha_loss = -(self.log_alpha[agent_id] * log_prob).mean()
                        self.alpha_optimizer[agent_id].zero_grad()
                        alpha_loss.backward()
                        self.alpha_optimizer[agent_id].step()
                        self.alpha[agent_id] = torch.exp(
                            self.log_alpha[agent_id].detach()
                        )
                    result = self.actor[agent_id].get_actions_with_logprobs(
                        sp_obs[agent_id],
                        sp_available_actions[agent_id]
                        if sp_available_actions is not None
                        else None,
                        agent_id=agent_id
                    )
                    if len(result) == 3:  # Transformer-enhanced policy
                        actions[agent_id], _, _ = result
                    else:  # Standard policy
                        actions[agent_id], _ = result
                # train critic's alpha
                if self.algo_args["algo"]["auto_alpha"]:
                    self.critic.update_alpha(logp_actions, np.sum(self.target_entropy))
            else:
                if self.args["algo"] == "had3qn":
                    actions = []
                    with torch.no_grad():
                        for agent_id in range(self.num_agents):
                            actions.append(
                                self.actor[agent_id].get_actions(
                                    sp_obs[agent_id], False
                                )
                            )
                    # actions shape: (n_agents, batch_size, 1)
                    update_actions, get_values = self.critic.train_values(
                        sp_share_obs, actions
                    )
                    if self.fixed_order:
                        agent_order = list(range(self.num_agents))
                    else:
                        agent_order = list(np.random.permutation(self.num_agents))
                    for agent_id in agent_order:
                        self.actor[agent_id].turn_on_grad()
                        # actor preds
                        actor_values = self.actor[agent_id].train_values(
                            sp_obs[agent_id], actions[agent_id]
                        )
                        # critic preds
                        critic_values = get_values()
                        # update
                        actor_loss = torch.mean(F.mse_loss(actor_values, critic_values))
                        self.actor[agent_id].actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor[agent_id].actor_optimizer.step()
                        self.actor[agent_id].turn_off_grad()
                        update_actions(agent_id)
                else:
                    actions = []
                    with torch.no_grad():
                        for agent_id in range(self.num_agents):
                            actions.append(
                                self.actor[agent_id].get_actions(
                                    sp_obs[agent_id], False
                                )
                            )
                    # actions shape: (n_agents, batch_size, dim)
                    if self.fixed_order:
                        agent_order = list(range(self.num_agents))
                    else:
                        agent_order = list(np.random.permutation(self.num_agents))
                    for agent_id in agent_order:
                        self.actor[agent_id].turn_on_grad()
                        # train this agent
                        actions[agent_id] = self.actor[agent_id].get_actions(
                            sp_obs[agent_id], False
                        )
                        actions_t = torch.cat(actions, dim=-1)
                        value_pred = self.critic.get_values(sp_share_obs, actions_t)
                        actor_loss = -torch.mean(value_pred)
                        self.actor[agent_id].actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor[agent_id].actor_optimizer.step()
                        self.actor[agent_id].turn_off_grad()
                        actions[agent_id] = self.actor[agent_id].get_actions(
                            sp_obs[agent_id], False
                        )
                # soft update
                for agent_id in range(self.num_agents):
                    self.actor[agent_id].soft_update()
            self.critic.soft_update()
