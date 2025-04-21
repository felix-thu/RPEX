import copy
import torch

from pex.utils.util import (DEFAULT_DEVICE, epsilon_greedy_sample,
                            extract_sub_dict)
from pex.algorithms.iql import IQL, EXP_ADV_MAX,expectile_loss
from pex.algorithms.riql import RIQL
def huber_loss(diff, sigma=1):
    beta = 1. / (sigma ** 2)
    diff = torch.abs(diff)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return loss

class RIQLPEX(RIQL):
    def __init__(self, critic, vf, policy,online_policy, optimizer_ctor,
                 tau, beta,sigma, quantile, discount, target_update_rate, ckpt_path, inv_temperature,
                 copy_to_target=False):
        super().__init__(critic=critic, vf=vf, policy=policy,online_policy=online_policy,
                         optimizer_ctor=optimizer_ctor,
                         max_steps=None,
                         tau=tau, beta=beta,sigma=sigma, quantile=quantile,
                         discount=discount,
                         target_update_rate=target_update_rate,
                         use_lr_scheduler=False)

        self.policy_offline = copy.deepcopy(self.policy).to(DEFAULT_DEVICE)
        self.policy = self.online_policy.to(DEFAULT_DEVICE)

        self._inv_temperature = inv_temperature

        # load checkpoint if ckpt_path is not None
        if ckpt_path is not None:

            map_location = DEFAULT_DEVICE
            if not torch.cuda.is_available():
                map_location = torch.device('cpu')
            checkpoint = torch.load(ckpt_path, map_location=map_location)

            # extract sub-dictionary
            policy_state_dict = extract_sub_dict("policy", checkpoint)
            critic_state_dict = extract_sub_dict("critic", checkpoint)

            self.policy_offline.load_state_dict(policy_state_dict)
            self.critic.load_state_dict(critic_state_dict)
            self.vf.load_state_dict(extract_sub_dict("vf", checkpoint))

            if copy_to_target:
                self.target_critic.load_state_dict(critic_state_dict)
            else:
                target_critic_state_dict = extract_sub_dict("target_critic", checkpoint)
                self.target_critic.load_state_dict(target_critic_state_dict)
            torch.cuda.empty_cache()


    def select_action(self, observations, evaluate=False, return_all_actions=False):
        if len(observations.shape) == 1:
            observations = observations.unsqueeze(0)
        a1 = self.policy_offline.act(observations, deterministic=True)

        dist = self.policy(observations)
        if evaluate:
            a2 = epsilon_greedy_sample(dist, eps=0.1)
        else:
            a2 = epsilon_greedy_sample(dist, eps=1.0)

        q1 = self.critic(observations, a1)
        q2 = self.critic(observations, a2)
        # print(q1.shape,q2.shape)
        q1 = torch.quantile(q1, self.quantile, dim=0)
        q2 = torch.quantile(q2, self.quantile, dim=0)
        # print(q1.shape,q2.shape)
        

        q = torch.stack([q1,q2], dim=-1)
        # print(q.shape)
        logits = q * self._inv_temperature
        w_dist = torch.distributions.Categorical(logits=logits)

        if evaluate:
            w = epsilon_greedy_sample(w_dist, eps=0.1)
        else:
            w = epsilon_greedy_sample(w_dist, eps=1.0)

        w = w.unsqueeze(-1)
        action = (1 - w) * a1 + w * a2
        
        # print(action.squeeze(0).shape)

        if not return_all_actions:
            return action.squeeze(0)
        else:
            return action.squeeze(0), a1.squeeze(0), a2.squeeze(0)



    def policy_update(self, observations, adv, actions):
        actions = self.select_action(observations)
        # print(actions.shape)
        with torch.no_grad():
            target_q_all = self.target_critic(observations, actions)
            target_q = torch.quantile(target_q_all.detach(),self.quantile,dim=0)
        v = self.vf(observations)
        adv = target_q.detach() - v
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.policy(observations)
        
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions.detach())
        elif torch.is_tensor(policy_out):
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=-1)
        else:
            raise NotImplementedError


        policy_loss = torch.mean(exp_adv * bc_losses)
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        if self.use_lr_scheduler:
            self.policy_lr_schedule.step()