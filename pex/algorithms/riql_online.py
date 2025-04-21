import torch

from ..utils.util import DEFAULT_DEVICE, extract_sub_dict
from .riql import RIQL

class RIQL_online(RIQL):
    def __init__(self, critic, vf, policy,online_policy, optimizer_ctor,
                 tau, beta,sigma, quantile, discount, target_update_rate, ckpt_path, 
                 copy_to_target=False):
        super().__init__(critic=critic, vf=vf, policy=policy,online_policy=online_policy,
                         optimizer_ctor=optimizer_ctor,
                         max_steps=None,
                         tau=tau, beta=beta,sigma=sigma, quantile=quantile,
                         discount=discount,
                         target_update_rate=target_update_rate,
                         use_lr_scheduler=False)

        # load checkpoint if ckpt_path is not None
        if ckpt_path is not None:

            map_location = DEFAULT_DEVICE
            if not torch.cuda.is_available():
                map_location = torch.device('cpu')
            checkpoint = torch.load(ckpt_path, map_location=map_location)

            # extract sub-dictionary
            policy_state_dict = extract_sub_dict("policy", checkpoint)
            critic_state_dict = extract_sub_dict("critic", checkpoint)

            self.policy.load_state_dict(policy_state_dict)
            self.critic.load_state_dict(critic_state_dict)
            if copy_to_target:
                self.target_critic.load_state_dict(critic_state_dict)
            else:
                target_critic_state_dict = extract_sub_dict("target_critic", checkpoint)
                self.target_critic.load_state_dict(target_critic_state_dict)

            self.vf.load_state_dict(extract_sub_dict("vf", checkpoint))
            torch.cuda.empty_cache()

    def select_action(self, state, evaluate=False):
        policy_out = self.policy(state)
        
        if evaluate is False:
            if isinstance(policy_out, torch.distributions.Distribution):
                action_sample, _, _ = self.policy.sample(state)
            elif torch.is_tensor(policy_out):
                action_sample = self.policy.act(state)+ torch.randn_like(self.policy.act(state))*0.01

            return action_sample
        else:
            if isinstance(policy_out, torch.distributions.Distribution):
                _, _, action_mode = self.policy.sample(state)
            elif torch.is_tensor(policy_out):
                action_mode = self.policy.act(state)
            
            return action_mode            


