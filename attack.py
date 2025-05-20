from typing import Dict

import os
import gym
import d4rl
import torch
import torch.nn as nn
import numpy as np


NAME_DICT = {
    "obs": "observations",
    "act": "actions",
    "rew": "rewards",
    "next_obs": "next_observations",
}

MODEL_PATH = {
    "EDAC": "./pretrained_model/EDAC/EDAC_baseline_seed0-",  ### to be added
}

dataset_path = "./adversarial_data/"




def adversarial_attack(original, std, obs,act,corruption_tag,config,actor,critic,scale=1.0):
    original_torch = torch.from_numpy(original).to(config.device).view(1,-1)
    std_torch = torch.from_numpy(std).to(config.device).view(1,-1)
    obs_torch = torch.from_numpy(obs).to(config.device).view(1,-1)
    act_torch = torch.from_numpy(act).to(config.device).view(1,-1)
    # load model 
    # corruption_agent = 'EDAC'
    # model_path = MODEL_PATH[corruption_agent] + config.env_name + '/2999.pt'
    # state_dict = torch.load(model_path, map_location=config.device)
    
    # assert corruption_agent == "EDAC"
    # from EDAC import Actor, VectorizedCritic

    # actor = (
    #     Actor(
    #         config.state_dim,
    #         config.action_dim,
    #         hidden_dim=256,
    #         max_action=config.max_action,
    #     )
    #     .to(config.device)
    #     .eval()
    # )
    # critic = (
    #     VectorizedCritic(config.state_dim, config.action_dim, num_critics=10, hidden_dim=256)
    #     .to(config.device)
    #     .eval()
    # )
    # actor.load_state_dict(state_dict["actor"])
    # critic.load_state_dict(state_dict["critic"])
    # print(f"Load model from {model_path}")
    
    para =2* scale* std_torch * (torch.rand(original_torch.shape, generator=torch.Generator()).to(config.device) - 0.5)

    
    for _ in range(2):
        para = torch.nn.Parameter(para.clone(), requires_grad=True)
        optimizer = torch.optim.Adam([para], lr=0.1 * scale)
        if corruption_tag == 'observations':
            noised_obs = original_torch + para * std_torch
            qvalue = critic(noised_obs.float(), act_torch.float())
            loss = qvalue.mean()
        elif corruption_tag == 'actions':
            noised_act = original_torch + para * std_torch
            qvalue = critic(obs_torch.float(), noised_act.float())
            loss = qvalue.mean()
        elif corruption_tag == 'next_observations':
            noised_obs = original_torch + para * std_torch
            action = actor(noised_obs.float())
            # print(action)
            # print(noised_obs.shape)
            qvalue = critic(noised_obs.float(), action[0])
            loss = qvalue.mean()
        else:
            raise NotImplementedError
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        para = torch.clamp(para, -scale, scale).detach()
        
    para = para * std_torch
    # del actor
    # del critic
    # torch.cuda.empty_cache()
    noise = para.cpu().numpy()
    if corruption_tag == 'observations' or corruption_tag == 'next_observations':
        attack_data = noise + original_torch.cpu().numpy()
    elif corruption_tag == 'actions':
        attack_data = noise + original_torch.cpu().numpy()
    else:
        raise NotImplementedError
        
    return attack_data

class Attack:
    def __init__(
        self,
        env_name: str,
        agent_name: str,
        dataset: Dict[str, np.ndarray],
        model_path: str,
        dataset_path: str,
        update_times: int = 100,
        step_size: float = 0.01,
        force_attack: bool = False,
        resample_indexs: bool = False,
        seed: int = 2023,
        device: str = "cpu",
    ):
        self.env_name = env_name
        self.agent_name = agent_name
        self.dataset = dataset
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.update_times = update_times
        self.step_size = step_size
        self.force_attack = force_attack
        self.device = device
        self.resample_indexs = resample_indexs

        self._np_rng = np.random.RandomState(seed)
        self._th_rng = torch.Generator()
        self._th_rng.manual_seed(seed)

        self.attack_indexs = None
        self.original_indexs = None

        env = gym.make(env_name)
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])
        env.close()

    def set_attack_config(
        self,
        corruption_name,
        corruption_tag,
        corruption_rate,
        corruption_range,
        corruption_random,
    ):
        self.corruption_tag = NAME_DICT[corruption_tag]
        self.corruption_rate = corruption_rate
        self.corruption_range = corruption_range
        self.corruption_random = corruption_random
        self.new_dataset_path = os.path.expanduser(
            os.path.join(self.dataset_path, self.env_name)
        )
        attack_mode = "random" if self.corruption_random else "adversarial"
        self.new_dataset_file = f"{self.agent_name}_{attack_mode}_{corruption_name}range{corruption_range}_rate{corruption_rate}.pth"

        self.corrupt_func = getattr(self, f"corrupt_{corruption_tag}")
        self.loss_Q = getattr(self, f"loss_Q_for_{corruption_tag}")
        if self.attack_indexs is None or self.resample_indexs:
            self.attack_indexs, self.original_indexs = self.sample_indexs()

    def load_model(self):
        model_path = self.model_path
        state_dict = torch.load(model_path, map_location=self.device)
      
        assert self.agent_name == "EDAC"
        from EDAC import Actor, VectorizedCritic

        self.actor = (
            Actor(
                self.state_dim,
                self.action_dim,
                hidden_dim=256,
                max_action=self.max_action,
            )
            .to(self.device)
            .eval()
        )
        self.critic = (
            VectorizedCritic(self.state_dim, self.action_dim, num_critics=10, hidden_dim=256)
            .to(self.device)
            .eval()
        )
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        print(f"Load model from {model_path}")

    
    def sample_indexs(self):
        indexs = np.arange(len(self.dataset["rewards"]))
        # print(indexs)
        random_num = self._np_rng.random(len(indexs))
        # print(random_num)
        # print(random_number.shape())
        attacked = np.where(random_num < self.corruption_rate)[0]
        original = np.where(random_num >= self.corruption_rate)[0]
        return indexs[attacked], indexs[original]

    def sample_para(self, shape, std):
        return (
            2
            * self.corruption_range
            * std
            * (torch.rand(shape, generator=self._th_rng).to(self.device) - 0.5)
        )

    def sample_data(self, shape):
        return self._np_rng.uniform(-self.corruption_range, self.corruption_range, size=shape)


    def optimize_para(self, para, std, obs, act=None):
        for _ in range(self.update_times):
            para = torch.nn.Parameter(para.clone(), requires_grad=True)
            optimizer = torch.optim.Adam([para], lr=self.step_size * self.corruption_range)
            loss = self.loss_Q(para, obs, act, std)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            para = torch.clamp(para, -self.corruption_range, self.corruption_range).detach()
        return para * std

    def loss_Q_for_obs(self, para, observation, action, std):
        noised_obs = observation + para * std
        qvalue = self.critic(noised_obs, action)
        return qvalue.mean()

    def loss_Q_for_act(self, para, observation, action, std):
        noised_act = action + para * std
        qvalue = self.critic(observation, noised_act)
        return qvalue.mean()

    def loss_Q_for_next_obs(self, para, observation, action, std):
        noised_obs = observation + para * std
        action = self.actor(noised_obs)
        qvalue = self.critic(noised_obs, action)
        return qvalue.mean()

    def loss_Q_for_rew(self):
        # Just Placeholder
        raise NotImplementedError

    def split_gradient_attack(self, original_obs_torch, original_act_torch, std_torch):
        if self.corruption_tag == 'observations' or self.corruption_tag == 'next_observations':
            attack_data = np.zeros(original_obs_torch.shape)
        elif self.corruption_tag == 'actions':
            attack_data = np.zeros(original_act_torch.shape)
        else:
            raise NotImplementedError

        split = 10
        pointer = 0
        M = original_obs_torch.shape[0]
        for i in range(split):
            number = M // split if i < split - 1 else M - pointer
            temp_act = original_act_torch[pointer : pointer + number]
            temp_obs = original_obs_torch[pointer : pointer + number]

            if self.corruption_tag == 'observations' or self.corruption_tag == 'next_observations':
                para = self.sample_para(temp_obs.shape, std_torch)
            elif self.corruption_tag == 'actions':
                para = self.sample_para(temp_act.shape, std_torch)
            else:
                raise NotImplementedError

            para = self.optimize_para(para, std_torch, temp_obs, temp_act)
            noise = para.cpu().numpy()
            if self.corruption_tag == 'observations' or self.corruption_tag == 'next_observations':
                attack_data[pointer : pointer + number] = noise + temp_obs.cpu().numpy()
            elif self.corruption_tag == 'actions':
                attack_data[pointer : pointer + number] = noise + temp_act.cpu().numpy()
            else:
                raise NotImplementedError

            pointer += number
        return attack_data
    

    
    def corrupt_obs(self, dataset):
        # load original obs
        original_obs = self.dataset[self.corruption_tag][self.attack_indexs].copy()

        if self.corruption_random:
            std = np.std(self.dataset[self.corruption_tag], axis=0, keepdims=True)
            attack_obs = original_obs + self.sample_data(original_obs.shape) * std
            print(f"Random attack {self.corruption_tag}")
        else:
            self.load_model()
            original_act = self.dataset["actions"][self.attack_indexs].copy()
            original_act_torch = torch.from_numpy(original_act.copy()).to(self.device)
            original_obs_torch = torch.from_numpy(original_obs.copy()).to(self.device)
            std_torch = torch.from_numpy(self.dataset[self.corruption_tag].std(axis=0)).view(1, -1).to(self.device)

            # adversarial attack obs
            attack_obs = self.split_gradient_attack(original_obs_torch, original_act_torch, std_torch)

            self.clear_gpu_cache()
            print(f"Adversarial attack {self.corruption_tag}")
            self.save_dataset(attack_obs)

        dataset[self.corruption_tag][self.attack_indexs] = attack_obs
        return dataset,std

    def corrupt_act(self, dataset):
        # load original act
        original_act = self.dataset[self.corruption_tag][self.attack_indexs].copy()
        # print(original_act.shape)

        if self.corruption_random:
            std = np.std(self.dataset[self.corruption_tag], axis=0, keepdims=True)
            attack_act = original_act + self.sample_data(original_act.shape) * std
            print(f"Random attack {self.corruption_tag}")
        else:
            self.load_model()
        
            original_obs = self.dataset["observations"][self.attack_indexs].copy()
            original_obs_torch = torch.from_numpy(original_obs.copy()).to(self.device)
            original_act_torch = torch.from_numpy(original_act.copy()).to(self.device)
            std_torch = torch.from_numpy(self.dataset[self.corruption_tag].std(axis=0)).view(1, -1).to(self.device)

            # adversarial attack act
            attack_act = self.split_gradient_attack(original_obs_torch, original_act_torch, std_torch)

            self.clear_gpu_cache()
            print(f"Adversarial attack {self.corruption_tag}")
            self.save_dataset(attack_act)

        dataset[self.corruption_tag][self.attack_indexs] = attack_act
        return dataset,std

    def corrupt_rew(self, dataset):
        # load original rew
        original_rew = self.dataset[self.corruption_tag][self.attack_indexs].copy()

        if self.corruption_random:
            std = np.std(self.dataset[self.corruption_tag], axis=0, keepdims=True)
            attack_rew = self.sample_data(original_rew.shape) * 30
            print(f"Random attack {self.corruption_tag}")
        else:
            attack_rew = original_rew.copy() * -self.corruption_range
            print(f"Adversarial attack {self.corruption_tag}")

            self.save_dataset(attack_rew)
        dataset[self.corruption_tag][self.attack_indexs] = attack_rew
        return dataset,std

    def corrupt_next_obs(self, dataset):
        # load original obs
        original_obs = self.dataset[self.corruption_tag][self.attack_indexs].copy()
        # original_obs = self.dataset[self.corruption_tag][self.attack_indexs].clone()

        if self.corruption_random:
            std = np.std(self.dataset[self.corruption_tag], axis=0, keepdims=True)
            attack_obs = original_obs + self.sample_data(original_obs.shape) * std
            print(f"Random attack {self.corruption_tag}")
        else:
            self.load_model()
            original_act = self.dataset["actions"][self.attack_indexs].copy()
            original_act_torch = torch.from_numpy(original_act.copy()).to(self.device)
            original_obs_torch = torch.from_numpy(original_obs.copy()).to(self.device)
            std_torch = torch.from_numpy(self.dataset[self.corruption_tag].std(axis=0)).view(1, -1).to(self.device)

            # adversarial attack obs
            attack_obs = self.split_gradient_attack(original_obs_torch, original_act_torch, std_torch)

            self.clear_gpu_cache()
            print(f"Adversarial attack {self.corruption_tag}")

            self.save_dataset(attack_obs)

        dataset[self.corruption_tag][self.attack_indexs] = attack_obs
        return dataset,std

    def clear_gpu_cache(self):
        self.actor.to("cpu")
        self.critic.to("cpu")
        torch.cuda.empty_cache()

    def save_dataset(self, attack_datas):
        ### save data
        save_dict = {}
        save_dict["attack_indexs"] = self.attack_indexs
        save_dict["original_indexs"] = self.original_indexs
        save_dict[self.corruption_tag] = attack_datas
        if not os.path.exists(self.new_dataset_path):
            os.makedirs(self.new_dataset_path)
        dataset_path = os.path.join(self.new_dataset_path, self.new_dataset_file)
        torch.save(save_dict, dataset_path)
        print(f"Save attack dataset in {dataset_path}")

    def get_original_data(self, indexs):
        dataset = {}
        dataset["observations"] = self.dataset["observations"][indexs]
        dataset["actions"] = self.dataset["actions"][indexs]
        dataset["rewards"] = self.dataset["rewards"][indexs]
        dataset["next_observations"] = self.dataset["next_observations"][indexs]
        dataset["terminals"] = self.dataset["terminals"][indexs]
        return dataset

    def attack(self, dataset):
        dataset_path = os.path.join(self.new_dataset_path, self.new_dataset_file)
        if os.path.exists(dataset_path) and not self.force_attack:
            new_dataset = torch.load(dataset_path)
            print(f"Load new dataset from {dataset_path}")
            original_indexs, attack_indexs, attack_datas = (
                new_dataset["original_indexs"],
                new_dataset["attack_indexs"],
                new_dataset[self.corruption_tag],
            )
            ori_dataset = self.get_original_data(original_indexs)
            dataset[self.corruption_tag][attack_indexs] = attack_datas
            self.attack_indexs = attack_indexs
            return ori_dataset, dataset
        else:
            ori_dataset = self.get_original_data(self.original_indexs)
            att_dataset,std = self.corrupt_func(dataset)
            return ori_dataset, att_dataset,std


def attack_dataset(config, dataset, use_original=False): 
    corruption_agent = 'EDAC'
    attack_agent = Attack(
        env_name=config.env_name,
        agent_name=corruption_agent,
        dataset=dataset,
        model_path=MODEL_PATH[corruption_agent] + config.env_name + '/2999.pt',
        dataset_path=dataset_path, 
        resample_indexs=True,
        force_attack=False, 
        device=config.device,
        seed=config.seed,
    )
    corruption_random = config.corruption_mode == "random"
    attack_params = {
        "corruption_rate": config.corruption_rate,
        "corruption_range": config.corruption_range,
        "corruption_random": corruption_random,
    }
    name = ""
    #### the ori_dataset refers to the part of unattacked data
    ### the att_dataset refers to attacked data + unattacked data
    if config.corrupt_obs:
        name += "obs_"
        attack_agent.set_attack_config(name, "obs", **attack_params)
        ori_dataset, att_dataset,std = attack_agent.attack(dataset)
        dataset = ori_dataset if use_original else att_dataset

    if config.corrupt_acts:
        name += "act_"
        attack_agent.set_attack_config(name, "act", **attack_params)
        ori_dataset, att_dataset,std = attack_agent.attack(dataset)
        dataset = ori_dataset if use_original else att_dataset

    if config.corrupt_reward:
        name += "rew_"
        attack_agent.set_attack_config(name, "rew", **attack_params)
        ori_dataset, att_dataset,std = attack_agent.attack(dataset)
        dataset = ori_dataset if use_original else att_dataset

    if config.corrupt_dynamics:
        name += "next_obs_"
        attack_agent.set_attack_config(name, "next_obs", **attack_params)
        ori_dataset, att_dataset,std = attack_agent.attack(dataset)
        dataset = ori_dataset if use_original else att_dataset


    return dataset, attack_agent.attack_indexs,std

def corrupt_trans(data,std,obs,act,actor,critic,corruption_random=True,corrupt_reward=False,corruption_tag=None,config=None):
    # load original obs
    original_data = data.copy()
    scale = config.corruption_range
    corruption = np.random.uniform(0,1)
    is_corrupted = np.array(corruption < config.corruption_rate).astype(np.int32)  # Convert boolean to NumPy integer


    if corruption_random:
        # print("corruption_random")
        if corrupt_reward:
            # is_corrupted = np.array(corruption < 0.3).astype(np.int32)
            attack_data = original_data*(1-is_corrupted) + np.random.uniform(-1,1,size=original_data.shape)*is_corrupted*30
        else:
            attack_data = original_data + scale * np.random.uniform(-1,1,size=original_data.shape)*std*is_corrupted
    else:
        # print("corruption_adversarial")
        if corrupt_reward:
            attack_data = original_data*(1-is_corrupted) + np.random.uniform(-1,1,size=original_data.shape)*is_corrupted
        else:
            if is_corrupted == 1:
                attack_data = adversarial_attack(original_data,std,obs,act,corruption_tag,config,actor,critic,scale)
            else:
                attack_data = original_data

    return attack_data.squeeze(),is_corrupted