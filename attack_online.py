from pathlib import Path

import gym
import d4rl
import numpy as np
import itertools
import os
import torch
from tqdm import trange
import uuid
from pex.algorithms.pex import PEX
from pex.algorithms.riql_pex import RIQLPEX
from pex.algorithms.iql_online import IQL_online
from pex.algorithms.riql_pex import RIQL
from pex.algorithms.riql_online import RIQL_online
from pex.algorithms.rpex import RPEX,RPEX2
from pex.networks.policy import GaussianPolicy, DeterministicPolicy
from pex.networks.value_functions import DoubleCriticNetwork, ValueNetwork,QFunction,VectorizedQ
from pex.utils.util import (
    set_seed, ReplayMemory, torchify, torchify, DEFAULT_DEVICE,
    get_batch_from_dataset_and_buffer,
    eval_policy, set_default_device, get_env_and_dataset,torchify,compute_mean_std,normalize_states,wrap_env)

from attack import attack_dataset, corrupt_trans
from RIQL_TRAIN_CONFIG import get_config
import logging
import sys 
from rich.pretty import pretty_repr
import json
import wandb 

use_wandb = False

def main(args):
    # torch.set_num_threads(1)
    
    name = f"attack-online-{args.algorithm.upper()}-{args.seed}-{str(uuid.uuid4())[:4]}"
    args.log_dir = os.path.join(args.log_dir, args.env_name, name)
    # if os.path.exists(args.log_dir):
    #     print(f"The directory {args.log_dir} exists. Please specify a different one.")
    #     return
    # else:
    #     print(f"Creating directory {args.log_dir}")
    #     # os.mkdir(args.log_dir)
    os.makedirs(args.log_dir, exist_ok=True)
    if use_wandb:
        wandb.init(
        config=args,
        project="ours_pex",
        group=args.algorithm.upper(),
        name=name,
        id=str(uuid.uuid4()),
    )
    
    logging.root.handlers = []
    logging.basicConfig(
                format='%(asctime)s %(levelname)s-%(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[
        logging.FileHandler(os.path.join(args.log_dir,'result.log')),
        logging.StreamHandler()
    ],level=logging.INFO,
                )
    

    
    # args.device = DEFAULT_DEVICE
    env, dataset, reward_transformer = get_env_and_dataset(args.env_name, args.max_episode_steps)
    dataset_size = dataset['observations'].shape[0]
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]
    '''
    Corrupt offline dataset by introducing adversarial modifications to the next observations
    '''
    if (args.corrupt_reward or args.corrupt_dynamics or args.corrupt_obs or args.corrupt_acts):
        # Initialize an array to track attacked data points
        attack_indexes = np.zeros(dataset["rewards"].shape)
        # Apply the attack function to corrupt the dataset
        dataset, indexes, std = attack_dataset(args, dataset)
        # Mark the attacked indexes in the tracking array
        attack_indexes[indexes] = 1.0
    
    if args.normalize_states:
        state_mean, state_std = compute_mean_std(np.concatenate([dataset["observations"], dataset['next_observations']], axis=0), eps=1e-3)
    else:
        state_mean, state_std = 0, 1
        
    dataset['observations'] = normalize_states(dataset['observations'], state_mean, state_std)
    dataset['next_observations'] = normalize_states(dataset['next_observations'], state_mean, state_std)
    
    max_steps = env._max_episode_steps
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    for k, v in dataset.items():
        dataset[k] = torchify(v)
    

    if args.seed is not None:
        set_seed(args.seed, env=env)

    # if torch.cuda.is_available():
    #     set_default_device()

    action_space = env.action_space
    if args.deterministic_policy:
        policy = DeterministicPolicy(obs_dim, act_dim, max_action=float(env.action_space.high[0]))
        online_policy = GaussianPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num, action_space=action_space, scale_distribution=False, state_dependent_std=False)
    
    else:
        policy = GaussianPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num, action_space=action_space, scale_distribution=False, state_dependent_std=False)
    
    algorithm_option = args.algorithm.upper()

    if algorithm_option == "IQLDIRECT":
        with open(os.path.join(args.log_dir,f'{args.env_name}.json'), 'wt') as f:
            json.dump(vars(args), f, indent=4)
        double_buffer = True
        alg = IQL_online(
            critic=DoubleCriticNetwork(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            vf=ValueNetwork(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            policy=policy,
            online_policy=online_policy if args.deterministic_policy else policy,
            optimizer_ctor=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
            tau=args.tau,
            beta=args.beta,
            target_update_rate=args.target_update_rate,
            discount=args.discount,
            ckpt_path=args.ckpt_path
        )

    elif algorithm_option == "HYBRIDIQL":
        with open(os.path.join(args.log_dir,f'{args.env_name}.json'), 'wt') as f:
            json.dump(vars(args), f, indent=4)        
        double_buffer = True
        alg = IQL_online(
            critic=DoubleCriticNetwork(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            vf=ValueNetwork(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            policy=policy,
            online_policy=online_policy if args.deterministic_policy else policy,
            optimizer_ctor=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
            tau=args.tau,
            beta=args.beta,
            target_update_rate=args.target_update_rate,
            discount=args.discount,
            ckpt_path=None
        )

    elif algorithm_option == "PEX":
        with open(os.path.join(args.log_dir,f'{args.env_name}.json'), 'wt') as f:
            json.dump(vars(args), f, indent=4)        
        double_buffer = True
        assert args.ckpt_path, "need to provide a valid checkpoint path"
        alg = PEX(
            critic=DoubleCriticNetwork(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            vf=ValueNetwork(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            policy=policy,
            online_policy=online_policy if args.deterministic_policy else policy,
            optimizer_ctor=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
            tau=args.tau,
            beta=args.beta,
            target_update_rate=args.target_update_rate,
            discount=args.discount,
            ckpt_path=args.ckpt_path,
            inv_temperature=args.inv_temperature,
        )
        
    elif algorithm_option == "RPEXIQL":
        with open(os.path.join(args.log_dir,f'{args.env_name}.json'), 'wt') as f:
            json.dump(vars(args), f, indent=4)        
        double_buffer = True
        assert args.ckpt_path, "need to provide a valid checkpoint path"
        alg = RPEX2(
            critic=DoubleCriticNetwork(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),      
            vf=ValueNetwork(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            policy=policy,
            online_policy=online_policy if args.deterministic_policy else policy,
            optimizer_ctor=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
            tau=args.tau,
            beta=args.beta,
            target_update_rate=args.target_update_rate,
            discount=args.discount,
            ckpt_path=args.ckpt_path,
            inv_temperature=args.inv_temperature,
            kappa = args.kappa,
        )        
        
    elif algorithm_option == "RIQLDIRECT":  # riql direct
        args = get_config(args)
        double_buffer = True
        with open(os.path.join(args.log_dir,f'{args.env_name}.json'), 'wt') as f:
            json.dump(vars(args), f, indent=4)        
        assert args.ckpt_path, "need to provide a valid checkpoint path"
        alg = RIQL_online(
            critic=VectorizedQ(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num, num_critics=args.num_critics),
            vf=ValueNetwork(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            policy=policy,
            online_policy=online_policy if args.deterministic_policy else policy,
            optimizer_ctor=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
            tau=args.tau,
            beta=args.beta,
            sigma=args.sigma,
            quantile=args.quantile,
            target_update_rate=args.target_update_rate,
            discount=args.discount,
            ckpt_path=args.ckpt_path,
        )
        
    elif algorithm_option == "RIQL": # riql pex
        args = get_config(args)
        double_buffer = True
        with open(os.path.join(args.log_dir,f'{args.env_name}.json'), 'wt') as f:
            json.dump(vars(args), f, indent=4)        
        assert args.ckpt_path, "need to provide a valid checkpoint path"
        alg = RIQLPEX(
            critic=VectorizedQ(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num, num_critics=args.num_critics),
            vf=ValueNetwork(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            policy=policy,
            online_policy=online_policy if args.deterministic_policy else policy,
            optimizer_ctor=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
            tau=args.tau,
            beta=args.beta,
            sigma=args.sigma,
            quantile=args.quantile,
            target_update_rate=args.target_update_rate,
            discount=args.discount,
            ckpt_path=args.ckpt_path,
            inv_temperature=args.inv_temperature,
        )
    
    elif algorithm_option == "RPEX":
        args = get_config(args)
        double_buffer = True
        with open(os.path.join(args.log_dir,f'{args.env_name}.json'), 'wt') as f:
            json.dump(vars(args), f, indent=4)        
        assert args.ckpt_path, "need to provide a valid checkpoint path"
        alg = RPEX(
            critic=VectorizedQ(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num, num_critics=args.num_critics),
            vf=ValueNetwork(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            policy=policy,
            online_policy=online_policy if args.deterministic_policy else policy,
            optimizer_ctor=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
            tau=args.tau,
            beta=args.beta,
            sigma=args.sigma,
            quantile=args.quantile,
            target_update_rate=args.target_update_rate,
            discount=args.discount,
            ckpt_path=args.ckpt_path,
            inv_temperature=args.inv_temperature,
            kappa = args.kappa,
        )



    memory = ReplayMemory(args.replay_size, args.seed)

    total_numsteps = 0

    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        while not done:

            action = alg.select_action(torchify(state).to(DEFAULT_DEVICE)).detach().cpu().numpy()
            
            if len(memory) > args.initial_collection_steps:
                for i in range(args.updates_per_step):
                    if use_wandb:
                        log_dict = alg.update(*get_batch_from_dataset_and_buffer(dataset, memory, args.batch_size, double_buffer))
                        wandb.log({"timesteps": total_numsteps, **log_dict})
                    else:
                        alg.update(*get_batch_from_dataset_and_buffer(dataset, memory, args.batch_size, double_buffer))

            next_state, reward, done, _ = env.step(action)
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            reward_for_replay = reward_transformer(reward)


            terminal = 0 if episode_steps == max_steps else float(done)
            
            # corruprt data
            attacked_next_state = next_state
            if args.corrupt_reward:
                reward_for_replay,_ = corrupt_trans(reward_for_replay, std,corrupt_reward=True)
            if args.corrupt_dynamics:
                attacked_next_state,_ = corrupt_trans(next_state, 1) if args.normalize_states else corrupt_trans(next_state, std)
            if args.corrupt_obs:
                state,_ = corrupt_trans(state, 1) if args.normalize_states else corrupt_trans(next_state, std)
            if args.corrupt_acts:
                action,_ = corrupt_trans(action, std)

            # memory.push(state, action, reward_for_replay, next_state, terminal)
            memory.push(state, action, reward_for_replay, attacked_next_state, terminal)            
            state = next_state
            # state = attacked_next_state

            if total_numsteps % args.eval_period == 0 and args.eval is True:

                logging.info("Episode: {}, total env-steps: {}".format(i_episode, total_numsteps))
                eval_log = eval_policy(env, args.env_name, alg, args.max_episode_steps, args.eval_episode_num)
                if use_wandb:
                    wandb.log({"timesteps": total_numsteps, **eval_log})
                logging.info(pretty_repr(eval_log))

        if total_numsteps > args.total_env_steps:
            break

        env.close()

    torch.save(alg.state_dict(), args.log_dir + '/{}_online_ckpt'.format(args.algorithm))

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # general parameters
    parser.add_argument('--algorithm', required=True)  # ['direct', 'buffer', 'pex']
    parser.add_argument('--device_number', type=int, default=0)
    parser.add_argument('--env_name', required=True)
    parser.add_argument('--log_dir',default='./rpex_results')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--hidden_num', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--target_update_rate', type=float, default=0.005)
    parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--eval_period', type=int, default=10000)
    parser.add_argument('--eval_episode_num', type=int, default=10,
                        help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--max_episode_steps', type=int, default=1000)
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--total_env_steps', type=int, default=1000001, metavar='N',
                        help='total number of env steps (default: 1000000)')
    parser.add_argument('--initial_collection_steps', type=int, default=5000, metavar='N',
                        help='Initial environmental steps before training starts (default: 5000)')
    parser.add_argument('--updates_per_step', type=int, default=4, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--ckpt_path', default='./riql_results/halfcheetah-medium-replay-v2/offline-riql-attack-0-8da3/offline_ckpt',
                help='path to the offline checkpoint')
    
    # policy parameters
    parser.add_argument('--normalize_states', action='store_true', default=False)
    parser.add_argument('--deterministic_policy', action='store_true', default=False)
    # iql parameters
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=3.0,
                        help='IQL inverse temperature')
    parser.add_argument('--rho', type=float, default=0.5) # dro parameter
    # riql parameters
    parser.add_argument('--sigma', type=float, default=3.0) 
    parser.add_argument('--quantile', type=float, default=0.1) 
    parser.add_argument('--num_critics', type=int, default=5)  
    # rpex parameters
    parser.add_argument('--inv_temperature', type=float, default=3, metavar='G',
                        help='inverse temperature for PEX action selection (default: 10)')
    parser.add_argument('--kappa', type=float, default=0.1)
    # attack parameters
    parser.add_argument('--corrupt_reward', action='store_true', default=False)
    parser.add_argument('--corrupt_dynamics', action='store_true', default=False)
    parser.add_argument('--corrupt_acts', action='store_true', default=False)
    parser.add_argument('--corrupt_obs', action='store_true', default=False)
    parser.add_argument('--corruption_mode', type=str, default='random')
    parser.add_argument('--corruption_range', type=float, default=1)
    parser.add_argument('--corruption_rate', type=float, default=0.3)
    

    args = parser.parse_args()
    torch.cuda.set_device(args.device_number)
    args.device = f"cuda:{args.device_number}" if torch.cuda.is_available() else "cpu"
    main(args)
