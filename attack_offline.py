import os
from pathlib import Path
import torch
from tqdm import trange
import uuid
from pex.algorithms.iql import IQL
from pex.algorithms.aligniql import ALIGNIQL
from pex.algorithms.riql import RIQL
from pex.networks.policy import GaussianPolicy, DeterministicPolicy
from pex.networks.value_functions import DoubleCriticNetwork, ValueNetwork,QFunction,VectorizedQ
from pex.utils.util import (
    set_seed, DEFAULT_DEVICE, sample_batch,
    eval_policy, set_default_device, get_env_and_dataset,torchify,compute_mean_std,normalize_states,wrap_env)
from attack import attack_dataset, corrupt_trans
from RIQL_TRAIN_CONFIG import get_config
import logging
import sys 
from rich.pretty import pretty_repr
import numpy as np
import json
import wandb

use_wandb = False
def main(args):
    # torch.set_num_threads(1)
    name = f"offline-{args.algorithm.upper()}-attack-{args.seed}-{str(uuid.uuid4())[:4]}"
    args.log_dir = os.path.join(args.log_dir, args.env_name, name)

    os.makedirs(args.log_dir, exist_ok=True)
    
    if use_wandb:
        wandb.init(
        config=args,
        project="aligniql_icml_robust",
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


    env, dataset, _ = get_env_and_dataset(args.env_name, args.max_episode_steps)
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]
    # args.device = DEFAULT_DEVICE
    ##### corrupt offline dataset
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
           
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    for k, v in dataset.items():
        dataset[k] = torchify(v)
    
    if args.seed is not None:
        set_seed(args.seed, env=env)

    if torch.cuda.is_available():
        set_default_device()
        

    action_space = env.action_space
    if args.deterministic_policy:
        policy = DeterministicPolicy(obs_dim, act_dim, max_action=float(env.action_space.high[0]))
    else:
        policy = GaussianPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num, action_space=action_space, scale_distribution=False, state_dependent_std=False)
    
    algorithm_option = args.algorithm.upper()
    if algorithm_option == "RIQL":
        args = get_config(args)
        with open(os.path.join(args.log_dir,f'{args.env_name}.json'), 'wt') as f:
            json.dump(vars(args), f, indent=4)
        iql = RIQL(
        critic=VectorizedQ(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num, num_critics=args.num_critics),
        # dro_net = QFunction(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
        vf=ValueNetwork(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
        policy=policy,
        optimizer_ctor=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
        max_steps=args.num_steps,
        tau=args.tau,
        beta=args.beta,
        sigma=args.sigma,
        quantile=args.quantile,
        # rho  = args.rho,
        target_update_rate=args.target_update_rate,
        discount=args.discount
        )
    elif algorithm_option == "IQL":
        with open(os.path.join(args.log_dir,f'{args.env_name}.json'), 'wt') as f:
            json.dump(vars(args), f, indent=4)
        iql = IQL(
        critic=DoubleCriticNetwork(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
        # dro_net = QFunction(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
        vf=ValueNetwork(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
        policy=policy,
        optimizer_ctor=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
        max_steps=args.num_steps,
        tau=args.tau,
        beta=args.beta,
        # rho  = args.rho,
        target_update_rate=args.target_update_rate,
        discount=args.discount
        )
    elif algorithm_option == "ALIGNIQL":
        with open(os.path.join(args.log_dir,f'{args.env_name}.json'), 'wt') as f:
            json.dump(vars(args), f, indent=4)
        iql = ALIGNIQL(
        critic=DoubleCriticNetwork(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
        # dro_net = QFunction(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
        vf=ValueNetwork(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
        policy=policy,
        optimizer_ctor=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
        max_steps=args.num_steps,
        tau=args.tau,
        beta=args.beta,
        # rho  = args.rho,
        target_update_rate=args.target_update_rate,
        discount=args.discount
        )
    for step in trange(args.num_steps):
        if use_wandb:
            log_dict = iql.update(**sample_batch(dataset, args.batch_size))
            wandb.log({"timesteps": step, **log_dict})
        else:
            iql.update(**sample_batch(dataset, args.batch_size))
        if (step + 1) % args.eval_period == 0:
            eval_log = eval_policy(env, args.env_name, iql, args.max_episode_steps, args.eval_episode_num)
            if use_wandb:
                wandb.log({"timesteps": step, **eval_log})
            logging.info(pretty_repr(eval_log))

    torch.save(iql.state_dict(), args.log_dir + '/offline_ckpt')


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # general parameters
    parser.add_argument('--algorithm', required=True)  # ['direct', 'buffer', 'pex']
    parser.add_argument('--device_number', type=int, default=0)
    parser.add_argument('--env_name', required=True)
    parser.add_argument('--log_dir', default='./offline_results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--hidden_num', type=int, default=2)
    parser.add_argument('--num_steps', type=int, default=2000001, metavar='N',
                        help='maximum number of training steps (default: 1000000)')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--target_update_rate', type=float, default=0.005)
    parser.add_argument('--eval_period', type=int, default=10000)
    parser.add_argument('--eval_episode_num', type=int, default=10,
                        help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--max_episode_steps', type=int, default=1000)
    # algorithm parameters
    parser.add_argument('--normalize_states', action='store_true', default=False)
    parser.add_argument('--deterministic_policy', action='store_true', default=False)
    # iql parameters
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=3.0,
                        help='IQL inverse temperature')
    # only used in DRO
    parser.add_argument('--rho', type=float, default=0.5) 
    # only used in RIQL
    parser.add_argument('--sigma', type=float, default=3.0) 
    parser.add_argument('--quantile', type=float, default=0.1) 
    parser.add_argument('--num_critics', type=int, default=5) 

    
    # attack parameters
    parser.add_argument('--corrupt_reward', action='store_true', default=False)
    parser.add_argument('--corrupt_dynamics', action='store_true', default=False)
    parser.add_argument('--corrupt_acts', action='store_true', default=False)
    parser.add_argument('--corrupt_obs', action='store_true', default=False)
    parser.add_argument('--corruption_mode', type=str, default='random')
    parser.add_argument('--corruption_range', type=float, default=0.5)
    parser.add_argument('--corruption_rate', type=float, default=0.5)
    
    
    args = parser.parse_args()
    torch.cuda.set_device(args.device_number)
    args.device = f"cuda:{args.device_number}" if torch.cuda.is_available() else "cpu"
    main(args)