'''
Plot learning curves in multiple figures
'''

import re
import matplotlib.pyplot as plt # type: ignore
import numpy as np
import os
import pandas as pd
import json

def extract_results(log_file):
    results = []
    std = []
    with open(log_file, 'r') as file:
        for line in file:
            # 匹配 'normalized return mean' 后的数值
            match1 = re.search(r"'normalized return mean'\s*:\s*([-+]?[0-9]*\.?[0-9]+)", line)
            match2 = re.search(r"'normalized return std'\s*:\s*([-+]?[0-9]*\.?[0-9]+)", line)
            if match1:
                results.append(float(match1.group(1)))
            if match2:
                std.append(float(match2.group(1)))
    return np.array(results), np.array(std)

def extract_info(directory):
    info = []
    for root, dirs, files in os.walk(directory):    
        for file in files:
            if file.endswith('.log'):
                log_path = os.path.join(root, file)
                # Get the directory containing the checkpoint file
                # Only proceed if a _ckpt file exists in the log_dir

                log_dir = os.path.dirname(log_path)
                # Get the grandparent directory name
                grandparent_dir = os.path.basename(os.path.dirname(os.path.dirname(log_path)))
                has_ckpt = any(f.endswith('_ckpt') for f in os.listdir(log_dir))
                if not has_ckpt:
                    continue
                # Look for config.json in the same directory as the checkpoint
                config_path = os.path.join(log_dir, f'{grandparent_dir}.json')
                seed = None
                algorithm = None
                
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r') as f:
                            hp_dict = {}
                            config = json.load(f)
                            hp_dict['seed'] = config.get('seed')
                            hp_dict['algorithm'] = config.get('algorithm') if config.get('algorithm') else 'riql'
                            hp_dict['utd'] = config.get('updates_per_step') if config.get('updates_per_step') else 1
                            
                            hp_dict['kappa'] = config.get('kappa') if config.get('kappa') else 0.1
                            hp_dict['inv_temperature'] = config.get('inv_temperature') if config.get('inv_temperature') else 3
                            hp_dict['env'] = config.get('env_name')
                            hp_dict['log_path'] = log_path
                            hp_dict['corrupt_reward'] = "False"
                            hp_dict['corrupt_dynamics'] = "False"
                            hp_dict['corrupt_acts'] = "False"
                            hp_dict['corrupt_obs'] = "False"
                            if config.get('corrupt_reward', True):
                                hp_dict['corrupt_reward'] = "True"
                            if config.get('corrupt_dynamics', True):
                                hp_dict['corrupt_dynamics'] = "True"
                            if config.get('corrupt_acts', True):
                                hp_dict['corrupt_acts'] = "True"
                            if config.get('corrupt_obs', True):
                                hp_dict['corrupt_obs'] = "True"
                            if config.get('normalize_states', True):
                                hp_dict['normalize_states'] = "True"
                            else:
                                hp_dict['normalize_states'] = "False"
                            if config.get('deterministic_policy', True):
                                hp_dict['deterministic_policy'] = "True"
                            else:
                                hp_dict['deterministic_policy'] = "False"
                            
                    except (json.JSONDecodeError, IOError) as e:
                        print(f"Warning: Could not read config file {config_path}: {e}")

                    info.append(hp_dict)
    return info

def get_dataframe(directory):
    info = extract_info(directory)
    # print(len(info))
    # create a new empyty dataframe
    df = pd.DataFrame()
    
    for hp_dict in info:
        mean,std = extract_results(hp_dict['log_path'])
        hp_dict['mean'] = np.mean(mean[-3:])
        hp_dict['std'] = np.std(mean[-3:])
        new_Df = pd.DataFrame([hp_dict])
        df = pd.concat([df, new_Df], ignore_index=True)

    return df

df = get_dataframe('experiments/hybrid_attack')
print(df)


df.to_csv('hybrid_attack.csv', index=False)

