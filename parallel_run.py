import time
import subprocess
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from tqdm import tqdm
import os
import json
import numpy as np
# extract ckpt path from certain directory


def extract_ckpt_path(directory):
    ckpt_info = []
    for root, dirs, files in os.walk(directory):    
        for file in files:
            if file.endswith('ckpt'):
                ckpt_path = os.path.join(root, file)
                # Get the grandparent directory name
                grandparent_dir = os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))
                
                # Look for the corresponding JSON file in the same directory
                json_file = os.path.join(os.path.dirname(ckpt_path), f"{grandparent_dir}.json")
                corruption_flags = []
                
                if os.path.exists(json_file):
                    try:
                        with open(json_file, 'r') as f:
                            config = json.load(f)
                            if config.get('corrupt_reward', True):
                                corruption_flags.append('--corrupt_reward')
                            if config.get('corrupt_dynamics', True):
                                corruption_flags.append('--corrupt_dynamics')
                                corrupt_dynamics = True
                            else:
                                corrupt_dynamics = False
                            if config.get('corrupt_acts', True):
                                corruption_flags.append('--corrupt_acts')
                            if config.get('corrupt_obs', True):
                                corruption_flags.append('--corrupt_obs')
                            if config.get('normalize_states', True):
                                normalize_states = True
                            else:
                                normalize_states = False
                            if config.get('deterministic_policy', True):
                                deterministic_policy = True
                            else:
                                deterministic_policy = False
                    except Exception as e:
                        print(f"Error reading JSON file {json_file}: {e}")
                turple = [grandparent_dir, ckpt_path, corruption_flags[0]]
                if normalize_states:
                    turple.append('--normalize_states')
                else:
                    turple.append(' ')
                if deterministic_policy:
                    turple.append('--deterministic_policy')
                else:
                    turple.append(' ')
                if "medium-replay-v2" in grandparent_dir:
                    if not corrupt_dynamics:
                        ckpt_info.append(tuple(turple))
    return ckpt_info



riql_attack = extract_ckpt_path('./riql_offline_results/stochastic_norm')

# print(riql_attack)
# print(len(riql_attack))
# sweep parameters
seeds = np.random.randint(0, 10000, size=3) 
kappas = [0.1,0.3]
corrupt_list = [(1.0,0.3)]
invs = [3,10]


# RPEX
rpex_commands = [["python3", f"attack_online.py", f"--algorithm=rpex", f"--env_name={env}",f"{normalize_states}", \
    f"--kappa={kappa}",f"--seed={seed}",f"--inv_temperature={inv}",f"{corruption_flag}",f"--log_dir=attack_online_results/rpex",f"--ckpt_path={ckpt_path}", \
        f"--corruption_range={ranges}",f"--corruption_rate={rate}"] 
    for seed in seeds  for env, ckpt_path,corruption_flag,normalize_states,_ in riql_attack  \
        for ranges,rate in corrupt_list for kappa in kappas for inv in invs]


# RIQL_PEX
riql_pex_commands = [["python3", f"attack_online.py", f"--algorithm=riql", f"--env_name={env}",f"{normalize_states}", \
    f"--inv_temperature={inv}",f"--seed={seed}",f"{corruption_flag}",f"--log_dir=attack_online_results/riql_pex",f"--ckpt_path={ckpt_path}", \
        f"--corruption_range={ranges}",f"--corruption_rate={rate}"] 
    for seed in seeds  for env, ckpt_path,corruption_flag,normalize_states,_ in riql_attack for ranges,rate in corrupt_list for inv in invs]

# RIQL_DIRECT
riql_direct_commands = [["python3", f"attack_online.py", f"--algorithm=riqldirect", f"--env_name={env}",f"{normalize_states}", \
f"--seed={seed}",f"{corruption_flag}",f"--log_dir=attack_online_results/rpex",f"--ckpt_path={ckpt_path}", \
        f"--corruption_range={ranges}",f"--corruption_rate={rate}"] 
    for seed in seeds  for env, ckpt_path,corruption_flag,normalize_states,_ in riql_attack for ranges,rate in corrupt_list]



# Combine all commands
commands = rpex_commands


for i,command in enumerate(commands):
    commands[i] = command + [f"--device_number={0}"]

# print(commands[7])
print(len(commands))


'''
Execute the commands in parallel
'''
max_processes =  6
processes = set()

finished_commands = []
# Create or load the JSON file to store finished commands
json_file = 'sweeps_results.json'
if os.path.exists(json_file):
    with open(json_file, 'r') as f:
        finished_commands = json.load(f)
else:
    with open(json_file, 'w') as f:
        json.dump(finished_commands, f)




for i, command in enumerate(commands):
    # Ensure the pool is filled up to max_processes
    while len(processes) >= max_processes:
        # Temporary set to collect processes still running
        still_running = set()
        for process in processes:
            retcode = process.poll()
            if retcode is None:  # Process is still running
                still_running.add(process)
            else:
                # Process finished, log it and don't re-add to still_running
                finished_commands.append({
                    "command": command,
                    "message": f"Process {process.pid} completed with exit code {retcode}"
                })
                with open(json_file, 'w') as f:
                    json.dump(finished_commands, f, indent=4)
                print(f"Process {process.pid} completed with exit code {retcode}")
        processes = still_running
        if len(processes) < max_processes:
            break  # Exit the while loop if there's room for more processes
        time.sleep(0.5)  # Sleep briefly to avoid too tight a loop

    # Start a new subprocess
    process = subprocess.Popen(command)
    processes.add(process)
    print(f"Started command {i} for command: {' '.join(command)}")

# Wait for all remaining processes
for process in processes:
    process.wait()
    finished_commands.append({
        "command": command,
        "message": f"Process {process.pid} completed with exit code {process.returncode}"
    })
    with open(json_file, 'w') as f:
        json.dump(finished_commands, f, indent=4)
    print(f"Process {process.pid} completed.")

print("All commands completed.")
