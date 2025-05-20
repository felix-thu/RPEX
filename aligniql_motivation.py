import time
import subprocess
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from tqdm import tqdm
import os
import json
import numpy as np
# extract ckpt path from certain directory



# sweep parameters
seeds = np.random.randint(0, 10000, size=1) 
corrupt_list = [(0.5,0.5)]
envs  =[  #'walker2d-medium-replay-v2',
                            'halfcheetah-medium-replay-v2']
                            # 'hopper-medium-replay-v2',]
algs = ['aligniql','iql']
corruption_flags = ['--corrupt_reward', '--corrupt_dynamics', '--corrupt_acts', '--corrupt_obs']
# RPEX
offline_commands = [["python3", f"attack_offline.py", f"--algorithm={alg}", f"--env_name={env}",f"--normalize_states", \
    f"--seed={seed}", '--corrupt_dynamics',f"--corrupt_reward",f"--num_steps=1000001",f"--corrupt_obs", '--corrupt_acts',f"--log_dir=aligniql_motivation2/hybrid", \
        f"--corruption_range={ranges}",f"--corruption_rate={rate}"] 
    for seed in seeds  for env in envs for ranges,rate in corrupt_list for alg in algs]



# RPEX
cql_commands = [
    [
        "python3", f"pex/algorithms/cql.py", f"--env_name={env}", 
        f"--seed={seed}", f"--corrupt_dynamics",f"--corrupt_reward",f"--corrupt_obs", '--corrupt_acts',f"--corruption_range={ranges}",f"--corruption_rate={rate}",f"--checkpoints_path=aligniql_motivation2/hybrid"
    ] 
    for seed in seeds for env in envs  for ranges,rate in corrupt_list
]

# Combine all commands
commands =  offline_commands + cql_commands


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
