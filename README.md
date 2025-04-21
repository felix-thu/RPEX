# RPEX: Robust Policy Execution

This repository contains the official implementation of our NeurIPS 2025 submission titled "RPEX: Robust Policy Expansion for Offline-to-Online RL". 

## Overview

RPEX is a novel approach for robust policy execution in reinforcement learning, addressing the challenges of policy deployment in real-world environments with uncertainties and perturbations.

## Key Features

- Robust policy execution framework
- Offline and online attack scenarios
- Parallel execution capabilities
- Comprehensive evaluation metrics

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/RPEX.git
cd RPEX
```

2. Create and activate a virtual environment:
```bash
conda create -n your_name python==3.10
conda activate your_name  
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run RPEX with random dynamics corruption:
### Offline Pretrain (RIQL)
```bash
python attack_offline.py --env_name="hopper-medium-replay-v2" --algorithm="riql" --normalize_states --corrupt_dynamics
```
'env_name' can be 'halfcheetah-medium-replay-v2', 'walker2d-medium-replay-v2', 'hopper-medium-replay-v2', ....

'corruption_range' and 'corruption_rate' are set to 1.0 and 0.3 by default.

Replace '--corrupt_dynamics' with '--corrupt_reward', '--corrupt_acts', and '--corrupt_obs' to enforce corruption on rewards, actions, and dynamics.
### Offline-to-Online Attack (Ours)
```bash
python attack_online.py --env_name="hopper-medium-replay-v2" --algorithm="rpex" --normalize_states --corrupt_dynamics --ckpt_path="./riql_offline_results/stochastic_norm/hopper-medium-replay-v2/offline-RIQL-attack-7031-a4ee/offline_ckpt"
```
The online attack configuration inherits the same default parameters as the offline attack (corruption_range=1.0, corruption_rate=0.3). These parameters can be customized using the `--corruption_range` and `--corruption_rate` flags, following the same syntax as the offline configuration.

To run the algorithm with a clean dataset (without any corruption), simply omit the corruption-related flags:

```bash
python attack_offline.py --env_name="hopper-medium-replay-v2" --algorithm="riql" --normalize_states
```

### Parallel Execution
```bash
python parallel_run.py
```

## Project Structure

```
RPEX/
├── attack_offline.py      # Offline attack implementation
├── attack_online.py       # Online attack implementation
├── parallel_run.py        # Parallel execution script
├── read_res.py           # Results analysis utilities
├── RIQL_TRAIN_CONFIG.py  # Training configuration
├── pex/                  # Core algorithm implementation
│   ├── algorithm/        # Algorithm implementations
│   └── network/          # Network architectures
├── iql_offline_results/  # IQL offline experiment results
└── riql_offline_results/ # RIQL offline experiment results
```

## Results

Offline experimental results and checkpoint can be found in the `iql_offline_results/` and `riql_offline_results/` directories.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{rpex2024,
  title={RPEX: Robust Policy Execution},
  author={Your Name and Co-authors},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please contact [Your Email].

## Acknowledgments

We would like to acknowledge the following works that inspired and contributed to our research:

- [RIQL](https://github.com/YangRui2015/RIQL): Towards Robust Offline Reinforcement Learning under Diverse Data Corruption
- [PEX](https://github.com/Haichao-Zhang/PEX): Policy Expansion for Bridging Offline-to-Online Reinforcement Learning

We also thank the open-source community for their valuable contributions to reinforcement learning research.



