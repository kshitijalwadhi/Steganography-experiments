#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --job-name stego_proj
#SBATCH --output stego_proj.out
#SBATCH --error stego_proj.err

# Load our conda environment
source stego_env/bin/activate

export WANDB_API_KEY=38619006c8e9a24d38d1e4433dcd882a7259ee6b

# python3 main.py --mode train --config configs/baseline_no_transforms.json --checkpoint best
#python3 -u -m src.main train --config configs/default.yaml
#python3 -u -m src.main experiment_rotation --config configs/default.yaml
python3 -u -m src.derive_robust_secret --config configs/default.yaml --num_secrets 10 --output_dir ./outputs/derived_secrets_equivariant
#python3 -u -m src.evaluate_derived --config configs/default.yaml --secrets_dir ./outputs/derived_secrets_equivariant --num_secrets 10