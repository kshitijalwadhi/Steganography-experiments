import argparse

import torch
from omegaconf import OmegaConf

from src.evaluate import evaluate
from src.experiment_rotation import experiment_rotation
from src.models import SteganoModel
from src.train import train
from src.utils import get_device, load_config, run_embedding, run_extraction


def main():
    parser = argparse.ArgumentParser(
        description="Deep Steganography Experiment Framework"
    )
    parser.add_argument(
        "mode",
        choices=["train", "evaluate", "embed", "extract", "experiment_rotation"],
        help="Operation mode: train, evaluate, embed, extract, or experiment_rotation",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML configuration file",
    )

    # Allow overriding specific config values from command line
    # Example: python src/main.py train --config configs/cifar_binary.yaml training.epochs=50
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Configuration overrides in hydra format (e.g., training.lr=0.0001 data.image_size=128)",
    )

    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)

    # Apply command-line overrides
    if args.overrides:
        print(f"Applying overrides: {args.overrides}")
        override_conf = OmegaConf.from_cli(args.overrides)
        cfg = OmegaConf.merge(cfg, override_conf)
        print("--- Updated Configuration ---")
        print(OmegaConf.to_yaml(cfg))
        print("---------------------------")

    # Execute the chosen mode
    if args.mode == "train":
        train(cfg)
    elif args.mode == "evaluate":
        evaluate(cfg)  # evaluate function needs cfg, loads model inside
    elif args.mode == "experiment_rotation":
        experiment_rotation(cfg)  # Run the rotation experiment
    elif args.mode == "embed" or args.mode == "extract":
        # Embedding and Extraction need an initialized model
        device = get_device(cfg.training.device)  # Use training device setting
        model = SteganoModel(
            cfg
        )  # No need to load weights here, handled inside run_ funcs
        if args.mode == "embed":
            run_embedding(cfg, model, device)
        else:  # extract
            run_extraction(cfg, model, device)


if __name__ == "__main__":
    main()
