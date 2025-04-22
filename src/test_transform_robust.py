"""
Test script for evaluating geometric transformation robustness of the steganography model.

This script loads a trained model and evaluates its performance on transformed stego images,
applying combinations of rotation, scaling, and translation.
Applies inverse transforms to revealed secret before calculating metrics (e.g., Accuracy for binary).
Includes visualization for a subset of specified transformations.
"""

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import \
    InterpolationMode  # Import InterpolationMode
from tqdm import tqdm

from src.datasets import get_dataloaders
from src.models import SteganoModel
# Import the new transform functions
from src.transforms import (apply_affine_transform,
                            apply_inverse_affine_transform)
from src.utils import bit_accuracy, get_device, load_checkpoint, load_config


def tensor_to_numpy(tensor):
    """Convert a torch tensor to numpy array (H, W, C or H, W) for visualization."""
    tensor = tensor.detach().cpu()
    if len(tensor.shape) == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    elif len(tensor.shape) == 3 and tensor.shape[0] == 3:
        return tensor.numpy().transpose(1, 2, 0)
    elif len(tensor.shape) == 2:
        return tensor.numpy()
    return tensor.numpy()


def visualize_transform_results(
    cover,
    secret,
    stego_identity,
    transformed_stegos,  # Dict mapping transform_key -> transformed_stego_tensor
    recovered_secrets_transformed,  # Dict mapping transform_key -> recovered_secret_tensor
    recovered_secrets_aligned,  # Dict mapping transform_key -> aligned_secret_tensor
    transform_keys_to_plot,  # List of specific transform keys (tuples) to visualize
    filename="transform_test_results.png",
):
    """Visualize results for a specified subset of transformations."""
    num_transforms_plot = len(transform_keys_to_plot)
    cols = max(4, num_transforms_plot)  # Adjust columns based on number of transforms
    rows = (
        4  # Rows: Original, Transformed Stego, Recovered Transformed, Recovered Aligned
    )

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    fig.suptitle(
        "Geometric Transformation Robustness Test Results",
        fontsize=16,
    )

    # Flatten axes array for easier indexing if necessary
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    # --- Row 1: Originals --- (Span first 3 columns if possible)
    axes[0, 0].imshow(tensor_to_numpy(cover[0]))
    axes[0, 0].set_title("Cover")
    axes[0, 0].axis("off")

    secret_np = tensor_to_numpy(secret[0])
    cmap = "gray" if secret_np.ndim == 2 else None
    vmin = 0 if secret_np.ndim == 2 else None
    vmax = 1 if secret_np.ndim == 2 else None
    axes[0, 1].imshow(secret_np, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("Original Secret")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(tensor_to_numpy(stego_identity[0]))
    axes[0, 2].set_title("Stego (Identity)")
    axes[0, 2].axis("off")
    # Hide remaining columns in the first row
    for j in range(3, cols):
        axes[0, j].axis("off")

    # --- Subsequent Rows for each transformation --- #
    for i, t_key in enumerate(transform_keys_to_plot):
        if i >= cols:
            break  # Don't exceed plot columns
        # Unpack correctly: angle, scale, and the (tx, ty) tuple
        angle, scale, translate_tuple = t_key
        tx, ty = translate_tuple  # Unpack the translation tuple
        transform_title = f"A={angle:.0f}, S={scale:.2f}\nT=({tx},{ty})"

        # Row 2: Transformed Stegos
        if t_key in transformed_stegos:
            axes[1, i].imshow(tensor_to_numpy(transformed_stegos[t_key][0]))
            axes[1, i].set_title(f"Stego\n{transform_title}")
        else:
            axes[1, i].set_title(f"Stego\n{transform_title}\n(Not Found)")
        axes[1, i].axis("off")

        # Row 3: Recovered Secrets (Transformed)
        if t_key in recovered_secrets_transformed:
            rec_trans_np = tensor_to_numpy(
                torch.sigmoid(recovered_secrets_transformed[t_key][0])
            )
            axes[2, i].imshow(rec_trans_np, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[2, i].set_title(f"Recovered (T)")
        else:
            axes[2, i].set_title(f"Recovered (T)\n(Not Found)")
        axes[2, i].axis("off")

        # Row 4: Recovered Secrets (Aligned)
        if t_key in recovered_secrets_aligned:
            rec_aligned_np = tensor_to_numpy(
                torch.sigmoid(recovered_secrets_aligned[t_key][0])
            )
            axes[3, i].imshow(rec_aligned_np, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[3, i].set_title(f"Recovered (Aligned)")
        else:
            axes[3, i].set_title(f"Recovered (Aligned)\n(Not Found)")
        axes[3, i].axis("off")

    # Hide unused columns in subsequent rows
    for r in range(1, rows):
        for j in range(num_transforms_plot, cols):
            if r < axes.shape[0] and j < axes.shape[1]:
                axes[r, j].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Visualization saved to {filename}")


def test_transform_robustness(cfg):
    """Test the model's robustness to geometric transformations using inverse transform alignment."""
    print("--- Testing Geometric Transformation Robustness (Aligned Metrics) ---")
    device = get_device(cfg.training.device)
    ckpt_path = cfg.embed.checkpoint_path
    image_size = cfg.data.image_size

    if not ckpt_path or not Path(ckpt_path).is_file():
        raise FileNotFoundError(f"Checkpoint path not found: {ckpt_path}")
    print(f"Loading checkpoint: {ckpt_path}")

    # --- Data ---
    _, val_loader = get_dataloaders(cfg)
    print(f"Evaluating on {len(val_loader.dataset)} samples from validation set.")

    # --- Model ---
    model = SteganoModel(cfg).to(device)
    _ = load_checkpoint(ckpt_path, model, device=device)
    model.eval()

    # --- Define Transformations to Test --- #
    test_angles = cfg.evaluation.get("test_angles", [0, 90, 180, 270])
    test_scales = cfg.evaluation.get("test_scales", [1.0])  # Default to identity scale
    test_translations_fraction = cfg.evaluation.get(
        "test_translations_fraction", [0.0]
    )  # Default to no translation
    test_translations_pixels = [
        [int(f * image_size) for f in test_translations_fraction] for _ in range(2)
    ]
    # Generate pairs of translations (tx, ty) - this might need refinement based on how you want to test translations
    # Example: Test max shift in each cardinal direction and center
    max_shift = int(max(test_translations_fraction) * image_size)
    test_translations = [[0, 0]]  # Identity
    if max_shift > 0:
        test_translations.extend(
            [[max_shift, 0], [-max_shift, 0], [0, max_shift], [0, -max_shift]]
        )
    # Make unique
    test_translations = [list(t) for t in set(tuple(t) for t in test_translations)]

    print(f"Testing Angles: {test_angles}")
    print(f"Testing Scales: {test_scales}")
    print(f"Testing Translations (pixels): {test_translations}")

    # Create all combinations of transformations to test
    transform_combinations = list(
        itertools.product(test_angles, test_scales, test_translations)
    )
    print(f"Total transformation combinations to test: {len(transform_combinations)}")

    # --- Define Transformations for Visualization --- #
    # Select a subset for visualization (e.g., identity, max rotation, max translate, min/max scale)
    vis_transforms = []
    vis_transforms.append((0.0, 1.0, [0, 0]))  # Identity
    if max(test_angles) > 0:
        vis_transforms.append((max(test_angles), 1.0, [0, 0]))
    if max_shift > 0:
        vis_transforms.append((0.0, 1.0, [max_shift, 0]))  # Max +X shift
    if min(test_scales) < 1.0:
        vis_transforms.append((0.0, min(test_scales), [0, 0]))
    if max(test_scales) > 1.0:
        vis_transforms.append((0.0, max(test_scales), [0, 0]))
    # Ensure unique tuples and convert translation list to tuple for dict keys
    vis_transform_keys = list(set((a, s, tuple(t)) for a, s, t in vis_transforms))
    print(f"Visualizing transformations: {vis_transform_keys}")

    # --- Initialize Results Storage --- #
    # Use tuple (angle, scale, tx, ty) as key
    results = {
        (angle, scale, tuple(translate)): {"secret_acc_aligned": []}
        for angle, scale, translate in transform_combinations
    }

    sample_for_vis = None
    vis_batch_idx = 0  # Visualize the first batch

    # Determine interpolation mode for transformations
    interp_mode = (
        InterpolationMode.NEAREST
        if cfg.data.secret_type == "binary"
        else InterpolationMode.BILINEAR
    )

    # --- Evaluation Loop --- #
    with torch.no_grad():
        for batch_idx, (cover, secret) in enumerate(
            tqdm(val_loader, desc="Evaluating Transforms")
        ):
            cover, secret = cover.to(device), secret.to(device)

            # Generate stego image (identity transform)
            stego = model.hide_secret(cover, secret)

            # Store tensors for the visualization sample if this is the target batch
            current_batch_transformed_stegos = {}
            current_batch_recovered_transformed = {}
            current_batch_recovered_aligned = {}

            # Test each transformation combination
            for angle, scale, translate_pixels in transform_combinations:
                transform_key = (angle, scale, tuple(translate_pixels))

                # Apply affine transform to stego
                transformed_stego = apply_affine_transform(
                    stego, angle, scale, translate_pixels, interpolation=interp_mode
                )

                # Reveal from transformed stego
                recovered_secret_transformed = model.reveal_secret(transformed_stego)

                # Apply inverse affine transform to align recovered secret
                recovered_secret_aligned_logits = apply_inverse_affine_transform(
                    recovered_secret_transformed,
                    angle,
                    scale,
                    translate_pixels,
                    interpolation=interp_mode,
                )

                # Calculate Accuracy using aligned logits and original secret
                # Adapt metric calculation if needed (e.g., MSE for image secrets)
                if cfg.data.secret_type == "binary":
                    secret_acc_aligned = bit_accuracy(
                        recovered_secret_aligned_logits, secret
                    )
                    if transform_key in results:
                        results[transform_key]["secret_acc_aligned"].append(
                            secret_acc_aligned
                        )
                    else:  # Should not happen with pre-initialization
                        print(
                            f"Warning: Transform key {transform_key} not found in results."
                        )
                # Add MSE calculation here if needed for image secrets
                # elif cfg.data.secret_type == "image":
                #     secret_mse_aligned = F.mse_loss(recovered_secret_aligned_logits, secret).item()
                #     if transform_key in results:
                #          results[transform_key]["secret_mse_aligned"].append(secret_mse_aligned)

                # Store tensors for visualization if this batch/transform is selected
                if batch_idx == vis_batch_idx and transform_key in vis_transform_keys:
                    current_batch_transformed_stegos[transform_key] = (
                        transformed_stego.clone()
                    )
                    current_batch_recovered_transformed[transform_key] = (
                        recovered_secret_transformed.clone()
                    )
                    current_batch_recovered_aligned[transform_key] = (
                        recovered_secret_aligned_logits.clone()
                    )

            # Finalize visualization sample storage for the target batch
            if batch_idx == vis_batch_idx:
                sample_for_vis = {
                    "cover": cover.clone(),
                    "secret": secret.clone(),
                    "stego_identity": stego.clone(),  # Store the untransformed stego
                    "transformed_stegos": current_batch_transformed_stegos,
                    "recovered_secrets_transformed": current_batch_recovered_transformed,
                    "recovered_secrets_aligned": current_batch_recovered_aligned,
                }

    # --- Aggregate and Report Results --- #
    print("\n--- Transformation Robustness Results (Aligned Secret Accuracy) ---")
    avg_results = {}
    # Sort results for cleaner reporting (optional, based on angle/scale/etc.)
    # Sorting complex keys might be tricky, just iterate through dict for now
    for transform_key, metrics in results.items():
        angle, scale, translate = transform_key
        if metrics["secret_acc_aligned"]:
            avg_acc = np.mean(metrics["secret_acc_aligned"])
            metric_key_name = (
                f"acc_A{angle:.0f}_S{scale:.2f}_T({translate[0]},{translate[1]})"
            )
            avg_results[metric_key_name] = avg_acc
            print(
                f"Transform (A={angle:.0f}, S={scale:.2f}, T={translate}): Aligned Secret Acc = {avg_acc*100:.1f}%"
            )
        # Add reporting for MSE if implemented
        # elif metrics.get("secret_mse_aligned"):
        # ... report MSE ...

    # --- Visualize Sample Results --- #
    if sample_for_vis:
        output_dir = Path(cfg.training.checkpoint_dir).parent / "experiments"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"transform_test_vis_{Path(ckpt_path).stem}.png"

        visualize_transform_results(
            sample_for_vis["cover"],
            sample_for_vis["secret"],
            sample_for_vis["stego_identity"],
            sample_for_vis["transformed_stegos"],
            sample_for_vis["recovered_secrets_transformed"],
            sample_for_vis["recovered_secrets_aligned"],
            vis_transform_keys,
            filename=str(output_file),
        )

    return avg_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test Geometric Transformation Robustness (Aligned Metrics) with Visualization"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rotation_robust_binary.yaml",  # Keep default or change
        help="Config file",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint override"
    )
    # Add arguments to override test transformations if needed
    # parser.add_argument('--test_angles', nargs='+', type=float, help='Override test angles')
    # parser.add_argument('--test_scales', nargs='+', type=float, help='Override test scales')
    # parser.add_argument('--test_translations', nargs='+', type=float, help='Override test translation fractions')

    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.checkpoint:
        cfg.embed.checkpoint_path = args.checkpoint
        print(f"Using command-line checkpoint: {cfg.embed.checkpoint_path}")

    # Add config overrides from command-line args if implemented
    # if args.test_angles: cfg.evaluation.test_angles = args.test_angles
    # ... etc ...

    # Optionally add a check for secret type if the script is specific (e.g., binary only)
    # if cfg.data.secret_type != "binary":
    #     raise ValueError("This script currently expects binary secrets for accuracy calculation.")

    test_transform_robustness(cfg)
