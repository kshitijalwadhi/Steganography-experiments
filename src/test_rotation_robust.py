"""
Test script for evaluating rotation robustness of the steganography model.

This script loads a trained model and evaluates its performance on rotated stego images
at angles from 0 to 359 degrees (in 15-degree steps), using BINARY secrets.
Applies inverse rotation to revealed secret before calculating Accuracy.
Includes visualization of rotated revealed secrets and aligned secrets for CARDINAL angles (0, 90, 180, 270).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm

from src.datasets import get_dataloaders
from src.models import SteganoModel
from src.utils import bit_accuracy, get_device, load_checkpoint, load_config


def apply_rotation(img, angle):
    """Rotate image tensor by specified angle in degrees using NEAREST interpolation."""
    angle = float(angle)
    if angle == 0:
        return img
    else:
        # Use NEAREST interpolation for binary-like data
        return TF.rotate(img, angle, interpolation=TF.InterpolationMode.NEAREST)


def get_inverse_rotation_angle(angle):
    """Calculate the inverse rotation angle."""
    # For cardinal angles, return the exact inverse multiple of 90
    if angle == 0:
        return 0
    elif angle == 90:
        return 270
    elif angle == 180:
        return 180
    elif angle == 270:
        return 90
    else:
        return -angle  # Use negative angle for others


def tensor_to_numpy(tensor):
    """Convert a torch tensor to numpy array (H, W, C or H, W) for visualization."""
    tensor = tensor.detach().cpu()

    # If channel dim exists and is 1, squeeze it for grayscale plotting
    if len(tensor.shape) == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    # If channel dim exists and is 3, transpose for RGB
    elif len(tensor.shape) == 3 and tensor.shape[0] == 3:
        return tensor.numpy().transpose(1, 2, 0)
    # If it's already 2D (H, W) or squeezed 1-channel
    elif len(tensor.shape) == 2:
        return tensor.numpy()

    # Fallback if shape is unexpected
    return tensor.numpy()


def visualize_results(
    cover,
    secret,
    stego,
    rotated_stegos,
    recovered_secrets_rotated,
    recovered_secrets_aligned,
    angles_to_plot,
    filename="rotation_test_results.png",
):
    """Visualize results for cardinal angles (0, 90, 180, 270)."""
    num_angles_plot = len(angles_to_plot)
    # Set fixed columns for cardinal angle visualization
    cols = 4  # One column per cardinal angle + one empty for alignment?
    rows = 4

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    fig.suptitle(
        "Rotation Robustness Test Results (Binary Secrets, Cardinal Angles)",
        fontsize=16,
    )

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    # --- Row 1: Originals (Span first 3 columns) ---
    axes[0, 0].imshow(tensor_to_numpy(cover[0]))
    axes[0, 0].set_title("Cover")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(tensor_to_numpy(secret[0]), cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Original Secret")
    axes[0, 1].axis("off")
    axes[0, 2].imshow(tensor_to_numpy(stego[0]))
    axes[0, 2].set_title("Stego (0°)")
    axes[0, 2].axis("off")
    axes[0, 3].axis("off")  # Hide last column in this row

    # --- Row 2: Rotated Stegos (Cardinal Angles) ---
    for i, angle in enumerate(angles_to_plot):
        if i >= cols:
            break
        if angle in rotated_stegos:
            axes[1, i].imshow(tensor_to_numpy(rotated_stegos[angle][0]))
            axes[1, i].set_title(f"Stego ({angle}°)")
        else:
            axes[1, i].set_title(
                f"Stego ({angle}°)\n(Not Found)"
            )  # Should not happen if vis_angles are computed
        axes[1, i].axis("off")
    for j in range(num_angles_plot, cols):
        axes[1, j].axis("off")

    # --- Row 3: Recovered Secrets Rotated (Cardinal Angles) ---
    for i, angle in enumerate(angles_to_plot):
        if i >= cols:
            break
        if angle in recovered_secrets_rotated:
            recovered_rotated_probs = torch.sigmoid(recovered_secrets_rotated[angle][0])
            axes[2, i].imshow(
                tensor_to_numpy(recovered_rotated_probs), cmap="gray", vmin=0, vmax=1
            )
            axes[2, i].set_title(f"Recovered Rotated ({angle}°)")
        else:
            axes[2, i].set_title(f"Recovered Rotated ({angle}°)\n(Not Found)")
        axes[2, i].axis("off")
    for j in range(num_angles_plot, cols):
        axes[2, j].axis("off")

    # --- Row 4: Recovered Secrets Aligned (Cardinal Angles) ---
    for i, angle in enumerate(angles_to_plot):
        if i >= cols:
            break
        if angle in recovered_secrets_aligned:
            recovered_aligned_probs = torch.sigmoid(recovered_secrets_aligned[angle][0])
            axes[3, i].imshow(
                tensor_to_numpy(recovered_aligned_probs), cmap="gray", vmin=0, vmax=1
            )
            axes[3, i].set_title(f"Recovered Aligned ({angle}°)")
        else:
            axes[3, i].set_title(f"Recovered Aligned ({angle}°)\n(Not Found)")
        axes[3, i].axis("off")
    for j in range(num_angles_plot, cols):
        axes[3, j].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=200)
    plt.close()

    print(f"Visualization saved to {filename}")


def test_rotation_robustness(cfg):
    """Test the model's robustness to rotations using BINARY secrets and inverse transform."""
    print(
        "--- Testing Rotation Robustness (Binary Secrets, 0-359 degrees, Aligned Acc) ---"
    )
    device = get_device(cfg.training.device)
    ckpt_path = cfg.embed.checkpoint_path

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

    # Define the full range of angles to test
    test_rotation_angles = list(range(0, 360, 15))
    print(f"Testing angles: {test_rotation_angles}")

    # Define the subset of angles for visualization (cardinal angles)
    vis_angles = [0, 90, 180, 270]
    print(f"Visualizing angles: {vis_angles}")

    results = {angle: {"secret_acc_aligned": []} for angle in test_rotation_angles}

    sample_for_vis = None
    vis_batch_idx = 0

    # --- Evaluation Loop ---
    with torch.no_grad():
        for batch_idx, (cover, secret) in enumerate(
            tqdm(val_loader, desc="Evaluating All Angles")
        ):
            cover, secret = cover.to(device), secret.to(device)

            # Generate stego image
            stego = model.hide_secret(cover, secret)

            current_batch_rotated_stegos = {}
            current_batch_recovered_rotated = {}
            current_batch_recovered_aligned = {}

            # Test each rotation angle in the full test list
            for angle in test_rotation_angles:
                # Rotate stego
                rotated_stego = apply_rotation(stego, angle)
                # Reveal from rotated stego
                recovered_secret_rotated = model.reveal_secret(rotated_stego)
                # Apply inverse rotation to align recovered secret
                inv_angle = get_inverse_rotation_angle(angle)
                recovered_secret_aligned_logits = apply_rotation(
                    recovered_secret_rotated, inv_angle
                )

                # Calculate Accuracy using logits and target
                secret_acc_aligned = bit_accuracy(
                    recovered_secret_aligned_logits, secret
                )
                results[angle]["secret_acc_aligned"].append(secret_acc_aligned)

                # Store tensors for visualization ONLY if the angle is in vis_angles
                if batch_idx == vis_batch_idx and angle in vis_angles:
                    current_batch_rotated_stegos[angle] = rotated_stego.clone()
                    current_batch_recovered_rotated[angle] = (
                        recovered_secret_rotated.clone()
                    )
                    current_batch_recovered_aligned[angle] = (
                        recovered_secret_aligned_logits.clone()
                    )

            # Finalize visualization sample storage for the target batch
            if batch_idx == vis_batch_idx:
                sample_for_vis = {
                    "cover": cover.clone(),
                    "secret": secret.clone(),
                    "stego": stego.clone(),
                    "rotated_stegos": current_batch_rotated_stegos,
                    "recovered_secrets_rotated": current_batch_recovered_rotated,
                    "recovered_secrets_aligned": current_batch_recovered_aligned,
                }

    # --- Aggregate and Report Results (All Tested Angles) ---
    print("\n--- Rotation Robustness Results (Aligned Secret Accuracy, 0-359°) ---")
    avg_results = {}
    # Sort angles for cleaner reporting
    sorted_angles = sorted(results.keys())
    for angle in sorted_angles:
        avg_acc = (
            np.mean(results[angle]["secret_acc_aligned"])
            if results[angle]["secret_acc_aligned"]
            else float("nan")
        )
        avg_results[f"{angle}_secret_acc_aligned"] = avg_acc
        print(f"Rotation {angle}°: Aligned Secret Acc = {avg_acc*100:.1f}%")

    # --- Visualize Sample Results (Using CARDINAL angles) ---
    if sample_for_vis:
        output_dir = Path(cfg.training.checkpoint_dir).parent / "experiments"
        output_dir.mkdir(parents=True, exist_ok=True)
        # Update filename for the new visualization
        output_file = (
            output_dir / f"rotation_test_binary_cardinal_vis_{Path(ckpt_path).stem}.png"
        )

        visualize_results(
            sample_for_vis["cover"],
            sample_for_vis["secret"],
            sample_for_vis["stego"],
            sample_for_vis["rotated_stegos"],
            sample_for_vis["recovered_secrets_rotated"],
            sample_for_vis["recovered_secrets_aligned"],
            vis_angles,
            filename=str(output_file),
        )

    return avg_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test Full Range Rotation Robustness (Binary Secrets, Aligned Acc) with Cardinal Angle Visualization"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rotation_robust_binary.yaml",
        help="Config file",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint override"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.checkpoint:
        cfg.embed.checkpoint_path = args.checkpoint
        print(f"Using command-line checkpoint: {cfg.embed.checkpoint_path}")
    # Ensure config specifies binary secret type for this test
    if cfg.data.secret_type != "binary":
        raise ValueError(
            "This script is designed for binary secrets. Please use a config with data.secret_type = 'binary'"
        )
    test_rotation_robustness(cfg)
