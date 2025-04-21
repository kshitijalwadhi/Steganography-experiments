"""
Test script for evaluating rotation robustness of the steganography model.

This script loads a trained model and evaluates its performance on rotated inputs
at angles from 0 to 359 degrees (in 15-degree steps), using BINARY secrets.
It supports rotating either the cover image before embedding or the stego image after embedding,
controlled by the config flag `cfg.rotation.apply_to`.
Applies inverse rotation to revealed secret before calculating Accuracy.
Includes visualization of rotated revealed secrets and aligned secrets for CARDINAL angles (0, 90, 180, 270).
Also includes a focused visualization for a specific angle (e.g., 90 degrees).
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
    """Rotate image tensor by specified angle in degrees using interpolation.

    Uses NEAREST for robustness testing where appropriate, but could be adaptive.
    """
    angle = float(angle)
    if angle == 0:
        return img
    else:
        # Defaulting to NEAREST for binary/potentially sharp secrets
        # Could use BILINEAR if visual fidelity of rotated image is more important
        # Let's switch to BILINEAR for potentially smoother MNIST results visualization
        return TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)


def get_inverse_rotation_angle(angle):
    """Calculate the inverse rotation angle."""
    # General case, works for all angles
    return -angle


def tensor_to_numpy(tensor):
    """Convert a torch tensor to numpy array (H, W, C or H, W) for visualization."""
    tensor = tensor.detach().cpu()

    # If channel dim exists and is 1, squeeze it for grayscale plotting
    if len(tensor.shape) == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    # If channel dim exists and is 3, transpose for RGB
    elif len(tensor.shape) == 3 and tensor.shape[0] == 3:
        tensor = tensor.numpy().transpose(1, 2, 0)
    # If it's already 2D (H, W) or squeezed 1-channel
    elif len(tensor.shape) == 2:
        return tensor.numpy()

    # Fallback if shape is unexpected
    return tensor


def visualize_results_overview(
    cover,
    secret,
    stego_base, # Stego generated from original cover
    rotated_inputs, # Could be rotated stego or stego from rotated cover
    recovered_secrets_rotated,
    recovered_secrets_aligned,
    angles_to_plot, # Should be [0, 90, 180, 270]
    apply_to, # Mode used ('cover' or 'stego')
    filename="rotation_test_overview.png",
):
    """Visualize overview results for cardinal angles (0, 90, 180, 270)."""
    num_angles_plot = len(angles_to_plot)
    cols = 4 # Fixed cols for cardinal angles
    rows = 4 # Fixed rows: Original, Input, Recovered Rotated, Recovered Aligned

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle(
        f"Rotation Robustness Overview (Apply to: {apply_to}, Cardinal Angles)",
        fontsize=14,
    )

    # Ensure axes is 2D array
    if rows == 1 and cols == 1: axes = np.array([[axes]])
    elif rows == 1: axes = np.array([axes])
    elif cols == 1: axes = np.array([[ax] for ax in axes])

    # --- Row 1: Originals --- 
    axes[0, 0].imshow(tensor_to_numpy(cover[0]))
    axes[0, 0].set_title("Cover (0°)")
    axes[0, 1].imshow(tensor_to_numpy(secret[0]), cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Secret (0°)")
    axes[0, 2].imshow(tensor_to_numpy(stego_base[0]))
    axes[0, 2].set_title("Base Stego (0°)")
    axes[0, 3].axis("off") # Hide last column
    for ax in axes[0, :3]: ax.axis("off")

    # --- Subsequent Rows for Cardinal Angles --- 
    input_label = "Rot. Stego" if apply_to == "stego" else "Stego from Rot. Cov."
    rows_data = [
        (rotated_inputs, input_label, None, None), # Row 2: Inputs
        (recovered_secrets_rotated, "Rec. Rotated", "gray", torch.sigmoid), # Row 3: Recovered Rotated
        (recovered_secrets_aligned, "Rec. Aligned", "gray", torch.sigmoid), # Row 4: Recovered Aligned
    ]

    for r, (data_dict, label_prefix, cmap, transform_func) in enumerate(rows_data, start=1):
        for i, angle in enumerate(angles_to_plot):
            if i >= cols: break
            ax = axes[r, i]
            if angle in data_dict:
                img_tensor = data_dict[angle][0]
                if transform_func:
                    img_tensor = transform_func(img_tensor)
                ax.imshow(tensor_to_numpy(img_tensor), cmap=cmap, vmin=0, vmax=1 if transform_func else None)
                ax.set_title(f"{label_prefix} ({angle}°)")
            else:
                ax.set_title(f"{label_prefix} ({angle}°)\n(Not Found)")
            ax.axis("off")
        # Turn off remaining axes in the row
        for j in range(num_angles_plot, cols):
            axes[r, j].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Overview visualization saved to {filename}")

def visualize_single_angle_focus(
    cover, # Original cover
    secret, # Original secret
    rotated_cover, # Cover rotated by focus_angle
    stego_from_rotated_cover, # Stego generated from rotated_cover
    recovered_rotated, # Secret recovered from stego_from_rotated_cover
    recovered_aligned, # Recovered secret after inverse rotation
    focus_angle, # The angle being visualized (e.g., 90)
    filename="rotation_test_focus.png",
):
    """Visualize the state for a single specific rotation angle."""
    cols = 4
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle(f"Rotation Focus: State at {focus_angle}° (Apply to: cover)", fontsize=14)

    if rows == 1 and cols == 1: axes = np.array([[axes]])
    elif rows == 1: axes = np.array([axes])
    elif cols == 1: axes = np.array([[ax] for ax in axes])

    # --- Row 1: Inputs --- 
    axes[0, 0].imshow(tensor_to_numpy(cover[0]))
    axes[0, 0].set_title("Original Cover (0°)")
    axes[0, 1].imshow(tensor_to_numpy(secret[0]), cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title("Original Secret (0°)")
    axes[0, 2].imshow(tensor_to_numpy(rotated_cover[0]))
    axes[0, 2].set_title(f"Rotated Cover ({focus_angle}°)")
    axes[0, 3].imshow(tensor_to_numpy(stego_from_rotated_cover[0]))
    axes[0, 3].set_title(f"Stego from Rot. Cov ({focus_angle}°)")

    # --- Row 2: Outputs --- 
    rec_rot_prob = torch.sigmoid(recovered_rotated[0])
    rec_aln_prob = torch.sigmoid(recovered_aligned[0])
    axes[1, 0].imshow(tensor_to_numpy(rec_rot_prob), cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title(f"Recovered Rotated ({focus_angle}°)")
    axes[1, 1].imshow(tensor_to_numpy(rec_aln_prob), cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title(f"Recovered Aligned ({focus_angle}°)")
    # Hide remaining axes
    axes[1, 2].axis('off')
    axes[1, 3].axis('off')

    # Turn off all axes borders
    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Focused visualization saved to {filename}")


def test_rotation_robustness(cfg):
    """Test the model's robustness to rotations using specified secrets and inverse transform."""

    # Check rotation config
    use_rotation = hasattr(cfg, "rotation") and cfg.rotation.get("enabled", False)
    apply_to = cfg.rotation.get("apply_to", "stego") if use_rotation else "stego"
    secret_type = cfg.data.secret_type
    print(
        f"--- Testing Rotation Robustness ({secret_type.capitalize()} Secrets, 0-359°, Apply to: {apply_to}, Aligned Acc) ---"
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

    # Define the subset of angles for overview visualization (cardinal angles)
    vis_angles_overview = [0, 90, 180, 270]
    print(f"Visualizing overview for angles: {vis_angles_overview}")
    # Define angle for focused visualization
    vis_angle_focus = 90
    print(f"Visualizing focus for angle: {vis_angle_focus}°")

    results = {angle: {"secret_acc_aligned": []} for angle in test_rotation_angles}

    sample_for_vis = None
    vis_batch_idx = 0 # Visualize the first batch

    # --- Evaluation Loop ---
    with torch.no_grad():
        for batch_idx, (cover, secret) in enumerate(
            tqdm(val_loader, desc=f"Evaluating All Angles (Apply to: {apply_to})")
        ):
            cover, secret = cover.to(device), secret.to(device)

            # Generate base stego image (from original cover) for overview visualization
            stego_base = model.hide_secret(cover, secret)

            # Prepare dicts to store data needed for visualizations
            current_batch_rotated_inputs = {} # For overview vis
            current_batch_recovered_rotated = {} # For overview vis
            current_batch_recovered_aligned = {} # For overview vis
            focused_vis_data = {} # For focused vis

            # Test each rotation angle in the full test list
            for angle in test_rotation_angles:
                # --- Prepare input for RevealNet based on rotation mode ---
                rotated_cover_tensor = None
                if apply_to == "cover":
                    # Rotate cover, then embed
                    rotated_cover_tensor = apply_rotation(cover, angle)
                    stego_input = model.hide_secret(rotated_cover_tensor, secret)
                else: # apply_to == "stego" (default/original behavior)
                    # Embed into original cover first (already done as stego_base)
                    # Rotate the resulting stego image
                    stego_input = apply_rotation(stego_base, angle)

                # Reveal from the prepared input
                recovered_secret_rotated = model.reveal_secret(stego_input)

                # Apply inverse rotation to align recovered secret
                inv_angle = get_inverse_rotation_angle(angle)
                recovered_secret_aligned_logits = apply_rotation(
                    recovered_secret_rotated, inv_angle
                )

                # Calculate Accuracy using aligned logits and original target secret
                # Note: bit_accuracy works for binary, might need different func for MNIST classification
                # Assuming bit_accuracy here computes pixel-wise accuracy for image secrets too.
                secret_acc_aligned = bit_accuracy(
                    recovered_secret_aligned_logits, secret
                )
                results[angle]["secret_acc_aligned"].append(secret_acc_aligned)

                # Store tensors for OVERVIEW visualization ONLY if the angle is in vis_angles_overview
                if batch_idx == vis_batch_idx and angle in vis_angles_overview:
                    current_batch_rotated_inputs[angle] = stego_input.clone()
                    current_batch_recovered_rotated[angle] = (
                        recovered_secret_rotated.clone()
                    )
                    current_batch_recovered_aligned[angle] = (
                        recovered_secret_aligned_logits.clone()
                    )

                # Store tensors for FOCUSED visualization if it's the target angle
                if batch_idx == vis_batch_idx and angle == vis_angle_focus:
                    focused_vis_data["rotated_cover"] = rotated_cover_tensor.clone() if rotated_cover_tensor is not None else apply_rotation(cover, angle) # Ensure we have it
                    focused_vis_data["stego_from_rotated_cover"] = stego_input.clone()
                    focused_vis_data["recovered_rotated"] = recovered_secret_rotated.clone()
                    focused_vis_data["recovered_aligned"] = recovered_secret_aligned_logits.clone()


            # Finalize visualization sample storage for the target batch
            if batch_idx == vis_batch_idx:
                sample_for_vis = {
                    "cover": cover.clone(),
                    "secret": secret.clone(),
                    "stego_base": stego_base.clone(),
                    "overview": {
                        "rotated_inputs": current_batch_rotated_inputs,
                        "recovered_secrets_rotated": current_batch_recovered_rotated,
                        "recovered_secrets_aligned": current_batch_recovered_aligned,
                    },
                    "focus": focused_vis_data,
                    "apply_to": apply_to,
                    "focus_angle": vis_angle_focus
                }

    # --- Aggregate and Report Results (All Tested Angles) ---
    print(f"\n--- Rotation Robustness Results (Apply to: {apply_to}, Aligned Secret Accuracy, 0-359°) ---")
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

    # --- Generate Visualizations --- 
    if sample_for_vis:
        output_dir = Path(cfg.training.checkpoint_dir).parent / "experiments"
        output_dir.mkdir(parents=True, exist_ok=True)
        base_filename = f"rotation_test_{secret_type}_apply-{apply_to}_{Path(ckpt_path).stem}"

        # Generate Overview Visualization
        overview_file = output_dir / f"{base_filename}_overview.png"
        visualize_results_overview(
            sample_for_vis["cover"],
            sample_for_vis["secret"],
            sample_for_vis["stego_base"],
            sample_for_vis["overview"]["rotated_inputs"],
            sample_for_vis["overview"]["recovered_secrets_rotated"],
            sample_for_vis["overview"]["recovered_secrets_aligned"],
            vis_angles_overview,
            sample_for_vis["apply_to"],
            filename=str(overview_file),
        )

        # Generate Focused Visualization (only if data exists)
        if sample_for_vis["focus"] and sample_for_vis["apply_to"] == 'cover':
            focus_file = output_dir / f"{base_filename}_focus{sample_for_vis['focus_angle']}deg.png"
            visualize_single_angle_focus(
                sample_for_vis["cover"],
                sample_for_vis["secret"],
                sample_for_vis["focus"]["rotated_cover"],
                sample_for_vis["focus"]["stego_from_rotated_cover"],
                sample_for_vis["focus"]["recovered_rotated"],
                sample_for_vis["focus"]["recovered_aligned"],
                sample_for_vis["focus_angle"],
                filename=str(focus_file),
            )
        elif sample_for_vis["apply_to"] != 'cover':
             print(f"Skipping focused visualization because apply_to mode is '{sample_for_vis['apply_to']}'.")

    return avg_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test Full Range Rotation Robustness with Overview and Focused Visualizations"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml", # Changed default
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

    # # Ensure config specifies binary secret type for this test - REMOVED this restriction
    # if cfg.data.secret_type != "binary":
    #     raise ValueError(
    #         "This script is designed for binary secrets. Please use a config with data.secret_type = 'binary'"
    #     )

    # Add default rotation config if missing (for testing pre-existing models/configs)
    if not hasattr(cfg, 'rotation'):
        print("Warning: 'rotation' section not found in config. Assuming defaults (enabled=False).")
        from omegaconf import DictConfig
        cfg.rotation = DictConfig({'enabled': False, 'apply_to': 'stego'}) # Assume default apply_to if rotation isn't enabled anyway
    elif not hasattr(cfg.rotation, 'apply_to'):
        print("Warning: 'rotation.apply_to' not found in config. Defaulting to 'stego'.")
        cfg.rotation.apply_to = 'stego'


    test_rotation_robustness(cfg)
