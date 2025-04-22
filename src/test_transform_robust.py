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
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import \
    InterpolationMode  # Import InterpolationMode
from tqdm import tqdm
import math

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


def calculate_psnr(img1, img2):
    """Calculate PSNR between two image tensors.
    Higher values indicate better image quality/similarity (max is infinity for identical images).
    """
    mse = F.mse_loss(img1, img2).item()
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


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
    # Select a varied subset of transformations for visualization
    vis_transforms = []
    
    # 1. Identity transformation (no change)
    vis_transforms.append((0.0, 1.0, [0, 0]))
    
    # 2-3. Pure rotations (different angles)
    if len(test_angles) >= 2:
        # Add 90° rotation if available
        if 90.0 in test_angles:
            vis_transforms.append((90.0, 1.0, [0, 0]))
        # Add 180° rotation if available
        if 180.0 in test_angles:
            vis_transforms.append((180.0, 1.0, [0, 0]))
    
    # 4-5. Pure scaling (min and max)
    if len(test_scales) >= 2:
        # Minimum scale
        if min(test_scales) < 1.0:
            vis_transforms.append((0.0, min(test_scales), [0, 0]))
        # Maximum scale
        if max(test_scales) > 1.0:
            vis_transforms.append((0.0, max(test_scales), [0, 0]))
    
    # 6-7. Pure translations (different directions)
    if max_shift > 0:
        # X-axis translation
        vis_transforms.append((0.0, 1.0, [max_shift, 0]))
        # Y-axis translation
        vis_transforms.append((0.0, 1.0, [0, max_shift]))
    
    # 8-10. Combined transformations
    # Rotation + Scale
    if len(test_angles) >= 2 and len(test_scales) >= 2:
        if 90.0 in test_angles and min(test_scales) < 1.0:
            vis_transforms.append((90.0, min(test_scales), [0, 0]))
    
    # Rotation + Translation
    if len(test_angles) >= 2 and max_shift > 0:
        if 90.0 in test_angles:
            vis_transforms.append((90.0, 1.0, [max_shift, max_shift]))
    
    # Scale + Translation
    if len(test_scales) >= 2 and max_shift > 0:
        if max(test_scales) > 1.0:
            vis_transforms.append((0.0, max(test_scales), [max_shift, 0]))
    
    # 11. All three: Rotation + Scale + Translation (if all are being tested)
    if len(test_angles) >= 2 and len(test_scales) >= 2 and max_shift > 0:
        if 180.0 in test_angles and max(test_scales) > 1.0:
            vis_transforms.append((180.0, max(test_scales), [max_shift, max_shift]))
    
    # Ensure unique tuples and convert translation list to tuple for dict keys
    vis_transform_keys = list(set((a, s, tuple(t)) for a, s, t in vis_transforms))
    # Limit to a reasonable number to avoid overcrowding (max 10-12 transforms)
    if len(vis_transform_keys) > 12:
        vis_transform_keys = vis_transform_keys[:12]
    
    # IMPORTANT: Make sure all visualization transformations are included in the test combinations
    # Add any visualization transformations that aren't already in the test combinations
    existing_transform_keys = set((a, s, tuple(t)) for a, s, t in transform_combinations)
    for vis_key in vis_transform_keys:
        if vis_key not in existing_transform_keys:
            angle, scale, trans_tuple = vis_key
            transform_combinations.append((angle, scale, list(trans_tuple)))
            print(f"Added visualization transformation to test combinations: {vis_key}")
    
    print(f"Visualizing transformations: {vis_transform_keys}")
    print(f"Total visualization examples: {len(vis_transform_keys)}")

    # --- Initialize Results Storage --- #
    # Use tuple (angle, scale, tx, ty) as key
    results = {}
    for angle, scale, translate in transform_combinations:
        transform_key = (angle, scale, tuple(translate))
        results[transform_key] = {
            "secret_acc_aligned": []
        }

    sample_for_vis = None
    vis_batch_idx = 1  # Visualize the second batch for more variety

    # Store visualization data for each batch until we find a good one
    all_vis_samples = []
    max_vis_samples = 3  # Store data for first 3 batches to choose from

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
            if batch_idx >= max_vis_samples:
                # Skip visualization data collection after max_vis_samples
                # but continue with the evaluation
                pass
            
            cover, secret = cover.to(device), secret.to(device)

            # Generate stego image (identity transform)
            stego = model.hide_secret(cover, secret)

            # Store tensors for the visualization sample if within first few batches
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

                # Calculate bit accuracy for all secret types (binary and image)
                secret_acc_aligned = bit_accuracy(
                    recovered_secret_aligned_logits, secret
                )
                results[transform_key]["secret_acc_aligned"].append(
                    secret_acc_aligned
                )

                # Store tensors for visualization if within first few batches
                if batch_idx < max_vis_samples and transform_key in vis_transform_keys:
                    current_batch_transformed_stegos[transform_key] = (
                        transformed_stego.clone()
                    )
                    current_batch_recovered_transformed[transform_key] = (
                        recovered_secret_transformed.clone()
                    )
                    current_batch_recovered_aligned[transform_key] = (
                        recovered_secret_aligned_logits.clone()
                    )

            # Finalize visualization sample storage for each batch within the first few
            if batch_idx < max_vis_samples:
                # Check if this batch has all the required transformations
                missing_transforms = set(vis_transform_keys) - set(current_batch_transformed_stegos.keys())
                coverage = len(current_batch_transformed_stegos) / len(vis_transform_keys)
                
                all_vis_samples.append({
                    "batch_idx": batch_idx,
                    "coverage": coverage,
                    "missing": len(missing_transforms),
                    "cover": cover.clone(),
                    "secret": secret.clone(),
                    "stego_identity": stego.clone(),
                    "transformed_stegos": current_batch_transformed_stegos,
                    "recovered_secrets_transformed": current_batch_recovered_transformed,
                    "recovered_secrets_aligned": current_batch_recovered_aligned,
                })
                
                print(f"Batch {batch_idx}: Visualization coverage {coverage*100:.1f}% ({len(current_batch_transformed_stegos)}/{len(vis_transform_keys)} transforms)")

    # Select the best sample for visualization (most complete)
    if all_vis_samples:
        # Sort by coverage (highest first) and then by batch index
        all_vis_samples.sort(key=lambda x: (-x["coverage"], x["batch_idx"]))
        sample_for_vis = all_vis_samples[0]
        vis_batch_idx = sample_for_vis["batch_idx"]
        print(f"Selected batch {vis_batch_idx} for visualization with {sample_for_vis['coverage']*100:.1f}% coverage")
        
        # Update visualization keys to only include those actually present in the sample
        available_keys = set(sample_for_vis["transformed_stegos"].keys())
        vis_transform_keys = [k for k in vis_transform_keys if k in available_keys]

    # --- Aggregate and Report Results --- #
    print("\n--- Transformation Robustness Results (Secret Bit Accuracy) ---")
    avg_results = {}
    
    # Organize results by angle and scale for better presentation
    organized_results = defaultdict(lambda: defaultdict(dict))
    angle_averages = defaultdict(list)
    scale_averages = defaultdict(list)
    translation_averages = defaultdict(list)
    
    # First, calculate averages and organize by angle -> scale -> translation
    for transform_key, metrics in results.items():
        angle, scale, translate = transform_key
        if metrics["secret_acc_aligned"]:
            avg_acc = np.mean(metrics["secret_acc_aligned"])
            avg_results[f"acc_A{angle:.0f}_S{scale:.2f}_T({translate[0]},{translate[1]})"] = avg_acc
            
            # Store in our organized structure
            organized_results[angle][scale][translate] = avg_acc
            
            # Collect for averages
            angle_averages[angle].append(avg_acc)
            scale_averages[scale].append(avg_acc)
            translation_averages[translate].append(avg_acc)
    
    # Print results grouped by angle and scale
    print("\n=== RESULTS BY ROTATION ANGLE ===")
    for angle in sorted(organized_results.keys()):
        angle_avg = np.mean(angle_averages[angle])
        print(f"\n>> ANGLE {angle}° (Avg: {angle_avg*100:.1f}%)")
        
        for scale in sorted(organized_results[angle].keys()):
            scale_results = organized_results[angle][scale]
            scale_values = list(scale_results.values())
            scale_in_angle_avg = np.mean(scale_values)
            
            print(f"  Scale {scale:.2f} (Avg: {scale_in_angle_avg*100:.1f}%)")
            
            # Format the translation results in a tabular-like way
            trans_results = []
            for translate, acc in scale_results.items():
                trans_str = f"T({translate[0]:2d},{translate[1]:2d}): {acc*100:.1f}%"
                trans_results.append(trans_str)
            
            # Display translations in a compact way (multiple per line)
            for i in range(0, len(trans_results), 3):
                chunk = trans_results[i:i+3]
                print(f"    {' | '.join(chunk)}")
    
    # Print summary by scale factor
    print("\n=== SUMMARY BY SCALE FACTOR ===")
    for scale in sorted(scale_averages.keys()):
        scale_avg = np.mean(scale_averages[scale])
        print(f"Scale {scale:.2f}: Average Accuracy = {scale_avg*100:.1f}%")
    
    # Print summary by translation factor
    print("\n=== SUMMARY BY TRANSLATION FACTOR ===")
    # Define a function to sort translations for logical ordering
    def translation_sort_key(trans):
        # Sort by Euclidean distance from origin, then by quadrant
        tx, ty = trans
        distance = tx**2 + ty**2
        # Center point first, then axes points, then others
        if tx == 0 and ty == 0:
            priority = 0
        elif tx == 0 or ty == 0:
            priority = 1
        else:
            priority = 2
        return (priority, distance, tx, ty)
    
    for translate in sorted(translation_averages.keys(), key=translation_sort_key):
        trans_avg = np.mean(translation_averages[translate])
        print(f"Translation ({translate[0]:2d},{translate[1]:2d}): Average Accuracy = {trans_avg*100:.1f}%")
    
    # Print overall average
    all_accs = [acc for sublist in angle_averages.values() for acc in sublist]
    overall_avg = np.mean(all_accs)
    print(f"\n=== OVERALL AVERAGE: {overall_avg*100:.1f}% ===")

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

    test_transform_robustness(cfg)
