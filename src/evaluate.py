from pathlib import Path

import lpips
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim  # Use skimage for SSIM
from tqdm import tqdm

from src.datasets import get_dataloaders  # Re-use dataloader logic
from src.models import SteganoModel
from src.utils import bit_accuracy, get_device, load_checkpoint, load_config, psnr


# Need to compute SSIM on CPU numpy arrays
def calculate_ssim(img1_tensor, img2_tensor):
    """Calculates SSIM between two image tensors (B, C, H, W)."""
    img1_np = img1_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img2_np = img2_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    # Ensure data range is appropriate for skimage ssim, typically [0, 1] or [-1, 1]
    # Our images are [0, 1] from ToTensor or sigmoid
    # Need channel_axis for multichannel images
    return ssim(
        img1_np, img2_np, data_range=img1_np.max() - img1_np.min(), channel_axis=-1
    )


def evaluate(cfg):
    """Evaluates a trained model on a dataset."""
    print("--- Running Evaluation ---")
    device = get_device(cfg.training.device)  # Use training device setting
    eval_cfg = cfg.evaluation
    ckpt_path = cfg.embed.checkpoint_path  # Reuse embed checkpoint path for eval model

    if not ckpt_path:
        raise ValueError(
            "Checkpoint path must be specified in config (e.g., embed.checkpoint_path) for evaluation."
        )

    # --- Data ---
    # Typically evaluate on the validation set, or add a dedicated test set option
    _, val_loader = get_dataloaders(cfg)
    print(f"Evaluating on {len(val_loader.dataset)} samples from validation set.")

    # --- Model ---
    model = SteganoModel(cfg).to(device)
    _ = load_checkpoint(ckpt_path, model, device=device)  # Load weights only
    model.eval()

    # --- Metrics Setup ---
    metrics_to_compute = eval_cfg.metrics
    results = {metric: [] for metric in metrics_to_compute}
    lpips_model = None
    if "lpips" in metrics_to_compute:
        print("Loading LPIPS model...")
        lpips_model = lpips.LPIPS(net="alex").to(device)  # Or 'vgg'
        print("LPIPS model loaded.")

    # --- Evaluation Loop ---
    with torch.no_grad():
        for cover, secret in tqdm(val_loader, desc="Evaluating"):
            cover, secret = cover.to(device), secret.to(device)

            stego, rec_secret = model(cover, secret)

            # Calculate requested metrics for each sample in the batch
            for i in range(cover.size(0)):
                single_cover = cover[i : i + 1]
                single_secret = secret[i : i + 1]
                single_stego = stego[i : i + 1]
                single_rec_secret = rec_secret[i : i + 1]

                if "psnr" in metrics_to_compute:
                    mse_hide = F.mse_loss(single_stego, single_cover)
                    results["psnr"].append(psnr(mse_hide))

                if "ssim" in metrics_to_compute:
                    ssim_val = calculate_ssim(single_stego, single_cover)
                    results["ssim"].append(ssim_val)

                if "lpips" in metrics_to_compute and lpips_model:
                    # LPIPS expects input in range [-1, 1], normalize from [0, 1]
                    lpips_val = lpips_model(
                        single_stego * 2 - 1, single_cover * 2 - 1
                    ).item()
                    results["lpips"].append(lpips_val)

                if "bit_acc" in metrics_to_compute and cfg.data.secret_type == "binary":
                    acc = bit_accuracy(single_rec_secret, single_secret)
                    results["bit_acc"].append(acc)
                # Could add secret reconstruction PSNR/SSIM/LPIPS if needed

    # --- Aggregate and Report Results ---
    print("--- Evaluation Results ---")
    avg_results = {}
    for metric, values in results.items():
        if values:
            avg_value = np.mean(values)
            avg_results[f"avg_{metric}"] = avg_value
            print(f"Average {metric.upper()}: {avg_value:.4f}")
        else:
            print(f"Metric {metric.upper()} not computed (check config/data type).")

    # Optional: Save results to a file
    results_df = pd.DataFrame(results)
    output_dir = Path(cfg.training.checkpoint_dir).parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"evaluation_results_{Path(ckpt_path).stem}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Saved detailed results to {results_path}")
    print("--------------------------")

    return avg_results


# Example usage (typically called from main.py):
# if __name__ == '__main__':
#     config = load_config() # Assumes config path is correct
#     evaluate(config)
