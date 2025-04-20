# Stego Image Rotation Experiment

This experiment tests the robustness of the steganography model against geometric transformations, specifically rotations. It evaluates how rotating the stego image (output of HideNet, which is also the input to RevealNet) affects the model's ability to recover the hidden secret.

## Experiment Details

The experiment:
1. Loads a trained steganography model
2. Processes each sample in the validation dataset:
   - Generates a stego image using the cover image and secret
   - Applies rotations of 0°, 90°, 180°, and 270° to the stego image
   - Feeds each rotated stego image to the reveal network
   - Computes metrics for each rotation angle
3. Aggregates and reports the results for each rotation angle

## Metrics

For all secret types:
- **PSNR**: Peak Signal-to-Noise Ratio between cover and stego images
- **SSIM**: Structural Similarity Index between cover and stego images

For binary secrets:
- **Bit Accuracy**: Percentage of correctly recovered bits

For image secrets:
- **Secret PSNR**: PSNR between original and recovered secret images

## Running the Experiment

To run the rotation experiment:

```bash
python src/main.py experiment_rotation --config configs/default.yaml
```

You can override specific configuration parameters:

```bash
python src/main.py experiment_rotation --config configs/default.yaml embed.checkpoint_path=./outputs/checkpoints/my_model.pth
```

## Important Configuration Parameters

- `embed.checkpoint_path`: Path to the trained model checkpoint (**required**)
- `data.secret_type`: Type of secret data ("binary" or "image")
- `evaluation.metrics`: List of metrics to compute

## Results Interpretation

The results are saved to `outputs/experiments/rotation_experiment_<checkpoint_name>.csv` and show how each rotation angle affects the metrics.

If the model is robust to rotations, you should expect similar performance across all rotation angles. Significantly lower performance at certain angles indicates vulnerability to those transformations.

For a fully rotation-invariant model, all rotations should yield similar metrics to the 0° (no rotation) case. 