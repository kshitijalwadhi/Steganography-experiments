# Rotation Invariant Steganography

This extension adds rotation invariance to the steganography model, allowing the RevealNet to recover hidden secrets from stego images that have been rotated by 0°, 90°, 180°, or 270°.

## Requirements

To use the rotation-invariant model, you need to install the `escnn` package:

```bash
pip install escnn
```

## Usage

### Training

To train a model with rotation invariance:

1. Use the provided rotation-invariant configuration:

```bash
python train.py --config configs/rotation_invariant.yaml
```

Or modify your existing configuration to include the `rotation_invariant` flag:

```yaml
model:
  prep_out_channels: 32
  rotation_invariant: true
```

### Evaluation

To compare the standard model with the rotation-invariant model:

```bash
python -m src.experiment_rotation_invariance --config configs/your_config.yaml
```

If you have separate checkpoints for the standard and invariant models:

```bash
python -m src.experiment_rotation_invariance --config configs/your_config.yaml --invariant_checkpoint path/to/invariant_model.pth
```

## How It Works

The rotation invariance is implemented using the `escnn` library, which provides equivariant neural networks. The RevealNet is modified to use the C4 group (rotations by 0°, 90°, 180°, and 270°) to achieve invariance to these transformations.

Key components:

1. **Group Symmetry**: The model uses the C4 rotation group to define symmetry constraints.
2. **Equivariant Layers**: Convolutional layers are replaced with equivariant versions that respect the group action.
3. **Group Pooling**: The output features are pooled over the group to achieve invariance.

## Experiment Results

The experiment compares the standard RevealNet with the rotation-invariant version on the following metrics:

- **PSNR**: Peak Signal-to-Noise Ratio between cover and stego images
- **SSIM**: Structural Similarity Index between cover and stego images
- **Bit Accuracy**: Accuracy of recovered binary secrets (for binary secrets)
- **Secret PSNR**: PSNR between original and recovered image secrets (for image secrets)

Results are saved in the `experiments` directory as CSV files for further analysis.

## Configuration

The `rotation_invariant.yaml` configuration file includes the following settings:

```yaml
model:
  rotation_invariant: true  # Enable rotation invariance in RevealNet
```

## Notes

- The rotation-invariant model may have different capacity and training characteristics compared to the standard model.
- Training the invariant model might require more computational resources due to the increased complexity of equivariant layers.
- The performance impact of using rotation invariance will depend on your specific use case and dataset. 