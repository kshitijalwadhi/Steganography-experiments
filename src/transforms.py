import math
import random

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


def get_affine_params(cfg, image_size, device):
    """Generates random affine transformation parameters based on config."""
    angle = 0.0
    scale = 1.0
    translate_pixels = [0, 0]

    # Rotation
    if cfg.rotation.get("enabled", False):
        prob_rot = cfg.rotation.get(
            "prob", 1.0
        )  # Default to always applying if section enabled
        if random.random() < prob_rot:
            angles = cfg.rotation.get("rotation_angles", [0, 90, 180, 270])
            probs = cfg.rotation.get("probs", None)  # Weights for angles
            if probs:
                angle_idx = random.choices(range(len(angles)), weights=probs, k=1)[0]
                angle = float(angles[angle_idx])
            else:
                angle = float(random.choice(angles))  # Uniform choice if no probs

    # Scaling
    if cfg.scaling.get("enabled", False):
        prob_scale = cfg.scaling.get("prob", 0.5)
        if random.random() < prob_scale:
            min_scale = cfg.scaling.get("min_scale_factor", 0.9)
            max_scale = cfg.scaling.get("max_scale_factor", 1.1)
            scale = random.uniform(min_scale, max_scale)

    # Translation
    if cfg.translation.get("enabled", False):
        prob_trans = cfg.translation.get("prob", 0.5)
        if random.random() < prob_trans:
            max_shift_fraction = cfg.translation.get("max_shift_fraction", 0.1)
            max_dx = int(max_shift_fraction * image_size)
            max_dy = int(max_shift_fraction * image_size)
            tx = random.randint(-max_dx, max_dx)
            ty = random.randint(-max_dy, max_dy)
            translate_pixels = [tx, ty]

    return angle, scale, translate_pixels


def apply_affine_transform(
    img,
    angle,
    scale,
    translate_pixels,
    interpolation=InterpolationMode.NEAREST,
    padding_mode="zeros",
):
    """Applies affine transformation (scale, rotation, translation) to the image tensor."""
    if angle == 0 and scale == 1.0 and translate_pixels == [0, 0]:
        return img

    # TF.affine applies transformations wrt image center: translate -> scale -> rotate
    # Padding mode 'zeros' fills outside pixels with 0
    # Use NEAREST for binary data, BILINEAR for continuous images usually
    return TF.affine(
        img,
        angle=angle,
        translate=translate_pixels,
        scale=scale,
        shear=[0.0, 0.0],
        interpolation=interpolation,
        fill=0,
    )  # fill=0 is equivalent to padding_mode='zeros'


def apply_inverse_affine_transform(
    img,
    angle,
    scale,
    translate_pixels,
    interpolation=InterpolationMode.NEAREST,
    padding_mode="zeros",
):
    """
    Applies the inverse affine transformation step-by-step in reverse order.
    Inverse: Rotate(-angle) -> Scale(1/scale) -> Translate(-tx, -ty)
    Note: This sequential application approximates the true inverse.
          Calculating the exact inverse matrix might be more accurate but is more complex.
          For small transformations, this should be sufficient.
    """
    if angle == 0 and scale == 1.0 and translate_pixels == [0, 0]:
        return img

    # 1. Inverse Rotation
    # Need to be careful with floating point precision if scale is ~0
    if abs(angle) > 1e-6:  # Apply only if rotation is significant
        img = TF.affine(
            img,
            angle=-angle,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, 0.0],
            interpolation=interpolation,
            fill=0,
        )

    # 2. Inverse Scaling
    inv_scale = 1.0 / scale if abs(scale) > 1e-6 else 1.0  # Avoid division by zero
    if abs(inv_scale - 1.0) > 1e-6:  # Apply only if scaling is significant
        img = TF.affine(
            img,
            angle=0,
            translate=[0, 0],
            scale=inv_scale,
            shear=[0.0, 0.0],
            interpolation=interpolation,
            fill=0,
        )

    # 3. Inverse Translation
    inv_translate = [-t for t in translate_pixels]
    if (
        abs(inv_translate[0]) > 1e-6 or abs(inv_translate[1]) > 1e-6
    ):  # Apply only if translation is significant
        img = TF.affine(
            img,
            angle=0,
            translate=inv_translate,
            scale=1.0,
            shear=[0.0, 0.0],
            interpolation=interpolation,
            fill=0,
        )

    return img


# --- Helper for specific inverse calculation (May not be needed if sequential inverse is used) ---
# def get_inverse_affine_params(angle, scale, translate_pixels):
#     """Calculates parameters for the inverse affine transformation."""
#     # This is complex due to the order of operations in TF.affine.
#     # The sequential application in apply_inverse_affine_transform is preferred.
#     inv_angle = -angle
#     inv_scale = 1.0 / scale if scale != 0 else 1.0

#     # Inverse translation calculation needs to account for rotation/scaling center and order
#     # For simplicity, we'll rely on the sequential application method.
#     # Placeholder - Don't use this unless verified.
#     inv_translate_pixels = [-t for t in translate_pixels] # This is likely incorrect

#     return inv_angle, inv_scale, inv_translate_pixels
