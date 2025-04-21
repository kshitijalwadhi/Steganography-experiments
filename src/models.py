import torch
import torch.nn as nn
import torch.nn.functional as F


# Utility functions for rotation handling
def to_polar(x):
    """Convert Cartesian coordinates to polar coordinates.

    Args:
        x: Tensor of shape (B, C, H, W)

    Returns:
        Tensor of shape (B, 4, H, W) containing [r, theta, r^2, theta^2]
        Note: Output channels = 4, not C+4. Concatenation happens later.
    """
    # Create coordinate grids
    B, _, H, W = x.shape # Use _ for C as we only need B, H, W
    h_center, w_center = H // 2, W // 2

    y_grid = torch.arange(H, device=x.device).view(-1, 1).repeat(1, W) - h_center
    x_grid = torch.arange(W, device=x.device).view(1, -1).repeat(H, 1) - w_center
    y_grid = y_grid.float() / H * 2
    x_grid = x_grid.float() / W * 2

    # Convert to polar
    r = torch.sqrt(x_grid.pow(2) + y_grid.pow(2))
    theta = torch.atan2(y_grid, x_grid)

    # Normalize r to [0, 1] and theta to [0, 1]
    # Avoid division by zero if max(r) is 0 (e.g., for a single pixel image)
    max_r = torch.max(r)
    r = r / (max_r + 1e-8)  # Add epsilon for stability
    theta = (theta + torch.pi) / (2 * torch.pi)

    # Calculate powers safely
    r_squared = r.pow(2)
    theta_squared = theta.pow(2)

    # Create 4-channel polar features [r, theta, r^2, theta^2]
    polar_grid = (
        torch.stack([r, theta, r_squared, theta_squared], dim=0)
        .unsqueeze(0)
        .repeat(B, 1, 1, 1)
    )

    return polar_grid # Return only the 4 polar channels


# Network components
def conv_block(in_c, out_c, kernel=3, stride=1):
    """Create a convolution block with batch norm and ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel, stride, padding=kernel // 2, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class PrepNet(nn.Module):
    """Network to prepare the secret message (binary or image)."""

    def __init__(self, in_c=1, out_c=32):  # Default in_c assumes binary/grayscale
        super().__init__()
        print(f"Initializing PrepNet with in_channels={in_c}")
        self.net = nn.Sequential(
            conv_block(in_c, out_c, 5),
            conv_block(out_c, out_c * 2, 3),
            conv_block(out_c * 2, out_c, 3),
        )

    def forward(self, x):
        return self.net(x)


class HideNet(nn.Module):
    """Enhanced U-Net (can be rotation aware or not) to hide secret in cover image.

    Accepts cover, polar coordinates of cover, and prepared secret.
    """

    def __init__(self, in_c=35 + 4): # Should match 3 (cover) + 4 (polar) + prep_out_channels
        super().__init__()
        print(f"Initializing HideNet with in_channels={in_c} (including polar)")
        # Encoder
        self.enc1 = conv_block(in_c, 64, 3)
        self.enc2 = conv_block(64, 128, 3, stride=2)
        # Bottleneck
        self.bottleneck = conv_block(128, 128, 3)
        # Decoder
        self.up = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(64 + 64, 64, 3)
        self.out_conv = nn.Conv2d(
            64, 3, 1
        )  # Output is always 3 channels (RGB stego image)

    def forward(self, cover, polar_grid_cover, pre_secret):
        # Concatenate cover, its polar grid, and the prepared secret
        x = torch.cat([cover, polar_grid_cover, pre_secret], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        u = self.up(b)
        # Handle potential size mismatch
        if u.shape[2:] != e1.shape[2:]:
            u = F.interpolate(
                u, size=e1.shape[2:], mode="bilinear", align_corners=False
            )
        u = torch.cat([u, e1], dim=1)
        d1 = self.dec1(u)
        # Sigmoid ensures stego image pixels are in [0, 1] range
        return torch.sigmoid(self.out_conv(d1))


class RotationInvariantRevealNet(nn.Module):
    """Rotation-invariant network using polar coords to reveal secret."""

    def __init__(self, out_c=1):  # Default out_c assumes binary/grayscale
        super().__init__()
        print(f"Initializing RotationInvariantRevealNet with out_channels={out_c}")
        # Initial convolution expects 3-channel stego image
        self.initial_conv = conv_block(3, 32, 5)

        # Polar branch: input is initial features (32) + polar coords (4) = 36 channels
        self.polar_branch = nn.Sequential(
            conv_block(32 + 4, 64, 3),
            conv_block(64, 64, 3),
            conv_block(64, 64, 3),
        )

        # Global pooling branch: operates on initial 32 features
        self.global_branch = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(32, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Combined processing: Polar branch output (64) + Global branch output (32) = 96 channels
        self.combined = nn.Sequential(
            conv_block(64 + 32, 64, 3),
            conv_block(64, 32, 3),
            nn.Conv2d(32, out_c, 1),  # Final output has out_c channels
        )

    def forward(self, stego):
        # Initial feature extraction from stego image (B, 3, H, W) -> (B, 32, H, W)
        features = self.initial_conv(stego)

        # Add polar coordinate information to features (B, 32, H, W) -> (B, 36, H, W)
        stego_polar_grid = to_polar(features) # Use features? Or original stego?
                                             # Original RevealNet paper used polar on input image. Let's try polar on stego input.
        stego_polar_grid_input = to_polar(stego) # Get polar coords from the input stego itself.
        polar_features = torch.cat([features, stego_polar_grid_input], dim=1)


        # Process through polar branch (B, 36, H, W) -> (B, 64, H, W)
        polar_branch_out = self.polar_branch(polar_features)

        # Process through global branch (B, 32, H, W) -> (B, 32, 1, 1)
        global_features = self.global_branch(features)

        # Expand global features to match spatial dimensions (B, 32, 1, 1) -> (B, 32, H, W)
        global_features = global_features.expand(
            -1, -1, features.size(2), features.size(3)
        )

        # Combine branches (B, 64, H, W) + (B, 32, H, W) -> (B, 96, H, W)
        combined = torch.cat([polar_branch_out, global_features], dim=1)

        # Final processing (B, 96, H, W) -> (B, out_c, H, W)
        output = self.combined(combined)

        # Output raw logits/values, loss function will handle normalization if needed
        return output


class SteganoModel(nn.Module):
    """Full steganography model using potentially rotation-aware components."""

    def __init__(self, cfg):
        super().__init__()

        # Determine input channels for PrepNet based on secret type and dataset
        if cfg.data.secret_type == "binary":
            secret_in_c = 1
        elif cfg.data.secret_type == "image":
            if cfg.data.get("secret_dataset") == "MNIST":
                secret_in_c = 1  # MNIST is grayscale
            else:  # Assume 3 channels for other image types
                secret_in_c = 3
        else:
            raise ValueError(f"Unsupported secret_type: {cfg.data.secret_type}")

        # Determine output channels for RevealNet (should match secret channels)
        reveal_out_c = secret_in_c

        prep_out_c = cfg.model.prep_out_channels
        # HideNet input channels: 3 (cover) + 4 (polar grid) + prep_out_c
        expected_hide_in_c = 3 + 4 + prep_out_c

        # If config has hide_in_channels, check if it matches the expected value
        config_hide_in_c = cfg.model.get("hide_in_channels", None)
        if config_hide_in_c is not None and config_hide_in_c != expected_hide_in_c:
            print(
                f"Warning: Config mismatch! model.hide_in_channels ({config_hide_in_c}) != 3 (cover) + 4 (polar) + model.prep_out_channels ({prep_out_c}) = {expected_hide_in_c}. Using calculated value {expected_hide_in_c}."
            )
            hide_in_c = expected_hide_in_c
        else:
            # Use expected value if config doesn't specify or matches
            hide_in_c = expected_hide_in_c

        print(
            f"Initializing SteganoModel: PrepNet in={secret_in_c}, RevealNet out={reveal_out_c}, HideNet in={hide_in_c} (incl. polar)"
        )

        self.prep = PrepNet(in_c=secret_in_c, out_c=prep_out_c)
        self.hider = HideNet(in_c=hide_in_c)
        # Use the rotation invariant reveal network
        self.reveal = RotationInvariantRevealNet(out_c=reveal_out_c)

    def forward(self, cover, secret):
        """Standard forward pass: Cover + Secret -> Stego -> Recovered Secret"""
        pre = self.prep(secret)
        # Calculate polar grid for the cover
        cover_polar_grid = to_polar(cover)
        # Pass cover, its polar grid, and prepared secret to hider
        stego = self.hider(cover, cover_polar_grid, pre)
        recovered = self.reveal(stego)  # Reveal from non-rotated stego
        return stego, recovered

    def hide_secret(self, cover, secret):
        """Generate stego image, including polar coords for HideNet."""
        pre = self.prep(secret)
        # Calculate polar grid for the input cover (could be rotated or not)
        cover_polar_grid = to_polar(cover)
        # Pass cover, its polar grid, and prepared secret to hider
        stego = self.hider(cover, cover_polar_grid, pre)
        return stego

    def reveal_secret(self, stego):
        """Reveal secret from stego image using RotationInvariantRevealNet."""
        recovered = self.reveal(stego)
        return recovered
