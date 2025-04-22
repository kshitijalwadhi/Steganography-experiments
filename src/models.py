import torch
import torch.nn as nn

# Try importing escnn and handle potential import error
try:
    import escnn.gspaces
    import escnn.nn
    from escnn.nn import GeometricTensor
    ESCnn_available = True
except ImportError:
    ESCnn_available = False
    print("WARNING: escnn library not found. EquivariantRevealNet will not be available.")


# Network components
def conv_block(in_c, out_c, kernel=3, stride=1):
    """Create a convolution block with batch norm and ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel, stride, padding=kernel // 2, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class PrepNet(nn.Module):
    """Network to prepare the secret binary message."""

    def __init__(self, in_c=1, out_c=32):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(in_c, out_c, 5),  # Adjusted based on config
            conv_block(out_c, out_c * 2, 3),
            conv_block(out_c * 2, out_c, 3),
        )

    def forward(self, x):
        return self.net(x)


class HideNet(nn.Module):
    """Tiny U-Net (1 downsample + 1 upsample) to hide secret in cover image."""

    def __init__(self, in_c=35):  # Adjusted based on config (3 + prep_out_channels)
        super().__init__()
        # Encoder
        self.enc1 = conv_block(in_c, 64, 3)
        self.enc2 = conv_block(64, 128, 3, stride=2)
        # Bottleneck
        self.bottleneck = conv_block(128, 128, 3)
        # Decoder
        self.up = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(64 + 64, 64, 3)
        self.out_conv = nn.Conv2d(64, 3, 1)  # Output is always 3 channels (RGB image)

    def forward(self, cover, pre_secret):
        x = torch.cat([cover, pre_secret], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        u = self.up(b)
        # Potential size mismatch correction if needed (unlikely with kernel=2, stride=2)
        # u = F.interpolate(u, size=e1.shape[2:], mode='bilinear', align_corners=False)
        u = torch.cat([u, e1], dim=1)
        d1 = self.dec1(u)
        return torch.sigmoid(self.out_conv(d1))


# Standard RevealNet (keep for comparison or fallback)
class RevealNet(nn.Module):
    """Network to reveal the hidden secret from a stego image."""

    def __init__(self, out_c=1):  # Adjusted based on config (1 for binary, 3 for image)
        super().__init__()
        self.conv1 = conv_block(3, 32, 5)
        self.conv2 = conv_block(32, 32, 3)
        self.conv3 = conv_block(32, 64, 3)
        self.conv4 = conv_block(64, 64, 3)
        self.conv5 = nn.Conv2d(64, out_c, 1)

    def forward(self, stego):
        x = self.conv1(stego)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.conv5(
            x
        )  # logits for binary, sigmoid/tanh for image? Output raw for now.


# Equivariant RevealNet using escnn (C4 Rotation Equivariance)
class EquivariantRevealNet(nn.Module):
    def __init__(self, out_c=1, channels=32):
        super().__init__()

        if not ESCnn_available:
             raise RuntimeError("Cannot initialize EquivariantRevealNet: escnn library not installed.")

        self.channels = channels
        self.out_channels = out_c

        # Define the symmetry group: C4 rotations on R2
        self.gspace = escnn.gspaces.rot2dOnR2(N=4)

        # Define the input field type: 3 trivial fields (standard RGB image)
        self.in_type = escnn.nn.FieldType(self.gspace, [self.gspace.trivial_repr] * 3)

        # Define hidden field types using regular representations
        # Number of output channels needs to be scaled by group order for regular repr?
        # Let's try keeping channels similar to original for now.
        feat_type_hid1 = escnn.nn.FieldType(self.gspace, [self.gspace.regular_repr] * channels)
        feat_type_hid2 = escnn.nn.FieldType(self.gspace, [self.gspace.regular_repr] * (channels * 2))

        # Define the output field type: MUST be invariant (trivial representation)
        # Since we want a single scalar output per pixel for binary secret
        self.out_type = escnn.nn.FieldType(self.gspace, [self.gspace.trivial_repr] * self.out_channels)

        # Build the equivariant network layers
        self.block1 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(self.in_type, feat_type_hid1, kernel_size=5, padding=2, bias=False),
            escnn.nn.InnerBatchNorm(feat_type_hid1),
            escnn.nn.ReLU(feat_type_hid1, inplace=False)
        )
        self.block2 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(feat_type_hid1, feat_type_hid1, kernel_size=3, padding=1, bias=False),
            escnn.nn.InnerBatchNorm(feat_type_hid1),
            escnn.nn.ReLU(feat_type_hid1, inplace=False)
        )
        self.block3 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(feat_type_hid1, feat_type_hid2, kernel_size=3, padding=1, bias=False),
            escnn.nn.InnerBatchNorm(feat_type_hid2),
            escnn.nn.ReLU(feat_type_hid2, inplace=False)
        )
        self.block4 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(feat_type_hid2, feat_type_hid2, kernel_size=3, padding=1, bias=False),
            escnn.nn.InnerBatchNorm(feat_type_hid2),
            escnn.nn.ReLU(feat_type_hid2, inplace=False)
        )

        # Final layer: Conv + Group Pooling to produce invariant output
        self.final_conv = escnn.nn.R2Conv(feat_type_hid2, self.out_type, kernel_size=1, bias=True)
        # GroupPooling aggregates over the group dimension, resulting in an invariant field.
        # self.pool = escnn.nn.GroupPooling(feat_type_hid2) # Alternative: pool before final conv? No, final conv needs to map to out_type.


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Wrap input tensor in a GeometricTensor
        x_geom = escnn.nn.GeometricTensor(x, self.in_type)

        # Pass through equivariant blocks
        out1 = self.block1(x_geom)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)

        # Final convolution to get invariant output type
        out5 = self.final_conv(out4)

        # The output of final_conv should already be of the invariant out_type.
        # Extract the torch.Tensor from the GeometricTensor.
        return out5.tensor


class SteganoModel(nn.Module):
    """Full steganography model combining prep, hide, and reveal networks."""

    def __init__(self, cfg, use_equivariant_reveal=True): # Add flag
        super().__init__()
        secret_in_c = 3 if cfg.data.secret_type == "image" else 1
        reveal_out_c = 3 if cfg.data.secret_type == "image" else 1
        prep_out_c = cfg.model.prep_out_channels
        hide_in_c = 3 + prep_out_c  # 3 for cover image

        self.prep = PrepNet(in_c=secret_in_c, out_c=prep_out_c)
        self.hider = HideNet(in_c=hide_in_c)

        # Conditionally use EquivariantRevealNet
        if use_equivariant_reveal and ESCnn_available:
            print("Using Equivariant RevealNet (escnn)")
            # TODO: Consider passing channel size from config?
            self.reveal = EquivariantRevealNet(out_c=reveal_out_c)
        else:
            if use_equivariant_reveal and not ESCnn_available:
                print("WARNING: Requested EquivariantRevealNet but escnn is not installed. Falling back to standard RevealNet.")
            else:
                 print("Using Standard RevealNet")
            self.reveal = RevealNet(out_c=reveal_out_c)

    def forward(self, cover, secret):
        pre = self.prep(secret)
        stego = self.hider(cover, pre)
        recovered = self.reveal(stego)
        return stego, recovered
