import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import escnn
    from escnn import gspaces
    from escnn import nn as enn
    ESCNN_AVAILABLE = True
except ImportError:
    ESCNN_AVAILABLE = False


# Network components
def conv_block(in_c, out_c, kernel=3, stride=1):
    """Create a convolution block with batch norm and ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel, stride, padding=kernel // 2, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


def gconv_block(in_type, out_type, kernel=3, stride=1):
    """Create a group-equivariant convolution block with batch norm and ReLU."""
    return nn.Sequential(
        enn.R2Conv(in_type, out_type, kernel_size=kernel, stride=stride, 
                  padding=kernel // 2, bias=False),
        enn.InnerBatchNorm(out_type),
        enn.ReLU(out_type, inplace=True),
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


class HideNetEquivariant(nn.Module):
    """G-equivariant U-Net to hide secret in cover image, respecting C4 rotation symmetry."""

    def __init__(self, in_c=35):  # Adjusted based on config (3 + prep_out_channels)
        super().__init__()
        if not ESCNN_AVAILABLE:
            raise ImportError("escnn package is required for equivariant networks")
        
        # Setup the group symmetry (C4 - 4-fold rotations)
        self.gspace = gspaces.rot2dOnR2(N=4)
        
        # Input has (3 cover + in_c-3 secret) channels with trivial representation
        # Using trivial representation for input ensures the tensor dimensions match
        cover_channels = 3
        secret_channels = in_c - cover_channels
        self.in_type = enn.FieldType(self.gspace, 
                                    (cover_channels + secret_channels)*[self.gspace.regular_repr])
        
        # First layer converts from trivial to regular representation
        first_hidden_type = enn.FieldType(self.gspace, 16*[self.gspace.regular_repr])
        self.initial_map = enn.R2Conv(self.in_type, first_hidden_type, kernel_size=1)
        
        # Define intermediate feature types
        self.enc1_type = enn.FieldType(self.gspace, 16*[self.gspace.regular_repr])
        self.enc2_type = enn.FieldType(self.gspace, 32*[self.gspace.regular_repr])
        self.bottleneck_type = enn.FieldType(self.gspace, 32*[self.gspace.regular_repr])
        self.dec_type = enn.FieldType(self.gspace, 16*[self.gspace.regular_repr])
        
        # Output has 3 trivial channels (RGB image)
        self.out_type = enn.FieldType(self.gspace, 3*[self.gspace.trivial_repr])
        
        # Encoder
        self.enc1 = gconv_block(first_hidden_type, self.enc1_type, kernel=3)
        self.enc2 = gconv_block(self.enc1_type, self.enc2_type, kernel=3, stride=2)
        
        # Bottleneck
        self.bottleneck = gconv_block(self.enc2_type, self.bottleneck_type, kernel=3)
        
        # Decoder - Using a regular upsampling + conv approach instead of ConvTransposed
        # This avoids the empty basis issue
        self.up = enn.R2Upsampling(self.bottleneck_type, scale_factor=2)
        self.post_up_conv = gconv_block(self.bottleneck_type, self.dec_type, kernel=3)
        
        # For the skip connection with concatenation
        self.skip_type = enn.FieldType(self.gspace, 
                                      16*[self.gspace.regular_repr] + 
                                      16*[self.gspace.regular_repr])
        self.dec1 = gconv_block(self.skip_type, self.dec_type, kernel=3)
        
        # Output layer
        self.out_conv = enn.R2Conv(self.dec_type, self.out_type, kernel_size=1)

    def forward(self, cover, pre_secret):
        # Create geometric tensor with trivial representation
        x = torch.cat([cover, pre_secret], dim=1)
        B, C, H, W = x.shape
        x = x.unsqueeze(2).repeat(1,1,4,1,1)         # [B,35,4,H,W]
        x = x.view(B, C * 4, H, W)
        x = enn.GeometricTensor(x, self.in_type)
        
        # Initial mapping to regular representation
        x = self.initial_map(x)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        
        # Bottleneck
        b = self.bottleneck(e2)
        
        # Decoder with skip connection
        u = self.up(b)
        u = self.post_up_conv(u)
        
        # Handle concatenation (needs to combine appropriate field types)
        cat_tensor = enn.tensor_directsum([u, e1])
        
        d1 = self.dec1(cat_tensor)
        out = self.out_conv(d1)
        
        # Convert back to standard tensor
        return torch.sigmoid(out.tensor)


class RevealNetRotInvariant(nn.Module):
    """Rotation invariant network to reveal the hidden secret from a stego image using escnn."""

    def __init__(self, out_c=1):
        super().__init__()
        if not ESCNN_AVAILABLE:
            raise ImportError("escnn package is required for rotation invariant networks")
        
        # Define the symmetry group C4 for 90-degree rotations
        self.r4_act = gspaces.rot2dOnR2(N=4)
        
        # Define input type (3 channels RGB image with C4 symmetry)
        self.input_type = enn.FieldType(self.r4_act, 3 * [self.r4_act.trivial_repr])
        
        # First conv layer - from input type to 32 regular features
        hidden_type1 = enn.FieldType(self.r4_act, 8 * [self.r4_act.regular_repr])
        self.conv1 = enn.R2Conv(self.input_type, hidden_type1, kernel_size=5, padding=2)
        self.bn1 = enn.InnerBatchNorm(hidden_type1)
        self.relu1 = enn.ReLU(hidden_type1)
        
        # Second conv layer
        hidden_type2 = enn.FieldType(self.r4_act, 8 * [self.r4_act.regular_repr])
        self.conv2 = enn.R2Conv(hidden_type1, hidden_type2, kernel_size=3, padding=1)
        self.bn2 = enn.InnerBatchNorm(hidden_type2)
        self.relu2 = enn.ReLU(hidden_type2)
        
        # Third conv layer
        hidden_type3 = enn.FieldType(self.r4_act, 16 * [self.r4_act.regular_repr])
        self.conv3 = enn.R2Conv(hidden_type2, hidden_type3, kernel_size=3, padding=1)
        self.bn3 = enn.InnerBatchNorm(hidden_type3)
        self.relu3 = enn.ReLU(hidden_type3)
        
        # Fourth conv layer
        hidden_type4 = enn.FieldType(self.r4_act, 16 * [self.r4_act.regular_repr])
        self.conv4 = enn.R2Conv(hidden_type3, hidden_type4, kernel_size=3, padding=1)
        self.bn4 = enn.InnerBatchNorm(hidden_type4)
        self.relu4 = enn.ReLU(hidden_type4)
        
        # Output conv layer (invariant mapping to out_c channels)
        self.output_type = enn.FieldType(self.r4_act, out_c * [self.r4_act.trivial_repr])
        self.conv5 = enn.R2Conv(hidden_type4, self.output_type, kernel_size=1)
        
        # Invariant map to convert output back to standard tensor
        self.invariant_map = enn.GroupPooling(self.output_type)
        
        self.out_c = out_c

    def forward(self, stego):
        # Convert standard tensor to geometric tensor
        x = enn.GeometricTensor(stego, self.input_type)
        
        # Apply equivariant layers
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        
        # Pool over group to achieve invariance
        x = self.invariant_map(x)
        
        # Return as standard tensor
        return x.tensor


class RevealNet(nn.Module):
    """Network to reveal the hidden secret from a stego image."""

    def __init__(self, out_c=1, rotation_invariant=False):  # Added rotation_invariant flag
        super().__init__()
        self.rotation_invariant = rotation_invariant
        
        if rotation_invariant:
            if not ESCNN_AVAILABLE:
                raise ImportError("escnn package is required for rotation invariant networks")
            self.net = RevealNetRotInvariant(out_c=out_c)
        else:
            self.conv1 = conv_block(3, 32, 5)
            self.conv2 = conv_block(32, 32, 3)
            self.conv3 = conv_block(32, 64, 3)
            self.conv4 = conv_block(64, 64, 3)
            self.conv5 = nn.Conv2d(64, out_c, 1)

    def forward(self, stego):
        if self.rotation_invariant:
            return self.net(stego)
        else:
            x = self.conv1(stego)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            return self.conv5(x)  # logits for binary, sigmoid/tanh for image? Output raw for now.


class SteganoModel(nn.Module):
    """Full steganography model combining prep, hide, and reveal networks."""

    def __init__(self, cfg):
        super().__init__()
        secret_in_c = 3 if cfg.data.secret_type == "image" else 1
        reveal_out_c = 3 if cfg.data.secret_type == "image" else 1
        prep_out_c = cfg.model.prep_out_channels
        hide_in_c = 3 + prep_out_c  # 3 for cover image
        # Get rotation invariance flag from config (default to False if not specified)
        rot_invariant = getattr(cfg.model, "rotation_invariant", False)

        self.prep = PrepNet(in_c=secret_in_c, out_c=prep_out_c)
        
        # Use equivariant HideNet when rotation invariance is enabled
        if rot_invariant and ESCNN_AVAILABLE:
            self.hider = HideNetEquivariant(in_c=hide_in_c)
        else:
            self.hider = HideNet(in_c=hide_in_c)
            
        self.reveal = RevealNet(out_c=reveal_out_c, rotation_invariant=rot_invariant)

    def forward(self, cover, secret):
        pre = self.prep(secret)
        stego = self.hider(cover, pre)
        recovered = self.reveal(stego)
        return stego, recovered
