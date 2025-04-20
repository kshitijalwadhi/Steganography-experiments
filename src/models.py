import torch
import torch.nn as nn


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


class SteganoModel(nn.Module):
    """Full steganography model combining prep, hide, and reveal networks."""

    def __init__(self, cfg):
        super().__init__()
        secret_in_c = 3 if cfg.data.secret_type == "image" else 1
        reveal_out_c = 3 if cfg.data.secret_type == "image" else 1
        prep_out_c = cfg.model.prep_out_channels
        hide_in_c = 3 + prep_out_c  # 3 for cover image

        self.prep = PrepNet(in_c=secret_in_c, out_c=prep_out_c)
        self.hider = HideNet(in_c=hide_in_c)
        self.reveal = RevealNet(out_c=reveal_out_c)

    def forward(self, cover, secret):
        pre = self.prep(secret)
        stego = self.hider(cover, pre)
        recovered = self.reveal(stego)
        return stego, recovered
