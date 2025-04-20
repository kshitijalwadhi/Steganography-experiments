import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from PIL import Image


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
            conv_block(in_c, 32, 5),
            conv_block(32, 64, 3),
            conv_block(64, out_c, 3),
        )

    def forward(self, x):
        return self.net(x)


class HideNet(nn.Module):
    """Tiny U-Net (1 downsample + 1 upsample) to hide secret in cover image."""
    def __init__(self, in_c=3 + 32):
        super().__init__()
        # Encoder
        self.enc1 = conv_block(in_c, 64, 3)
        self.enc2 = conv_block(64, 128, 3, stride=2)
        # Bottleneck
        self.bottleneck = conv_block(128, 128, 3)
        # Decoder
        self.up = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(64 + 64, 64, 3)
        self.out_conv = nn.Conv2d(64, 3, 1)

    def forward(self, cover, pre_secret):
        x = torch.cat([cover, pre_secret], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        u = self.up(b)
        u = torch.cat([u, e1], dim=1)
        d1 = self.dec1(u)
        return torch.sigmoid(self.out_conv(d1))


class RevealNet(nn.Module):
    """Network to reveal the hidden secret from a stego image."""
    def __init__(self, out_c=1):
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
        return self.conv5(x)  # logits


class SteganoModel(nn.Module):
    """Full steganography model combining prep, hide, and reveal networks."""
    def __init__(self):
        super().__init__()
        self.prep = PrepNet()
        self.hider = HideNet()
        self.reveal = RevealNet()

    def forward(self, cover, secret):
        pre = self.prep(secret)
        stego = self.hider(cover, pre)
        recovered = self.reveal(stego)
        return stego, recovered


# Helper metrics
def psnr(mse: torch.Tensor):
    """Calculate Peak Signal-to-Noise Ratio from MSE."""
    return 10 * math.log10(1.0 / mse.clamp(min=1e-10))


def bit_accuracy(logits: torch.Tensor, target: torch.Tensor):
    """Calculate bit accuracy between binary predictions and targets."""
    preds = (torch.sigmoid(logits) > 0.5).float()
    return (preds == target).float().mean()


# Training and embedding utilities
def train_demo(device=None, steps=100):
    """Quick training run on CIFAR-10 covers. Returns trained model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data preparation
    tfm = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
    ])
    data = datasets.CIFAR10("./data", train=True, download=True, transform=tfm)
    loader = DataLoader(data, batch_size=8, shuffle=True, num_workers=2)

    # Model setup
    model = SteganoModel().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    bce = nn.BCEWithLogitsLoss()
    lam_hide, lam_reveal = 0.25, 1.0

    # Training loop
    it = iter(loader)
    for step in range(steps):
        try:
            cover, _ = next(it)
        except StopIteration:
            it = iter(loader)
            cover, _ = next(it)
            
        cover = cover.to(device)
        secret = torch.randint(0, 2, (cover.size(0), 1, 64, 64), device=device).float()

        stego, rec = model(cover, secret)
        loss_hide = F.mse_loss(stego, cover)
        loss_reveal = bce(rec, secret)
        loss = lam_hide * loss_hide + lam_reveal * loss_reveal

        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % 10 == 0:
            acc = bit_accuracy(rec, secret).item()
            print(f"Step {step:03d} | L_hide={loss_hide.item():.4f} | L_msg={loss_reveal.item():.4f} "
                  f"| PSNR={psnr(loss_hide):.2f}dB | Acc={acc*100:.1f}%")
    
    return model.cpu()


def embed_image(model: SteganoModel, cover_path, out_path="stego.png", device=None):
    """Embed a random binary secret in a cover image and save the result."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device).eval()

    # Load and prepare image
    pil = Image.open(cover_path).convert("RGB")
    tfm = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])
    cover = tfm(pil).unsqueeze(0).to(device)
    secret = torch.randint(0, 2, (1, 1, 64, 64), device=device).float()

    # Generate stego image
    with torch.no_grad():
        stego, rec = model(cover, secret)
        acc = bit_accuracy(rec, secret).item()
        mse = F.mse_loss(stego, cover)
    
    # Save and report
    TF.to_pil_image(stego.squeeze(0).cpu()).save(out_path)
    print(f"Embedded! PSNR={psnr(mse):.2f} dB | Acc={acc*100:.1f}% | saved to {out_path}")


# Auto-run when inside IPython/Jupyter
def jupyter_auto_run():
    """Auto-run quick demo when in Jupyter environment."""
    if "get_ipython" in globals():
        print("🔄 Detected Jupyter environment – starting quick demo (100 steps)…")
        demo_model = train_demo(steps=100)
        print("✅ Demo finished. You now have `demo_model` in memory.")
        return demo_model
    return None


# CLI functionality
def main():
    """Command-line interface for training and embedding."""
    if "get_ipython" in globals():
        return jupyter_auto_run()
        
    import argparse

    p = argparse.ArgumentParser("Simple Deep Stego – CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Train command
    t = sub.add_parser("train", help="train demo and save checkpoint")
    t.add_argument("--ckpt", type=str, default="", help="where to save .pt weights")
    t.add_argument("--steps", type=int, default=100)

    # Embed command
    e = sub.add_parser("embed", help="embed one image")
    e.add_argument("cover", type=str)
    e.add_argument("--ckpt", type=str, default="", help="load weights")
    e.add_argument("--out", type=str, default="stego.png")

    args = p.parse_args()

    if args.cmd == "train":
        m = train_demo(steps=args.steps)
        if args.ckpt:
            torch.save(m.state_dict(), args.ckpt)
            print(f"Weights saved to {args.ckpt}")
    else:  # embed
        m = SteganoModel()
        if args.ckpt:
            m.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
            print(f"Loaded weights from {args.ckpt}")
        embed_image(m, args.cover, args.out)


if __name__ == "__main__":
    main()
else:
    jupyter_auto_run()