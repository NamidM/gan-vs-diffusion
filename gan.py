import torch
import torch.nn as nn
import torch.nn.functional as F

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, apply_norm=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels) if apply_norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.net(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return x


class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.encoder = nn.ModuleList([
            DownBlock(in_channels, 64, apply_norm=False),
            DownBlock(64, 128),
            DownBlock(128, 256),
            DownBlock(256, 512),
        ])
        self.decoder = nn.ModuleList([
            UpBlock(512, 256),
            UpBlock(512, 128),
            UpBlock(256, 64),
            UpBlock(128, 64),
        ])
        self.head = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)

        x = self.decoder[0](x, skips[-2])
        x = self.decoder[1](x, skips[-3])
        x = self.decoder[2](x, skips[-4])
        x = self.decoder[3](x)
        return torch.tanh(self.head(x))


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))


def train_step(generator, discriminator, optimizer_G, optimizer_D, x, y, l1_weight=10.0):
    generator.train()
    discriminator.train()

    fake = generator(x)

    d_real = discriminator(x, y)
    d_fake = discriminator(x, fake.detach())
    loss_D = (
        F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
        + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
    )
    optimizer_D.zero_grad()
    loss_D.backward()
    optimizer_D.step()

    d_fake = discriminator(x, fake)
    adv_loss = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
    l1_loss = F.l1_loss(fake, y)
    loss_G = adv_loss + l1_weight * l1_loss
    optimizer_G.zero_grad()
    loss_G.backward()
    optimizer_G.step()

    return {
        "loss_G": loss_G.item(),
        "loss_D": loss_D.item(),
        "adv_loss": adv_loss.item(),
        "l1_loss": l1_loss.item(),
    }