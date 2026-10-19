import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None, :]
        return self.mlp(torch.cat([args.sin(), args.cos()], dim=1))


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, groups=8):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x) + self.time_proj(t_emb)[:, :, None, None]
        return self.block2(h) + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, channels, groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).transpose(2, 3).unbind(dim=1)
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    """Predicts the noise in y_t from [low-res image, y_t] and the timestep. For 64x64 inputs: 64 -> 32 -> 16 -> 8."""

    def __init__(self, in_channels=2, out_channels=1, base_channels=32, channel_mult=(1, 2, 4, 4),
                 res_blocks=2, attn_levels=(3,), groups=8):
        super().__init__()
        time_dim = base_channels * 4
        self.time_embedding = TimeEmbedding(base_channels)
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.down = nn.ModuleList()
        skip_channels = [base_channels]
        ch = base_channels
        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            for _ in range(res_blocks):
                self.down.append(ResBlock(ch, out_ch, time_dim, groups))
                ch = out_ch
                if level in attn_levels:
                    self.down.append(SelfAttention(ch, groups))
                skip_channels.append(ch)
            if level != len(channel_mult) - 1:
                self.down.append(Downsample(ch))
                skip_channels.append(ch)

        self.mid = nn.ModuleList([
            ResBlock(ch, ch, time_dim, groups),
            SelfAttention(ch, groups),
            ResBlock(ch, ch, time_dim, groups),
        ])

        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult
            for _ in range(res_blocks + 1):
                self.up.append(ResBlock(ch + skip_channels.pop(), out_ch, time_dim, groups))
                ch = out_ch
                if level in attn_levels:
                    self.up.append(SelfAttention(ch, groups))
            if level != 0:
                self.up.append(Upsample(ch))

        self.head = nn.Sequential(
            nn.GroupNorm(groups, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, t):
        t_emb = self.time_embedding(t)

        x = self.init_conv(x)
        skips = [x]
        for layer in self.down:
            x = layer(x, t_emb) if isinstance(layer, ResBlock) else layer(x)
            if isinstance(layer, SelfAttention):
                skips[-1] = x
            else:
                skips.append(x)

        for layer in self.mid:
            x = layer(x, t_emb) if isinstance(layer, ResBlock) else layer(x)

        for layer in self.up:
            if isinstance(layer, ResBlock):
                x = layer(torch.cat([x, skips.pop()], dim=1), t_emb)
            else:
                x = layer(x)

        return self.head(x)


class GaussianDiffusion(nn.Module):
    def __init__(self, denoiser, timesteps=1000, beta_start=1e-4, beta_end=2e-2):
        super().__init__()
        self.denoiser = denoiser
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        self.register_buffer("alphas_cumprod", torch.cumprod(1.0 - betas, dim=0))

    def loss(self, x, y):
        """Add noise to the target y at a random timestep and train the denoiser to predict that noise."""
        t = torch.randint(0, self.timesteps, (len(y),), device=y.device)
        noise = torch.randn_like(y)
        a = self.alphas_cumprod[t][:, None, None, None]
        y_t = a.sqrt() * y + (1.0 - a).sqrt() * noise
        return F.mse_loss(self.denoiser(torch.cat([x, y_t], dim=1), t), noise)

    @torch.no_grad()
    def sample(self, x, steps=None):
        """DDPM sampling from pure noise, conditioned on the low-res image x. Fewer steps skip timesteps."""
        timesteps = torch.linspace(self.timesteps - 1, 0, steps or self.timesteps, device=x.device).long()
        alphas = self.alphas_cumprod[timesteps]
        alphas_prev = torch.cat([alphas[1:], alphas.new_ones(1)])

        y = torch.randn_like(x)
        for t, a_t, a_prev in zip(timesteps, alphas, alphas_prev):
            eps = self.denoiser(torch.cat([x, y], dim=1), t.expand(len(x)))
            y0 = ((y - (1.0 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1.0, 1.0)
            sigma = ((1.0 - a_prev) / (1.0 - a_t)).sqrt() * (1.0 - a_t / a_prev).sqrt()
            direction = (1.0 - a_prev - sigma ** 2).clamp(min=0.0).sqrt() * eps
            y = a_prev.sqrt() * y0 + direction + sigma * torch.randn_like(y)
        return y


def build_diffusion(timesteps=1000):
    return GaussianDiffusion(UNet(), timesteps=timesteps)


def train_step(diffusion, optimizer, x, y):
    diffusion.train()
    loss = diffusion.loss(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {"loss": loss.item()}