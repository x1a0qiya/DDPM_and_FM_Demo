import torch
import torch.nn as nn
import math

"""
Time Positional Embedding
"""
class TiPE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t : torch.tensor):
        t = t * 1000.0

        emb = math.log(10000.0) / (self.dim // 2 - 1)
        emb = torch.exp(torch.arange(self.dim // 2, device = t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb


class Double_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, tim_dim):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )

        self.tim = nn.Sequential(
            nn.SiLU(),
            nn.Linear(tim_dim, out_channels * 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )

        self.align_conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x : torch.tensor, tipe : torch.tensor):
        xt = self.conv1(x)

        t = self.tim(tipe).unsqueeze(-1).unsqueeze(-1)
        scaling, translation = t.chunk(2, dim = 1)

        xt = xt * (1 + scaling) + translation

        xt = self.conv2(xt)
        return xt + self.align_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 3, base_channels = 64, tim_dim = 256):
        super().__init__()

        """
        Encoder
        """

        self.tipe = TiPE(base_channels)
        self.tipe_p = nn.Sequential(
            nn.Linear(base_channels, tim_dim),
            nn.SiLU(),
            nn.Linear(tim_dim, tim_dim)
        )

        self.inp = Double_Conv(in_channels, base_channels, tim_dim)

        self.pool = nn.MaxPool2d(kernel_size = 2)

        self.down1 = Double_Conv(base_channels, base_channels * 2, tim_dim)        
        self.down2 = Double_Conv(base_channels * 2, base_channels * 4, tim_dim)     
        self.down3 = Double_Conv(base_channels * 4, base_channels * 8, tim_dim)           
        self.down4 = Double_Conv(base_channels * 8, base_channels * 16, tim_dim)            

        """
        Decoder
        """

        self.up = nn.Upsample(scale_factor = 2, mode = "bilinear", align_corners = True)

        self.up_conv1 = Double_Conv(base_channels * 16 + base_channels * 8, base_channels * 8, tim_dim)
        self.up_conv2 = Double_Conv(base_channels * 8 + base_channels * 4, base_channels * 4, tim_dim)
        self.up_conv3 = Double_Conv(base_channels * 4 + base_channels * 2, base_channels * 2, tim_dim)
        self.up_conv4 = Double_Conv(base_channels * 2 + base_channels, base_channels, tim_dim)

        self.outp = nn.Conv2d(base_channels, out_channels, kernel_size = 3, padding = 1)


    def forward(self, x : torch.tensor, t : torch.tensor):
        t_ = self.tipe(t)
        t_ = self.tipe_p(t_)

        x0 = self.inp(x, t_)

        x1 = self.down1(self.pool(x0), t_)
        x2 = self.down2(self.pool(x1), t_)
        x3 = self.down3(self.pool(x2), t_)
        x4 = self.down4(self.pool(x3), t_)

        xt = self.up(x4)
        xt = torch.cat([x3, xt], dim = 1)
        xt = self.up_conv1(xt, t_)

        xt = self.up(xt)
        xt = torch.cat([x2, xt], dim = 1)
        xt = self.up_conv2(xt, t_)

        xt = self.up(xt)
        xt = torch.cat([x1, xt], dim = 1)
        xt = self.up_conv3(xt, t_)

        xt = self.up(xt)
        xt = torch.cat([x0, xt], dim = 1)
        xt = self.up_conv4(xt, t_)

        xt = self.outp(xt)

        return xt