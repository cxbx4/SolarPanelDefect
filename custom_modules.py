import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class LWGA(nn.Module):
    def __init__(self, c1, c2=None):
        super().__init__()
        c = c1
        self.c1 = c // 4
        self.c2 = c // 4
        self.c3 = c // 4
        self.c4 = c - 3 * (c // 4)

        assert self.c1 > 0 and self.c2 > 0 and self.c3 > 0 and self.c4 > 0, (
            f"LWGA requires input channels >= 4, got c1={c1}."
        )

        self.conv1 = nn.Conv2d(self.c1, self.c1, 1, bias=False)
        self.att1 = nn.Sequential(
            nn.Conv2d(self.c1, self.c1, 1, bias=False),
            nn.BatchNorm2d(self.c1),
            nn.ReLU(inplace=True),
        )

        self.dwconv2 = nn.Conv2d(self.c2, self.c2, 3, 1, 1, groups=self.c2, bias=False)
        self.att2 = nn.Sequential(
            nn.Conv2d(self.c2, self.c2, 1, bias=False),
            nn.BatchNorm2d(self.c2),
            nn.ReLU(inplace=True),
        )

        self.dwconv3 = nn.Conv2d(self.c3, self.c3, 5, 1, 2, groups=self.c3, bias=False)
        self.att3 = nn.Sequential(
            nn.Conv2d(self.c3, self.c3, 1, bias=False),
            nn.BatchNorm2d(self.c3),
            nn.ReLU(inplace=True),
        )

        self.att4 = nn.Sequential(
            nn.Conv2d(self.c4, self.c4, 1, bias=False),
            nn.BatchNorm2d(self.c4),
            nn.ReLU(inplace=True),
        )

        self.fusion = nn.Conv2d(c1, c1, 1, bias=False)
        self.bn = nn.BatchNorm2d(c1)
        self.act = nn.SiLU()

    def forward(self, x):
        B, C, H, W = x.shape
        x1, x2, x3, x4 = torch.split(x, [self.c1, self.c2, self.c3, self.c4], dim=1)

        f1 = self.conv1(x1) * torch.sigmoid(self.att1(x1))

        gap2 = F.adaptive_avg_pool2d(x2, 1)
        f2 = self.dwconv2(x2) * torch.sigmoid(self.att2(gap2))

        gap3 = F.adaptive_avg_pool2d(x3, 1)
        f3 = self.dwconv3(x3) * torch.sigmoid(self.att3(gap3))

        gap4 = F.adaptive_avg_pool2d(x4, 1)
        f4_up = F.interpolate(gap4, size=(H, W), mode="bilinear", align_corners=False)
        f4 = f4_up * torch.sigmoid(self.att4(x4))

        out = torch.cat([f1, f2, f3, f4], dim=1)
        out = self.act(self.bn(self.fusion(out)))
        return out


class PATConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, r=0.2, d=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.att = nn.Sequential(
            nn.Conv2d(c2, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.r = r
        self.residual = (c1 == c2 and s == 1)

    def forward(self, x):
        res = x if self.residual else 0
        B, C = x.shape[:2]

        m = F.adaptive_max_pool2d(x, 1)

        k = max(1, int(C * self.r))
        topk_vals = torch.topk(m.view(B, C), k, dim=1)[0]
        threshold = topk_vals[:, -1:].view(B, 1, 1, 1)
        mask = (m >= threshold).float()

        x_masked = x * mask
        x_conv = self.conv(x_masked)
        w = torch.sigmoid(self.att(F.adaptive_avg_pool2d(x_conv, 1)))

        out = self.act(self.bn(x_conv * w))
        return out + res if self.residual else out