import math
import torch
import torch.nn as nn

class Multi_Spatiotemporal_AttentionLayer(nn.Module):
    """Multi Spatio_temporal Attention(MSTA)"""

    def __init__(self, dim, h_w, kernel_size, dilation=3, gamma=2, b=1):
        super().__init__()
        self.h, self.w = h_w
        self.hw = self.h + self.w
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = (dilation * (dd_k - 1) // 2)

        t = int(abs((math.log2(dim) + b) / gamma))
        k = t if (t % 2) else (t + 1)

        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_c = nn.Conv1d(self.hw, self.hw, kernel_size=k,padding=int(k/2),bias=False)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # append a LKA
        u = x.clone()
        attn = self.conv0(x)           # depth-wise conv
        attn = self.conv_spatial(attn) # depth-wise dilation convolution
        f_x = self.conv1(attn)         # 1x1 conv
        # append a MCA
        b, c, _, _ = x.size()
        x_h = self.pool_h(x).squeeze(-1).permute(0,2,1)         # for W AvgPool, (B, C, H, 1) -> (B, H, C)
        x_w = self.pool_w(x).squeeze(-2).permute(0,2,1)         # for H AvgPool, (B, C, 1, W) -> (B, W, C)
        y = torch.cat([x_h, x_w], dim=1)                        # cat x_h & x_w -> (B, H+W, C)
        y = self.conv_c(y)                                      # 1D conv
        x_h, x_w = torch.split(y, [self.h, self.w], dim=1)      # split x_h & x_w
        a_h = self.pool(x_h.permute(0,2,1)).view(b, c, 1, 1)    # AvgPool 1D & reshape, (B, H, C) -> (B, C, 1, 1)
        a_w = self.pool(x_w.permute(0,2,1)).view(b, c, 1, 1)    # AvgPool 1D & reshape, (B, W, C) -> (B, C, 1, 1)
        a_h = self.sigmoid(a_h)                                 # sigmoid
        a_w = self.sigmoid(a_w)                                 # sigmoid
        return (a_w.expand_as(u) * a_h.expand_as(u)) * f_x * u