import torch
import torch.nn as nn

class ContextAggregation(nn.Module):

    def __init__(self, in_channels, reduction=1, conv_cfg=None):
        super(ContextAggregation, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.inter_channels = max(in_channels // reduction, 1)

        self.a = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.k = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.v = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.m = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)

        self.init_weights()

    def init_weights(self):
        for m in (self.a, self.k, self.v):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.m.weight, 0)
        if self.m.bias is not None:
            nn.init.constant_(self.m.bias, 0)

    def forward(self, x):
        n, c = x.size(0), self.inter_channels

        # a: [N, 1, H, W]
        a = self.a(x).sigmoid()

        # k: [N, 1, HW, 1]
        k = self.k(x).view(n, 1, -1, 1).softmax(2)

        # v: [N, 1, C, HW]
        v = self.v(x).view(n, 1, c, -1)

        # y: [N, C, 1, 1]
        y = torch.matmul(v, k).view(n, c, 1, 1)
        y = self.m(y) * a

        return x + y
