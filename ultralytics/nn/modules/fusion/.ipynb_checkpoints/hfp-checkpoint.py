# 论文: HS-FPN: High Frequency and Spatial Perception FPN for Tiny Object Detection (AAAI 2025)
# 链接: https://arxiv.org/abs/2412.10116
# 模块作用: 对融合后特征进行频域高频感知与空间/通道门控，突出细粒度目标与边缘信息，抑制跨模态低频冗余。
import torch
import torch.nn as nn

class HighFrequencyPerception(nn.Module):
    def __init__(self, ratio=(0.25, 0.25), patch=(8, 8), groups=32):
        super().__init__()
        self.ratio = ratio
        self.ph, self.pw = int(patch[0]), int(patch[1])
        self.groups = int(groups)
        self._built = False
        self._c = None

    def _build_if_needed(self, x):
        c = x.shape[1]
        if self._built and self._c == c:
            return
        
        device = x.device
        dtype = x.dtype
        
        g = max(1, min(self.groups, c))
        # 显式将新层移动到对应的设备和精度
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(c, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        ).to(device).to(dtype)
        
        self.channel_conv1 = nn.Conv2d(c, c, kernel_size=1, groups=g).to(device).to(dtype)
        self.channel_conv2 = nn.Conv2d(c, c, kernel_size=1, groups=g).to(device).to(dtype)
        self.out_conv = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, c), num_channels=c),
        ).to(device).to(dtype)
        
        self._built = True
        self._c = c

    def _mask_fft(self, x):
        B, C, H, W = x.shape
        orig_dtype = x.dtype
        # 强制用 float32 计算 FFT，避开 ComplexHalf 错误
        x_f32 = x.to(torch.float32)
        xf = torch.fft.rfft2(x_f32, dim=(-2, -1))
        
        h0 = int(H * self.ratio[0])
        w0 = int((W // 2 + 1) * self.ratio[1])
        mask = torch.ones_like(xf)
        mask[:, :, :h0, :w0] = 0
        
        xf = xf * mask
        xh = torch.fft.irfft2(xf, s=(H, W))
        # 🚨 关键：转回原始精度（Half），否则后面卷积会报类型错误
        return xh.to(orig_dtype)

    def forward(self, x):
        B, C, H, W = x.shape
        self._build_if_needed(x)
        
        # 1. 获取高频特征
        hf = self._mask_fft(x)
        
        # 2. 空间感知
        spa = self.spatial_conv(hf) * x
        
        # 3. 通道感知 (保持在原有精度)
        amax = torch.nn.functional.adaptive_max_pool2d(hf, output_size=(self.ph, self.pw))
        aavg = torch.nn.functional.adaptive_avg_pool2d(hf, output_size=(self.ph, self.pw))
        
        amax = torch.sum(torch.relu(amax), dim=(2, 3), keepdim=True)
        aavg = torch.sum(torch.relu(aavg), dim=(2, 3), keepdim=True)
        
        ch = self.channel_conv1(amax) + self.channel_conv1(aavg)
        ch = torch.sigmoid(self.channel_conv2(ch))
        cha = ch * x
        
        # 4. 融合输出
        out = self.out_conv(spa + cha)
        return out