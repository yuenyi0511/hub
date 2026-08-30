"""
基于卷积自编码器的异常声音检测模型
仅用Normal数据训练；Anomaly数据重建误差更大 -> 检测异常
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAutoEncoder(nn.Module):
    """2D卷积自编码器 - 输入: (B, 1, n_mels, time_frames)"""

    def __init__(self, n_mels=64, time_frames=32, latent_dim=32):
        super().__init__()
        self.n_mels = n_mels
        self.time_frames = time_frames

        # 经过3次 stride=2 的卷积/池化，尺寸变为原来的 1/8
        h3 = n_mels // 8
        w3 = time_frames // 8
        flat_dim = 128 * h3 * w3

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
        )

        # Decoder 线性部分
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, flat_dim),
            nn.ReLU(inplace=True),
        )

        # Decoder 卷积部分
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # 归一化到 [0, 1] 范围，方便计算 MSE
        )

    def forward(self, x):
        latent = self.encoder(x)
        h = self.decoder_linear(latent)
        
        # 恢复到 3次池化后的 4D 形状
        h3 = self.n_mels // 8
        w3 = self.time_frames // 8
        h = h.view(-1, 128, h3, w3)
        
        recon = self.decoder_conv(h)
        # 裁剪/插值回原始尺寸
        if recon.size(2) != self.n_mels or recon.size(3) != self.time_frames:
            recon = F.interpolate(recon, size=(self.n_mels, self.time_frames), mode='bilinear', align_corners=False)
        return recon


class DeepAutoEncoder(nn.Module):
    """全连接自编码器 - 简化版，适合小数据"""

    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, input_dim),
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon