"""
MIMII 异常检测 - 单条音频推理脚本
用法: python src/predict.py <音频路径> --device <设备名>
示例: python src/predict.py data/mimii/fan/fan/test/section_00_source_test_anomaly_0000_m-n_W.wav --device fan
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import librosa
import torchaudio

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.models.mimii_dataset import MIMIIOnlyNormalDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

DEVICES = ["fan", "bearing", "gearbox", "slider", "valve"]


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.features.eval()

    @torch.no_grad()
    def forward(self, x):
        feat = self.features(x)
        b, c, h, w = feat.shape
        feat = feat.view(b, c, -1)
        feat = torch.mean(feat, dim=2)
        return feat


def load_report():
    """加载最终报告中的阈值信息"""
    report_path = os.path.join(project_root, "data", "models", "mimii_final_report.json")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def build_memory_bank_for_device(extractor, device_type, device, memory_bank_size=3000):
    """为指定设备构建记忆库"""
    data_dir = os.path.join(project_root, "data", "mimii")
    train_dataset = MIMIIOnlyNormalDataset(
        data_dir, device_type=device_type, n_mels=64,
        segment_duration=1.0, augment=False
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)

    memory_bank = []
    for x, _ in tqdm(train_loader, desc="  构建记忆库", leave=False):
        x = x.to(device)
        feats = extractor(x)
        memory_bank.append(feats.cpu().numpy())

    memory_bank = np.concatenate(memory_bank, axis=0)
    if len(memory_bank) > memory_bank_size:
        indices = np.random.choice(len(memory_bank), memory_bank_size, replace=False)
        memory_bank = memory_bank[indices]
    return memory_bank


def process_audio(audio_path, n_mels=64, segment_duration=1.0, sr=22050):
    """将音频文件转换为梅尔频谱张量"""
    y, sr_actual = librosa.load(audio_path, sr=sr)

    # 如果音频太短，补零
    min_length = int(sr * segment_duration)
    if len(y) < min_length:
        y = np.pad(y, (0, min_length - len(y)))

    # 只取前 segment_duration 秒
    y = y[:int(sr * segment_duration)]

    # 生成梅尔频谱
    mel = librosa.feature.melspectrogram(y=y, sr=sr_actual, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # 归一化到 [0, 1]
    mel_min, mel_max = mel_db.min(), mel_db.max()
    if mel_max - mel_min > 0:
        mel_norm = (mel_db - mel_min) / (mel_max - mel_min)
    else:
        mel_norm = np.zeros_like(mel_db)

    # 转为张量 [1, 1, H, W]
    mel_tensor = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return mel_tensor


def predict(audio_path, device_type, extractor, memory_bank, device, threshold):
    """对单条音频进行异常检测"""
    # 处理音频
    mel_tensor = process_audio(audio_path).to(device)

    # 提取特征
    with torch.no_grad():
        feat = extractor(mel_tensor)

    # 计算与记忆库的最小距离
    mem_tensor = torch.tensor(memory_bank, device=device)
    dist_matrix = torch.cdist(feat, mem_tensor, p=2)
    min_dist = torch.min(dist_matrix, dim=1).values.item()

    # 判定
    is_anomaly = min_dist > threshold
    confidence = min_dist / threshold if threshold > 0 else 0

    return {
        "audio_path": audio_path,
        "device_type": device_type,
        "knn_distance": min_dist,
        "threshold": threshold,
        "is_anomaly": is_anomaly,
        "confidence": confidence,
        "prediction": "ANOMALY ⚠️" if is_anomaly else "NORMAL ✅",
    }


def main():
    parser = argparse.ArgumentParser(description="MIMII 单条音频异常检测推理")
    parser.add_argument("audio_path", type=str, help="待检测的音频文件路径")
    parser.add_argument("--device", type=str, required=True, choices=DEVICES,
                        help="设备类型: fan, bearing, gearbox, slider, valve")
    parser.add_argument("--threshold", type=float, default=None,
                        help="判定阈值（不指定则从报告自动读取）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 检查音频文件
    if not os.path.exists(args.audio_path):
        print(f"❌ 音频文件不存在: {args.audio_path}")
        sys.exit(1)

    # 加载阈值
    threshold = args.threshold
    if threshold is None:
        report = load_report()
        if report and "per_device" in report:
            for item in report["per_device"]:
                if item["device"] == args.device:
                    threshold = item["threshold"]
                    break
        if threshold is None:
            print("⚠ 未找到预设阈值，使用默认值 0.5")
            threshold = 0.5

    # 初始化模型
    extractor = FeatureExtractor().to(device)

    # 构建记忆库
    print(f"\n正在为 [{args.device}] 构建记忆库...")
    memory_bank = build_memory_bank_for_device(extractor, args.device, device)
    print(f"记忆库大小: {memory_bank.shape[0]} 个特征\n")

    # 推理
    print(f"正在检测: {args.audio_path}")
    result = predict(args.audio_path, args.device, extractor, memory_bank, device, threshold)

    # 输出结果
    print(f"\n{'='*50}")
    print(f"  检测结果")
    print(f"{'='*50}")
    print(f"  音频文件  : {result['audio_path']}")
    print(f"  设备类型  : {result['device_type']}")
    print(f"  KNN距离   : {result['knn_distance']:.4f}")
    print(f"  判定阈值  : {result['threshold']:.4f}")
    print(f"  置信度    : {result['confidence']:.2%}")
    print(f"  判定结果  : {result['prediction']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()