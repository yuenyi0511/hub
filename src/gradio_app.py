"""
MIMII 异常检测 - Gradio Web 交互界面
用法:
  pip install gradio
  python src/gradio_app.py
  浏览器打开 http://127.0.0.1:7860
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import librosa
import gradio as gr
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.models.mimii_dataset import MIMIIOnlyNormalDataset

DEVICES = ["fan", "bearing", "gearbox", "slider", "valve"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== 特征提取器 =====
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


# ===== 全局变量 =====
extractor = FeatureExtractor().to(device)
memory_banks = {}
thresholds = {}


def load_thresholds():
    """从报告文件加载各设备的阈值"""
    global thresholds
    report_path = os.path.join(project_root, "data", "models", "mimii_final_report.json")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        for item in report.get("per_device", []):
            thresholds[item["device"]] = item["threshold"]
    # 默认阈值
    for dev in DEVICES:
        if dev not in thresholds:
            thresholds[dev] = 0.5


def build_memory_bank(device_type):
    """为指定设备构建记忆库（懒加载）"""
    if device_type in memory_banks:
        return memory_banks[device_type]

    print(f"正在为 [{device_type}] 构建记忆库...")
    data_dir = os.path.join(project_root, "data", "mimii")
    train_dataset = MIMIIOnlyNormalDataset(
        data_dir, device_type=device_type, n_mels=64,
        segment_duration=1.0, augment=False
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)

    bank = []
    for x, _ in tqdm(train_loader, desc=f"  {device_type}", leave=False):
        x = x.to(device)
        feats = extractor(x)
        bank.append(feats.cpu().numpy())

    bank = np.concatenate(bank, axis=0)
    if len(bank) > 3000:
        indices = np.random.choice(len(bank), 3000, replace=False)
        bank = bank[indices]

    memory_banks[device_type] = bank
    print(f"  [{device_type}] 记忆库构建完成: {bank.shape[0]} 个特征")
    return bank


def process_audio(audio_path, n_mels=64, segment_duration=1.0, sr=22050):
    """将音频转换为梅尔频谱张量"""
    y, sr_actual = librosa.load(audio_path, sr=sr)
    min_length = int(sr * segment_duration)
    if len(y) < min_length:
        y = np.pad(y, (0, min_length - len(y)))
    y = y[:int(sr * segment_duration)]

    mel = librosa.feature.melspectrogram(y=y, sr=sr_actual, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_min, mel_max = mel_db.min(), mel_db.max()
    if mel_max - mel_min > 0:
        mel_norm = (mel_db - mel_min) / (mel_max - mel_min)
    else:
        mel_norm = np.zeros_like(mel_db)

    mel_tensor = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return mel_tensor


def detect_anomaly(audio_path, device_type):
    """核心检测函数，供 Gradio 调用"""
    if audio_path is None:
        return "⚠️ 请先上传一个 .wav 音频文件", ""

    # 构建记忆库（首次调用时）
    memory_bank = build_memory_bank(device_type)
    threshold = thresholds.get(device_type, 0.5)

    # 处理音频
    mel_tensor = process_audio(audio_path).to(device)

    # 提取特征 & 计算距离
    with torch.no_grad():
        feat = extractor(mel_tensor)
    mem_tensor = torch.tensor(memory_bank, device=device)
    dist_matrix = torch.cdist(feat, mem_tensor, p=2)
    min_dist = torch.min(dist_matrix, dim=1).values.item()

    # 判定
    is_anomaly = min_dist > threshold
    confidence = min_dist / threshold if threshold > 0 else 0

    # 构造结果
    if is_anomaly:
        result_text = f"⚠️ 异常 (ANOMALY)\n\nKNN距离: {min_dist:.4f}\n阈值: {threshold:.4f}\n置信度: {confidence:.1%}"
    else:
        result_text = f"✅ 正常 (NORMAL)\n\nKNN距离: {min_dist:.4f}\n阈值: {threshold:.4f}\n置信度: {confidence:.1%}"

    detail_text = (
        f"设备类型: {device_type}\n"
        f"音频路径: {os.path.basename(audio_path)}\n"
        f"KNN距离: {min_dist:.6f}\n"
        f"判定阈值: {threshold:.6f}\n"
        f"距离/阈值: {confidence:.4f}\n"
        f"判定结果: {'异常' if is_anomaly else '正常'}"
    )

    return result_text, detail_text


# ===== 构建 Gradio 界面 =====
def create_ui():
    load_thresholds()

    with gr.Blocks() as app:
        gr.Markdown("""
        # 🔧 MIMII 工业设备异常检测系统
        ### 基于 ResNet18 + Memory Bank KNN 的声音异常检测
        上传一段设备运行音频，选择对应设备类型，系统将自动判断设备是否异常。
        """)

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="上传设备音频 (.wav)",
                    type="filepath",
                    sources=["upload"]
                )
                device_dropdown = gr.Dropdown(
                    choices=DEVICES,
                    value="fan",
                    label="设备类型",
                    info="选择音频对应的设备"
                )
                detect_btn = gr.Button("🔍 开始检测", variant="primary", size="lg")

            with gr.Column(scale=1):
                result_output = gr.Textbox(
                    label="检测结果",
                    lines=6,
                    interactive=False
                )
                detail_output = gr.Textbox(
                    label="详细信息",
                    lines=6,
                    interactive=False
                )

        detect_btn.click(
            fn=detect_anomaly,
            inputs=[audio_input, device_dropdown],
            outputs=[result_output, detail_output]
        )

        gr.Markdown("""
        ---
        **技术栈**: ResNet18 (ImageNet预训练) + Memory Bank KNN + 自动阈值搜索  
        **训练数据**: MIMII Dataset (5种工业设备)  
        **模型性能**: 平均 AUC-ROC = 0.9651 | 平均 F1 = 0.6877
        """)

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Soft()
    )