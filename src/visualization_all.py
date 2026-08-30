"""
MIMII 异常检测 - 全套可视化图表生成
生成：ROC曲线、混淆矩阵、t-SNE特征分布、分数分布
用法: python src/visualization_all.py
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ===== 路径配置 =====
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.models.mimii_dataset import MIMIIDataset, MIMIIOnlyNormalDataset

DEVICES = ["fan", "bearing", "gearbox", "slider", "valve"]
FIGURE_DIR = os.path.join(project_root, "data", "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)


# ===== 特征提取器（与训练时完全一致）=====
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


def build_memory_bank(extractor, train_loader, device, memory_bank_size=3000):
    """构建正常特征记忆库"""
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


def compute_scores(extractor, test_loader, memory_bank, device):
    """计算测试集每个样本的KNN距离分数，同时收集特征用于t-SNE"""
    all_labels = []
    all_scores = []
    all_feats = []
    mem_tensor = torch.tensor(memory_bank, device=device)

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="  测试集推理", leave=False):
            x = x.to(device)
            feats = extractor(x)
            all_feats.append(feats.cpu().numpy())

            dist_matrix = torch.cdist(feats, mem_tensor, p=2)
            min_dists = torch.min(dist_matrix, dim=1).values
            all_scores.extend(min_dists.cpu().numpy())
            all_labels.extend(y.numpy())

    all_feats = np.concatenate(all_feats, axis=0)
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    return all_scores, all_labels, all_feats


def find_best_threshold(scores, labels):
    """搜索最优F1阈值"""
    best_f1, best_thresh = 0.0, 0.0
    min_s, max_s = np.percentile(scores, 5), np.percentile(scores, 99)
    for thresh in np.linspace(min_s, max_s, 100):
        preds = (scores > thresh).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1


# ===== 绘图函数 =====

def plot_roc_curves(results_dict):
    """绘制5个设备的ROC曲线（同一张图）"""
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    for color, (dev, (scores, labels)) in zip(colors, results_dict.items()):
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f"{dev} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.set_title("ROC Curves - All Devices", fontsize=16, fontweight="bold")
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "roc_curves_all.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  ✅ ROC曲线已保存: {path}")


def plot_confusion_matrices(results_dict, thresholds):
    """绘制5个设备的混淆矩阵热力图"""
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle("Confusion Matrices - All Devices", fontsize=18, fontweight="bold")

    for ax, (dev, (scores, labels)) in zip(axes, results_dict.items()):
        thresh = thresholds[dev]
        preds = (scores > thresh).astype(int)
        cm = confusion_matrix(labels, preds)

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Normal", "Anomaly"],
                    yticklabels=["Normal", "Anomaly"])
        ax.set_title(f"{dev}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  ✅ 混淆矩阵已保存: {path}")


def plot_tsne(results_dict, feats_dict, memory_banks):
    """绘制t-SNE特征分布散点图"""
    all_sampled_feats = []
    all_labels = []
    all_devices = []

    for dev in DEVICES:
        if dev not in feats_dict:
            continue
        feats = feats_dict[dev]
        scores, labels = results_dict[dev]

        # 正常样本最多取500个
        normal_idx = np.where(labels == 0)[0]
        if len(normal_idx) > 500:
            normal_idx = np.random.choice(normal_idx, 500, replace=False)

        # 异常样本最多取500个
        anomaly_idx = np.where(labels == 1)[0]
        if len(anomaly_idx) > 500:
            anomaly_idx = np.random.choice(anomaly_idx, 500, replace=False)

        selected_idx = np.concatenate([normal_idx, anomaly_idx])
        selected_feats = feats[selected_idx]
        selected_labels = labels[selected_idx]

        all_sampled_feats.append(selected_feats)
        all_labels.append(selected_labels)
        all_devices.extend([dev] * len(selected_idx))

    if len(all_sampled_feats) == 0:
        print("  ⚠ 无数据可绘制t-SNE")
        return

    all_sampled_feats = np.concatenate(all_sampled_feats, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print("  正在运行t-SNE降维（可能需要1-2分钟）...")
    import numpy as np

    if len(all_sampled_feats) > 1000:
        print("  数据量较大，随机抽取 1000 个样本加速计算...")
        indices = np.random.choice(len(all_sampled_feats), 1000, replace=False)
        all_sampled_feats = all_sampled_feats[indices]
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    feats_2d = tsne.fit_transform(all_sampled_feats)

    fig, ax = plt.subplots(figsize=(12, 9))

    normal_mask = all_labels == 0
    anomaly_mask = all_labels == 1

    ax.scatter(feats_2d[normal_mask, 0], feats_2d[normal_mask, 1],
               c="#3498db", alpha=0.5, s=15, label="Normal", edgecolors="none")
    ax.scatter(feats_2d[anomaly_mask, 0], feats_2d[anomaly_mask, 1],
               c="#e74c3c", alpha=0.8, s=30, label="Anomaly", marker="x", linewidths=1.5)

    ax.set_title("t-SNE Feature Distribution (Normal vs Anomaly)",
                 fontsize=16, fontweight="bold")
    ax.legend(fontsize=13, loc="best")
    ax.set_xlabel("t-SNE Dim 1", fontsize=12)
    ax.set_ylabel("t-SNE Dim 2", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "tsne_features.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  ✅ t-SNE图已保存: {path}")


def plot_score_distributions(results_dict, thresholds):
    """绘制每个设备的正常/异常距离分数分布"""
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle("Anomaly Score Distributions - All Devices",
                 fontsize=18, fontweight="bold")

    for ax, (dev, (scores, labels)) in zip(axes, results_dict.items()):
        normal_scores = scores[labels == 0]
        anomaly_scores = scores[labels == 1]

        ax.hist(normal_scores, bins=50, alpha=0.6, color="#3498db",
                label="Normal", density=True)
        ax.hist(anomaly_scores, bins=50, alpha=0.6, color="#e74c3c",
                label="Anomaly", density=True)

        thresh = thresholds[dev]
        ax.axvline(x=thresh, color="black", linestyle="--", lw=2,
                    label=f"Threshold={thresh:.3f}")

        ax.set_title(f"{dev}", fontsize=14, fontweight="bold")
        ax.set_xlabel("KNN Distance", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "score_distributions.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  ✅ 分数分布图已保存: {path}")


# ===== 主函数 =====
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    data_dir = os.path.join(project_root, "data", "mimii")
    segment_duration = 1.0
    memory_bank_size = 3000

    extractor = FeatureExtractor().to(device)

    results_dict = {}   # {dev: (scores, labels)}
    feats_dict = {}     # {dev: feats_array}
    thresholds = {}     # {dev: best_threshold}
    memory_banks = {}   # {dev: memory_bank}

    for dev in DEVICES:
        print(f"\n{'='*50}")
        print(f" 处理设备: {dev.upper()}")
        print(f"{'='*50}")

        train_dataset = MIMIIOnlyNormalDataset(
            data_dir, device_type=dev, n_mels=64,
            segment_duration=segment_duration, augment=False
        )
        test_dataset = MIMIIDataset(
            data_dir, device_type=dev, n_mels=64,
            segment_duration=segment_duration, augment=False
        )

        if len(train_dataset) == 0 or len(test_dataset) == 0:
            print(f"  ⚠ 跳过")
            continue

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

        # 构建记忆库
        memory_bank = build_memory_bank(extractor, train_loader, device, memory_bank_size)
        memory_banks[dev] = memory_bank
        print(f"  记忆库: {memory_bank.shape}")

        # 测试集推理
        scores, labels, feats = compute_scores(extractor, test_loader, memory_bank, device)
        results_dict[dev] = (scores, labels)
        feats_dict[dev] = feats

        # 搜索最优阈值
        thresh, f1 = find_best_threshold(scores, labels)
        thresholds[dev] = thresh
        print(f"  最优阈值: {thresh:.4f}  F1: {f1:.4f}")

    # ===== 生成全部图表 =====
    print(f"\n{'='*50}")
    print(f" 开始生成可视化图表")
    print(f"{'='*50}")

    print("\n[1/4] ROC曲线...")
    plot_roc_curves(results_dict)

    print("\n[2/4] 混淆矩阵...")
    plot_confusion_matrices(results_dict, thresholds)

    print("\n[3/4] t-SNE特征分布...")
    plot_tsne(results_dict, feats_dict, memory_banks)

    print("\n[4/4] 分数分布...")
    plot_score_distributions(results_dict, thresholds)

    print(f"\n{'='*50}")
    print(f" 全部图表已保存至: {FIGURE_DIR}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()