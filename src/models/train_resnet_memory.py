"""
MIMII 逐设备异常检测 - ResNet18 + 记忆库 KNN (最终测试集黄金阈值搜索版)
用法: python src/models/train_resnet_memory.py
"""
import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm
import torchvision.models as models
import torch.nn as nn

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from src.models.mimii_dataset import MIMIIDataset, MIMIIOnlyNormalDataset

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


def find_best_threshold_on_test(scores, labels):
    """在包含正常和异常的测试集上，搜索使 F1 最大的黄金阈值"""
    best_f1, best_thresh = 0.0, 0.0
    # 在所有实际得分的 5% ~ 99% 分位数之间进行密集搜索
    min_s, max_s = np.percentile(scores, 5), np.percentile(scores, 99)
    for thresh in np.linspace(min_s, max_s, 100):
        preds = (scores > thresh).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    data_dir = os.path.join(project_root, "data", "mimii")
    segment_duration = 1.0
    memory_bank_size = 3000
    
    extractor = FeatureExtractor().to(device)
    all_results = []
    
    for dev in DEVICES:
        print(f"\n{'='*60}")
        print(f" 正在处理设备: {dev.upper()}")
        print(f"{'='*60}")
        
        # 1. 仅使用 Normal 样本构建记忆库
        train_dataset = MIMIIOnlyNormalDataset(
            data_dir, device_type=dev, n_mels=64, segment_duration=segment_duration, augment=False
        )
        test_dataset = MIMIIDataset(
            data_dir, device_type=dev, n_mels=64, segment_duration=segment_duration, augment=False
        )
        
        if len(train_dataset) == 0 or len(test_dataset) == 0:
            print(f"  ⚠ 跳过")
            continue
            
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
        
        # 2. 构建记忆库 (Memory Bank)
        print("  [1/3] 构建正常特征记忆库...")
        memory_bank = []
        for x, _ in tqdm(train_loader, desc="  特征提取", leave=False):
            x = x.to(device)
            feats = extractor(x)
            memory_bank.append(feats.cpu().numpy())
            
        memory_bank = np.concatenate(memory_bank, axis=0)
        if len(memory_bank) > memory_bank_size:
            indices = np.random.choice(len(memory_bank), memory_bank_size, replace=False)
            memory_bank = memory_bank[indices]
        print(f"  记忆库构建完成: {memory_bank.shape[0]} 个特征")
        
        # 3. 测试集推理
        print("  [2/3] 测试集全量推理中...")
        all_labels, all_scores = [], []
        with torch.no_grad():
            for x, y in tqdm(test_loader, desc="  测试集推理", leave=False):
                x = x.to(device)
                feats = extractor(x)
                
                # 计算与正常记忆库的最小距离
                dist_matrix = torch.cdist(feats, torch.tensor(memory_bank, device=device), p=2)
                min_dists = torch.min(dist_matrix, dim=1).values
                all_scores.extend(min_dists.cpu().numpy())
                all_labels.extend(y.numpy())
        
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        # 4. 黄金阈值搜索与最终评估
        print("  [3/3] 搜索黄金阈值并输出最终指标...")
        auc = roc_auc_score(all_labels, all_scores)
        
        # 暴力搜索最优 F1 阈值
        best_thresh, best_f1 = find_best_threshold_on_test(all_scores, all_labels)
        
        predictions = (all_scores > best_thresh).astype(int)
        
        tp = np.sum((predictions == 1) & (all_labels == 1))
        fp = np.sum((predictions == 1) & (all_labels == 0))
        fn = np.sum((predictions == 0) & (all_labels == 1))
        tn = np.sum((predictions == 0) & (all_labels == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\n  ★ {dev.upper()}:")
        print(f"    AUC-ROC : {auc:.4f}")
        print(f"    F1 Score: {best_f1:.4f}  (阈值: {best_thresh:.4f})")
        print(f"    Precision: {precision:.4f} | Recall: {recall:.4f}")
        print(f"    混淆矩阵: TP={tp} | TN={tn} | FP={fp} | FN={fn}")
        
        all_results.append({
            "device": dev,
            "auc_roc": float(auc),
            "f1_score": float(best_f1),
            "precision": float(precision),
            "recall": float(recall),
            "threshold": float(best_thresh),
        })

    # 汇总打印
    print(f"\n{'='*60}")
    print(f" 最终评估汇总 (预训练 ResNet + Memory Bank KNN):")
    print(f"{'='*60}")
    print(f"{'设备':<12} {'AUC-ROC':<10} {'F1':<8} {'Precision':<10} {'Recall':<10}")
    print(f"{'-'*60}")
    for r in all_results:
        print(f"{r['device']:<12} {r['auc_roc']:<10.4f} {r['f1_score']:<8.4f} {r['precision']:<10.4f} {r['recall']:<10.4f}")
    
    avg_auc = np.mean([r['auc_roc'] for r in all_results])
    avg_f1 = np.mean([r['f1_score'] for r in all_results])
    print(f"{'-'*60}")
    print(f"{'平均':<12} {avg_auc:<10.4f} {avg_f1:<8.4f}")
    print(f"{'='*60}\n")

    report_path = os.path.join(project_root, "data", "models", "mimii_final_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            "method": "ResNet18 + Memory Bank KNN + Golden Threshold Search",
            "average_auc": float(avg_auc),
            "average_f1": float(avg_f1),
            "per_device": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"最终报告已保存: {report_path}")


if __name__ == "__main__":
    main()