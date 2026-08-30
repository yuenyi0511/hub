import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

FEATURE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "features")

def load_features():
    """加载特征数据集"""
    files = glob.glob(os.path.join(FEATURE_DIR, "features_*.csv"))
    if not files:
        raise FileNotFoundError("未找到特征文件！请先运行 feature_engineering.py")
    latest = max(files, key=os.path.getmtime)
    print(f"加载特征文件: {latest}")
    return pd.read_csv(latest)

def anomaly_detection(df):
    """无监督异常检测（孤立森林）"""
    print("\n" + "=" * 60)
    print("【模块一：异常检测】")
    print("=" * 60)
    
    # 选取振动和温度相关特征
    feature_cols = [
        "vibration_x_rms", "vibration_x_kurtosis", "vibration_x_crest_factor",
        "vibration_y_rms", "vibration_y_kurtosis", "vibration_y_crest_factor",
        "vibration_z_rms", "vibration_z_kurtosis", "vibration_z_crest_factor",
        "bearing_temp_mean", "motor_temp_mean", "temp_gradient_bearing_ambient"
    ]
    
    # 过滤存在的列
    available_cols = [c for c in feature_cols if c in df.columns]
    X = df[available_cols].values
    
    # 训练孤立森林
    model = IsolationForest(contamination=0.1, random_state=42)
    predictions = model.fit_predict(X)
    scores = model.decision_function(X)
    
    df["anomaly_score"] = scores
    df["is_anomaly"] = (predictions == -1).astype(int)
    
    print(f"\n检测样本数: {len(df)}")
    print(f"异常样本数: {df['is_anomaly'].sum()}")
    print(f"异常比例: {df['is_anomaly'].mean() * 100:.1f}%")
    
    # 打印每个样本的健康状态
    print(f"\n{'样本来源':<30} {'异常得分':>10} {'健康状态':>8}")
    print("-" * 50)
    for _, row in df.iterrows():
        status = "异常 " if row["is_anomaly"] else "正常 "
        label = row.get('source_file', row.get('window_start', '未知'))
        print(f"{str(label):<25} {row['anomaly_score']:>10.4f} {status:>8}")
    
    return df

def health_assessment(df):
    """健康度评估（0-100分）"""
    print("\n" + "=" * 60)
    print("【模块二：健康度评估】")
    print("=" * 60)
    
    # 基于退化程度计算健康度
    if "degradation_level" in df.columns:
        # 退化程度越高，健康度越低
        deg = df["degradation_level"].values
        # 归一化到0-1范围
        deg_normalized = (deg - deg.min()) / (deg.max() - deg.min() + 1e-8)
        health_score = (1 - deg_normalized) * 100
        df["health_score"] = health_score
    else:
        # 用振动RMS代替
        rms = df.get("rms_mean", pd.Series([0]))
        health_score = np.clip(100 - rms.values * 10, 0, 100)
        df["health_score"] = health_score
    
    print(f"\n当前健康度评分:")
    print(f"  健康度: {df['health_score'].iloc[-1]:.1f} / 100")
    
    # 健康等级判定
    score = df["health_score"].iloc[-1]
    if score >= 80:
        level = "良好 —— 设备运行正常，无需干预"
    elif score >= 60:
        level = "注意 —— 存在轻微劣化，建议关注"
    elif score >= 40:
        level = "警告 —— 劣化明显，建议计划维护"
    else:
        level = "危险 —— 需立即停机检修！"
    
    print(f"  等级判定: {level}")
    
    return df

def fault_classification(df):
    """故障分类（基于规则的专家系统）"""
    print("\n" + "=" * 60)
    print("【模块三：故障分类】")
    print("=" * 60)
    
    fault_rules = []
    
    for _, row in df.iterrows():
        faults = []
        
        # 规则1：振动超标
        if row.get("vibration_x_rms", 0) > 4.5:
            faults.append("轴承磨损（振动X轴超标）")
        if row.get("vibration_y_rms", 0) > 4.5:
            faults.append("轴承磨损（振动Y轴超标）")
        if row.get("vibration_z_rms", 0) > 4.5:
            faults.append("轴向不对中（振动Z轴超标）")
        
        # 规则2：峭度异常（冲击故障）
        if row.get("vibration_x_kurtosis", 0) > 5:
            faults.append("滚动体局部缺陷（峭度异常）")
        
        # 规则3：温度异常
        if row.get("bearing_temp_mean", 0) > 70:
            faults.append("轴承过热")
        if row.get("motor_temp_mean", 0) > 80:
            faults.append("电机过热")
        
        # 规则4：温度梯度异常
        if row.get("temp_gradient_bearing_ambient", 0) > 20:
            faults.append("散热不良/润滑失效")


        # 规则5：振动上升趋势报警
        if row.get("vibration_x_rms", 0) > 3.0:
            faults.append("⚠️ 振动X轴偏高，疑似早期磨损")
        if row.get("vibration_x_kurtosis", 0) > 3.0:
            faults.append("⚠️ 峭度升高，疑似滚动体点蚀")
        
        # 规则6：温度异常上升趋势
        if row.get("bearing_temp_trend", 0) > 0.01:
            faults.append("⚠️ 轴承温度持续上升，疑似润滑劣化")
        
        fault_str = "; ".join(faults) if faults else "无故障（正常运行）"
        fault_rules.append(fault_str)
    
    df["fault_diagnosis"] = fault_rules
    
    print(f"\n故障诊断结果:")
    for _, row in df.iterrows():
        label = row.get('source_file', row.get('window_start', '未知'))
        print(f"  [{str(label)[:20]}] → {row['fault_diagnosis']}")
    
    return df

def remaining_useful_life(df):
    """剩余寿命预测（RUL） - 基于健康度反推"""
    print("\n" + "=" * 60)
    print("【模块四：剩余寿命预测(RUL)】")
    print("=" * 60)
    
    rul_list = []
    for _, row in df.iterrows():
        health = row.get("health_score", 0)
        
        # 基于健康度反推剩余寿命（假设满健康对应 1000 小时）
        if health >= 80:
            # 健康良好：剩余寿命充足，按线性外推
            rul = health * 12.5  # 100分 → 1250小时, 80分 → 1000小时
        elif health >= 50:
            # 中等退化：加速衰减
            rul = health * 8  # 50分 → 400小时
        elif health >= 20:
            # 严重退化：急剧缩短
            rul = health * 3  # 20分 → 60小时
        else:
            # 濒临损坏
            rul = max(0, health * 0.5)  # 0分 → 0小时
        
        rul_list.append(rul)
    
    df["rul_hours"] = rul_list
    
    print(f"\n  剩余寿命计算完成：")
    for i, row in df.iterrows():
        label = row.get('source_file', f'样本{i}')
        print(f"  [{str(label)[:20]}] 健康度 {row['health_score']:.1f} → RUL: {row['rul_hours']:.0f} 小时")
    
    return df

def generate_report(df):
    """生成综合诊断报告"""
    print("\n" + "=" * 60)
    print("【综合诊断报告】")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"""
  报告生成时间: {timestamp}
  
  ┌─────────────────────────────────────────────┐
  │  设备状态总览                               │
  ├─────────────────────────────────────────────┤""")
    
    for _, row in df.iterrows():
        print(f"  │  健康度: {row['health_score']:.1f}/100")
        print(f"  │  异常检测: {'⚠️ 发现异常' if row['is_anomaly'] else '✅ 正常'}")
        print(f"  │  故障诊断: {row['fault_diagnosis']}")
        if not pd.isna(row.get('rul_hours', np.nan)) and row['rul_hours'] != float('inf'):
            print(f"  │  剩余寿命: 约 {row['rul_hours']:.0f} 小时")
        elif row.get('rul_hours') == float('inf'):
            print(f"  │  剩余寿命: 充足")
    
    print(f"""  │
  │  建议措施: """)
    
    score = df["health_score"].iloc[-1]
    if score >= 80:
        print("  │    → 继续正常运行，定期巡检")
    elif score >= 60:
        print("  │    → 缩短巡检周期，准备备件")
    elif score >= 40:
        print("  │    → 安排计划性停机维护")
    else:
        print("  │    → 立即停机！联系维修人员")
    
    print("  └─────────────────────────────────────────────┘")
    
    # 保存报告
    report_path = os.path.join(os.path.dirname(__file__), "..", "docs", "diagnosis_report.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# 预测性维护 - 故障诊断报告\n\n")
        f.write(f"**生成时间**: {timestamp}\n\n")
        f.write(f"## 健康度评估\n\n")
        f.write(f"| 指标 | 数值 |\n")
        f.write(f"|------|------|\n")
        for _, row in df.iterrows():
            f.write(f"| 健康度 | {row['health_score']:.1f}/100 |\n")
            f.write(f"| 异常状态 | {'异常' if row['is_anomaly'] else '正常'} |\n")
            f.write(f"| 故障诊断 | {row['fault_diagnosis']} |\n")
            if not pd.isna(row.get('rul_hours', np.nan)) and row['rul_hours'] != float('inf'):
                f.write(f"| 剩余寿命 | {row['rul_hours']:.0f} 小时 |\n")
        #f.write(f"\n## 原始特征数据\n\n")
        #f.write(df.to_csv(index=False))
    
    print(f"\n  📄 报告已保存到: {report_path}")

def main():
    print("=" * 60)
    print("  预测性维护 - 故障诊断系统")
    print("=" * 60)
    
    # 加载特征
    df = load_features()
    print(f"共加载 {len(df)} 条特征样本")
    
    # 四大模块
    df = anomaly_detection(df)
    df = health_assessment(df)
    df = fault_classification(df)
    df = remaining_useful_life(df)
    
    # 生成报告
    generate_report(df)

if __name__ == "__main__":
    main()