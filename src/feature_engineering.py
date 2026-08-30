import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
FEATURE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "features")

os.makedirs(FEATURE_DIR, exist_ok=True)

def load_latest_csv():
    """加载最新采集的CSV文件"""
    files = glob.glob(os.path.join(RAW_DATA_DIR, "plc_data_*.csv"))
    if not files:
        raise FileNotFoundError("未找到任何采集数据文件！")
    latest = max(files, key=os.path.getmtime)
    print(f"加载数据: {latest}")
    return pd.read_csv(latest)

def extract_sliding_features(df, window_size=30, step=10):
    """滑动窗口特征提取，每 step 秒提取一次特征"""
    all_features = []
    for start in range(0, len(df) - window_size + 1, step):
        window = df.iloc[start:start + window_size]
        feat = extract_features(window)
        feat["window_start"] = window["timestamp"].iloc[0]
        feat["window_end"] = window["timestamp"].iloc[-1]
        all_features.append(feat)
    return pd.DataFrame(all_features)
def extract_features(df):
    """从原始时序数据中提取时域和频域特征"""
    features = {}
    
    # ========== 振动特征（三轴） ==========
    for axis in ["vibration_x", "vibration_y", "vibration_z"]:
        data = df[axis].values
        
        # 时域统计特征
        features[f"{axis}_mean"] = np.mean(data)
        features[f"{axis}_std"] = np.std(data)
        features[f"{axis}_max"] = np.max(data)
        features[f"{axis}_min"] = np.min(data)
        features[f"{axis}_peak2peak"] = np.max(data) - np.min(data)
        
        # RMS（有效值）
        features[f"{axis}_rms"] = np.sqrt(np.mean(data ** 2))
        
        # 峭度（Kurtosis）—— 冲击故障敏感指标
        features[f"{axis}_kurtosis"] = pd.Series(data).kurtosis()
        
        # 偏度（Skewness）
        features[f"{axis}_skewness"] = pd.Series(data).skew()
        
        # 波形因子
        if features[f"{axis}_mean"] != 0:
            features[f"{axis}_form_factor"] = features[f"{axis}_rms"] / abs(np.mean(data))
        else:
            features[f"{axis}_form_factor"] = 0
        
        # 峰值因子（Crest Factor）
        features[f"{axis}_crest_factor"] = np.max(np.abs(data)) / features[f"{axis}_rms"]
    
    # ========== 温度特征 ==========
    for temp_col in ["bearing_temp", "motor_temp", "ambient_temp"]:
        data = df[temp_col].values
        features[f"{temp_col}_mean"] = np.mean(data)
        features[f"{temp_col}_max"] = np.max(data)
        features[f"{temp_col}_trend"] = np.polyfit(range(len(data)), data, 1)[0]  # 温度上升斜率
        features[f"{temp_col}_std"] = np.std(data)
    
    # ========== 温度梯度（温差） ==========
    features["temp_gradient_bearing_ambient"] = df["bearing_temp"].mean() - df["ambient_temp"].mean()
    features["temp_gradient_motor_ambient"] = df["motor_temp"].mean() - df["ambient_temp"].mean()
    
    # ========== RMS 综合指标 ==========
    data = df["rms_value"].values
    features["rms_mean"] = np.mean(data)
    features["rms_std"] = np.std(data)
    features["rms_max"] = np.max(data)
    features["rms_crest_factor"] = np.max(np.abs(data)) / np.sqrt(np.mean(data ** 2))
    
    # ========== 运行状态 ==========
    features["runtime_hours"] = df["runtime_hours"].iloc[-1]
    features["rpm_mean"] = df["rpm"].mean()
    features["load_percent_mean"] = df["load_percent"].mean()
    features["degradation_level"] = df["degradation_level"].iloc[-1]
    features["fault_type"] = df["fault_type"].iloc[-1]
    
    return features

def generate_dataset():
    """将多个CSV文件转换为特征数据集，用于模型训练"""
    files = glob.glob(os.path.join(RAW_DATA_DIR, "plc_data_*.csv"))
    if not files:
        print("未找到任何采集数据文件！")
        return
    
    all_features = []
    for f in sorted(files):
        try:
            df = pd.read_csv(f)
            feat = extract_features(df)
            feat["source_file"] = os.path.basename(f)
            feat["record_time"] = datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d %H:%M:%S")
            all_features.append(feat)
        except Exception as e:
            print(f"处理 {f} 失败: {e}")
    
    if not all_features:
        print("没有成功处理任何文件！")
        return
    
    df_features = pd.DataFrame(all_features)
    
    # 保存特征数据集
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(FEATURE_DIR, f"features_{timestamp}.csv")
    df_features.to_csv(save_path, index=False, encoding="utf-8")
    
    print(f"\n特征数据集已保存: {save_path}")
    print(f"共生成 {len(df_features)} 条样本，{len(df_features.columns)} 个特征")
    print(f"\n特征列名:")
    for col in df_features.columns:
        print(f"  - {col}")
    
    return df_features

def main():
    print("=" * 60)
    print("  预测性维护 - 特征工程（滑动窗口版）")
    print("=" * 60)
    
    df = load_latest_csv()
    print(f"原始数据: {len(df)} 条记录")
    
    # 滑动窗口提取
    df_features = extract_sliding_features(df, window_size=30, step=10)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(FEATURE_DIR, f"features_{timestamp}.csv")
    df_features.to_csv(save_path, index=False, encoding="utf-8")
    
    print(f"\n特征数据集已保存: {save_path}")
    print(f"共生成 {len(df_features)} 个窗口，{len(df_features.columns)} 个特征")

if __name__ == "__main__":
    main()