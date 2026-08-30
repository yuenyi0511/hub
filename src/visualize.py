import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import glob
from datetime import datetime

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
FEATURE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "features")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")

os.makedirs(REPORT_DIR, exist_ok=True)

def load_latest_raw():
    files = glob.glob(os.path.join(RAW_DATA_DIR, "plc_data_*.csv"))
    if not files:
        raise FileNotFoundError("未找到原始数据文件！")
    return pd.read_csv(max(files, key=os.path.getmtime))

def plot_dashboard(df_raw):
    """生成四合一仪表盘"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("预测性维护 - 实时状态仪表盘", fontsize=18, fontweight='bold')
    
    # 子图1：三轴振动趋势
    ax1 = axes[0, 0]
    ax1.plot(df_raw["timestamp"], df_raw["vibration_x"], label="X轴", linewidth=1.2)
    ax1.plot(df_raw["timestamp"], df_raw["vibration_y"], label="Y轴", linewidth=1.2)
    ax1.plot(df_raw["timestamp"], df_raw["vibration_z"], label="Z轴", linewidth=1.2)
    ax1.axhline(y=4.5, color='r', linestyle='--', alpha=0.7, label="报警阈值 4.5")
    ax1.set_title("三轴振动趋势 (mm/s)", fontsize=13)
    ax1.set_xlabel("时间")
    ax1.set_ylabel("振动幅值 (mm/s)")
    ax1.legend(fontsize=9)
    ax1.tick_params(axis='x', rotation=45)
    
    # 子图2：温度趋势
    ax2 = axes[0, 1]
    ax2.plot(df_raw["timestamp"], df_raw["bearing_temp"], label="轴承温度", color='red', linewidth=1.2)
    ax2.plot(df_raw["timestamp"], df_raw["motor_temp"], label="电机温度", color='orange', linewidth=1.2)
    ax2.plot(df_raw["timestamp"], df_raw["ambient_temp"], label="环境温度", color='blue', linewidth=1.2)
    ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label="轴承报警 70℃")
    ax2.set_title("温度趋势 (℃)", fontsize=13)
    ax2.set_xlabel("时间")
    ax2.set_ylabel("温度 (℃)")
    ax2.legend(fontsize=9)
    ax2.tick_params(axis='x', rotation=45)
    
    # 子图3：RMS + 退化程度
    ax3 = axes[1, 0]
    color_rms = 'purple'
    ax3.plot(df_raw["timestamp"], df_raw["rms_value"], color=color_rms, linewidth=1.2, label="RMS值")
    ax3.set_ylabel("RMS (mm/s)", color=color_rms)
    ax3.tick_params(axis='y', labelcolor=color_rms)
    ax3.set_title("RMS值 & 退化程度", fontsize=13)
    ax3.legend(loc='upper left', fontsize=9)
    ax3.tick_params(axis='x', rotation=45)
    
    ax3b = ax3.twinx()
    color_deg = 'green'
    ax3b.plot(df_raw["timestamp"], df_raw["degradation_level"], color=color_deg, linewidth=1.5, linestyle='--', label="退化程度")
    ax3b.set_ylabel("退化程度", color=color_deg)
    ax3b.tick_params(axis='y', labelcolor=color_deg)
    ax3b.legend(loc='upper right', fontsize=9)
    
    # 子图4：振动频谱分析（简易FFT）
    ax4 = axes[1, 1]
    import numpy as np
    vib_x = df_raw["vibration_x"].values
    fft_vals = np.abs(np.fft.rfft(vib_x))
    freqs = np.fft.rfftfreq(len(vib_x), d=1.0)  # 采样间隔1秒
    ax4.plot(freqs, fft_vals, color='teal', linewidth=1)
    ax4.set_title("振动X轴频谱分析 (FFT)", fontsize=13)
    ax4.set_xlabel("频率 (Hz)")
    ax4.set_ylabel("幅值")
    ax4.set_xlim(0, 0.5)  # 只显示低频部分
    
    plt.tight_layout()
    
    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(REPORT_DIR, f"dashboard_{timestamp}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"仪表盘已保存: {save_path}")
    plt.show()

def main():
    print("=" * 60)
    print("  预测性维护 - 可视化仪表盘")
    print("=" * 60)
    
    df_raw = load_latest_raw()
    print(f"加载数据: {len(df_raw)} 条记录")
    
    plot_dashboard(df_raw)

if __name__ == "__main__":
    main()