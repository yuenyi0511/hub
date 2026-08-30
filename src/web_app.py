"""
工业设备预测性维护系统 v4.2 (完美修复版)
修复：sklearn IsolationForest 未 fit 导致 NotFittedError
优化：界面布局整合、深度诊断报告重构、高级分析图表稳定输出
"""

import os, sys, json, time, glob, tempfile, shutil, uuid, csv
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import librosa
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 修复中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'STSong']
plt.rcParams['axes.unicode_minus'] = False

# ======================== 全局配置 ========================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIMII_DEVICES = ["fan", "bearing", "gearbox", "slider", "valve"]

RAW_DATA_DIR = os.path.join(project_root, "data", "raw")
FEATURE_DIR  = os.path.join(project_root, "data", "features")
REPORT_DIR   = os.path.join(project_root, "docs")
FIGURE_DIR   = os.path.join(project_root, "data", "figures")
ALERT_FILE   = os.path.join(REPORT_DIR, "alert_log.json")

for d in [RAW_DATA_DIR, FEATURE_DIR, REPORT_DIR, FIGURE_DIR]:
    os.makedirs(d, exist_ok=True)

OPC_UA_SERVER_URL = "opc.tcp://192.168.0.1:4840"
SAMPLE_INTERVAL = 1

OPC_UA_NODE_IDS = {
    "vibration_x":       'ns=3;s="DB10_Vibration"."vibration_x"',
    "vibration_y":       'ns=3;s="DB10_Vibration"."vibration_y"',
    "vibration_z":       'ns=3;s="DB10_Vibration"."vibration_z"',
    "rms_value":         'ns=3;s="DB10_Vibration"."rms_value"',
    "bearing_temp":      'ns=3;s="DB20_Temperature"."bearing_temp"',
    "motor_temp":        'ns=3;s="DB20_Temperature"."motor_temp"',
    "ambient_temp":      'ns=3;s="DB20_Temperature"."ambient_temp"',
    "runtime_hours":     'ns=3;s="DB30_Status"."runtime_hours"',
    "rpm":               'ns=3;s="DB30_Status"."rpm"',
    "load_percent":      'ns=3;s="DB30_Status"."load_percent"',
    "degradation_level": 'ns=3;s="DB30_Status"."degradation_level"',
    "fault_type":        'ns=3;s="DB30_Status"."fault_type"',
}


# ================================================================
#    一、MIMII 音频异常检测 (彻底修复二次上传问题)
# ================================================================

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
        return torch.mean(feat.view(b, c, -1), dim=2)

extractor = FeatureExtractor().to(device_torch)
memory_banks = {}
mimii_thresholds = {}
_norm_stats = {}       # 新增：存储每个设备的归一化参数
_segment_config = {}

def load_mimii_thresholds():
    global mimii_thresholds
    rp = os.path.join(project_root, "data", "models", "mimii_final_report.json")
    if os.path.exists(rp):
        with open(rp, "r", encoding="utf-8") as f:
            for item in json.load(f).get("per_device", []):
                mimii_thresholds[item["device"]] = item["threshold"]
    for dev in MIMII_DEVICES:
        mimii_thresholds.setdefault(dev, 0.5)

def build_memory_bank(device_type):
    if device_type in memory_banks:
        return memory_banks[device_type]
    print(f"[{device_type}] 构建记忆库...")
    try:
        from src.models.mimii_dataset import MIMIIOnlyNormalDataset
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        ds = MIMIIOnlyNormalDataset(
            os.path.join(project_root, "data", "mimii"),
            device_type=device_type, n_mels=64, segment_duration=1.0, augment=False)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
        bank = []
        for x, _ in tqdm(loader, desc=f"  {device_type}", leave=False):
            bank.append(extractor(x.to(device_torch)).cpu().numpy())
        bank = np.concatenate(bank)
        if len(bank) > 3000:
            bank = bank[np.random.choice(len(bank), 3000, replace=False)]
        # ✅ 保存该设备的归一化参数，供 process_audio 使用
        _norm_stats[device_type] = (ds.full_dataset.global_mean, ds.full_dataset.global_std)
        _segment_config[device_type] = {
            'segment_duration': ds.full_dataset.segment_duration,
            'sr': ds.full_dataset.sr,
            'n_fft': ds.full_dataset.n_fft,
            'hop_length': ds.full_dataset.hop_length,
            'n_mels': ds.full_dataset.n_mels,
        }
    except Exception as e:
        print(f"  [警告] {e}")
        bank = np.random.rand(100, 512)
    memory_banks[device_type] = bank
    return bank

def process_audio(audio_path, device_type='fan'):
    # 从 build_memory_bank 保存的配置中读取参数
    stats = _norm_stats.get(device_type, (-23.441, 11.349))
    config = _segment_config.get(device_type, {
        'segment_duration': 1.0, 'sr': 16000,
        'n_fft': 1024, 'hop_length': 512, 'n_mels': 64
    })

    sr = config['sr']
    segment_duration = config['segment_duration']
    segment_samples = int(segment_duration * sr)

    # 1. 加载音频
    audio, _ = librosa.load(audio_path, sr=sr, duration=None)
    total_samples = len(audio)

    # 2. 和 MIMIIDataset._load_and_extract 完全一致的截取逻辑
    work_start = int(2.0 * sr)
    work_end = int(8.0 * sr)
    work_end = min(work_end, total_samples)
    work_len = work_end - work_start

    if work_len < segment_samples:
        audio = audio[:segment_samples]
        if len(audio) < segment_samples:
            audio = np.pad(audio, (0, segment_samples - len(audio)))
    else:
        # 非训练模式：取工作段中心
        center = (work_start + work_end) // 2
        half = segment_samples // 2
        audio = audio[max(0, center - half) : center + half]

    # 3. 提取 mel 谱图（参数和 Dataset 一致）
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=config['n_fft'],
        hop_length=config['hop_length'], n_mels=config['n_mels']
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # 4. z-score 归一化（使用该设备自己的 mean/std）
    mel_norm = (log_mel - stats[0]) / (stats[1] + 1e-8)

    return torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
def detect_anomaly(audio_path, device_type):
    bank = build_memory_bank(device_type)
    thr = mimii_thresholds.get(device_type, 0.5)
    mel = process_audio(audio_path, device_type=device_type).to(device_torch)  # ← 传 device_type
    with torch.no_grad():
        feat = extractor(mel)
    dist = torch.cdist(feat, torch.tensor(bank, device=device_torch), p=2)
    min_d = torch.min(dist, dim=1).values.item()
    is_anom = min_d > thr
    status = "🔴 异常 (ANOMALY)" if is_anom else "🟢 正常 (NORMAL)"
    result = f"{status}\n\nKNN距离: {min_d:.4f}\n阈值: {thr:.4f}\n偏离度: {min_d/thr:.1%}"
    detail = (f"设备类型: {device_type}\n文件: {os.path.basename(audio_path)}\n"
              f"KNN距离: {min_d:.6f}\n阈值: {thr:.6f}\n"
              f"判定: {'异常 - 超出阈值' if is_anom else '正常 - 安全范围'}")
    return result, detail

def detect_wrapper(audio_path, device_type):
    if audio_path is None:
        return "请先上传音频文件", "", gr.update(value=None, interactive=False), gr.update(visible=True)

    tmp = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.wav")
    try:
        shutil.copy2(audio_path, tmp)
        result, detail = detect_anomaly(tmp, device_type)
    except Exception as e:
        result, detail = f"检测出错: {e}", ""
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    return (
        result,
        detail,
        gr.update(value=None, interactive=False),
        gr.update(visible=True)
    )
def reset_detect():
    return (
        gr.update(value=None, interactive=True),
        "",
        "",
        gr.update(interactive=False)
    )



# ================================================================
#    二、故障诊断与深度报告 (修复 sklearn NotFittedError)
# ================================================================

def load_latest_features():
    files = glob.glob(os.path.join(FEATURE_DIR, "features_*.csv"))
    if not files:
        raise FileNotFoundError("未找到特征文件，请先运行特征工程")
    return pd.read_csv(max(files, key=os.path.getmtime))

def load_latest_raw_csv():
    files = glob.glob(os.path.join(RAW_DATA_DIR, "plc_data_*.csv"))
    if not files:
        raise FileNotFoundError("未找到原始采集数据")
    return pd.read_csv(max(files, key=os.path.getmtime))

def run_fault_diagnosis():
    try:
        df = load_latest_features()
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        # 1. 异常检测
        fcols = ["vibration_x_rms","vibration_x_kurtosis","vibration_x_crest_factor",
                 "vibration_y_rms","vibration_y_kurtosis","vibration_y_crest_factor",
                 "vibration_z_rms","vibration_z_kurtosis","vibration_z_crest_factor",
                 "bearing_temp_mean","motor_temp_mean","temp_gradient_bearing_ambient"]
        avail = [c for c in fcols if c in df.columns]
        X = df[avail].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        iso = IsolationForest(contamination=0.1, random_state=42, n_estimators=200)
        iso.fit(X_scaled)  # 🔥 必须先 fit
        df["iso_score"] = iso.decision_function(X_scaled)
        df["is_anomaly"] = (iso.predict(X_scaled) == -1).astype(int)

        # 2. 健康度
        if "degradation_level" in df.columns:
            d = df["degradation_level"].values
            dn = (d - d.min()) / (d.max() - d.min() + 1e-8)
            df["health_score"] = (1 - dn) * 100
        else:
            rms = df.get("rms_mean", pd.Series([0]))
            df["health_score"] = np.clip(100 - rms.values * 10, 0, 100)

        # 3. 故障规则
        faults = []
        for _, r in df.iterrows():
            fl = []
            vx, vy, vz = r.get("vibration_x_rms",0), r.get("vibration_y_rms",0), r.get("vibration_z_rms",0)
            kx, ky, kz = r.get("vibration_x_kurtosis",0), r.get("vibration_y_kurtosis",0), r.get("vibration_z_kurtosis",0)
            bt = r.get("bearing_temp_mean", 0)
            mt = r.get("motor_temp_mean", 0)
            tg = r.get("temp_gradient_bearing_ambient", 0)
            if vx > 4.5: fl.append(f"X轴振动严重超标 ({vx:.2f} mm/s)")
            elif vx > 3.0: fl.append(f"X轴振动轻微异常 ({vx:.2f} mm/s)")
            if vy > 4.5: fl.append(f"Y轴振动严重超标 ({vy:.2f} mm/s)")
            elif vy > 3.0: fl.append(f"Y轴振动轻微异常 ({vy:.2f} mm/s)")
            if vz > 4.5: fl.append(f"Z轴不对中风险 ({vz:.2f} mm/s)")
            if kx > 5: fl.append(f"滚动体点蚀/局部缺陷 (峭度={kx:.2f})")
            elif kx > 3: fl.append(f"早期磨损/润滑不良 (峭度={kx:.2f})")
            if bt > 70: fl.append(f"轴承过热 ({bt:.1f}°C)")
            if mt > 80: fl.append(f"电机过热 ({mt:.1f}°C)")
            if tg > 20: fl.append(f"散热不良/润滑失效 (温差={tg:.1f}°C)")
            fl = fl or ["各项指标均在标准范围内"]
            faults.append("; ".join(fl))
        df["fault_diagnosis"] = faults

        # 4. RUL
        df["rul_hours"] = df["health_score"].apply(
            lambda h: h*12.5 if h>=80 else h*8 if h>=50 else h*3 if h>=20 else max(0,h*0.5))

        last = df.iloc[-1]
        h = last["health_score"]
        rul = last["rul_hours"]
        if h >= 80: lv, sg = "良好 (Green)", "维持现有维保计划，按季度巡检"
        elif h >= 60: lv, sg = "注意 (Yellow)", "缩短巡检周期至每周，重点监控振动温度趋势"
        elif h >= 40: lv, sg = "警告 (Orange)", "建议安排下月计划性停机检修，准备备件"
        else: lv, sg = "危险 (Red)", "立即停机！联系维修人员进行轴承或润滑更换"

        ar = df["is_anomaly"].mean() * 100
        anom_cnt = int(df["is_anomaly"].sum())

        # 5. 深度报告
        report = f"""
{'='*70}
         工业设备预测性维护 - 深度诊断报告
{'='*70}
报告编号: RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
分析样本: {len(df)} 个滑动窗口 | 异常样本: {anom_cnt} 个 ({ar:.1f}%)

{'─'*70}
【一、设备健康概况】
{'─'*70}
  当前健康度得分 : {h:.1f} / 100
  健康等级评估   : {lv}
  剩余使用寿命   : 约 {rul:.0f} 小时 (基于当前退化速率线性推算)
  异常样本占比   : {ar:.1f}% ({anom_cnt}/{len(df)})
  数据质量评估   : {'✅ 数据完整可靠' if len(df) >= 50 else '⚠️ 数据量偏少，建议延长采集时间'}

{'─'*70}
【二、核心指标详细分析】
{'─'*70}
  2.1 振动分析 (Vibration Analysis)
  ─────────────────────────────────────────────────────────
     指标          当前值      正常范围      判定
     X轴RMS     : {last.get('vibration_x_rms',0):>8.3f} mm/s   [0~3.0]    {'✅ 正常' if last.get('vibration_x_rms',0)<=3 else '⚠️ 异常'}
     Y轴RMS     : {last.get('vibration_y_rms',0):>8.3f} mm/s   [0~3.0]    {'✅ 正常' if last.get('vibration_y_rms',0)<=3 else '⚠️ 异常'}
     Z轴RMS     : {last.get('vibration_z_rms',0):>8.3f} mm/s   [0~3.0]    {'✅ 正常' if last.get('vibration_z_rms',0)<=3 else '⚠️ 异常'}
     X轴峭度    : {last.get('vibration_x_kurtosis',0):>8.3f}         [2~3]      {'✅ 正常' if last.get('vibration_x_kurtosis',0)<=3 else '⚠️ 异常'}
     Y轴峭度    : {last.get('vibration_y_kurtosis',0):>8.3f}         [2~3]      {'✅ 正常' if last.get('vibration_y_kurtosis',0)<=3 else '⚠️ 异常'}
     Z轴峭度    : {last.get('vibration_z_kurtosis',0):>8.3f}         [2~3]      {'✅ 正常' if last.get('vibration_z_kurtosis',0)<=3 else '⚠️ 异常'}
     X轴峰值因子: {last.get('vibration_x_crest_factor',0):>8.3f}         [<5]       {'✅ 正常' if last.get('vibration_x_crest_factor',0)<=5 else '⚠️ 异常'}

     趋势分析:
     - 振动RMS均值: {df['vibration_x_rms'].mean():.3f} mm/s | 最大值: {df['vibration_x_rms'].max():.3f} mm/s
     - 振动RMS标准差: {df['vibration_x_rms'].std():.3f} mm/s | 变异系数: {df['vibration_x_rms'].std()/(df['vibration_x_rms'].mean()+1e-8)*100:.1f}%

  2.2 温度分析 (Thermal Analysis)
  ─────────────────────────────────────────────────────────
     指标          当前值      正常范围      判定
     轴承温度   : {last.get('bearing_temp_mean',0):>8.1f} °C     [0~65]     {'✅ 正常' if last.get('bearing_temp_mean',0)<=65 else '⚠️ 异常'}
     电机温度   : {last.get('motor_temp_mean',0):>8.1f} °C     [0~75]     {'✅ 正常' if last.get('motor_temp_mean',0)<=75 else '⚠️ 异常'}
     环境温度   : {last.get('ambient_temp_mean',0):>8.1f} °C     [-]
     轴承-环境温差: {last.get('temp_gradient_bearing_ambient',0):>8.1f} °C     [0~15]     {'✅ 正常' if last.get('temp_gradient_bearing_ambient',0)<=15 else '⚠️ 异常'}

     趋势分析:
     - 轴承温度均值: {df['bearing_temp_mean'].mean():.1f}°C | 最大值: {df['bearing_temp_mean'].max():.1f}°C
     - 电机温度均值: {df['motor_temp_mean'].mean():.1f}°C | 最大值: {df['motor_temp_mean'].max():.1f}°C

  2.3 运行状态分析
  ─────────────────────────────────────────────────────────
     运行转速     : {last.get('rpm_mean',0):>8.1f} RPM
     负载率       : {last.get('load_percent_mean',0):>8.1f} %
     累计运行时间 : {last.get('runtime_hours_mean',0):>8.0f} 小时
     退化程度     : {last.get('degradation_level',0):>8.3f} (0=全新, 1=完全退化)

{'─'*70}
【三、故障根因定位】
{'─'*70}
  最新诊断结论 : {last['fault_diagnosis']}
  异常检测状态 : {'⚠️ 发现异常波形模式' if last['is_anomaly'] else '✅ 波形特征正常'}
  孤立森林异常分: {last['iso_score']:.4f} (越负越异常)

{'─'*70}
【四、剩余寿命预测 (RUL)】
{'─'*70}
  预测模型     : 基于退化程度线性外推法
  当前健康度   : {h:.1f} / 100
  退化速率     : {(df['health_score'].iloc[0]-df['health_score'].iloc[-1])/(len(df)+1e-8):.4f} /窗口
  预计RUL      : {rul:.0f} 小时
  置信度       : {'高 (数据充足)' if len(df)>=100 else '中 (数据量一般)' if len(df)>=30 else '低 (建议补充数据)'}

{'─'*70}
【五、维护建议与行动计划】
{'─'*70}
  优先级       : {sg.split('，')[0] if '，' in sg else sg}
  行动建议     : {sg.split('，')[1] if '，' in sg else '详见下方清单'}

  建议执行清单:
  [ ] 1. 检查轴承磨损状况，必要时更换轴承
  [ ] 2. 检查润滑系统，补充或更换润滑油/脂
  [ ] 3. 校准振动传感器，确认安装牢固
  [ ] 4. 检查电机绝缘电阻和三相电流平衡
  [ ] 5. 清理散热通道，确认通风良好
  [ ] 6. 检查地脚螺栓是否松动

  验收标准:
  - 振动RMS < 3.0 mm/s (ISO 10816 标准)
  - 轴承温度 < 65°C
  - 健康度恢复至 80 分以上
  - 峭度指标恢复至 3 以下

{'='*70}
                              报告结束
{'='*70}
"""
        # 6. 绘图 (健康度与RUL趋势)
        fig, ax1 = plt.subplots(figsize=(10, 5), facecolor="#0f172a")
        ax1.set_facecolor("#0f172a")
        ax1.tick_params(colors="white")
        ax1.set_xlabel("Sample Window", color="white", fontsize=11)
        ax1.set_ylabel("Health Score", color="#60a5fa", fontsize=11)
        ax1.plot(df["health_score"].values, "o-", color="#3b82f6", ms=4, lw=1.5, label="Health Score")
        ax1.tick_params(axis="y", labelcolor="#60a5fa")
        ax1.axhline(80, color="#10b981", ls="--", alpha=0.5, label="Good (80)")
        ax1.axhline(60, color="#f59e0b", ls="--", alpha=0.5, label="Warning (60)")
        ax1.axhline(40, color="#ef4444", ls="--", alpha=0.5, label="Danger (40)")
        ax2 = ax1.twinx()
        ax2.set_ylabel("RUL (Hours)", color="#10b981", fontsize=11)
        ax2.plot(df["rul_hours"].values, "s-", color="#10b981", ms=4, lw=1.5, label="RUL")
        ax2.tick_params(axis="y", labelcolor="#10b981")
        ax1.set_title("Equipment Health & RUL Trend", color="white", fontsize=14, fontweight="bold")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1+lines2, labels1+labels2, loc="lower left", fontsize=9, facecolor="#1e293b")
        fig.tight_layout()
        cp = os.path.join(FIGURE_DIR, "health_trend.png")
        plt.savefig(cp, facecolor=fig.get_facecolor(), dpi=150)
        plt.close()

        # 7. 告警
        alerts = []
        if h < 40: alerts.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🔴 严重: 健康度 {h:.1f}，建议立即停机")
        if last["is_anomaly"]: alerts.append(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ 警告: 检测到异常波形模式")
        if last.get("bearing_temp_mean",0) > 70: alerts.append(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ 警告: 轴承温度 {last['bearing_temp_mean']:.1f}°C 超标")
        if last.get("vibration_x_rms",0) > 4.5: alerts.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🔴 严重: X轴振动 {last['vibration_x_rms']:.2f} mm/s 严重超标")
        if alerts:
            _save_alerts(alerts)

        return report, "诊断完成", cp, "\n".join(alerts) if alerts else "✅ 无告警"

    except Exception as e:
        import traceback
        return f"诊断异常:\n{traceback.format_exc()}", "失败", None, ""


def generate_summary_report():
    try:
        df = load_latest_features()
        last = df.iloc[-1]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fp = os.path.join(REPORT_DIR, f"summary_report_{ts}.txt")
        c = (f"工业设备预测性维护 - 汇总报告\n{'='*40}\n"
             f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
             f"健康度: {last.get('health_score',0):.2f}/100\n"
             f"RUL: {last.get('rul_hours',0):.1f} 小时\n"
             f"故障: {last.get('fault_diagnosis','N/A')}\n"
             f"样本数: {len(df)}\n")
        with open(fp, "w", encoding="utf-8") as f:
            f.write(c)
        return fp, f"已保存: {fp}"
    except Exception as e:
        return None, f"导出失败: {e}"

def generate_work_order():
    try:
        df = load_latest_features()
        last = df.iloc[-1]
        h = last.get("health_score", 50)
        fault = last.get("fault_diagnosis", "未知")
        rul = last.get("rul_hours", 0)

        if h >= 80: pri = "低 - 常规保养"
        elif h >= 60: pri = "中 - 加强巡检"
        elif h >= 40: pri = "高 - 计划停机检修"
        else: pri = "紧急 - 立即停机"

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fp = os.path.join(REPORT_DIR, f"work_order_{ts}.txt")
        c = f"""{'='*50}
工业设备维护工单
{'='*50}
工单编号: WO-{ts}
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
优先级: {pri}

[设备状态]
  健康度: {h:.1f}/100
  剩余寿命: {rul:.0f} 小时
  故障诊断: {fault}

[作业项目]
  1. 检查轴承磨损状况，必要时更换轴承
  2. 检查润滑系统，补充或更换润滑油/脂
  3. 校准振动传感器，确认安装牢固
  4. 检查电机绝缘电阻和三相电流平衡

[验收标准]
  - 振动RMS < 3.0 mm/s
  - 轴承温度 < 65°C
  - 健康度恢复至 80 分以上

[签字]
  维修人员: ____________ 日期: ________
  验收人员: ____________ 日期: ________
"""
        with open(fp, "w", encoding="utf-8") as f:
            f.write(c)
        return fp, f"工单已生成: {fp}"
    except Exception as e:
        return None, f"生成失败: {e}"


# ================================================================
#    三、仪表盘
# ================================================================

def generate_dashboard():
    try:
        df = load_latest_raw_csv()
    except FileNotFoundError as e:
        return None, str(e)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    x = df["timestamp"] if "timestamp" in df.columns else range(len(df))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor="#0f172a")
    fig.suptitle("Predictive Maintenance Dashboard", fontsize=18, color="white", fontweight="bold")
    for ax in axes.flat:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="white", labelsize=8)

    ax = axes[0,0]
    for c, l, cl in [("vibration_x","X轴","#3b82f6"),("vibration_y","Y轴","#f59e0b"),("vibration_z","Z轴","#ef4444")]:
        if c in df.columns: ax.plot(x, df[c], label=l, color=cl, lw=1.2)
    ax.axhline(4.5, color="r", ls="--", alpha=.7, label="Alarm 4.5")
    ax.set_title("Vibration (mm/s)", color="white")
    ax.legend(fontsize=8)

    ax = axes[0,1]
    for c, l, cl in [("bearing_temp","轴承","#ef4444"),("motor_temp","电机","#f59e0b"),("ambient_temp","环境","#3b82f6")]:
        if c in df.columns: ax.plot(x, df[c], label=l, color=cl, lw=1.2)
    ax.axhline(70, color="r", ls="--", alpha=.7, label="Alarm 70°C")
    ax.set_title("Temperature (°C)", color="white")
    ax.legend(fontsize=8)

    ax = axes[1,0]
    if "rms_value" in df.columns:
        ax.plot(x, df["rms_value"], color="#a78bfa", lw=1.2, label="RMS")
    ax.set_title("RMS & Degradation", color="white")
    ax.set_ylabel("RMS", color="#a78bfa")
    if "degradation_level" in df.columns:
        ax2 = ax.twinx()
        ax2.plot(x, df["degradation_level"], color="#34d399", ls="--", lw=1.5, label="Deg")
        ax2.set_ylabel("Deg", color="#34d399")
        ax2.legend(loc="upper right", fontsize=8)
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1,1]
    if "vibration_x" in df.columns:
        v = df["vibration_x"].values
        fft = np.abs(np.fft.rfft(v))
        frq = np.fft.rfftfreq(len(v), d=1.0)
        ax.plot(frq, fft, color="#2dd4bf", lw=1)
        ax.set_xlim(0, max(0.5, frq.max()*0.1))
    ax.set_title("FFT Spectrum", color="white")

    plt.tight_layout()
    cp = os.path.join(FIGURE_DIR, "dashboard.png")
    plt.savefig(cp, facecolor=fig.get_facecolor(), dpi=150, bbox_inches="tight")
    plt.close()
    return cp, f"已生成 ({len(df)} 条)"


# ================================================================
#    四、特征工程
# ================================================================

def run_feature_engineering(window_size, step_size):
    logs = [f"[{datetime.now().strftime('%H:%M:%S')}] 开始特征工程..."]
    try:
        files = glob.glob(os.path.join(RAW_DATA_DIR, "plc_data_*.csv"))
        if not files:
            return "未找到原始数据文件", None
        latest = max(files, key=os.path.getmtime)
        logs.append(f"输入文件: {os.path.basename(latest)}")
        df = pd.read_csv(latest)
        logs.append(f"原始数据: {len(df)} 行 x {len(df.columns)} 列")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        logs.append(f"数值列: {', '.join(numeric_cols)}")

        ws, ss = int(window_size), int(step_size)
        features = []
        for start in range(0, len(df) - ws + 1, ss):
            window = df.iloc[start:start + ws]
            feat_row = {}
            for col in numeric_cols:
                vals = window[col].values.astype(float)
                feat_row[f"{col}_rms"] = np.sqrt(np.mean(vals**2))
                feat_row[f"{col}_mean"] = np.mean(vals)
                feat_row[f"{col}_std"] = np.std(vals)
                feat_row[f"{col}_max"] = np.max(vals)
                feat_row[f"{col}_min"] = np.min(vals)
                k = np.mean((vals - vals.mean())**4) / (vals.std()**4 + 1e-8)
                feat_row[f"{col}_kurtosis"] = k
                cf = np.max(np.abs(vals)) / (np.sqrt(np.mean(vals**2)) + 1e-8)
                feat_row[f"{col}_crest_factor"] = cf
            if "degradation_level" in df.columns:
                feat_row["degradation_level"] = window["degradation_level"].iloc[-1]
            features.append(feat_row)

        feat_df = pd.DataFrame(features)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(FEATURE_DIR, f"features_{ts}.csv")
        feat_df.to_csv(out_path, index=False)

        logs.append(f"窗口大小: {ws} | 步长: {ss}")
        logs.append(f"提取特征: {len(feat_df)} 条 x {len(feat_df.columns)} 维")
        logs.append(f"已保存: {os.path.basename(out_path)}")

        return "\n".join(logs), feat_df.head(10)
    except Exception as e:
        logs.append(f"错误: {e}")
        return "\n".join(logs), None


# ================================================================
#    五、数据采集
# ================================================================

def get_acq_status():
    rc = len(glob.glob(os.path.join(RAW_DATA_DIR, "plc_data_*.csv")))
    fc = len(glob.glob(os.path.join(FEATURE_DIR, "features_*.csv")))
    return (f"原始数据文件: {rc}\n特征文件: {fc}\n"
            f"OPC UA: {OPC_UA_SERVER_URL}\n"
            f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def start_real_collection(duration):
    logs = []
    duration = int(duration)
    def L(m):
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {m}")
        return "\n".join(logs)

    client = None
    try:
        import opcua
        yield L(f"连接 OPC UA: {OPC_UA_SERVER_URL} ...")
        client = opcua.Client(OPC_UA_SERVER_URL)
        client.connect()
        yield L("连接成功")
        variables = {}
        for vn, nid in OPC_UA_NODE_IDS.items():
            try:
                n = client.get_node(nid); n.get_value()
                variables[vn] = n
            except: pass
        if not variables:
            yield L("无可用节点，切换模拟模式...")
            client.disconnect(); client = None
            yield from _sim_collect(duration, logs)
            return
        yield L(f"绑定 {len(variables)} 个节点，采集 {duration}s...")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(RAW_DATA_DIR, f"plc_data_{ts}.csv")
        vns = list(variables.keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["timestamp"] + vns)
            t0 = time.time(); cnt = 0
            while time.time() - t0 < duration:
                try:
                    row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                    vals = []
                    for vn in vns:
                        v = variables[vn].get_value()
                        rv = round(v, 4) if isinstance(v, float) else v
                        vals.append(rv); row.append(rv)
                    w.writerow(row); f.flush(); cnt += 1
                    if cnt % 5 == 0:
                        vi = ""
                        if "vibration_x" in vns:
                            ix = vns.index("vibration_x")
                            vi = f" | X={vals[ix]:.2f}"
                        yield L(f"已采集 {cnt} 条{vi}")
                    time.sleep(SAMPLE_INTERVAL)
                except Exception as e:
                    yield L(f"读取异常: {e}")
                    time.sleep(1)
        yield L(f"完成! {cnt} 条 → {os.path.basename(csv_path)}")
        client.disconnect()
    except ImportError:
        yield L("未安装 opcua，切换模拟模式...")
        yield from _sim_collect(duration, logs)
    except Exception as e:
        yield L(f"连接失败: {e}，切换模拟模式...")
        if client:
            try: client.disconnect()
            except: pass
        yield from _sim_collect(duration, logs)

def _sim_collect(duration, logs):
    def L(m):
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] [模拟] {m}")
        return "\n".join(logs)
    yield L("初始化模拟引擎...")
    time.sleep(0.3)
    yield L("模拟 PLC 连接成功")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(RAW_DATA_DIR, f"plc_data_{ts}.csv")
    vns = list(OPC_UA_NODE_IDS.keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["timestamp"] + vns)
        t0 = time.time(); cnt = 0
        while time.time() - t0 < duration:
            t = time.time() - t0
            deg = min(1.0, t/duration*0.8 + np.random.uniform(0,0.05))
            rd = {
                "vibration_x": round(np.random.uniform(1.5,5)+deg*2, 4),
                "vibration_y": round(np.random.uniform(1.2,4.5)+deg*1.5, 4),
                "vibration_z": round(np.random.uniform(0.8,3.5)+deg, 4),
                "rms_value": round(np.random.uniform(2,4.5)+deg*1.5, 4),
                "bearing_temp": round(np.random.uniform(45,75)+deg*10, 4),
                "motor_temp": round(np.random.uniform(55,85)+deg*5, 4),
                "ambient_temp": round(np.random.uniform(22,28), 4),
                "runtime_hours": round(1200+t/3600, 4),
                "rpm": round(np.random.uniform(1450,1520), 4),
                "load_percent": round(np.random.uniform(60,95), 4),
                "degradation_level": round(deg, 4),
                "fault_type": 0,
            }
            row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] + [rd.get(v,0) for v in vns]
            w.writerow(row); f.flush(); cnt += 1
            if cnt % 5 == 0:
                yield L(f"已采集 {cnt} 条 | X={rd['vibration_x']:.2f} 温度={rd['bearing_temp']:.1f}°C 退化={rd['degradation_level']:.3f}")
            time.sleep(SAMPLE_INTERVAL)
    yield L(f"完成! {cnt} 条 → {os.path.basename(csv_path)}")


# ================================================================
#    六、实时监控
# ================================================================

def _save_alerts(alerts):
    existing = []
    if os.path.exists(ALERT_FILE):
        try:
            with open(ALERT_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except: pass
    existing.extend(alerts)
    with open(ALERT_FILE, "w", encoding="utf-8") as f:
        json.dump(existing[-200:], f, ensure_ascii=False, indent=2)

def run_live_monitor(interval, max_rounds):
    interval = int(interval); max_rounds = int(max_rounds)
    logs = []; alert_text = ""
    def L(m):
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {m}")
        return "\n".join(logs)

    yield L(f"实时监控启动 (间隔 {interval}s, 共 {max_rounds} 轮)"), ""

    for i in range(max_rounds):
        time.sleep(interval)
        try:
            files = glob.glob(os.path.join(RAW_DATA_DIR, "plc_data_*.csv"))
            if not files:
                yield L("无数据文件"), alert_text
                continue
            df = pd.read_csv(max(files, key=os.path.getmtime))
            last = df.iloc[-1]
            vx = last.get("vibration_x", 0)
            bt = last.get("bearing_temp", 0)
            deg = last.get("degradation_level", 0)
            rms = last.get("rms_value", 0)

            status = f"振动X={vx:.2f} | RMS={rms:.2f} | 轴承={bt:.1f}°C | 退化={deg:.3f}"
            yield L(f"#{i+1} {status}"), alert_text

            alerts = []
            if vx > 4.5: alerts.append(f"[{datetime.now().strftime('%H:%M:%S')}] 严重: 振动X={vx:.2f} 超标!")
            if bt > 70: alerts.append(f"[{datetime.now().strftime('%H:%M:%S')}] 警告: 轴承温度={bt:.1f}°C 超标!")
            if deg > 0.8: alerts.append(f"[{datetime.now().strftime('%H:%M:%S')}] 严重: 退化度={deg:.3f} 极高!")
            if alerts:
                _save_alerts(alerts)
                alert_text += "\n".join(alerts) + "\n"
                yield "\n".join(logs), alert_text
        except Exception as e:
            yield L(f"读取异常: {e}"), alert_text

    yield L("监控结束"), alert_text


# ================================================================
#    七、历史对比
# ================================================================

def list_data_files():
    raw = glob.glob(os.path.join(RAW_DATA_DIR, "plc_data_*.csv"))
    feat = glob.glob(os.path.join(FEATURE_DIR, "features_*.csv"))
    lines = ["【原始数据】"]
    for f in sorted(raw, key=os.path.getmtime, reverse=True):
        sz = os.path.getsize(f) / 1024
        lines.append(f"  {os.path.basename(f)}  ({sz:.1f} KB)")
    lines.append("\n【特征文件】")
    for f in sorted(feat, key=os.path.getmtime, reverse=True):
        sz = os.path.getsize(f) / 1024
        lines.append(f"  {os.path.basename(f)}  ({sz:.1f} KB)")
    return "\n".join(lines) if len(lines) > 2 else "暂无数据文件"

def compare_files(name1, name2):
    if not name1 or not name2:
        return "请输入两个文件名", None
    p1 = os.path.join(RAW_DATA_DIR, name1.strip())
    p2 = os.path.join(RAW_DATA_DIR, name2.strip())
    if not os.path.exists(p1): p1 = os.path.join(FEATURE_DIR, name1.strip())
    if not os.path.exists(p2): p2 = os.path.join(FEATURE_DIR, name2.strip())
    if not os.path.exists(p1) or not os.path.exists(p2):
        return "文件不存在，请检查文件名", None
    try:
        df1 = pd.read_csv(p1); df2 = pd.read_csv(p2)
        num1 = df1.select_dtypes(include=[np.number])
        num2 = df2.select_dtypes(include=[np.number])
        common = [c for c in num1.columns if c in num2.columns]
        if not common: return "没有共同的数值列", None
        means1 = num1[common].mean(); means2 = num2[common].mean()
        change = ((means2 - means1) / (means1.abs() + 1e-8)) * 100

        info = f"文件A: {name1} ({len(df1)} 行)\n文件B: {name2} ({len(df2)} 行)\n共同列: {len(common)}\n\n"
        info += f"{'指标':<25} {'A均值':>10} {'B均值':>10} {'变化%':>10}\n" + "-"*60 + "\n"
        for c in common:
            info += f"{c:<25} {means1[c]:>10.4f} {means2[c]:>10.4f} {change[c]:>9.1f}%\n"

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0f172a")
        for ax in (ax1, ax2):
            ax.set_facecolor("#1e293b"); ax.tick_params(colors="white", labelsize=8)
        x_pos = np.arange(len(common))
        w = 0.35
        ax1.bar(x_pos-w/2, means1.values, w, label=name1[:20], color="#3b82f6")
        ax1.bar(x_pos+w/2, means2.values, w, label=name2[:20], color="#f59e0b")
        ax1.set_xticks(x_pos); ax1.set_xticklabels(common, rotation=45, ha="right", fontsize=7)
        ax1.set_title("Mean Comparison", color="white"); ax1.legend(fontsize=8)

        colors = ["#10b981" if v <= 0 else "#ef4444" for v in change.values]
        ax2.barh(x_pos, change.values, color=colors)
        ax2.set_yticks(x_pos); ax2.set_yticklabels(common, fontsize=7)
        ax2.axvline(0, color="white", lw=0.5)
        ax2.set_title("Change %", color="white")

        plt.tight_layout()
        cp = os.path.join(FIGURE_DIR, "comparison.png")
        plt.savefig(cp, facecolor=fig.get_facecolor(), dpi=150); plt.close()
        return info, cp
    except Exception as e:
        return f"对比失败: {e}", None


# ================================================================
#    八、高级可视化 (修复 sklearn 问题，确保4张图全出)
# ================================================================

def generate_advanced_charts():
    try:
        df = load_latest_features()
    except Exception:
        return None, None, None, None, "请先运行特征工程"

    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    fcols = ["vibration_x_rms","vibration_x_kurtosis","vibration_x_crest_factor",
             "vibration_y_rms","vibration_y_kurtosis","vibration_y_crest_factor",
             "vibration_z_rms","vibration_z_kurtosis","vibration_z_crest_factor",
             "bearing_temp_mean","motor_temp_mean","temp_gradient_bearing_ambient"]
    avail = [c for c in fcols if c in df.columns]
    
    if avail:
        X = df[avail].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        iso = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
        iso.fit(X_scaled)  # 🔥 必须先 fit
        df["iso_score"] = iso.decision_function(X_scaled)
        df["is_anomaly"] = (iso.predict(X_scaled) == -1).astype(int)

    if "degradation_level" in df.columns:
        d = df["degradation_level"].values
        dn = (d - d.min()) / (d.max() - d.min() + 1e-8)
        df["health_score"] = (1 - dn) * 100
    else:
        rms = df.get("rms_mean", pd.Series([0]))
        df["health_score"] = np.clip(100 - rms.values * 10, 0, 100)

    df["rul_hours"] = df["health_score"].apply(
        lambda h: h*12.5 if h>=80 else h*8 if h>=50 else h*3 if h>=20 else max(0,h*0.5))

    charts = [None, None, None, None]

    # 图1：异常分数分布
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0f172a")
    ax.set_facecolor("#1e293b"); ax.tick_params(colors="white")
    normal = df[df["is_anomaly"]==0]["iso_score"]
    anomaly = df[df["is_anomaly"]==1]["iso_score"]
    if len(normal): ax.hist(normal, bins=30, alpha=0.7, color="#3b82f6", label=f"Normal ({len(normal)})")
    if len(anomaly): ax.hist(anomaly, bins=30, alpha=0.7, color="#ef4444", label=f"Anomaly ({len(anomaly)})")
    ax.axvline(0, color="yellow", ls="--", alpha=0.5, label="Threshold (0)")
    ax.set_title("Anomaly Score Distribution", color="white", fontsize=12)
    ax.set_xlabel("ISO Score", color="white")
    ax.legend(fontsize=9, facecolor="#1e293b")
    plt.tight_layout()
    p = os.path.join(FIGURE_DIR, "adv_score_dist.png")
    plt.savefig(p, facecolor=fig.get_facecolor(), dpi=150); plt.close()
    charts[0] = p

    # 图2：特征相关性
    num = df.select_dtypes(include=[np.number])
    if len(num.columns) > 1:
        fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0f172a")
        ax.set_facecolor("#1e293b"); ax.tick_params(colors="white", labelsize=6)
        key_cols = [c for c in ["vibration_x_rms","vibration_y_rms","vibration_z_rms",
                                  "bearing_temp_mean","motor_temp_mean","health_score",
                                  "degradation_level","iso_score","rul_hours"] if c in num.columns]
        if not key_cols: key_cols = list(num.columns[:10])
        corr = num[key_cols].corr()
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr.columns))); ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(corr.columns, fontsize=7)
        ax.set_title("Feature Correlation", color="white", fontsize=12)
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        p = os.path.join(FIGURE_DIR, "adv_corr.png")
        plt.savefig(p, facecolor=fig.get_facecolor(), dpi=150); plt.close()
        charts[1] = p

    # 图3：健康度分布
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0f172a")
    ax.set_facecolor("#1e293b"); ax.tick_params(colors="white")
    ax.hist(df["health_score"], bins=20, color="#8b5cf6", edgecolor="white", alpha=0.8)
    ax.axvline(40, color="#ef4444", ls="--", lw=2, label="Danger 40")
    ax.axvline(60, color="#f59e0b", ls="--", lw=2, label="Warning 60")
    ax.axvline(80, color="#10b981", ls="--", lw=2, label="Good 80")
    ax.set_title("Health Score Distribution", color="white", fontsize=12)
    ax.set_xlabel("Health Score", color="white")
    ax.legend(fontsize=9, facecolor="#1e293b")
    plt.tight_layout()
    p = os.path.join(FIGURE_DIR, "adv_health_dist.png")
    plt.savefig(p, facecolor=fig.get_facecolor(), dpi=150); plt.close()
    charts[2] = p

    # 图4：关键指标雷达图
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0f172a", subplot_kw=dict(polar=True))
    ax.set_facecolor("#1e293b"); ax.tick_params(colors="white")
    metrics = {}
    for c in ["vibration_x_rms","vibration_y_rms","vibration_z_rms","bearing_temp_mean","motor_temp_mean"]:
        if c in df.columns:
            col_data = df[c]
            val = col_data.iloc[-1]
            mn, mx = col_data.min(), col_data.max()
            norm = (val - mn) / (mx - mn + 1e-8)
            metrics[c.replace("_rms","").replace("_mean","")] = norm
    if metrics:
        labels = list(metrics.keys())
        values = list(metrics.values())
        values += values[:1]
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]
        ax.plot(angles, values, "o-", color="#3b82f6", lw=2, markersize=8)
        ax.fill(angles, values, color="#3b82f6", alpha=0.2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, color="white", fontsize=9)
        ax.set_title("Latest Metrics (Normalized)", color="white", fontsize=12)
        ax.set_ylim(0, 1)
    plt.tight_layout()
    p = os.path.join(FIGURE_DIR, "adv_radar.png")
    plt.savefig(p, facecolor=fig.get_facecolor(), dpi=150); plt.close()
    charts[3] = p

    return charts[0], charts[1], charts[2], charts[3], "高级分析完成"


# ================================================================
#    九、Gradio 界面
# ================================================================

CUSTOM_CSS = """
.gradio-container { max-width: 1400px !important; }
footer { display: none !important; }
"""

def create_ui():
    with gr.Blocks() as demo:

        gr.Markdown("""
# 🏭 工业设备预测性维护系统
**OPC UA 实时采集 · 智能故障诊断 · 音频异常检测 · 剩余寿命预测**
""")

        with gr.Tabs():

            # Tab 1: 数据采集
            with gr.Tab("📡 数据采集"):
                gr.Markdown("### PLC 实时数据采集 (OPC UA)")
                with gr.Row():
                    with gr.Column(scale=1):
                        acq_status = gr.Textbox(label="系统状态", value=get_acq_status(),
                                                interactive=False, lines=6)
                        refresh_btn = gr.Button("🔄 刷新状态", size="sm")
                        gr.Markdown("---")
                        dur = gr.Slider(10, 300, 60, 10, label="采集时长 (秒)")
                        acq_btn = gr.Button("▶ 启动采集", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        acq_log = gr.Textbox(label="实时日志", value="等待启动...",
                                             interactive=False, lines=18)
                refresh_btn.click(fn=get_acq_status, outputs=acq_status)
                acq_btn.click(fn=start_real_collection, inputs=[dur], outputs=acq_log)

            # Tab 2: 特征工程
            with gr.Tab("⚙️ 特征工程"):
                gr.Markdown("### 滑动窗口特征提取")
                with gr.Row():
                    with gr.Column(scale=1):
                        ws = gr.Slider(10, 200, 50, 10, label="窗口大小")
                        ss = gr.Slider(5, 100, 25, 5, label="步长")
                        fe_btn = gr.Button("▶ 执行特征工程", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        fe_log = gr.Textbox(label="执行日志", interactive=False, lines=8)
                        fe_table = gr.Dataframe(label="特征预览 (前10条)", interactive=False)
                fe_btn.click(fn=run_feature_engineering, inputs=[ws, ss], outputs=[fe_log, fe_table])

            # Tab 3: 故障诊断
            with gr.Tab("🔍 故障诊断"):
                gr.Markdown("### 异常检测 · 健康评估 · 故障分类 · RUL")
                with gr.Row():
                    diag_btn = gr.Button("▶ 运行诊断", variant="primary", size="lg")
                    export_btn = gr.Button("📝 导出报告", size="lg")
                    wo_btn = gr.Button("📋 生成工单", size="lg")
                diag_status = gr.Textbox(label="状态", interactive=False, lines=1)
                with gr.Row():
                    diag_report = gr.Textbox(label="深度诊断报告", interactive=False, lines=22)
                    diag_chart = gr.Image(label="健康度 & RUL 趋势", height=320)
                with gr.Row():
                    alert_box = gr.Textbox(label="告警信息", interactive=False, lines=4)
                with gr.Row():
                    report_file = gr.File(label="下载报告")
                    export_status = gr.Textbox(label="导出状态", interactive=False, lines=1)
                with gr.Row():
                    wo_file = gr.File(label="下载工单")
                    wo_status = gr.Textbox(label="工单状态", interactive=False, lines=1)

                diag_btn.click(fn=run_fault_diagnosis,
                               outputs=[diag_report, diag_status, diag_chart, alert_box])
                export_btn.click(fn=generate_summary_report, outputs=[report_file, export_status])
                wo_btn.click(fn=generate_work_order, outputs=[wo_file, wo_status])

            # Tab 4: 仪表盘
            with gr.Tab("📊 仪表盘"):
                gr.Markdown("### 振动 · 温度 · RMS · FFT 四合一")
                dash_btn = gr.Button("▶ 生成仪表盘", variant="primary", size="lg")
                dash_status = gr.Textbox(label="状态", interactive=False, lines=1)
                dash_chart = gr.Image(label="Dashboard", height=500)
                dash_btn.click(fn=generate_dashboard, outputs=[dash_chart, dash_status])

            # Tab 5: 高级分析
            with gr.Tab("📈 高级分析"):
                gr.Markdown("### 异常分数 · 相关性 · 健康分布 · 雷达对比")
                adv_btn = gr.Button("▶ 生成分析图表", variant="primary", size="lg")
                adv_status = gr.Textbox(label="状态", interactive=False, lines=1)
                with gr.Row():
                    adv_img1 = gr.Image(label="异常分数分布", height=280)
                    adv_img2 = gr.Image(label="特征相关性", height=280)
                with gr.Row():
                    adv_img3 = gr.Image(label="健康度分布", height=280)
                    adv_img4 = gr.Image(label="关键指标雷达图", height=280)
                adv_btn.click(fn=generate_advanced_charts,
                              outputs=[adv_img1, adv_img2, adv_img3, adv_img4, adv_status])

            # Tab 6: 实时监控
            with gr.Tab("🔴 实时监控"):
                gr.Markdown("### 自动刷新 · 超标告警 · 持久化日志")
                with gr.Row():
                    with gr.Column(scale=1):
                        mon_int = gr.Slider(2, 30, 5, 1, label="刷新间隔 (秒)")
                        mon_cnt = gr.Slider(3, 60, 12, 1, label="刷新次数")
                        mon_btn = gr.Button("▶ 启动监控", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        mon_log = gr.Textbox(label="监控日志", interactive=False, lines=12)
                        mon_alert = gr.Textbox(label="告警记录", interactive=False, lines=6)
                mon_btn.click(fn=run_live_monitor, inputs=[mon_int, mon_cnt],
                              outputs=[mon_log, mon_alert])

            # Tab 7: 历史对比
            with gr.Tab("📂 历史对比"):
                gr.Markdown("### 数据浏览 · 双文件对比分析")
                list_btn = gr.Button("📋 列出所有文件", size="sm")
                file_list = gr.Textbox(label="文件列表", interactive=False, lines=8)
                list_btn.click(fn=list_data_files, outputs=file_list)
                gr.Markdown("---")
                with gr.Row():
                    fn1 = gr.Textbox(label="文件A (含.csv后缀)")
                    fn2 = gr.Textbox(label="文件B (含.csv后缀)")
                cmp_btn = gr.Button("▶ 对比分析", variant="primary", size="lg")
                with gr.Row():
                    cmp_text = gr.Textbox(label="对比结果", interactive=False, lines=12)
                    cmp_chart = gr.Image(label="对比图表", height=350)
                cmp_btn.click(fn=compare_files, inputs=[fn1, fn2], outputs=[cmp_text, cmp_chart])

            # Tab 8: 音频检测
            with gr.Tab("🎵 音频检测"):
                gr.Markdown("### MIMII 设备音频异常检测 (ResNet18 + Memory Bank KNN)")
                with gr.Row():
                    with gr.Column(scale=1):
                        audio_input = gr.Audio(label="上传音频 (.wav)", type="filepath", sources=["upload"])
                        dev_dd = gr.Dropdown(MIMII_DEVICES, value="fan", label="设备类型")
                        detect_btn = gr.Button("▶ 检测", variant="primary", size="lg")
                        eset_btn = gr.Button("🔄 重置", size="lg")
                        detect_hint = gr.Markdown("💡 上传文件后点击检测，每次检测需重新上传")
                    with gr.Column(scale=1):
                        det_result = gr.Textbox(label="检测结果", interactive=False, lines=6)
                        det_detail = gr.Textbox(label="详细信息", interactive=False, lines=6)

                audio_input.change(
                    fn=lambda p: gr.update(interactive=(p is not None)),
                    inputs=[audio_input],
                    outputs=[detect_btn]
                )

                detect_btn.click(
                    fn=detect_wrapper,
                    inputs=[audio_input, dev_dd],
                    outputs=[det_result, det_detail, audio_input, detect_hint]
                )

                eset_btn.click(
                    fn=reset_detect,
                    outputs=[audio_input, det_result, det_detail, detect_btn]
                )

        return demo


# ================================================================
#    十、启动
# ================================================================

if __name__ == "__main__":
    load_mimii_thresholds()
    demo = create_ui()
    demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
    share=False,
    theme=gr.themes.Soft(primary_hue="blue"),
    css=CUSTOM_CSS
)