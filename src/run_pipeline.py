"""
工业设备预测性维护系统 - 统一流水线入口
用法:
  python srcrun_pipeline.py collect          # 模式1: PLC数据采集
  python srcrun_pipeline.py diagnose         # 模式2: 基于PLC数据的故障诊断
  python srcrun_pipeline.py detect <wav> <device>  # 模式3: 单条音频异常检测
  python srcrun_pipeline.py visualize        # 模式4: 生成可视化图表
  python srcrun_pipeline.py web              # 模式5: 启动Gradio Web界面
  python srcrun_pipeline.py full             # 模式6: 全流程(采集→诊断→报告)
"""
import os
import sys
import json
import subprocess
import time
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


def banner(title):
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * width}\n")


# ===== 模式1: PLC数据采集 =====
def run_collect(duration=60):
    """从PLC采集传感器数据，存入SQLite"""
    banner("模式1: PLC 数据采集")

    from src.data_acquisition import DataAcquisition

    acq = DataAcquisition()
    print(f"  开始采集，持续 {duration} 秒...")
    print(f"  数据来源: Siemens S7-1200 PLC (Snap7)")
    print(f"  存储目标: data/pm_database.db + InfluxDB\n")

    try:
        acq.start_collection(duration=duration)
        print("\n  ✅ 数据采集完成")
    except Exception as e:
        print(f"\n  ⚠ 采集异常（可能PLC未连接）: {e}")
        print("  提示: 若PLC未连接，可先用仿真数据:")
        print("    python srcdata_generator.py")


# ===== 模式2: 基于PLC数据的故障诊断 =====
def run_diagnose():
    """对采集到的PLC数据进行故障诊断"""
    banner("模式2: PLC数据故障诊断")

    from src.fault_diagnosis import FaultDiagnosis

    diag = FaultDiagnosis()
    print("  正在从数据库读取最新采集数据...")
    print("  分析维度: 振动、温度、设备状态\n")

    try:
        report = diag.diagnose_all()
        report_path = os.path.join(project_root, "docs", "diagnosis_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"  ✅ 诊断报告已保存: docsdiagnosis_report.md")
    except Exception as e:
        print(f"  ⚠ 诊断异常: {e}")


# ===== 模式3: 单条音频异常检测 =====
def run_detect(audio_path, device_type):
    """对单条音频进行MIMII异常检测"""
    banner(f"模式3: 音频异常检测 [{device_type}]")

    from src.predict import predict_single

    print(f"  音频文件: {audio_path}")
    print(f"  设备类型: {device_type}")
    print(f"  模型: ResNet18 + Memory Bank KNN\n")

    result = predict_single(audio_path, device_type)
    print(f"\n  检测结果: {result}")


# ===== 模式4: 生成可视化图表 =====
def run_visualize():
    """运行全套可视化（ROC/混淆矩阵/t-SNE/分数分布）"""
    banner("模式4: 生成可视化图表")

    script = os.path.join(project_root, "src", "visualization_all.py")
    print("  将生成以下图表:")
    print("    1. ROC曲线 (5个设备)")
    print("    2. 混淆矩阵热力图")
    print("    3. t-SNE特征分布")
    print("    4. 异常分数分布\n")

    subprocess.run([sys.executable, script], cwd=project_root)


# ===== 模式5: 启动Gradio Web界面 =====
def run_web():
    """启动Gradio交互式Web界面"""
    banner("模式5: 启动 Gradio Web 界面")

    script = os.path.join(project_root, "src", "gradio_app.py")
    print("  启动后请在浏览器访问: http://127.0.0.1:7860")
    print("  按 Ctrl+C 停止服务\n")

    subprocess.run([sys.executable, script], cwd=project_root)


# ===== 模式6: 全流程 =====
def run_full():
    """端到端全流程: 采集 → 诊断 → 可视化"""
    banner("模式6: 全流程运行")

    steps = [
        ("Step 1/3: PLC数据采集 (60秒)", lambda: run_collect(duration=60)),
        ("Step 2/3: 故障诊断分析", run_diagnose),
        ("Step 3/3: 生成可视化报告", run_visualize),
    ]

    start_time = time.time()

    for i, (desc, func) in enumerate(steps):
        print(f"\n{'─' * 60}")
        print(f"  [{i+1}/{len(steps)}] {desc}")
        print(f"{'─' * 60}")
        try:
            func()
        except Exception as e:
            print(f"  ⚠ 步骤 {i+1} 执行异常: {e}")
            print("  跳过此步骤，继续执行...")

    elapsed = time.time() - start_time
    banner(f"全流程完成！总耗时: {elapsed:.1f} 秒")
    print("  输出文件:")
    print("    - docsdiagnosis_report.md  (诊断报告)")
    print("    - data*.png        (可视化图表)")
    print("    - data*.csv            (采集数据)")


# ===== 主入口 =====
USAGE = """
工业设备预测性维护系统 - 统一入口
═══════════════════════════════════════

用法: python srcrun_pipeline.py <模式> [参数]

可用模式:
  collect              PLC数据采集（默认60秒）
  collect <秒数>       PLC数据采集（指定时长）
  diagnose             基于PLC数据的故障诊断
  detect <wav> <dev>   单条音频异常检测
  visualize            生成MIMII可视化图表
  web                  启动Gradio Web界面
  full                 全流程（采集→诊断→可视化）

示例:
  python srcrun_pipeline.py collect 120
  python srcrun_pipeline.py detect data/mimii/fan/fan/test/xxx.wav fan
  python srcrun_pipeline.py web
  python srcrun_pipeline.py full
"""

def main():
    if len(sys.argv) < 2:
        print(USAGE)
        return

    mode = sys.argv[1].lower()

    if mode == "collect":
        duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        run_collect(duration)
    elif mode == "diagnose":
        run_diagnose()
    elif mode == "detect":
        if len(sys.argv) < 4:
            print("用法: python srcrun_pipeline.py detect <音频路径> <设备类型>")
            print("设备类型: fan / bearing / gearbox / slider / valve")
            return
        run_detect(sys.argv[2], sys.argv[3])
    elif mode == "visualize":
        run_visualize()
    elif mode == "web":
        run_web()
    elif mode == "full":
        run_full()
    else:
        print(f"未知模式: {mode}")
        print(USAGE)


if __name__ == "__main__":
    main()