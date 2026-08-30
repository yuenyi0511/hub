import opcua
import time
import csv
import os
from datetime import datetime

# ========== 配置区 ==========
OPC_UA_SERVER_URL = "opc.tcp://192.168.0.1:4840"
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
SAMPLE_INTERVAL = 1# 采集间隔（秒）

  
# ============================

os.makedirs(SAVE_DIR, exist_ok=True)

def connect_opcua():
    """连接到博图 OPC UA 服务器"""
    client = opcua.Client(OPC_UA_SERVER_URL)
    try:
        client.connect()
        print(f"✅ 已连接到 OPC UA 服务器: {OPC_UA_SERVER_URL}")
        return client
    except Exception as e:
        print(f" 连接失败: {e}")
        print("请检查：")
        print("  1. PLCSIM Advanced 是否已启动且为 RUN 状态")
        print("  2. 博图项目中是否已激活 OPC UA 服务器")
        print("  3. IP 地址和端口是否正确")
        return None

def get_node_safe(client, node_id):
    """安全获取节点"""
    try:
        node = client.get_node(node_id)
        node.get_value()
        return node
    except Exception as e:
        print(f"️  节点 {node_id} 无法访问: {e}")
        return None

def start_acquisition(client, duration=60):
    """开始数据采集 - 使用浏览到的实际节点路径"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(SAVE_DIR, f"plc_data_{timestamp}.csv")
    
    # 使用浏览到的实际节点 ID（ns=3，带完整DB名称）
    node_ids = {
        "vibration_x":     'ns=3;s="DB10_Vibration"."vibration_x"',
        "vibration_y":     'ns=3;s="DB10_Vibration"."vibration_y"',
        "vibration_z":     'ns=3;s="DB10_Vibration"."vibration_z"',
        "rms_value":       'ns=3;s="DB10_Vibration"."rms_value"',
        "bearing_temp":    'ns=3;s="DB20_Temperature"."bearing_temp"',
        "motor_temp":      'ns=3;s="DB20_Temperature"."motor_temp"',
        "ambient_temp":    'ns=3;s="DB20_Temperature"."ambient_temp"',
        "runtime_hours":   'ns=3;s="DB30_Status"."runtime_hours"',
        "rpm":             'ns=3;s="DB30_Status"."rpm"',
        "load_percent":    'ns=3;s="DB30_Status"."load_percent"',
        "degradation_level": 'ns=3;s="DB30_Status"."degradation_level"',
        "fault_type":      'ns=3;s="DB30_Status"."fault_type"',
    }
    
    # 先测试所有节点
    variables = {}
    for var_name, nid in node_ids.items():
        node = get_node_safe(client, nid)
        if node:
            variables[var_name] = node
            print(f"  ✅ {var_name} -> {nid}")
    
    if not variables:
        print("\n 未找到任何数据节点！")
        return
    
    print(f"\n✅ 成功找到 {len(variables)} 个变量节点")
    
    var_names = list(variables.keys())
    print(f"\n开始采集数据，持续 {duration} 秒...")
    print(f"数据保存到: {csv_path}")
    print(f"{'时间':>20} | {'振动X':>8} {'振动Y':>8} {'振动Z':>8} {'RMS':>8} | {'轴承温度':>8} {'电机温度':>8} {'环境温度':>8} | {'运行时间':>8} {'退化程度':>8}")
    print("-" * 130)
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp"] + var_names)
        
        start_time = time.time()
        while time.time() - start_time < duration:
            try:
                row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                values = []
                for var_name in var_names:
                    node = variables[var_name]
                    val = node.get_value()
                    values.append(round(val, 4) if isinstance(val, float) else val)
                    row.append(round(val, 4) if isinstance(val, float) else val)
                
                writer.writerow(row)
                f.flush()
                
                v = values
                print(f"{row[0]} | {v[0]:>8.2f} {v[1]:>8.2f} {v[2]:>8.2f} {v[3]:>8.2f} | {v[4]:>8.2f} {v[5]:>8.2f} {v[6]:>8.2f} | {v[7]:>8.2f} {v[8]:>8.4f}")
                
                time.sleep(SAMPLE_INTERVAL)
                
            except Exception as e:
                print(f"️  读取异常: {e}")
                time.sleep(1)
    
    print(f"\n✅ 采集完成！共保存到 {csv_path}")

def main():
    print("=" * 60)
    print("  预测性维护 - PLC 数据采集脚本")
    print("=" * 60)
    
    client = connect_opcua()
    if client is None:
        return
    
    try:
        start_acquisition(client, duration=60)
    finally:
        client.disconnect()
        print("已断开 OPC UA 连接")

if __name__ == "__main__":
    main()