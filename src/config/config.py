# ============================================================
# 预测性维护 - 全局配置
# ============================================================
import os

# ---------- 项目根目录 ----------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ---------- SQLite 配置（替代 MySQL）----------
SQLITE_DB_PATH = os.path.join(PROJECT_ROOT, "data", "pm_database.db")

# ---------- InfluxDB 配置 ----------
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "Gg3I6-noOQnBpS0xTsB7upZ9MExc2FSj3Go17Ccb7piXQeWHKvCyN5DX_qGcih0Sw4NrjyGyRIYpBaPzIxba6w=="
INFLUXDB_ORG = "predictive_maintenance"
INFLUXDB_BUCKET = "sensor_data"

# ---------- OPC UA 配置 ----------
OPCUA_SERVER_URL = "opc.tcp://192.168.0.1:4840"

# 振动数据节点
OPCUA_NODES_VIBRATION = {
    "vibration_x": "ns=2;s=\"DB10\".\"vibration_x\"",
    "vibration_y": "ns=2;s=\"DB10\".\"vibration_y\"",
    "vibration_z": "ns=2;s=\"DB10\".\"vibration_z\"",
    "rms_value": "ns=2;s=\"DB10\".\"rms_value\"",
}

# 温度数据节点
OPCUA_NODES_TEMP = {
    "bearing_temp": "ns=2;s=\"DB20\".\"bearing_temp\"",
    "motor_temp": "ns=2;s=\"DB20\".\"motor_temp\"",
    "ambient_temp": "ns=2;s=\"DB20\".\"ambient_temp\"",
}

# 设备状态节点
OPCUA_NODES_STATUS = {
    "runtime_hours": "ns=2;s=\"DB30\".\"runtime_hours\"",
    "rpm": "ns=2;s=\"DB30\".\"rpm\"",
    "load_percent": "ns=2;s=\"DB30\".\"load_percent\"",
    "degradation_level": "ns=2;s=\"DB30\".\"degradation_level\"",
    "fault_type": "ns=2;s=\"DB30\".\"fault_type\"",
}

# 报警写回节点
OPCUA_ALARM_NODE = "ns=2;s=\"DB30\".\"alarm_flag\""