"""
数据库初始化脚本
- 创建 InfluxDB Bucket
- 创建 SQLite 数据库和表
运行一次即可：python src/utils/db_init.py
"""
import os
import sys
import sqlite3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.config import SQLITE_DB_PATH, INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG, INFLUXDB_BUCKET


def init_influxdb():
    """初始化 InfluxDB（检查 Bucket 是否存在）"""
    print("=" * 50)
    print("【InfluxDB 初始化】")
    print("=" * 50)

    try:
        from influxdb_client import InfluxDBClient
        client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
        buckets_api = client.buckets_api()
        existing = buckets_api.find_bucket_by_name(INFLUXDB_BUCKET)

        if existing:
            print(f"  ✓ Bucket '{INFLUXDB_BUCKET}' 已存在，跳过创建")
        else:
            buckets_api.create_bucket(bucket_name=INFLUXDB_BUCKET, org_id=INFLUXDB_ORG)
            print(f"  ✓ Bucket '{INFLUXDB_BUCKET}' 创建成功")

        client.close()
    except Exception as e:
        print(f"  ⚠ InfluxDB 连接失败: {e}")
        print(f"    数据将仅保存到CSV，不写入InfluxDB")
        return False

    return True


def init_sqlite():
    """初始化 SQLite（创建数据库和表）"""
    print("\n" + "=" * 50)
    print("【SQLite 初始化】")
    print("=" * 50)

    # 确保 data 目录存在
    os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)

    print(f"  数据库路径: {SQLITE_DB_PATH}")

    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()

    # 表1：设备台账
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS equipment (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            equipment_code TEXT NOT NULL UNIQUE,
            equipment_name TEXT NOT NULL,
            equipment_type TEXT,
            install_date TEXT,
            location TEXT,
            rated_rpm INTEGER,
            rated_power REAL,
            status INTEGER DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now', 'localtime')),
            updated_at TEXT DEFAULT (datetime('now', 'localtime'))
        )
    """)
    print("  ✓ 表 equipment（设备台账）就绪")

    # 表2：报警记录
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alarm_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            equipment_code TEXT NOT NULL,
            alarm_time TEXT NOT NULL,
            alarm_level INTEGER NOT NULL,
            fault_type TEXT NOT NULL,
            fault_description TEXT,
            health_score REAL,
            rul_hours REAL,
            vibration_rms REAL,
            bearing_temp REAL,
            motor_temp REAL,
            is_acknowledged INTEGER DEFAULT 0,
            acknowledged_by TEXT,
            acknowledged_at TEXT,
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_alarm_equipment_time 
        ON alarm_records(equipment_code, alarm_time)
    """)
    print("  ✓ 表 alarm_records（报警记录）就绪")

    # 表3：维护记录
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS maintenance_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            equipment_code TEXT NOT NULL,
            maintenance_type TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT,
            fault_found TEXT,
            actions_taken TEXT,
            parts_replaced TEXT,
            cost REAL DEFAULT 0,
            technician TEXT,
            result TEXT,
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        )
    """)
    print("  ✓ 表 maintenance_records（维护记录）就绪")

    # 表4：模型推理日志
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inference_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            equipment_code TEXT NOT NULL,
            inference_time TEXT NOT NULL,
            model_name TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL,
            health_score REAL,
            rul_hours REAL,
            is_anomaly INTEGER,
            latency_ms REAL,
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        )
    """)
    print("  ✓ 表 inference_log（模型推理日志）就绪")

    # 插入默认设备
    cursor.execute("""
        INSERT OR IGNORE INTO equipment 
        (equipment_code, equipment_name, equipment_type, install_date, location, rated_rpm, rated_power)
        VALUES ('PUMP-001', '1号离心泵', '离心泵', '2026-01-15', '车间A-3号工位', 1800, 15.0)
    """)
    if cursor.rowcount > 0:
        print("  ✓ 默认设备 PUMP-001 已录入")
    else:
        print("  - 默认设备 PUMP-001 已存在，跳过")

    conn.commit()
    conn.close()
    return True


def main():
    print("\n" + "=" * 50)
    print("  预测性维护 - 数据库初始化")
    print("=" * 50 + "\n")

    ok1 = init_influxdb()
    ok2 = init_sqlite()

    print("\n" + "=" * 50)
    if ok2:
        print("  ✅ SQLite 数据库初始化完成！")
        print(f"  📁 数据库文件: {SQLITE_DB_PATH}")
    else:
        print("  ⚠️ SQLite 初始化失败")

    if not ok1:
        print("  ℹ️ InfluxDB 未连接，采集数据仅保存CSV")
    print("=" * 50)


if __name__ == "__main__":
    main()