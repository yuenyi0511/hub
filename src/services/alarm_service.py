"""
报警服务 - 将故障诊断结果写入 SQLite 报警记录表
"""
import os
import sys
import sqlite3
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.config import SQLITE_DB_PATH


class AlarmService:
    def __init__(self):
        self.conn = None

    def connect(self):
        self.conn = sqlite3.connect(SQLITE_DB_PATH)
        self.conn.row_factory = sqlite3.Row  # 让结果可以用列名访问

    def close(self):
        if self.conn:
            self.conn.close()

    def record_alarm(self, equipment_code, fault_type, fault_description,
                     health_score, rul_hours, vibration_rms,
                     bearing_temp, motor_temp, alarm_level=2):
        """记录一条报警"""
        if not self.conn:
            self.connect()

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO alarm_records 
            (equipment_code, alarm_time, alarm_level, fault_type, fault_description,
             health_score, rul_hours, vibration_rms, bearing_temp, motor_temp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            equipment_code, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            alarm_level, fault_type, fault_description,
            health_score, rul_hours, vibration_rms, bearing_temp, motor_temp
        ))
        self.conn.commit()
        cursor.close()
        print(f"  📢 报警已记录: [{fault_type}] 健康度={health_score:.1f}")

    def get_recent_alarms(self, equipment_code="PUMP-001", limit=20):
        """查询最近的报警记录"""
        if not self.conn:
            self.connect()

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM alarm_records 
            WHERE equipment_code = ? 
            ORDER BY alarm_time DESC 
            LIMIT ?
        """, (equipment_code, limit))
        records = [dict(row) for row in cursor.fetchall()]
        cursor.close()
        return records

    def acknowledge_alarm(self, alarm_id, acknowledged_by="系统"):
        """确认报警"""
        if not self.conn:
            self.connect()

        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE alarm_records 
            SET is_acknowledged = 1, acknowledged_by = ?, acknowledged_at = ?
            WHERE id = ?
        """, (acknowledged_by, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), alarm_id))
        self.conn.commit()
        cursor.close()


# 快捷使用
if __name__ == "__main__":
    service = AlarmService()
    service.connect()

    alarms = service.get_recent_alarms()
    print(f"最近 {len(alarms)} 条报警记录:")
    for a in alarms:
        print(f"  [{a['alarm_time']}] {a['fault_type']} - 健康度:{a['health_score']}")

    service.close()
    