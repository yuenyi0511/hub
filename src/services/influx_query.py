"""
InfluxDB 查询封装 - 供特征工程和可视化模块使用
"""
import os
import sys
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.config import INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG, INFLUXDB_BUCKET


class InfluxQuery:
    def __init__(self):
        self.client = InfluxDBClient(
            url=INFLUXDB_URL,
            token=INFLUXDB_TOKEN,
            org=INFLUXDB_ORG
        )

    def query_latest(self, minutes=30, equipment_code="PUMP-001"):
        """查询最近N分钟的数据"""
        query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")

          |> range(start: -{minutes}m)
          |> filter(fn: (r) => r["_measurement"] == "sensor_data")
          |> filter(fn: (r) => r["equipment_code"] == "{equipment_code}")
          |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> sort(columns: ["_time"])
        '''
        result = self.client.query_api().query_data_frame(query)
        return result

    def query_time_range(self, start_time, end_time, equipment_code="PUMP-001"):
        """查询指定时间范围的数据"""
        start_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")

          |> range(start: {start_str}, stop: {end_str})
          |> filter(fn: (r) => r["_measurement"] == "sensor_data")
          |> filter(fn: (r) => r["equipment_code"] == "{equipment_code}")
          |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> sort(columns: ["_time"])
        '''
        result = self.client.query_api().query_data_frame(query)
        return result

    def query_field_stats(self, field, hours=1, equipment_code="PUMP-001"):
        """查询某字段最近N小时的统计信息"""
        query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")

          |> range(start: -{hours}h)
          |> filter(fn: (r) => r["_measurement"] == "sensor_data")
          |> filter(fn: (r) => r["equipment_code"] == "{equipment_code}")
          |> filter(fn: (r) => r["_field"] == "{field}")
          |> mean()
        '''
        result = self.client.query_api().query_data_frame(query)
        return result

    def close(self):
        self.client.close()


if __name__ == "__main__":
    q = InfluxQuery()

    print("最近 30 分钟数据:")
    df = q.query_latest(minutes=30)
    if df is not None and len(df) > 0:
        print(f"  共 {len(df)} 条记录")
        print(df.tail())
    else:
        print("  暂无数据")

    q.close()