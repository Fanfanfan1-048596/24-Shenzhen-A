#!/usr/bin/env python3
import numpy as np
import pandas as pd
import math
from pathlib import Path

# WGS84椭球参数
WGS84_A = 6378137.0
WGS84_F = 1/298.257223563
WGS84_E2 = 2*WGS84_F - WGS84_F**2

def deg_to_rad(degrees):
    return degrees * math.pi / 180.0

def geodetic_to_ecef(lon, lat, alt):
    lat_rad = deg_to_rad(lat)
    lon_rad = deg_to_rad(lon)
    
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat_rad)**2)
    
    x = (N + alt) * math.cos(lat_rad) * math.cos(lon_rad)
    y = (N + alt) * math.cos(lat_rad) * math.sin(lon_rad)
    z = (N * (1 - WGS84_E2) + alt) * math.sin(lat_rad)
    
    return x, y, z

def ecef_to_enu(x, y, z, ref_lon, ref_lat, ref_alt):
    ref_x, ref_y, ref_z = geodetic_to_ecef(ref_lon, ref_lat, ref_alt)

    dx = x - ref_x
    dy = y - ref_y
    dz = z - ref_z

    lat_rad = deg_to_rad(ref_lat)
    lon_rad = deg_to_rad(ref_lon)

    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)

    e = -sin_lon * dx + cos_lon * dy
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    
    return e, n, u

def geodetic_to_enu_direct(lon, lat, alt, ref_lon, ref_lat, ref_alt):
    x, y, z = geodetic_to_ecef(lon, lat, alt)
    return ecef_to_enu(x, y, z, ref_lon, ref_lat, ref_alt)

def main():
    devices_data = {
        '设备': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        '经度(°)': [110.241, 110.780, 110.712, 110.251, 110.524, 110.467, 110.047],
        '纬度(°)': [27.204, 27.456, 27.785, 27.825, 27.617, 27.921, 27.121],
        '高程(m)': [824, 727, 742, 850, 786, 678, 575],
        '音爆抵达时间(s)': [100.767, 112.220, 188.020, 258.985, 118.443, 266.871, 163.024]
    }

    df = pd.DataFrame(devices_data)

    ref_lon = df.loc[0, '经度(°)']
    ref_lat = df.loc[0, '纬度(°)']
    ref_alt = df.loc[0, '高程(m)']

    enu_coordinates = []
    for index, row in df.iterrows():
        lon = row['经度(°)']
        lat = row['纬度(°)']
        alt = row['高程(m)']
        device = row['设备']

        e, n, u = geodetic_to_enu_direct(lon, lat, alt, ref_lon, ref_lat, ref_alt)
        enu_coordinates.append({
            '设备': device,
            '经度(°)': lon,
            '纬度(°)': lat,
            '高程(m)': alt,
            '音爆抵达时间(s)': row['音爆抵达时间(s)'],
            'X坐标(m)': round(e, 3), 
            'Y坐标(m)': round(n, 3),  
            'Z坐标(m)': round(u, 3)   
        })

    result_df = pd.DataFrame(enu_coordinates)
    # output_file = '/home/fanfanfan/Desktop/25数模国赛/培训作业/深圳杯A题/设备ENU坐标.csv'
    project_root = Path(__file__).resolve().parent.parent
    output_file = project_root / 'data' / '设备ENU坐标.csv'
    result_df.to_csv(output_file, index=False, encoding='utf-8')
    
    return result_df

if __name__ == "__main__":
    result = main()
