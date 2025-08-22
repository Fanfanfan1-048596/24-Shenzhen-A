#!/usr/bin/env python3
import numpy as np
import pandas as pd
import math
from pathlib import Path

WGS84_A = 6378137.0
WGS84_F = 1/298.257223563
WGS84_E2 = 2*WGS84_F - WGS84_F**2

def deg_to_rad(degrees):
    return degrees * math.pi / 180.0

def rad_to_deg(radians):
    return radians * 180.0 / math.pi

def geodetic_to_ecef(lon, lat, alt):
    lat_rad = deg_to_rad(lat)
    lon_rad = deg_to_rad(lon)
    
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat_rad)**2)
    
    x = (N + alt) * math.cos(lat_rad) * math.cos(lon_rad)
    y = (N + alt) * math.cos(lat_rad) * math.sin(lon_rad)
    z = (N * (1 - WGS84_E2) + alt) * math.sin(lat_rad)
    
    return x, y, z

def enu_to_ecef(e, n, u, ref_lon, ref_lat, ref_alt):
    ref_x, ref_y, ref_z = geodetic_to_ecef(ref_lon, ref_lat, ref_alt)
    
    lat_rad = deg_to_rad(ref_lat)
    lon_rad = deg_to_rad(ref_lon)
    
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)
    
    dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
    dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
    dz = cos_lat * n + sin_lat * u
    
    x = ref_x + dx
    y = ref_y + dy
    z = ref_z + dz
    
    return x, y, z

def ecef_to_geodetic(x, y, z):
    lon_rad = math.atan2(y, x)
    
    p = math.sqrt(x**2 + y**2)
    lat_rad = math.atan2(z, p * (1 - WGS84_E2))
    
    for _ in range(10):
        N = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat_rad)**2)
        alt = p / math.cos(lat_rad) - N
        lat_rad = math.atan2(z, p * (1 - WGS84_E2 * N / (N + alt)))
    
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat_rad)**2)
    alt = p / math.cos(lat_rad) - N
    
    lon = rad_to_deg(lon_rad)
    lat = rad_to_deg(lat_rad)
    
    return lon, lat, alt

def enu_to_geodetic_direct(e, n, u, ref_lon, ref_lat, ref_alt):
    x, y, z = enu_to_ecef(e, n, u, ref_lon, ref_lat, ref_alt)
    return ecef_to_geodetic(x, y, z)

def main():
    optimal_enu = {
        'E(m)': 25768.558787,
        'N(m)': 11174.229306,
        'U(m)': 32.258643
    }
    
    ref_lon = 110.241
    ref_lat = 27.204
    ref_alt = 824
    
    lon, lat, alt = enu_to_geodetic_direct(
        optimal_enu['E(m)'], 
        optimal_enu['N(m)'], 
        optimal_enu['U(m)'],
        ref_lon, ref_lat, ref_alt
    )
    
    print(f"转换后的地理坐标：")
    print(f"  经度: {lon:.3f}°")
    print(f"  纬度: {lat:.3f}°")
    print(f"  高程: {alt:.0f} m")
    print()
    
    return lon, lat, alt

if __name__ == "__main__":
    lon, lat, alt = main()
