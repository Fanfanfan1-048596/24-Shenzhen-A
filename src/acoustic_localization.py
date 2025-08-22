#!/usr/bin/env python3
import numpy as np
import pandas as pd
import math
from itertools import combinations
from pathlib import Path
from scipy.optimize import least_squares
import warnings
warnings.filterwarnings('ignore')

SOUND_SPEED = 340.0

class AcousticLocalizer:
    def __init__(self, devices_data):
        self.devices_data = devices_data
        self.results = []
    
    def chan_algorithm(self, selected_devices):
        try:
            positions = selected_devices[['X坐标(m)', 'Y坐标(m)', 'Z坐标(m)']].values
            times = selected_devices['音爆抵达时间(s)'].values
            
            N = len(positions)
            if N < 5:
                return None
            
            # A为参考设备
            ref_pos = positions[0]
            ref_time = times[0]
            
            # 构建线性方程组
            A = np.zeros((N-1, 4))
            b = np.zeros(N-1)
            
            for i in range(1, N):
                pos_i = positions[i]
                time_i = times[i]
                
                A[i-1, 0] = 2 * (pos_i[0] - ref_pos[0])  # x方向
                A[i-1, 1] = 2 * (pos_i[1] - ref_pos[1])  # y方向
                A[i-1, 2] = 2 * (pos_i[2] - ref_pos[2])  # z方向
                A[i-1, 3] = -2 * SOUND_SPEED**2 * (time_i - ref_time)  # ts项
                
                Ki = np.sum(pos_i**2)
                K1 = np.sum(ref_pos**2)
                b[i-1] = Ki - K1 - SOUND_SPEED**2 * (time_i**2 - ref_time**2)
            
            # 最小二乘求解
            if np.linalg.matrix_rank(A) < 4:
                return None
                
            u, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            x, y, z, ts = u
            if not np.isfinite([x, y, z, ts]).all():
                return None
                
            return x, y, z, ts
            
        except Exception as e:
            print(f"Chan算法计算错误: {e}")
            return None
    
    def objective_function(self, params, positions, times):
        x, y, z, ts = params
        residuals = []
        for i, (pos, t_measured) in enumerate(zip(positions, times)):
            distance = np.sqrt((x - pos[0])**2 + (y - pos[1])**2 + (z - pos[2])**2)
            t_theoretical = ts + distance / SOUND_SPEED

            residual = t_measured - t_theoretical
            residuals.append(residual)
        
        return np.array(residuals)
    
    def levenberg_marquardt_optimization(self, initial_solution, selected_devices):
        try:
            positions = selected_devices[['X坐标(m)', 'Y坐标(m)', 'Z坐标(m)']].values
            times = selected_devices['音爆抵达时间(s)'].values
            
            result = least_squares(
                self.objective_function,
                initial_solution,
                args=(positions, times),
                method='lm',
                max_nfev=1000
            )
            
            if result.success:
                x_opt, y_opt, z_opt, ts_opt = result.x

                final_residuals = self.objective_function(result.x, positions, times)
                rms_error = np.sqrt(np.mean(final_residuals**2))
                
                return {
                    'success': True,
                    'x': x_opt,
                    'y': y_opt, 
                    'z': z_opt,
                    'ts': ts_opt,
                    'rms_error': rms_error,
                    'residuals': final_residuals,
                    'cost': result.cost,
                    'nfev': result.nfev
                }
            else:
                return {'success': False, 'message': result.message}
                
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def solve_all_combinations(self):
        device_indices = list(range(len(self.devices_data)))
        combinations_5 = list(combinations(device_indices, 5))
        
        # print(f"共有= {len(combinations_5)} 种组合")
        results = []
        
        for i, combo in enumerate(combinations_5):           
            selected_devices = self.devices_data.iloc[list(combo)].copy()
            initial_solution = self.chan_algorithm(selected_devices)
            
            if initial_solution is None:
                print("  Chan算法失败，跳过")
                continue
            
            print(f"Chan算法初始解: x={initial_solution[0]:.2f}, y={initial_solution[1]:.2f}, z={initial_solution[2]:.2f}, ts={initial_solution[3]:.6f}")

            optimization_result = self.levenberg_marquardt_optimization(initial_solution, selected_devices)
            
            if optimization_result['success']:
                print(f"  LM优化成功: x={optimization_result['x']:.2f}, y={optimization_result['y']:.2f}, z={optimization_result['z']:.2f}, ts={optimization_result['ts']:.2f}")
                print(f"  RMS误差: {optimization_result['rms_error']:.6f} s")
                
                result_dict = {
                    'combination_id': i + 1,
                    'devices': [self.devices_data.iloc[j]['设备'] for j in combo],
                    'devices_indices': combo,
                    'initial_x': initial_solution[0],
                    'initial_y': initial_solution[1],
                    'initial_z': initial_solution[2],
                    'initial_ts': initial_solution[3],
                    'optimized_x': optimization_result['x'],
                    'optimized_y': optimization_result['y'],
                    'optimized_z': optimization_result['z'],
                    'optimized_ts': optimization_result['ts'],
                    'rms_error': optimization_result['rms_error'],
                    'cost': optimization_result['cost'],
                    'nfev': optimization_result['nfev']
                }
                
                results.append(result_dict)
            else:
                print(f"  LM优化失败: {optimization_result['message']}")
        
        return results
    
    def save_results(self, results, output_file):
        results_df = pd.DataFrame(results)
        results_df['devices_str'] = results_df['devices'].apply(lambda x: ','.join(x))
        
        save_columns = [
            'combination_id', 'devices_str', 
            'initial_x', 'initial_y', 'initial_z', 'initial_ts',
            'optimized_x', 'optimized_y', 'optimized_z', 'optimized_ts',
            'rms_error', 'cost', 'nfev'
        ]
        
        results_df[save_columns].to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n结果已保存到: {output_file}")
        
        print(f"成功求解的组合数: {len(results)}")
        print(f"最小RMS误差: {results_df['rms_error'].min():.6f} s")
        print(f"最大RMS误差: {results_df['rms_error'].max():.6f} s")
        print(f"平均RMS误差: {results_df['rms_error'].mean():.6f} s")
        
        best_idx = results_df['rms_error'].idxmin()
        best_result = results_df.iloc[best_idx]
        print(f"最优解 (组合{best_result['combination_id']}):")
        print(f"设备组合: {best_result['devices_str']}")
        print(f"坐标: ({best_result['optimized_x']:.6f}, {best_result['optimized_y']:.6f}, {best_result['optimized_z']:.6f})")
        print(f"发生时间: {best_result['optimized_ts']:.3f} s")
        print(f"RMS误差: {best_result['rms_error']:.3f} s")

def main():
    project_root = Path(__file__).resolve().parent.parent
    data_file = project_root / 'data' / '设备ENU坐标.csv'
    
    if not data_file.exists():
        print(f"文件不存在: {data_file}")
        return
    
    devices_data = pd.read_csv(data_file, encoding='utf-8')
    # print("坐标数据:")
    # print(devices_data)
    
    localizer = AcousticLocalizer(devices_data)    
    print("开始求解所有组合")
    results = localizer.solve_all_combinations()
    output_file = project_root / 'data' / '音爆定位结果.csv'
    localizer.save_results(results, output_file)

if __name__ == "__main__":
    main()
