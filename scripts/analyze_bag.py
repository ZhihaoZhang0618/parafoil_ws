#!/usr/bin/env python3
"""
分析翼伞仿真 bag 包，检查动力学是否正常。

检查项目:
1. 位置变化: NED坐标是否合理变化
2. 速度: 是否在合理范围内
3. 姿态: 是否稳定，响应控制输入
4. 下降率: 是否符合预期
5. 转弯响应: 非对称输入是否产生转弯
"""

import sys
import os
import numpy as np
from pathlib import Path

# 尝试导入 rosbag2
try:
    from rosbags.rosbag2 import Reader
    from rosbags.serde import deserialize_cdr
    HAS_ROSBAGS = True
except ImportError:
    HAS_ROSBAGS = False
    print("警告: rosbags 未安装，尝试使用 ros2 bag play + 订阅方式")


def euler_from_quaternion(q):
    """从四元数 [w,x,y,z] 计算欧拉角 [roll, pitch, yaw]"""
    w, x, y, z = q
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


def analyze_with_rosbags(bag_path):
    """使用 rosbags 库分析 bag 文件"""
    print(f"\n{'='*60}")
    print(f"分析 Bag: {bag_path}")
    print('='*60)
    
    # 存储数据
    times = []
    positions = []  # NED
    velocities = []  # body frame
    quaternions = []  # [w,x,y,z]
    angular_velocities = []
    
    odom_times = []
    odom_positions = []  # ENU (from odom)
    
    with Reader(bag_path) as reader:
        # 获取连接信息
        print("\n可用话题:")
        for conn in reader.connections:
            print(f"  {conn.topic}: {conn.msgtype}")
        
        # 读取消息
        for conn, timestamp, rawdata in reader.messages():
            t_sec = timestamp / 1e9
            
            if conn.topic == '/parafoil/state':
                # 自定义消息，需要解析
                msg = deserialize_cdr(rawdata, conn.msgtype)
                times.append(t_sec)
                positions.append([msg.position[0], msg.position[1], msg.position[2]])
                velocities.append([msg.velocity[0], msg.velocity[1], msg.velocity[2]])
                quaternions.append([msg.quaternion[0], msg.quaternion[1], 
                                   msg.quaternion[2], msg.quaternion[3]])
                angular_velocities.append([msg.omega[0], msg.omega[1], msg.omega[2]])
            
            elif conn.topic == '/parafoil/odom':
                msg = deserialize_cdr(rawdata, conn.msgtype)
                odom_times.append(t_sec)
                pos = msg.pose.pose.position
                odom_positions.append([pos.x, pos.y, pos.z])
    
    # 转换为 numpy 数组
    times = np.array(times)
    positions = np.array(positions)
    velocities = np.array(velocities)
    quaternions = np.array(quaternions)
    angular_velocities = np.array(angular_velocities)
    
    if len(times) == 0:
        print("\n错误: 没有读取到 /parafoil/state 消息!")
        
        if len(odom_times) > 0:
            print(f"\n但读取到 {len(odom_times)} 条 /parafoil/odom 消息")
            odom_positions = np.array(odom_positions)
            analyze_odom_data(np.array(odom_times), odom_positions)
        return
    
    print(f"\n读取到 {len(times)} 条状态消息")
    print(f"时间跨度: {times[-1] - times[0]:.2f} 秒")
    print(f"平均频率: {len(times) / (times[-1] - times[0]):.1f} Hz")
    
    # 分析位置
    analyze_position(times, positions)
    
    # 分析速度
    analyze_velocity(times, velocities)
    
    # 分析姿态
    analyze_attitude(times, quaternions)
    
    # 分析角速度
    analyze_angular_velocity(times, angular_velocities)
    
    # 综合诊断
    diagnose_dynamics(times, positions, velocities, quaternions, angular_velocities)


def analyze_odom_data(times, positions):
    """分析 odom 数据 (ENU 坐标)"""
    print("\n" + "="*60)
    print("Odom 数据分析 (ENU 坐标)")
    print("="*60)
    
    print(f"\n初始位置 (ENU): x={positions[0,0]:.2f}, y={positions[0,1]:.2f}, z={positions[0,2]:.2f}")
    print(f"最终位置 (ENU): x={positions[-1,0]:.2f}, y={positions[-1,1]:.2f}, z={positions[-1,2]:.2f}")
    
    # 位置变化
    delta = positions[-1] - positions[0]
    print(f"\n位置变化:")
    print(f"  Δx (East):  {delta[0]:+.2f} m")
    print(f"  Δy (North): {delta[1]:+.2f} m")
    print(f"  Δz (Up):    {delta[2]:+.2f} m")
    
    # 水平距离
    horizontal_dist = np.sqrt(delta[0]**2 + delta[1]**2)
    print(f"\n水平移动距离: {horizontal_dist:.2f} m")
    print(f"垂直下降 (应为负): {delta[2]:.2f} m")
    
    # 计算速度
    dt = np.diff(times)
    vel = np.diff(positions, axis=0) / dt[:, np.newaxis]
    
    print(f"\n速度统计:")
    print(f"  Vx (East):  min={vel[:,0].min():.2f}, max={vel[:,0].max():.2f}, mean={vel[:,0].mean():.2f} m/s")
    print(f"  Vy (North): min={vel[:,1].min():.2f}, max={vel[:,1].max():.2f}, mean={vel[:,1].mean():.2f} m/s")
    print(f"  Vz (Up):    min={vel[:,2].min():.2f}, max={vel[:,2].max():.2f}, mean={vel[:,2].mean():.2f} m/s")
    
    horizontal_speed = np.sqrt(vel[:,0]**2 + vel[:,1]**2)
    print(f"\n水平速度: min={horizontal_speed.min():.2f}, max={horizontal_speed.max():.2f}, mean={horizontal_speed.mean():.2f} m/s")
    
    # 下沉率
    sink_rate = -vel[:,2]  # 正值表示下沉
    print(f"下沉率: min={sink_rate.min():.2f}, max={sink_rate.max():.2f}, mean={sink_rate.mean():.2f} m/s")
    
    # 滑翔比
    if sink_rate.mean() > 0.1:
        glide_ratio = horizontal_speed.mean() / sink_rate.mean()
        print(f"平均滑翔比: {glide_ratio:.2f}")


def analyze_position(times, positions):
    """分析位置数据"""
    print("\n" + "-"*40)
    print("位置分析 (NED 坐标)")
    print("-"*40)
    
    print(f"初始位置: N={positions[0,0]:.2f}, E={positions[0,1]:.2f}, D={positions[0,2]:.2f}")
    print(f"最终位置: N={positions[-1,0]:.2f}, E={positions[-1,1]:.2f}, D={positions[-1,2]:.2f}")
    
    delta = positions[-1] - positions[0]
    print(f"\n位置变化:")
    print(f"  ΔN (北向): {delta[0]:+.2f} m")
    print(f"  ΔE (东向): {delta[1]:+.2f} m")
    print(f"  ΔD (向下): {delta[2]:+.2f} m")
    
    # 水平距离
    horizontal_dist = np.sqrt(delta[0]**2 + delta[1]**2)
    duration = times[-1] - times[0]
    
    print(f"\n水平移动: {horizontal_dist:.2f} m")
    print(f"垂直下降: {delta[2]:.2f} m (正值=下降)")
    print(f"平均水平速度: {horizontal_dist/duration:.2f} m/s")
    print(f"平均下沉率: {delta[2]/duration:.2f} m/s")
    
    if delta[2] > 0:
        glide_ratio = horizontal_dist / delta[2]
        print(f"滑翔比: {glide_ratio:.2f}")
    
    # 检查异常
    if abs(delta[2]) < 1.0:
        print("\n⚠️ 警告: 几乎没有垂直运动!")
    if horizontal_dist < 10.0:
        print("\n⚠️ 警告: 水平移动很小!")


def analyze_velocity(times, velocities):
    """分析速度数据"""
    print("\n" + "-"*40)
    print("速度分析 (体轴系)")
    print("-"*40)
    
    # 计算速度大小
    speed = np.linalg.norm(velocities, axis=1)
    
    print(f"速度大小: min={speed.min():.2f}, max={speed.max():.2f}, mean={speed.mean():.2f} m/s")
    print(f"u (前向): min={velocities[:,0].min():.2f}, max={velocities[:,0].max():.2f}, mean={velocities[:,0].mean():.2f} m/s")
    print(f"v (侧向): min={velocities[:,1].min():.2f}, max={velocities[:,1].max():.2f}, mean={velocities[:,1].mean():.2f} m/s")
    print(f"w (垂向): min={velocities[:,2].min():.2f}, max={velocities[:,2].max():.2f}, mean={velocities[:,2].mean():.2f} m/s")
    
    # 翼伞正常飞行 u 应该是主要分量
    if velocities[:,0].mean() < 5.0:
        print("\n⚠️ 警告: 前向速度过低!")
    if abs(velocities[:,1]).max() > 10.0:
        print("\n⚠️ 警告: 侧向速度过大!")


def analyze_attitude(times, quaternions):
    """分析姿态数据"""
    print("\n" + "-"*40)
    print("姿态分析")
    print("-"*40)
    
    # 计算欧拉角
    eulers = np.array([euler_from_quaternion(q) for q in quaternions])
    roll = np.degrees(eulers[:, 0])
    pitch = np.degrees(eulers[:, 1])
    yaw = np.degrees(eulers[:, 2])
    
    print(f"横滚角: min={roll.min():.2f}°, max={roll.max():.2f}°, mean={roll.mean():.2f}°")
    print(f"俯仰角: min={pitch.min():.2f}°, max={pitch.max():.2f}°, mean={pitch.mean():.2f}°")
    print(f"偏航角: 初始={yaw[0]:.2f}°, 最终={yaw[-1]:.2f}°, 变化={yaw[-1]-yaw[0]:.2f}°")
    
    # 翼伞正常飞行，俯仰角应该在 -30° 到 30° 之间
    if abs(pitch).max() > 45:
        print("\n⚠️ 警告: 俯仰角过大!")
    if abs(roll).max() > 60:
        print("\n⚠️ 警告: 横滚角过大!")


def analyze_angular_velocity(times, angular_velocities):
    """分析角速度数据"""
    print("\n" + "-"*40)
    print("角速度分析 (体轴系)")
    print("-"*40)
    
    p = np.degrees(angular_velocities[:, 0])
    q = np.degrees(angular_velocities[:, 1])
    r = np.degrees(angular_velocities[:, 2])
    
    print(f"p (滚转率): min={p.min():.2f}°/s, max={p.max():.2f}°/s, mean={p.mean():.2f}°/s")
    print(f"q (俯仰率): min={q.min():.2f}°/s, max={q.max():.2f}°/s, mean={q.mean():.2f}°/s")
    print(f"r (偏航率): min={r.min():.2f}°/s, max={r.max():.2f}°/s, mean={r.mean():.2f}°/s")


def diagnose_dynamics(times, positions, velocities, quaternions, angular_velocities):
    """综合诊断动力学问题"""
    print("\n" + "="*60)
    print("动力学诊断")
    print("="*60)
    
    issues = []
    
    # 检查1: 位置是否变化
    pos_change = np.linalg.norm(positions[-1] - positions[0])
    if pos_change < 10:
        issues.append("位置几乎没有变化")
    
    # 检查2: 下降
    d_change = positions[-1, 2] - positions[0, 2]
    if d_change < 0:
        issues.append(f"高度增加了 {-d_change:.2f}m (NED坐标中D应增加表示下降)")
    
    # 检查3: 速度
    speed = np.linalg.norm(velocities, axis=1)
    if speed.mean() < 3:
        issues.append(f"平均速度过低: {speed.mean():.2f} m/s")
    if speed.max() > 50:
        issues.append(f"最大速度过高: {speed.max():.2f} m/s")
    
    # 检查4: 四元数归一化
    quat_norms = np.linalg.norm(quaternions, axis=1)
    if np.abs(quat_norms - 1.0).max() > 0.01:
        issues.append("四元数未归一化")
    
    # 检查5: NaN 或 Inf
    if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
        issues.append("位置包含 NaN 或 Inf!")
    if np.any(np.isnan(velocities)) or np.any(np.isinf(velocities)):
        issues.append("速度包含 NaN 或 Inf!")
    
    # 检查6: 控制响应
    # 根据时间段分析偏航变化
    duration = times[-1] - times[0]
    eulers = np.array([euler_from_quaternion(q) for q in quaternions])
    yaw = eulers[:, 2]
    
    # 偏航角变化率
    yaw_rate = np.abs(np.diff(np.unwrap(yaw)) / np.diff(times))
    
    if yaw_rate.max() < 0.01:
        issues.append("偏航几乎没有变化，可能控制无响应")
    
    # 输出诊断结果
    if len(issues) == 0:
        print("\n✅ 未发现明显问题，动力学看起来正常!")
    else:
        print("\n❌ 发现以下问题:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    
    # 建议
    print("\n建议检查:")
    print("  1. 初始速度设置 (V_init)")
    print("  2. 气动系数 (c_L0, c_La, c_D0 等)")
    print("  3. 积分器设置 (dt, dt_max)")
    print("  4. 控制命令是否正确发布")


def main():
    if len(sys.argv) < 2:
        print("用法: python3 analyze_bag.py <bag_path>")
        print("示例: python3 analyze_bag.py test_bags/dynamics_test_xxx")
        sys.exit(1)
    
    bag_path = sys.argv[1]
    
    if not os.path.exists(bag_path):
        print(f"错误: Bag 路径不存在: {bag_path}")
        sys.exit(1)
    
    if HAS_ROSBAGS:
        analyze_with_rosbags(bag_path)
    else:
        print("请安装 rosbags: pip install rosbags")
        print("或者使用 ROS2 工具查看 bag:")
        print(f"  ros2 bag info {bag_path}")
        print(f"  ros2 bag play {bag_path}")
        sys.exit(1)


if __name__ == '__main__':
    main()
