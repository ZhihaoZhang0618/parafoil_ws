#!/usr/bin/env python3
"""
分析翼伞仿真 bag 包，检查动力学是否正常。
使用 rosbags 库读取 ROS2 bag 文件。
"""

import sys
import os
import numpy as np
from pathlib import Path

try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
except ImportError:
    print("错误: 请安装 rosbags: pip install rosbags")
    sys.exit(1)

# 获取 ROS2 Humble 类型存储
typestore = get_typestore(Stores.ROS2_HUMBLE)


def euler_from_quaternion(w, x, y, z):
    """从四元数 [w,x,y,z] 计算欧拉角 [roll, pitch, yaw]"""
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


def analyze_bag(bag_path):
    """分析 bag 文件"""
    print(f"\n{'='*60}")
    print(f"分析 Bag: {bag_path}")
    print('='*60)
    
    # 存储数据
    odom_times = []
    positions = []  # ENU
    orientations = []  # [w, x, y, z]
    linear_velocities = []
    angular_velocities = []
    
    cmd_times = []
    commands = []  # [delta_l, delta_r]
    
    with Reader(bag_path) as reader:
        # 获取连接信息
        print("\n可用话题:")
        for conn in reader.connections:
            print(f"  {conn.topic}: {conn.msgtype}")
        
        # 读取消息
        for conn, timestamp, rawdata in reader.messages():
            t_sec = timestamp / 1e9
            
            if conn.topic == '/parafoil/odom':
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                odom_times.append(t_sec)
                
                pos = msg.pose.pose.position
                positions.append([pos.x, pos.y, pos.z])
                
                ori = msg.pose.pose.orientation
                orientations.append([ori.w, ori.x, ori.y, ori.z])
                
                lin = msg.twist.twist.linear
                linear_velocities.append([lin.x, lin.y, lin.z])
                
                ang = msg.twist.twist.angular
                angular_velocities.append([ang.x, ang.y, ang.z])
            
            elif conn.topic == '/rockpara_actuators_node/auto_commands':
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                cmd_times.append(t_sec)
                commands.append([msg.vector.x, msg.vector.y])
    
    # 转换为 numpy 数组
    odom_times = np.array(odom_times)
    positions = np.array(positions)
    orientations = np.array(orientations)
    linear_velocities = np.array(linear_velocities)
    angular_velocities = np.array(angular_velocities)
    
    if len(odom_times) == 0:
        print("\n错误: 没有读取到 /parafoil/odom 消息!")
        return
    
    # 相对时间
    t0 = odom_times[0]
    odom_times = odom_times - t0
    cmd_times = np.array(cmd_times) - t0 if len(cmd_times) > 0 else np.array([])
    commands = np.array(commands) if len(commands) > 0 else np.array([])
    
    print(f"\n读取到 {len(odom_times)} 条 odom 消息")
    print(f"读取到 {len(cmd_times)} 条控制命令")
    print(f"时间跨度: {odom_times[-1]:.2f} 秒")
    print(f"平均频率: {len(odom_times) / odom_times[-1]:.1f} Hz")
    
    # 分析位置
    analyze_position(odom_times, positions)
    
    # 分析速度
    analyze_velocity(odom_times, linear_velocities)
    
    # 分析姿态
    analyze_attitude(odom_times, orientations)
    
    # 分析角速度
    analyze_angular_velocity(odom_times, angular_velocities)
    
    # 分析控制响应
    if len(cmd_times) > 0:
        analyze_control_response(odom_times, orientations, cmd_times, commands)
    
    # 综合诊断
    diagnose_dynamics(odom_times, positions, linear_velocities, orientations, angular_velocities)


def analyze_position(times, positions):
    """分析位置数据 (ENU 坐标)"""
    print("\n" + "-"*40)
    print("位置分析 (ENU 坐标)")
    print("-"*40)
    
    print(f"初始位置: E={positions[0,0]:.2f}, N={positions[0,1]:.2f}, U={positions[0,2]:.2f}")
    print(f"最终位置: E={positions[-1,0]:.2f}, N={positions[-1,1]:.2f}, U={positions[-1,2]:.2f}")
    
    delta = positions[-1] - positions[0]
    print(f"\n位置变化:")
    print(f"  ΔE (东向): {delta[0]:+.2f} m")
    print(f"  ΔN (北向): {delta[1]:+.2f} m")
    print(f"  ΔU (高度): {delta[2]:+.2f} m")
    
    # 水平距离
    horizontal_dist = np.sqrt(delta[0]**2 + delta[1]**2)
    duration = times[-1]
    
    print(f"\n水平移动: {horizontal_dist:.2f} m")
    print(f"高度变化: {delta[2]:.2f} m (负值=下降)")
    print(f"平均水平速度: {horizontal_dist/duration:.2f} m/s")
    print(f"平均下沉率: {-delta[2]/duration:.2f} m/s")
    
    if delta[2] < 0:
        glide_ratio = horizontal_dist / (-delta[2])
        print(f"滑翔比: {glide_ratio:.2f}")


def analyze_velocity(times, velocities):
    """分析速度数据 (ENU)"""
    print("\n" + "-"*40)
    print("速度分析 (ENU)")
    print("-"*40)
    
    # 计算速度大小
    speed = np.linalg.norm(velocities, axis=1)
    horizontal_speed = np.sqrt(velocities[:,0]**2 + velocities[:,1]**2)
    
    print(f"总速度: min={speed.min():.2f}, max={speed.max():.2f}, mean={speed.mean():.2f} m/s")
    print(f"水平速度: min={horizontal_speed.min():.2f}, max={horizontal_speed.max():.2f}, mean={horizontal_speed.mean():.2f} m/s")
    print(f"垂直速度 (Vz): min={velocities[:,2].min():.2f}, max={velocities[:,2].max():.2f}, mean={velocities[:,2].mean():.2f} m/s")
    
    # 下沉率
    sink_rate = -velocities[:,2]
    print(f"下沉率: min={sink_rate.min():.2f}, max={sink_rate.max():.2f}, mean={sink_rate.mean():.2f} m/s")


def analyze_attitude(times, orientations):
    """分析姿态数据"""
    print("\n" + "-"*40)
    print("姿态分析")
    print("-"*40)
    
    # 计算欧拉角
    eulers = np.array([euler_from_quaternion(q[0], q[1], q[2], q[3]) for q in orientations])
    roll = np.degrees(eulers[:, 0])
    pitch = np.degrees(eulers[:, 1])
    yaw = np.degrees(eulers[:, 2])
    
    print(f"横滚角 (Roll):  min={roll.min():.2f}°, max={roll.max():.2f}°, mean={roll.mean():.2f}°")
    print(f"俯仰角 (Pitch): min={pitch.min():.2f}°, max={pitch.max():.2f}°, mean={pitch.mean():.2f}°")
    print(f"偏航角 (Yaw):   初始={yaw[0]:.2f}°, 最终={yaw[-1]:.2f}°")
    
    # 计算偏航角总变化 (考虑绕圈)
    yaw_rad = eulers[:, 2]
    yaw_unwrapped = np.unwrap(yaw_rad)
    total_yaw_change = np.degrees(yaw_unwrapped[-1] - yaw_unwrapped[0])
    print(f"偏航角总变化: {total_yaw_change:.2f}°")


def analyze_angular_velocity(times, angular_velocities):
    """分析角速度数据"""
    print("\n" + "-"*40)
    print("角速度分析")
    print("-"*40)
    
    p = np.degrees(angular_velocities[:, 0])
    q = np.degrees(angular_velocities[:, 1])
    r = np.degrees(angular_velocities[:, 2])
    
    print(f"p (滚转率): min={p.min():.2f}°/s, max={p.max():.2f}°/s, mean={p.mean():.2f}°/s")
    print(f"q (俯仰率): min={q.min():.2f}°/s, max={q.max():.2f}°/s, mean={q.mean():.2f}°/s")
    print(f"r (偏航率): min={r.min():.2f}°/s, max={r.max():.2f}°/s, mean={r.mean():.2f}°/s")


def analyze_control_response(odom_times, orientations, cmd_times, commands):
    """分析控制响应"""
    print("\n" + "-"*40)
    print("控制响应分析")
    print("-"*40)
    
    print(f"接收到的控制命令:")
    for i, (t, cmd) in enumerate(zip(cmd_times, commands)):
        print(f"  t={t:.2f}s: delta_L={cmd[0]:.2f}, delta_R={cmd[1]:.2f}")
    
    # 计算偏航角
    eulers = np.array([euler_from_quaternion(q[0], q[1], q[2], q[3]) for q in orientations])
    yaw_unwrapped = np.degrees(np.unwrap(eulers[:, 2]))
    
    # 分析每个控制命令后的偏航变化
    for i, (t_cmd, cmd) in enumerate(zip(cmd_times, commands)):
        # 找到命令后3秒的数据
        mask = (odom_times >= t_cmd) & (odom_times < t_cmd + 3.0)
        if np.sum(mask) > 10:
            yaw_segment = yaw_unwrapped[mask]
            yaw_change = yaw_segment[-1] - yaw_segment[0]
            da = cmd[0] - cmd[1]  # 非对称量
            print(f"\n命令 t={t_cmd:.2f}s (da={da:+.2f}): 偏航变化 = {yaw_change:+.2f}°")
            
            if da > 0.1 and yaw_change < 1.0:
                print(f"  ⚠️ 左刹车输入但偏航变化很小!")
            if da < -0.1 and yaw_change > -1.0:
                print(f"  ⚠️ 右刹车输入但偏航变化很小!")


def diagnose_dynamics(times, positions, velocities, orientations, angular_velocities):
    """综合诊断动力学问题"""
    print("\n" + "="*60)
    print("动力学诊断")
    print("="*60)
    
    issues = []
    
    # 检查1: 高度变化
    alt_change = positions[-1, 2] - positions[0, 2]
    if alt_change > -5:
        issues.append(f"高度下降太少: {alt_change:.2f}m (翼伞应该持续下降)")
    
    # 检查2: 速度
    speed = np.linalg.norm(velocities, axis=1)
    if speed.mean() < 3:
        issues.append(f"平均速度过低: {speed.mean():.2f} m/s")
    if speed.max() > 50:
        issues.append(f"最大速度过高: {speed.max():.2f} m/s")
    
    # 检查3: 偏航变化
    eulers = np.array([euler_from_quaternion(q[0], q[1], q[2], q[3]) for q in orientations])
    yaw_unwrapped = np.unwrap(eulers[:, 2])
    yaw_rate = np.abs(np.diff(yaw_unwrapped) / np.diff(times))
    if yaw_rate.max() < 0.01:
        issues.append("偏航几乎没有变化，可能控制无响应")
    
    # 检查4: 四元数归一化
    quat_norms = np.linalg.norm(orientations, axis=1)
    if np.abs(quat_norms - 1.0).max() > 0.01:
        issues.append("四元数未归一化")
    
    # 检查5: NaN 或 Inf
    if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
        issues.append("位置包含 NaN 或 Inf!")
    if np.any(np.isnan(velocities)) or np.any(np.isinf(velocities)):
        issues.append("速度包含 NaN 或 Inf!")
    
    # 检查6: 滑翔比
    horizontal_dist = np.sqrt((positions[-1,0] - positions[0,0])**2 + 
                              (positions[-1,1] - positions[0,1])**2)
    if alt_change < -1:
        glide_ratio = horizontal_dist / (-alt_change)
        if glide_ratio < 1.5:
            issues.append(f"滑翔比太低: {glide_ratio:.2f} (正常应该 2-4)")
        if glide_ratio > 6:
            issues.append(f"滑翔比太高: {glide_ratio:.2f} (可能有问题)")
    
    # 输出诊断结果
    if len(issues) == 0:
        print("\n✅ 未发现明显问题，动力学看起来正常!")
    else:
        print("\n❌ 发现以下问题:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    
    # 总结
    print("\n动力学指标总结:")
    duration = times[-1]
    horizontal_dist = np.sqrt((positions[-1,0] - positions[0,0])**2 + 
                              (positions[-1,1] - positions[0,1])**2)
    print(f"  飞行时间: {duration:.1f} s")
    print(f"  水平距离: {horizontal_dist:.1f} m")
    print(f"  高度下降: {-alt_change:.1f} m")
    print(f"  平均速度: {speed.mean():.1f} m/s")
    if alt_change < -1:
        print(f"  滑翔比: {horizontal_dist / (-alt_change):.2f}")


def main():
    if len(sys.argv) < 2:
        print("用法: python3 analyze_odom_bag.py <bag_path>")
        print("示例: python3 analyze_odom_bag.py test_bags/dynamics_test_xxx")
        sys.exit(1)
    
    bag_path = sys.argv[1]
    
    if not os.path.exists(bag_path):
        print(f"错误: Bag 路径不存在: {bag_path}")
        sys.exit(1)
    
    analyze_bag(bag_path)


if __name__ == '__main__':
    main()
