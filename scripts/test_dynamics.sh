#!/bin/bash
# 翼伞动力学测试脚本 - 修正版
# 录制 bag 包用于分析

set -e

WORKSPACE="/home/aims/parafoil_ws"
BAG_DIR="$WORKSPACE/test_bags"
BAG_NAME="dynamics_test_$(date +%Y%m%d_%H%M%S)"

# 清理旧进程
echo "=== 清理旧进程 ==="
pkill -9 -f sim_node 2>/dev/null || true
pkill -9 -f "ros2 bag" 2>/dev/null || true
pkill -9 -f "ros2 topic pub" 2>/dev/null || true
sleep 2

# 设置环境
echo "=== 设置 ROS2 环境 ==="
source /opt/ros/humble/setup.zsh
source $WORKSPACE/install/setup.zsh

# 创建 bag 目录
mkdir -p $BAG_DIR

# 启动仿真器 (后台) - 使用参数文件
echo "=== 启动仿真器 ==="
ros2 run parafoil_simulator_ros sim_node --ros-args --params-file $WORKSPACE/src/parafoil_simulator_ros/config/params.yaml &
SIM_PID=$!
sleep 3

# 启动 bag 录制 (后台)
echo "=== 开始录制 bag: $BAG_NAME ==="
cd $BAG_DIR
ros2 bag record -o $BAG_NAME \
    /parafoil/odom \
    /rockpara_actuators_node/auto_commands &
BAG_PID=$!
sleep 2

# 定义发送控制命令的函数
# 使用 geometry_msgs/msg/Vector3Stamped
# x = delta_left, y = delta_right
send_cmd() {
    local dl=$1
    local dr=$2
    ros2 topic pub --once /rockpara_actuators_node/auto_commands geometry_msgs/msg/Vector3Stamped \
        "{header: {stamp: {sec: 0, nanosec: 0}, frame_id: ''}, vector: {x: $dl, y: $dr, z: 0.0}}" 2>/dev/null &
}

# 测试1: 无输入，自由滑翔 5 秒
echo "=== 测试1: 自由滑翔 (5秒) ==="
send_cmd 0.0 0.0
sleep 5

# 测试2: 左转 (左刹车 0.5, 右刹车 0.0) 持续 3 秒
echo "=== 测试2: 左转 (左刹车=0.5) (3秒) ==="
send_cmd 0.5 0.0
sleep 3

# 测试3: 右转 (左刹车 0.0, 右刹车 0.5) 持续 3 秒
echo "=== 测试3: 右转 (右刹车=0.5) (3秒) ==="
send_cmd 0.0 0.5
sleep 3

# 测试4: 对称刹车 (两边都 0.5) 持续 3 秒
echo "=== 测试4: 对称刹车 (3秒) ==="
send_cmd 0.5 0.5
sleep 3

# 测试5: 恢复自由滑翔 3 秒
echo "=== 测试5: 恢复自由滑翔 (3秒) ==="
send_cmd 0.0 0.0
sleep 3

# 停止录制和仿真
echo "=== 停止录制和仿真 ==="
kill $BAG_PID 2>/dev/null || true
sleep 1
kill $SIM_PID 2>/dev/null || true
sleep 2

echo "=== 录制完成 ==="
echo "Bag 文件: $BAG_DIR/$BAG_NAME"

# 显示 bag 信息
echo ""
echo "=== Bag 信息 ==="
ros2 bag info $BAG_DIR/$BAG_NAME

echo ""
echo "=== 运行分析脚本 ==="
python3 $WORKSPACE/scripts/analyze_odom_bag.py $BAG_DIR/$BAG_NAME
