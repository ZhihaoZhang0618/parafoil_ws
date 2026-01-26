#!/bin/bash
# 翼伞运动学测试脚本 - 使用正确的消息类型

set -e
WORKSPACE="/home/aims/parafoil_ws"

source /opt/ros/humble/setup.bash
source ${WORKSPACE}/install/setup.bash

echo "=========================================="
echo "翼伞运动学测试"
echo "=========================================="

# 清理之前的进程
pkill -f sim_node 2>/dev/null || true
sleep 1

# 启动仿真器 (后台)
echo "[1] 启动仿真器..."
ros2 run parafoil_simulator_ros sim_node &
SIM_PID=$!
sleep 3

if ! kill -0 $SIM_PID 2>/dev/null; then
    echo "错误: 仿真器启动失败"
    exit 1
fi

echo ""
echo "[2] 查看初始状态..."
echo "Topic: /rockpara_actuators_node/auto_commands (控制输入)"
echo ""

# 监控odom (3秒)
echo ">>> 自由滑翔阶段 - 监控odom (3秒)..."
timeout 3 ros2 topic echo /parafoil/odom --field pose.pose.position --field twist.twist.linear 2>/dev/null &
ECHO_PID=$!
sleep 3
kill $ECHO_PID 2>/dev/null || true

echo ""
echo ">>> 发送左转控制: delta_l=0.5, delta_r=0.0"
# 使用geometry_msgs/Vector3Stamped: x=delta_l, y=delta_r
ros2 topic pub -r 50 /rockpara_actuators_node/auto_commands geometry_msgs/msg/Vector3Stamped "{header: {stamp: {sec: 0, nanosec: 0}, frame_id: ''}, vector: {x: 0.5, y: 0.0, z: 0.0}}" &
PUB_PID=$!

# 监控5秒
echo ">>> 左转阶段 - 监控odom (5秒)..."
timeout 5 ros2 topic echo /parafoil/odom --field pose.pose.position --field twist.twist.linear 2>/dev/null &
ECHO_PID=$!
sleep 5
kill $ECHO_PID 2>/dev/null || true
kill $PUB_PID 2>/dev/null || true

echo ""
echo ">>> 发送右转控制: delta_l=0.0, delta_r=0.5"
ros2 topic pub -r 50 /rockpara_actuators_node/auto_commands geometry_msgs/msg/Vector3Stamped "{header: {stamp: {sec: 0, nanosec: 0}, frame_id: ''}, vector: {x: 0.0, y: 0.5, z: 0.0}}" &
PUB_PID=$!

# 监控5秒
echo ">>> 右转阶段 - 监控odom (5秒)..."
timeout 5 ros2 topic echo /parafoil/odom --field pose.pose.position --field twist.twist.linear 2>/dev/null &
ECHO_PID=$!
sleep 5
kill $ECHO_PID 2>/dev/null || true
kill $PUB_PID 2>/dev/null || true

# 清理
echo ""
echo "=========================================="
echo "测试完成，清理进程..."
kill $SIM_PID 2>/dev/null || true

echo "完成"
