#!/bin/bash
# 简单测试脚本 - 直接观察状态topic输出

set -e
WORKSPACE="/home/aims/parafoil_ws"

source /opt/ros/humble/setup.bash
source ${WORKSPACE}/install/setup.bash

echo "=========================================="
echo "启动仿真器并监控状态..."
echo "=========================================="

# 启动仿真器 (后台)
ros2 run parafoil_simulator_ros sim_node &
SIM_PID=$!
sleep 2

echo ""
echo ">>> 监控 /parafoil/state topic (5秒)..."
timeout 5 ros2 topic echo /parafoil/state --field position --field velocity --field orientation 2>/dev/null || true

echo ""
echo ">>> 发送左转控制指令 delta_l=0.5, delta_r=0.0"
ros2 topic pub --once /parafoil/control_cmd parafoil_simulator_ros/msg/ControlCmd "{delta_l: 0.5, delta_r: 0.0}"

echo ""
echo ">>> 继续监控状态 (5秒)..."
timeout 5 ros2 topic echo /parafoil/state --field position --field velocity --field orientation 2>/dev/null || true

echo ""
echo ">>> 发送右转控制指令 delta_l=0.0, delta_r=0.5"
ros2 topic pub --once /parafoil/control_cmd parafoil_simulator_ros/msg/ControlCmd "{delta_l: 0.0, delta_r: 0.5}"

echo ""
echo ">>> 继续监控状态 (5秒)..."
timeout 5 ros2 topic echo /parafoil/state --field position --field velocity --field orientation 2>/dev/null || true

# 停止仿真器
kill $SIM_PID 2>/dev/null || true

echo ""
echo "测试完成"
