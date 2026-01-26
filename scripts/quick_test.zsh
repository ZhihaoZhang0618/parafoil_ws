#!/bin/zsh
# 翼伞动力学快速测试脚本

WORKSPACE="/home/aims/parafoil_ws"
BAG_DIR="$WORKSPACE/test_bags"
BAG_NAME="dynamics_test_$(date +%Y%m%d_%H%M%S)"

# 清理
pkill -9 -f sim_node 2>/dev/null
pkill -9 -f "ros2 bag" 2>/dev/null
sleep 2

# 环境
source /opt/ros/humble/setup.zsh
source $WORKSPACE/install/setup.zsh

mkdir -p $BAG_DIR
cd $BAG_DIR

echo "=== 启动仿真器和录制 ==="
# 后台启动仿真器
ros2 run parafoil_simulator_ros sim_node --ros-args \
    --params-file $WORKSPACE/src/parafoil_simulator_ros/config/params.yaml &
SIM_PID=$!
sleep 3

# 后台录制
ros2 bag record -o $BAG_NAME /parafoil/odom /rockpara_actuators_node/auto_commands &
BAG_PID=$!
sleep 2

# 发送控制命令函数
send_cmd() {
    ros2 topic pub --once /rockpara_actuators_node/auto_commands geometry_msgs/msg/Vector3Stamped \
        "{header: {stamp: {sec: 0, nanosec: 0}, frame_id: ''}, vector: {x: $1, y: $2, z: 0.0}}" 2>/dev/null &
}

echo "=== 测试1: 自由滑翔 5秒 ==="
send_cmd 0.0 0.0
sleep 5

echo "=== 测试2: 左转 3秒 ==="
send_cmd 0.5 0.0
sleep 3

echo "=== 测试3: 右转 3秒 ==="
send_cmd 0.0 0.5
sleep 3

echo "=== 测试4: 对称刹车 3秒 ==="
send_cmd 0.5 0.5
sleep 3

echo "=== 测试5: 恢复滑翔 3秒 ==="
send_cmd 0.0 0.0
sleep 3

# 停止
echo "=== 停止 ==="
kill $BAG_PID 2>/dev/null
sleep 1
kill $SIM_PID 2>/dev/null
sleep 2

echo "=== Bag 信息 ==="
ros2 bag info $BAG_DIR/$BAG_NAME

echo ""
echo "=== 分析结果 ==="
python3 $WORKSPACE/scripts/analyze_odom_bag.py $BAG_DIR/$BAG_NAME
