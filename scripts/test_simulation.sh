#!/bin/bash
# 翼伞仿真测试脚本
# 启动仿真器、输入测试程序、bag录制

set -e

WORKSPACE="/home/aims/parafoil_ws"
BAG_DIR="${WORKSPACE}/bags"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BAG_NAME="parafoil_test_${TIMESTAMP}"

# 创建bag目录
mkdir -p ${BAG_DIR}

echo "=========================================="
echo "翼伞仿真测试脚本"
echo "=========================================="
echo "Bag录制目录: ${BAG_DIR}/${BAG_NAME}"
echo ""

# 确保source ROS2环境
source /opt/ros/humble/setup.bash
source ${WORKSPACE}/install/setup.bash

# 清理之前的进程
cleanup() {
    echo ""
    echo "正在清理进程..."
    pkill -f "ros2 bag record" 2>/dev/null || true
    pkill -f "sim_node" 2>/dev/null || true
    pkill -f "test_input_node" 2>/dev/null || true
    sleep 1
    echo "清理完成"
}

trap cleanup EXIT

echo "[1/3] 启动仿真器..."
ros2 run parafoil_simulator_ros sim_node &
SIM_PID=$!
sleep 2

# 检查仿真器是否启动
if ! kill -0 $SIM_PID 2>/dev/null; then
    echo "错误: 仿真器启动失败"
    exit 1
fi
echo "仿真器已启动 (PID: $SIM_PID)"

echo ""
echo "[2/3] 启动bag录制..."
ros2 bag record -o ${BAG_DIR}/${BAG_NAME} \
    /parafoil/state \
    /parafoil/sensors/position \
    /parafoil/sensors/imu \
    /parafoil/control_cmd \
    /parafoil/odom \
    /tf &
BAG_PID=$!
sleep 1
echo "Bag录制已启动 (PID: $BAG_PID)"

echo ""
echo "[3/3] 运行测试输入序列 (10秒)..."
echo "  - 前5秒: 无控制输入 (观察自由滑翔)"
echo "  - 后5秒: 左转控制"
echo ""

# 等待5秒观察自由滑翔
echo ">>> 自由滑翔阶段 (5秒)..."
for i in {5..1}; do
    echo "  剩余: ${i}秒"
    sleep 1
done

# 发送左转控制指令 (delta_l=0.3, delta_r=0.0)
echo ""
echo ">>> 左转控制阶段 (5秒)..."
echo "  发送控制: delta_l=0.3, delta_r=0.0"
for i in {5..1}; do
    ros2 topic pub --once /parafoil/control_cmd parafoil_simulator_ros/msg/ControlCmd "{delta_l: 0.3, delta_r: 0.0}" 2>/dev/null
    echo "  剩余: ${i}秒"
    sleep 1
done

echo ""
echo "=========================================="
echo "测试完成，停止录制..."
echo "=========================================="

# 停止bag录制
kill $BAG_PID 2>/dev/null || true
sleep 2

# 停止仿真器
kill $SIM_PID 2>/dev/null || true
sleep 1

echo ""
echo "Bag文件保存在: ${BAG_DIR}/${BAG_NAME}"
echo ""

# 显示bag信息
echo "Bag信息:"
ros2 bag info ${BAG_DIR}/${BAG_NAME}

echo ""
echo "=========================================="
echo "分析最后几条状态消息..."
echo "=========================================="

# 使用Python脚本分析bag数据
python3 << 'EOF'
import sqlite3
import os
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

bag_dir = os.environ.get('BAG_DIR', '/home/aims/parafoil_ws/bags')
bag_name = os.environ.get('BAG_NAME', '')

# 找到最新的bag
if not bag_name:
    bags = sorted([d for d in os.listdir(bag_dir) if d.startswith('parafoil_test_')])
    if bags:
        bag_name = bags[-1]

bag_path = os.path.join(bag_dir, bag_name)
db_path = os.path.join(bag_path, f"{bag_name}_0.db3")

if not os.path.exists(db_path):
    # 尝试其他命名格式
    db_files = [f for f in os.listdir(bag_path) if f.endswith('.db3')]
    if db_files:
        db_path = os.path.join(bag_path, db_files[0])

print(f"分析bag: {bag_path}")

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 获取topic信息
    cursor.execute("SELECT id, name, type FROM topics")
    topics = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}
    
    # 找到odom topic
    odom_topic_id = None
    for tid, (name, msg_type) in topics.items():
        if name == '/parafoil/odom':
            odom_topic_id = tid
            break
    
    if odom_topic_id:
        # 获取最后10条消息
        cursor.execute(f"""
            SELECT timestamp, data FROM messages 
            WHERE topic_id = {odom_topic_id}
            ORDER BY timestamp DESC
            LIMIT 10
        """)
        
        from nav_msgs.msg import Odometry
        
        print("\n最后10条Odometry消息 (时间倒序):")
        print("-" * 80)
        print(f"{'时间(s)':<12} {'X(m)':<12} {'Y(m)':<12} {'Z(m)':<12} {'Vx(m/s)':<10} {'Vy(m/s)':<10} {'Vz(m/s)':<10}")
        print("-" * 80)
        
        msgs = cursor.fetchall()
        for timestamp, data in reversed(msgs):
            msg = deserialize_message(data, Odometry)
            pos = msg.pose.pose.position
            vel = msg.twist.twist.linear
            t = timestamp / 1e9
            print(f"{t:<12.2f} {pos.x:<12.3f} {pos.y:<12.3f} {pos.z:<12.3f} {vel.x:<10.3f} {vel.y:<10.3f} {vel.z:<10.3f}")
        
        print("-" * 80)
        
        # 计算一些统计
        if len(msgs) >= 2:
            first_ts, first_data = msgs[-1]
            last_ts, last_data = msgs[0]
            first_msg = deserialize_message(first_data, Odometry)
            last_msg = deserialize_message(last_data, Odometry)
            
            dt = (last_ts - first_ts) / 1e9
            dx = last_msg.pose.pose.position.x - first_msg.pose.pose.position.x
            dy = last_msg.pose.pose.position.y - first_msg.pose.pose.position.y
            dz = last_msg.pose.pose.position.z - first_msg.pose.pose.position.z
            
            print(f"\n运动分析 (最后{dt:.1f}秒):")
            print(f"  位移: ΔX={dx:.2f}m, ΔY={dy:.2f}m, ΔZ={dz:.2f}m")
            print(f"  平均速度: Vx={dx/dt:.2f}m/s, Vy={dy/dt:.2f}m/s, Vz={dz/dt:.2f}m/s")
            
            # 检查Z方向 (高度变化)
            # 在ENU坐标系中，Z正向为上
            if dz < 0:
                print(f"  ✓ 高度下降 (符合预期，翼伞应该下沉)")
            else:
                print(f"  ✗ 高度上升 (异常!)")
    
    conn.close()
    
except Exception as e:
    print(f"分析失败: {e}")
    import traceback
    traceback.print_exc()

EOF

echo ""
echo "测试脚本执行完成"
