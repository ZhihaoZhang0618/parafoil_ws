#!/bin/bash

set -e

IMAGE_NAME="zhangzhihao0618/ros2-px4-dev:humble"
CONTAINER_NAME="parafoil"
SSH_HOST_PORT="2222"
ROOT_PASSWORD="radxa"

# 启动 Docker 服务
echo "正在启动 Docker 服务..."
sudo systemctl start docker 2>/dev/null || echo "Docker 可能已在运行"

# 等待 Docker 服务完全启动
sleep 2

# 启动 parafoil 容器
echo "正在启动 parafoil 容器..."
RECREATE_CONTAINER=0
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
        CURRENT_NET_MODE="$(docker inspect "$CONTAINER_NAME" --format '{{.HostConfig.NetworkMode}}')"
        if [ "$CURRENT_NET_MODE" != "host" ]; then
                echo "发现已有容器 $CONTAINER_NAME（NetworkMode=$CURRENT_NET_MODE），将重建为 host 网络..."
                RECREATE_CONTAINER=1
        fi
fi

if [ "$RECREATE_CONTAINER" = "1" ]; then
        docker rm -f "$CONTAINER_NAME" >/dev/null
fi

if ! docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
        echo "未发现容器 $CONTAINER_NAME，正在用镜像 $IMAGE_NAME 创建（host 网络 + /dev 共享）..."

        docker run -d \
                --name "$CONTAINER_NAME" \
                --network host \
                --privileged \
                -v /dev:/dev \
                "$IMAGE_NAME" \
                bash -lc "sleep infinity"
else
        docker start "$CONTAINER_NAME" 2>/dev/null || echo "容器可能已在运行"
fi

# 等待容器启动
sleep 2

# 在容器中启动 SSH 服务
echo "正在启动容器内的 SSH 服务（端口 2222）..."
docker exec "$CONTAINER_NAME" bash -lc "set -e; \
        echo 'root:${ROOT_PASSWORD}' | chpasswd; \
        if [[ -f /etc/ssh/sshd_config ]]; then \
                sed -i -E '/^(#\\s*)?
