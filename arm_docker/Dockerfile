# Use the official amd64 ROS2 Humble image as base
FROM --platform=linux/amd64 ros:humble

# Allow changing the root password at build time (default from the script)
ARG ROOT_PASSWORD=root
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install SSH server and any build tools we need
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        openssh-server \
        sudo \
        python3-colcon-common-extensions \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Prepare sshd
RUN mkdir -p /var/run/sshd

# Set root password and enable password authentication
RUN echo "root:${ROOT_PASSWORD}" | chpasswd \
    && sed -i 's/^#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config || true \
    && sed -i 's/^#PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config || true \
    && sed -i 's/^#PermitEmptyPasswords.*/PermitEmptyPasswords no/' /etc/ssh/sshd_config || true

# Copy the workspace into the image
WORKDIR /home/ros/parafoil_ws
COPY . /home/ros/parafoil_ws
RUN chown -R root:root /home/ros/parafoil_ws || true

# Try to install ROS package dependencies (best-effort); building is left to runtime
RUN apt-get update \
    && rosdep update || true \
    && rosdep install -y --from-paths src --ignore-src -r || true \
    && rm -rf /var/lib/apt/lists/*

# Expose SSH
EXPOSE 22

# Start sshd in foreground by default
CMD ["/usr/sbin/sshd", "-D"]
