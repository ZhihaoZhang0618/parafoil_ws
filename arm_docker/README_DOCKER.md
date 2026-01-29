# Docker for parafoil_ws 🔧

This repo contains a `Dockerfile` and `docker-compose.yml` to run the workspace in a container similar to the behaviour described in `start-docker-ssh.sh`.

Quick steps

1. Build the image:
   - Using Docker:
     ```bash
     docker build -t parafoil:latest . --build-arg ROOT_PASSWORD=root
     ```
   - Or using docker-compose:
     ```bash
     docker compose build
     ```

2. Run the container (host network + /dev shared, same as the script):
   ```bash
   docker compose up -d
   ```

   If you prefer not to use `network_mode: host`, edit `docker-compose.yml`, remove/comment `network_mode: "host"` and enable the ports mapping (example maps host 2222 -> container 22).

3. SSH into the container:
   - If using host networking: ssh root@localhost -p 2222 (map a host port to 22 if you didn't use host networking)
   - Default root password (build arg): `root` (change with `--build-arg ROOT_PASSWORD=<pw>` at build time)

Notes & Tips 💡
- The Dockerfile attempts to `rosdep install` and `colcon build` the main packages. Building inside the image can take time; you may prefer to mount the workspace and run a build inside the running container for faster iteration.
- The Dockerfile is intentionally best-effort; if some system dependencies are missing you may need to install them (e.g. `sudo apt-get install -y <package>`) and rebuild.
- For development it is convenient to bind-mount the workspace (docker-compose already does) so you can edit locally and run builds inside the container.

Platform & Build notes ⚠️
- If you see `exec /bin/bash: exec format error` or a warning like `The requested image's platform (linux/arm64/v8) does not match the detected host platform (linux/amd64/...)`, your host architecture (amd64) does not match the base image architecture (arm64).

Quick ways to resolve:
1. Use Docker Buildx with QEMU emulation (fast to try):
   ```bash
   # install QEMU emulation helpers (one-time)
   sudo docker run --rm --privileged tonistiigi/binfmt --install all

   # create and use a buildx builder
   docker buildx create --use --name parafoil-builder || true

   # build for amd64 (or for multiple platforms). Use --load to load into local docker
   docker buildx build --platform linux/amd64 -t parafoil:latest --load . --build-arg ROOT_PASSWORD=root
   ```

2. Pull or use an amd64 base image (recommended if available):
   - Try forcing platform in the Dockerfile `FROM --platform=linux/amd64 <image>` or pick an amd64 image (e.g. official ROS2 images) to avoid emulation.

3. Use BuildKit for faster builds and to avoid legacy builder warnings:
   ```bash
   docker buildx create --use
   DOCKER_BUILDKIT=1 docker build -t parafoil:latest . --build-arg ROOT_PASSWORD=root
   ```

Also: create a `.dockerignore` (already included in this repo) to avoid packaging `.git` and large files into the build context and speed up builds.


If you want, I can also:
- finish the `start-docker-ssh.sh` script so it works with this image,
- add a small `entrypoint` to build on container start, or
- set up a non-root user and passwordless sudo for safer development.
