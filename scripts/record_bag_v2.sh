#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TARGET_X="${TARGET_X:-150.0}"
TARGET_Y="${TARGET_Y:-50.0}"
INITIAL_ALTITUDE="${INITIAL_ALTITUDE:-300.0}"
DURATION="${DURATION:-45}"
L1_DISTANCE="${L1_DISTANCE:-15.0}"
K_YAW_RATE="${K_YAW_RATE:-1.5}"
MAX_DELTA_A="${MAX_DELTA_A:-0.5}"
CAPTURE_RADIUS="${CAPTURE_RADIUS:-30.0}"
TARGET_LOCK_RADIUS="${TARGET_LOCK_RADIUS:-40.0}"
TERMINAL_RADIUS="${TERMINAL_RADIUS:-40.0}"
TERMINAL_BRAKE="${TERMINAL_BRAKE:-0.5}"
TERMINAL_ALTITUDE="${TERMINAL_ALTITUDE:-80.0}"
DISTANCE_BLEND="${DISTANCE_BLEND:-1.0}"
ALTITUDE_RESERVE="${ALTITUDE_RESERVE:-5.0}"
ALTITUDE_SAFETY_MARGIN="${ALTITUDE_SAFETY_MARGIN:-5.0}"
MAX_BRAKE="${MAX_BRAKE:-0.5}"
DRIFT_COMP_ENABLE="${DRIFT_COMP_ENABLE:-true}"
DRIFT_COMP_ALTITUDE="${DRIFT_COMP_ALTITUDE:-80.0}"
DRIFT_COMP_BRAKE="${DRIFT_COMP_BRAKE:-0.9}"
DRIFT_COMP_SCALE="${DRIFT_COMP_SCALE:-1.0}"
WIND_USE_TOPIC="${WIND_USE_TOPIC:-true}"
WIND_TOPIC="${WIND_TOPIC:-/wind_estimate}"
WIND_EST_SIGMA="${WIND_EST_SIGMA:-0.5}"
WIND_EST_NOISE_SIGMA="${WIND_EST_NOISE_SIGMA:-0.5}"
WIND_EST_GUST_SCALE="${WIND_EST_GUST_SCALE:-1.0}"
WIND_EST_BIAS_N="${WIND_EST_BIAS_N:-0.0}"
WIND_EST_BIAS_E="${WIND_EST_BIAS_E:-0.0}"
WIND_EST_BIAS_D="${WIND_EST_BIAS_D:-0.0}"
WIND_EST_SEED="${WIND_EST_SEED:--1}"
WIND_ENABLE_STEADY="${WIND_ENABLE_STEADY:-true}"
WIND_ENABLE_GUST="${WIND_ENABLE_GUST:-true}"
WIND_ENABLE_COLORED="${WIND_ENABLE_COLORED:-false}"
WIND_STEADY_N="${WIND_STEADY_N:-0.0}"
WIND_STEADY_E="${WIND_STEADY_E:-2.0}"
WIND_STEADY_D="${WIND_STEADY_D:-0.0}"
WIND_GUST_INTERVAL="${WIND_GUST_INTERVAL:-10.0}"
WIND_GUST_DURATION="${WIND_GUST_DURATION:-2.0}"
WIND_GUST_MAGNITUDE="${WIND_GUST_MAGNITUDE:-3.0}"
WIND_COLORED_TAU="${WIND_COLORED_TAU:-2.0}"
WIND_COLORED_SIGMA="${WIND_COLORED_SIGMA:-1.0}"
WIND_SEED="${WIND_SEED:--1}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/bags/v2_${STAMP}}"

echo "[v2] Recording bag to: ${OUT_DIR}"

# Avoid unbound COLCON_TRACE in some environments
unset AMENT_CURRENT_PREFIX || true
unset COLCON_CURRENT_PREFIX || true
source "${ROOT_DIR}/install/setup.bash"

LAUNCH_CMD="ros2 launch parafoil_plannerv2 integrated_test_v2.launch.py \
  target_x:=${TARGET_X} target_y:=${TARGET_Y} initial_altitude:=${INITIAL_ALTITUDE} \
  l1_distance:=${L1_DISTANCE} k_yaw_rate:=${K_YAW_RATE} max_delta_a:=${MAX_DELTA_A} \
  capture_radius:=${CAPTURE_RADIUS} target_lock_radius:=${TARGET_LOCK_RADIUS} \
  terminal_radius:=${TERMINAL_RADIUS} terminal_brake:=${TERMINAL_BRAKE} \
  terminal_altitude:=${TERMINAL_ALTITUDE} distance_blend:=${DISTANCE_BLEND} \
  altitude_reserve:=${ALTITUDE_RESERVE} altitude_safety_margin:=${ALTITUDE_SAFETY_MARGIN} \
  max_brake:=${MAX_BRAKE} \
  drift_comp_enable:=${DRIFT_COMP_ENABLE} drift_comp_altitude:=${DRIFT_COMP_ALTITUDE} \
  drift_comp_brake:=${DRIFT_COMP_BRAKE} drift_comp_scale:=${DRIFT_COMP_SCALE} \
  wind_use_topic:=${WIND_USE_TOPIC} wind_topic:=${WIND_TOPIC} wind_estimate_sigma:=${WIND_EST_SIGMA} \
  wind_est_gust_noise_sigma:=${WIND_EST_NOISE_SIGMA} wind_est_gust_scale:=${WIND_EST_GUST_SCALE} \
  wind_est_bias_n:=${WIND_EST_BIAS_N} wind_est_bias_e:=${WIND_EST_BIAS_E} wind_est_bias_d:=${WIND_EST_BIAS_D} \
  wind_est_seed:=${WIND_EST_SEED} \
  wind_enable_steady:=${WIND_ENABLE_STEADY} wind_enable_gust:=${WIND_ENABLE_GUST} \
  wind_enable_colored:=${WIND_ENABLE_COLORED} wind_steady_n:=${WIND_STEADY_N} \
  wind_steady_e:=${WIND_STEADY_E} wind_steady_d:=${WIND_STEADY_D} \
  wind_gust_interval:=${WIND_GUST_INTERVAL} wind_gust_duration:=${WIND_GUST_DURATION} \
  wind_gust_magnitude:=${WIND_GUST_MAGNITUDE} wind_colored_tau:=${WIND_COLORED_TAU} \
  wind_colored_sigma:=${WIND_COLORED_SIGMA} wind_seed:=${WIND_SEED}"

echo "[v2] Launching: ${LAUNCH_CMD}"
setsid bash -lc "${LAUNCH_CMD}" >"${OUT_DIR}_launch.log" 2>&1 &
LAUNCH_PID=$!

sleep 2

TOPICS=(
  /position
  /parafoil/odom
  /planned_path
  /planner_status
  /guidance_debug
  /wind_estimate
  /rockpara_actuators_node/auto_commands
)

mkdir -p "${ROOT_DIR}/bags"
if [ -d "${OUT_DIR}" ]; then
  rm -rf "${OUT_DIR}"
fi

echo "[v2] Recording topics for ${DURATION}s..."
timeout "${DURATION}" ros2 bag record -o "${OUT_DIR}" "${TOPICS[@]}" >"${OUT_DIR}_bag.log" 2>&1 || true

echo "[v2] Stopping launch (pid ${LAUNCH_PID})"
kill -TERM -- -"${LAUNCH_PID}" >/dev/null 2>&1 || true
sleep 1
kill -KILL -- -"${LAUNCH_PID}" >/dev/null 2>&1 || true

echo "[v2] Done."
echo "[v2] BAG_DIR=${OUT_DIR}"
