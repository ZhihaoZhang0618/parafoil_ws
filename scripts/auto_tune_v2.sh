#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TARGET_X="${TARGET_X:-150.0}"
TARGET_Y="${TARGET_Y:-50.0}"
INITIAL_ALTITUDE="${INITIAL_ALTITUDE:-300.0}"
DURATION="${DURATION:-90}"
LANDING_ALT="${LANDING_ALT:-1.0}"
THRESHOLD="${THRESHOLD:-20.0}"

L1_LIST=(${L1_LIST:-15.0 20.0 25.0 30.0})
K_LIST=(${K_LIST:-0.8 1.0 1.2})
DELTA_LIST=(${DELTA_LIST:-0.3 0.5})
CAPTURE_LIST=(${CAPTURE_LIST:-20.0 30.0 40.0})
TARGET_LOCK_LIST=(${TARGET_LOCK_LIST:-30.0})
TERMINAL_RADIUS_LIST=(${TERMINAL_RADIUS_LIST:-30.0})
TERMINAL_BRAKE_LIST=(${TERMINAL_BRAKE_LIST:-0.9})
TERMINAL_ALT_LIST=(${TERMINAL_ALT_LIST:-60.0})
DIST_BLEND_LIST=(${DIST_BLEND_LIST:-0.7})
ALT_RESERVE_LIST=(${ALT_RESERVE_LIST:-10.0})
ALT_SAFETY_LIST=(${ALT_SAFETY_LIST:-15.0})
DRIFT_ENABLE_LIST=(${DRIFT_ENABLE_LIST:-true})
DRIFT_ALT_LIST=(${DRIFT_ALT_LIST:-80.0})
DRIFT_BRAKE_LIST=(${DRIFT_BRAKE_LIST:-0.9})
DRIFT_SCALE_LIST=(${DRIFT_SCALE_LIST:-1.0})

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/bags/tune_${STAMP}"
mkdir -p "${LOG_DIR}"

RESULTS_CSV="${LOG_DIR}/results.csv"
echo "run_id,l1,k_yaw_rate,max_delta_a,capture_radius,target_lock_radius,terminal_radius,terminal_brake,terminal_altitude,distance_blend,altitude_reserve,altitude_safety_margin,drift_enable,drift_altitude,drift_brake,drift_scale,landing_dist,landing_x,landing_y,landing_alt,bag_dir" > "${RESULTS_CSV}"

best_dist=1e9
best_line=""
run_id=0

echo "[tune] target=(${TARGET_X},${TARGET_Y}) alt0=${INITIAL_ALTITUDE} duration=${DURATION}s threshold=${THRESHOLD}m"
echo "[tune] logs: ${LOG_DIR}"

for l1 in "${L1_LIST[@]}"; do
  for k in "${K_LIST[@]}"; do
    for d in "${DELTA_LIST[@]}"; do
      for c in "${CAPTURE_LIST[@]}"; do
        for lock_r in "${TARGET_LOCK_LIST[@]}"; do
          for term_r in "${TERMINAL_RADIUS_LIST[@]}"; do
            for term_b in "${TERMINAL_BRAKE_LIST[@]}"; do
              for term_a in "${TERMINAL_ALT_LIST[@]}"; do
                for blend in "${DIST_BLEND_LIST[@]}"; do
                  for a_res in "${ALT_RESERVE_LIST[@]}"; do
                    for a_safe in "${ALT_SAFETY_LIST[@]}"; do
                      for drift_en in "${DRIFT_ENABLE_LIST[@]}"; do
                        for drift_alt in "${DRIFT_ALT_LIST[@]}"; do
                          for drift_brake in "${DRIFT_BRAKE_LIST[@]}"; do
                            for drift_scale in "${DRIFT_SCALE_LIST[@]}"; do
        run_id=$((run_id + 1))
        out_dir="${LOG_DIR}/bag_${run_id}"

        echo "[tune] run=${run_id} L1=${l1} K=${k} max_delta=${d} capture=${c} lock=${lock_r} term_r=${term_r} term_b=${term_b} term_alt=${term_a} blend=${blend} alt_res=${a_res} alt_safe=${a_safe} drift_en=${drift_en} drift_alt=${drift_alt} drift_brake=${drift_brake} drift_scale=${drift_scale}"

        TARGET_X="${TARGET_X}" TARGET_Y="${TARGET_Y}" INITIAL_ALTITUDE="${INITIAL_ALTITUDE}" \
        DURATION="${DURATION}" L1_DISTANCE="${l1}" K_YAW_RATE="${k}" MAX_DELTA_A="${d}" \
        CAPTURE_RADIUS="${c}" TARGET_LOCK_RADIUS="${lock_r}" \
        TERMINAL_RADIUS="${term_r}" TERMINAL_BRAKE="${term_b}" TERMINAL_ALTITUDE="${term_a}" \
        DISTANCE_BLEND="${blend}" ALTITUDE_RESERVE="${a_res}" ALTITUDE_SAFETY_MARGIN="${a_safe}" \
        DRIFT_COMP_ENABLE="${drift_en}" DRIFT_COMP_ALTITUDE="${drift_alt}" \
        DRIFT_COMP_BRAKE="${drift_brake}" DRIFT_COMP_SCALE="${drift_scale}" \
        OUT_DIR="${out_dir}" \
        "${ROOT_DIR}/scripts/record_bag_v2.sh" >"${out_dir}_record.log" 2>&1

        metrics="$(
          python3 "${ROOT_DIR}/scripts/analyze_bag_v2.py" "${out_dir}" \
            --target-x "${TARGET_X}" --target-y "${TARGET_Y}" --landing-alt "${LANDING_ALT}"
        )"

        landing_dist="$(echo "${metrics}" | awk -F': ' '/landing_dist/ {print $2; exit}')"
        landing_x="$(echo "${metrics}" | awk -F': ' '/landing_x/ {print $2; exit}')"
        landing_y="$(echo "${metrics}" | awk -F': ' '/landing_y/ {print $2; exit}')"
        landing_alt="$(echo "${metrics}" | awk -F': ' '/landing_alt/ {print $2; exit}')"

        echo "${run_id},${l1},${k},${d},${c},${lock_r},${term_r},${term_b},${term_a},${blend},${a_res},${a_safe},${drift_en},${drift_alt},${drift_brake},${drift_scale},${landing_dist},${landing_x},${landing_y},${landing_alt},${out_dir}" >> "${RESULTS_CSV}"

        if awk "BEGIN {exit !(${landing_dist} < ${best_dist})}"; then
          best_dist="${landing_dist}"
          best_line="run=${run_id} L1=${l1} K=${k} max_delta=${d} capture=${c} lock=${lock_r} term_r=${term_r} term_b=${term_b} term_alt=${term_a} blend=${blend} alt_res=${a_res} alt_safe=${a_safe} drift_en=${drift_en} drift_alt=${drift_alt} drift_brake=${drift_brake} drift_scale=${drift_scale} landing_dist=${landing_dist}"
        fi

        if awk "BEGIN {exit !(${landing_dist} <= ${THRESHOLD})}"; then
          echo "[tune] threshold met: ${best_line}"
          echo "[tune] results: ${RESULTS_CSV}"
          exit 0
        fi
                            done
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "[tune] done. best: ${best_line}"
echo "[tune] results: ${RESULTS_CSV}"
