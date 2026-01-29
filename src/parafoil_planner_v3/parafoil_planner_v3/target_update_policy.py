"""目标点更新策略模块

实现阵风鲁棒的目标点更新策略（方案 D）：
- 保守风模型：通过 gust_margin 收缩可达域
- 滞后机制：小幅度变化不触发切换
- 阶段锁定：APPROACH/FLARE 阶段限制更新
- 紧急重选：当前目标不可达时强制重选
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from parafoil_planner_v3.landing_site_selector import LandingSiteSelection
from parafoil_planner_v3.types import GuidancePhase, Target


class UpdateReason(Enum):
    """目标更新原因"""
    INITIAL = "initial"
    CRUISE_UPDATE = "cruise_update"
    CRUISE_HYSTERESIS = "cruise_hysteresis"
    APPROACH_LOCKED = "approach_locked"
    APPROACH_HYSTERESIS = "approach_hysteresis"
    APPROACH_SIGNIFICANT = "approach_significant_improvement"
    FLARE_LOCKED = "flare_locked"
    EMERGENCY_RESELECT = "emergency_reselect"
    EMERGENCY_COOLDOWN = "emergency_cooldown"
    UNREACHABLE_WIND = "unreachable_wind"
    MANUAL = "manual"
    DISABLED = "disabled"
    NO_CURRENT = "no_current_target"


@dataclass
class TargetUpdatePolicyConfig:
    """目标更新策略配置"""
    # 是否启用策略
    enabled: bool = True

    # 滞后机制
    enable_hysteresis: bool = True
    score_hysteresis: float = 0.5       # score 差阈值（无量纲）
    dist_hysteresis_m: float = 20.0     # 位置差阈值（米）

    # 阶段锁定
    cruise_allow_update: bool = True
    approach_allow_update: str = "emergency_only"  # true | false | emergency_only
    flare_lock: bool = True

    # 紧急重选
    emergency_margin_mps: float = -0.5  # 可达裕度低于此值触发强制重选
    emergency_cooldown_s: float = 2.0   # 紧急重选后的冷却时间

    # APPROACH 阶段显著改善阈值倍数
    approach_significant_factor: float = 2.0

    # 平滑过渡（可选，暂不实现）
    smooth_transition: bool = False
    smooth_rate_mps: float = 5.0


@dataclass
class TargetState:
    """目标点状态"""
    target: Target
    score: float
    margin: float
    selection: LandingSiteSelection
    update_time: float
    reason: UpdateReason


class TargetUpdatePolicy:
    """目标点更新策略管理器

    根据制导阶段和目标变化幅度决定是否切换目标点，
    实现阵风鲁棒的目标稳定化。
    """

    def __init__(self, config: TargetUpdatePolicyConfig):
        self.config = config
        self._lock = threading.RLock()
        self._current: Optional[TargetState] = None
        self._last_emergency_time: Optional[float] = None

    def reset(self) -> None:
        """重置状态（任务开始时调用）"""
        with self._lock:
            self._current = None
            self._last_emergency_time = None

    def force_update(
        self,
        target: Target,
        score: float,
        margin: float,
        selection: LandingSiteSelection,
        current_time: float,
        reason: UpdateReason = UpdateReason.MANUAL,
    ) -> None:
        """强制设置目标（用于手动模式或初始化）"""
        with self._lock:
            self._current = TargetState(
                target=target,
                score=score,
                margin=margin,
                selection=selection,
                update_time=current_time,
                reason=reason,
            )

    @property
    def current_target(self) -> Optional[Target]:
        """获取当前目标"""
        with self._lock:
            return self._current.target if self._current else None

    @property
    def current_state(self) -> Optional[TargetState]:
        """获取当前目标状态"""
        with self._lock:
            return self._current

    def update(
        self,
        new_selection: LandingSiteSelection,
        phase: GuidancePhase,
        current_time: float,
        current_margin: float,
    ) -> Tuple[Target, UpdateReason]:
        """
        决定是否切换目标点

        Args:
            new_selection: Selector 的新候选结果
            phase: 当前制导阶段
            current_time: 当前时间（秒）
            current_margin: 当前目标的实时可达裕度（m/s）

        Returns:
            (target, reason): 选定的目标点及决策原因
        """
        with self._lock:
            # 策略未启用，直接使用新候选
            if not self.config.enabled:
                self._switch_target(new_selection, current_time, UpdateReason.DISABLED)
                return new_selection.target, UpdateReason.DISABLED

            # 首次调用，初始化
            if self._current is None:
                self._switch_target(new_selection, current_time, UpdateReason.INITIAL)
                return new_selection.target, UpdateReason.INITIAL

            # 更新当前目标的实时裕度（用于状态监控）
            self._current.margin = float(current_margin)

            # 1. 紧急检查：当前目标是否仍可达
            if current_margin < self.config.emergency_margin_mps:
                if self._in_cooldown(current_time):
                    return self._current.target, UpdateReason.EMERGENCY_COOLDOWN
                self._trigger_emergency(new_selection, current_time)
                return new_selection.target, UpdateReason.EMERGENCY_RESELECT

            # 2. FLARE 阶段锁定
            if phase == GuidancePhase.FLARE and self.config.flare_lock:
                return self._current.target, UpdateReason.FLARE_LOCKED

            # 3. APPROACH 阶段处理
            if phase == GuidancePhase.APPROACH:
                return self._handle_approach(new_selection, current_time)

            # 4. CRUISE 阶段处理
            return self._handle_cruise(new_selection, current_time)

    def _handle_approach(
        self,
        new_selection: LandingSiteSelection,
        current_time: float,
    ) -> Tuple[Target, UpdateReason]:
        """处理 APPROACH 阶段的目标更新"""
        mode = self.config.approach_allow_update

        # 完全锁定
        if mode == "false" or mode is False:
            return self._current.target, UpdateReason.APPROACH_LOCKED

        # 仅紧急切换（显著改善）
        if mode == "emergency_only":
            if self._is_significantly_better(new_selection, self.config.approach_significant_factor):
                self._switch_target(new_selection, current_time, UpdateReason.APPROACH_SIGNIFICANT)
                return new_selection.target, UpdateReason.APPROACH_SIGNIFICANT
            return self._current.target, UpdateReason.APPROACH_HYSTERESIS

        # 允许更新（仍受滞后约束）
        if self._should_switch(new_selection):
            self._switch_target(new_selection, current_time, UpdateReason.CRUISE_UPDATE)
            return new_selection.target, UpdateReason.CRUISE_UPDATE
        return self._current.target, UpdateReason.APPROACH_HYSTERESIS

    def _handle_cruise(
        self,
        new_selection: LandingSiteSelection,
        current_time: float,
    ) -> Tuple[Target, UpdateReason]:
        """处理 CRUISE 阶段的目标更新"""
        if not self.config.cruise_allow_update:
            return self._current.target, UpdateReason.CRUISE_HYSTERESIS

        if self.config.enable_hysteresis and not self._should_switch(new_selection):
            return self._current.target, UpdateReason.CRUISE_HYSTERESIS

        self._switch_target(new_selection, current_time, UpdateReason.CRUISE_UPDATE)
        return new_selection.target, UpdateReason.CRUISE_UPDATE

    def _should_switch(self, new_selection: LandingSiteSelection) -> bool:
        """判断是否应该切换目标（滞后判断）"""
        if self._current is None:
            return True

        # score 差异检查
        delta_score = self._current.score - new_selection.score
        if delta_score > self.config.score_hysteresis:
            return True

        # 距离差异检查
        current_xy = self._current.target.position_xy
        new_xy = new_selection.target.position_xy
        delta_dist = float(np.linalg.norm(new_xy - current_xy))
        if delta_dist > self.config.dist_hysteresis_m:
            return True

        return False

    def _is_significantly_better(
        self,
        new_selection: LandingSiteSelection,
        factor: float,
    ) -> bool:
        """判断新目标是否显著优于当前目标"""
        if self._current is None:
            return True

        delta_score = self._current.score - new_selection.score
        threshold = self.config.score_hysteresis * factor
        return delta_score > threshold

    def _switch_target(
        self,
        selection: LandingSiteSelection,
        current_time: float,
        reason: UpdateReason,
    ) -> None:
        """切换到新目标"""
        self._current = TargetState(
            target=selection.target,
            score=selection.score,
            margin=selection.reach_margin_mps,
            selection=selection,
            update_time=current_time,
            reason=reason,
        )

    def _trigger_emergency(
        self,
        selection: LandingSiteSelection,
        current_time: float,
    ) -> None:
        """触发紧急重选"""
        self._last_emergency_time = current_time
        self._switch_target(selection, current_time, UpdateReason.EMERGENCY_RESELECT)

    def _in_cooldown(self, current_time: float) -> bool:
        """检查是否在紧急重选冷却期内"""
        if self._last_emergency_time is None:
            return False
        elapsed = current_time - self._last_emergency_time
        return elapsed < self.config.emergency_cooldown_s

    def get_status_dict(self) -> dict:
        """获取状态字典（用于日志/可视化）"""
        with self._lock:
            if self._current is None:
                return {"target_policy": "no_target"}
            return {
                "target_policy": {
                    "reason": self._current.reason.value,
                    "score": round(self._current.score, 3),
                    "margin_mps": round(self._current.margin, 3),
                    "update_time": round(self._current.update_time, 2),
                    "position_ne": [
                        round(float(self._current.target.p_I[0]), 2),
                        round(float(self._current.target.p_I[1]), 2),
                    ],
                }
            }
