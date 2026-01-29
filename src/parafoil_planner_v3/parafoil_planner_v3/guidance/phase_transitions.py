from __future__ import annotations

from parafoil_planner_v3.types import GuidancePhase, PhaseTransition, State, Target, Wind

from .phase_manager import PhaseManager


def update_phase(phase_manager: PhaseManager, state: State, target: Target, wind: Wind) -> PhaseTransition:
    """Compatibility wrapper around `PhaseManager.update()`."""
    return phase_manager.update(state, target, wind)


__all__ = ["update_phase", "GuidancePhase"]

