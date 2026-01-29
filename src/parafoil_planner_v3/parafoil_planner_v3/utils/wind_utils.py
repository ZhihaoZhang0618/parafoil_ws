from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple

import numpy as np


class WindInputFrame(str, Enum):
    """Frame for incoming wind vector messages."""

    AUTO = "auto"
    NED = "ned"
    ENU = "enu"


class WindConvention(str, Enum):
    """
    Convention for wind vectors.

    - TO: wind vector points where the air mass goes (velocity of air in inertial frame).
    - FROM: wind vector points where the air mass comes from (meteorological convention).
    """

    TO = "to"
    FROM = "from"


def parse_wind_input_frame(value: str) -> WindInputFrame:
    v = str(value).strip().lower()
    if v in ("auto", ""):
        return WindInputFrame.AUTO
    if v in ("ned",):
        return WindInputFrame.NED
    if v in ("enu",):
        return WindInputFrame.ENU
    raise ValueError(f"Unknown wind.input_frame '{value}' (expected: auto|ned|enu)")


def parse_wind_convention(value: str) -> WindConvention:
    v = str(value).strip().lower()
    if v in ("to", ""):
        return WindConvention.TO
    if v in ("from",):
        return WindConvention.FROM
    raise ValueError(f"Unknown wind.convention '{value}' (expected: to|from)")


def frame_from_frame_id(frame_id: str) -> Optional[WindInputFrame]:
    """
    Best-effort frame detection from message header.frame_id.

    Returns None when unknown.
    """
    fid = str(frame_id or "").strip().lower()
    if fid in ("ned", "nedsim", "nwu_ned"):
        return WindInputFrame.NED
    if fid in ("enu", "world", "map"):
        # Many ROS stacks use ENU for world/map frames.
        return WindInputFrame.ENU
    return None


def enu_to_ned(v_enu: np.ndarray) -> np.ndarray:
    """
    Convert a vector from ENU to NED.

    ENU: [E, N, U]
    NED: [N, E, D]
    """
    v = np.asarray(v_enu, dtype=float).reshape(3)
    return np.array([v[1], v[0], -v[2]], dtype=float)


def to_ned_wind_to(
    wind_vec: np.ndarray,
    *,
    input_frame: WindInputFrame,
    convention: WindConvention,
) -> np.ndarray:
    """
    Convert incoming wind vector to internal standard: NED + wind-to.
    """
    v = np.asarray(wind_vec, dtype=float).reshape(3)
    if input_frame == WindInputFrame.ENU:
        v = enu_to_ned(v)
    elif input_frame == WindInputFrame.NED:
        pass
    else:
        raise ValueError(f"input_frame must be NED or ENU, got: {input_frame}")

    if convention == WindConvention.FROM:
        v = -v
    elif convention == WindConvention.TO:
        pass
    else:
        raise ValueError(f"convention must be TO or FROM, got: {convention}")

    return v.astype(float)


def clip_wind_xy(wind_ned: np.ndarray, *, max_speed_mps: float) -> Tuple[np.ndarray, bool]:
    """
    Clip horizontal wind speed while preserving direction.

    Returns: (clipped_wind, did_clip)
    """
    v = np.asarray(wind_ned, dtype=float).reshape(3).copy()
    max_speed_mps = float(max_speed_mps)
    if max_speed_mps <= 0.0:
        return v, False

    s = float(np.linalg.norm(v[:2]))
    if not np.isfinite(s) or s <= max_speed_mps:
        return v, False

    scale = max_speed_mps / max(s, 1e-9)
    v[0] *= scale
    v[1] *= scale
    return v, True

