from __future__ import annotations

import json
from html import escape
from typing import Iterable, Sequence

import numpy as np


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (int, bool)):
        return str(v)
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _summary_table(rows: Sequence[tuple[str, object]]) -> str:
    tr = "\n".join([f"<tr><th>{escape(str(k))}</th><td>{escape(_fmt(v))}</td></tr>" for k, v in rows])
    return f"<table>{tr}</table>"


def histogram_svg(
    values: Sequence[float],
    bins: int = 20,
    width: int = 480,
    height: int = 200,
    color: str = "#4C78A8",
    x_label: str = "",
) -> str:
    vals = np.asarray(list(values), dtype=float).reshape(-1)
    if vals.size == 0:
        return '<div class="empty">No data</div>'

    counts, edges = np.histogram(vals, bins=int(max(bins, 1)))
    max_count = int(np.max(counts)) if counts.size else 0
    if max_count <= 0:
        return '<div class="empty">No data</div>'

    margin_l, margin_r, margin_t, margin_b = 32, 10, 10, 24
    w = width - margin_l - margin_r
    h = height - margin_t - margin_b
    bar_w = w / max(len(counts), 1)

    rects = []
    for i, c in enumerate(counts):
        bh = 0.0 if max_count == 0 else h * (float(c) / float(max_count))
        x = margin_l + i * bar_w
        y = margin_t + (h - bh)
        title = f"{edges[i]:.3g}–{edges[i+1]:.3g}: {int(c)}"
        rects.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bh:.2f}" '
            f'fill="{color}"><title>{escape(title)}</title></rect>'
        )

    x0 = margin_l
    y0 = margin_t + h
    svg = f"""
<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">
  <line x1="{x0}" y1="{y0}" x2="{x0 + w}" y2="{y0}" stroke="#444" stroke-width="1" />
  <line x1="{x0}" y1="{margin_t}" x2="{x0}" y2="{y0}" stroke="#444" stroke-width="1" />
  {''.join(rects)}
  <text x="{x0}" y="{height - 6}" font-size="10" fill="#555">{escape(f'{edges[0]:.3g}')}</text>
  <text x="{x0 + w - 8}" y="{height - 6}" font-size="10" fill="#555" text-anchor="end">{escape(f'{edges[-1]:.3g}')}</text>
  <text x="{x0 + w/2}" y="{height - 6}" font-size="10" fill="#555" text-anchor="middle">{escape(x_label)}</text>
</svg>
"""
    return svg.strip()


def scatter_svg(
    x: Sequence[float],
    y: Sequence[float],
    width: int = 400,
    height: int = 400,
    color: str = "#E45756",
    x_label: str = "X",
    y_label: str = "Y",
    title: str = "",
    show_origin: bool = True,
    radius: float = 4.0,
) -> str:
    """Generate SVG scatter plot for landing error visualization."""
    x_arr = np.asarray(list(x), dtype=float).reshape(-1)
    y_arr = np.asarray(list(y), dtype=float).reshape(-1)
    if x_arr.size == 0 or y_arr.size == 0:
        return '<div class="empty">No data</div>'

    margin_l, margin_r, margin_t, margin_b = 50, 20, 30, 40
    w = width - margin_l - margin_r
    h = height - margin_t - margin_b

    # Compute bounds (symmetric around origin for landing plots)
    x_max = float(np.max(np.abs(x_arr))) * 1.1 if x_arr.size else 1.0
    y_max = float(np.max(np.abs(y_arr))) * 1.1 if y_arr.size else 1.0
    bound = max(x_max, y_max, 1.0)

    def to_svg_x(val: float) -> float:
        return margin_l + (val + bound) / (2 * bound) * w

    def to_svg_y(val: float) -> float:
        return margin_t + (bound - val) / (2 * bound) * h

    # Grid lines
    grid_lines = []
    for g in [-bound, -bound / 2, 0, bound / 2, bound]:
        gx = to_svg_x(g)
        gy = to_svg_y(g)
        grid_lines.append(f'<line x1="{gx:.1f}" y1="{margin_t}" x2="{gx:.1f}" y2="{margin_t + h}" stroke="#eee" stroke-width="1"/>')
        grid_lines.append(f'<line x1="{margin_l}" y1="{gy:.1f}" x2="{margin_l + w}" y2="{gy:.1f}" stroke="#eee" stroke-width="1"/>')

    # Origin marker
    origin_markers = ""
    if show_origin:
        ox, oy = to_svg_x(0), to_svg_y(0)
        origin_markers = f'''
        <circle cx="{ox:.1f}" cy="{oy:.1f}" r="6" fill="none" stroke="#28a745" stroke-width="2"/>
        <line x1="{ox - 10}" y1="{oy}" x2="{ox + 10}" y2="{oy}" stroke="#28a745" stroke-width="2"/>
        <line x1="{ox}" y1="{oy - 10}" x2="{ox}" y2="{oy + 10}" stroke="#28a745" stroke-width="2"/>
        '''

    # Data points
    points = []
    for i, (px, py) in enumerate(zip(x_arr, y_arr)):
        sx, sy = to_svg_x(float(px)), to_svg_y(float(py))
        dist = float(np.sqrt(px * px + py * py))
        points.append(
            f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="{radius}" fill="{color}" opacity="0.7">'
            f'<title>({float(px):.2f}, {float(py):.2f}) dist={dist:.2f}m</title></circle>'
        )

    # CEP circles
    distances = np.sqrt(x_arr ** 2 + y_arr ** 2)
    cep50 = float(np.percentile(distances, 50))
    cep95 = float(np.percentile(distances, 95))
    cep_circles = ""
    if cep50 < bound:
        r50 = cep50 / bound * (w / 2)
        cep_circles += f'<circle cx="{to_svg_x(0):.1f}" cy="{to_svg_y(0):.1f}" r="{r50:.1f}" fill="none" stroke="#4a90d9" stroke-width="1" stroke-dasharray="4,2"/>'
    if cep95 < bound:
        r95 = cep95 / bound * (w / 2)
        cep_circles += f'<circle cx="{to_svg_x(0):.1f}" cy="{to_svg_y(0):.1f}" r="{r95:.1f}" fill="none" stroke="#ffc107" stroke-width="1" stroke-dasharray="4,2"/>'

    title_html = f'<text x="{width / 2}" y="18" font-size="14" font-weight="600" text-anchor="middle" fill="#333">{escape(title)}</text>' if title else ""

    svg = f"""
<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">
  {title_html}
  {''.join(grid_lines)}
  {origin_markers}
  {cep_circles}
  {''.join(points)}
  <line x1="{margin_l}" y1="{margin_t + h}" x2="{margin_l + w}" y2="{margin_t + h}" stroke="#444" stroke-width="1"/>
  <line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + h}" stroke="#444" stroke-width="1"/>
  <text x="{margin_l + w / 2}" y="{height - 8}" font-size="11" fill="#555" text-anchor="middle">{escape(x_label)}</text>
  <text x="12" y="{margin_t + h / 2}" font-size="11" fill="#555" text-anchor="middle" transform="rotate(-90 12 {margin_t + h / 2})">{escape(y_label)}</text>
  <text x="{margin_l}" y="{height - 24}" font-size="9" fill="#888">{-bound:.0f}</text>
  <text x="{margin_l + w}" y="{height - 24}" font-size="9" fill="#888" text-anchor="end">{bound:.0f}</text>
  <text x="{margin_l - 4}" y="{margin_t + 4}" font-size="9" fill="#888" text-anchor="end">{bound:.0f}</text>
  <text x="{margin_l - 4}" y="{margin_t + h}" font-size="9" fill="#888" text-anchor="end">{-bound:.0f}</text>
  <text x="{width - 10}" y="{margin_t + 16}" font-size="9" fill="#4a90d9">CEP50={cep50:.1f}m</text>
  <text x="{width - 10}" y="{margin_t + 28}" font-size="9" fill="#ffc107">CEP95={cep95:.1f}m</text>
</svg>
"""
    return svg.strip()


def trajectory_svg(
    x: Sequence[float],
    y: Sequence[float],
    width: int = 500,
    height: int = 400,
    color: str = "#4a90d9",
    x_label: str = "East (m)",
    y_label: str = "North (m)",
    target_xy: tuple[float, float] | None = None,
    start_marker: bool = True,
) -> str:
    """Generate SVG for XY trajectory plot."""
    x_arr = np.asarray(list(x), dtype=float).reshape(-1)
    y_arr = np.asarray(list(y), dtype=float).reshape(-1)
    if x_arr.size < 2:
        return '<div class="empty">Insufficient data</div>'

    margin_l, margin_r, margin_t, margin_b = 50, 20, 20, 40
    w = width - margin_l - margin_r
    h = height - margin_t - margin_b

    x_min, x_max = float(np.min(x_arr)), float(np.max(x_arr))
    y_min, y_max = float(np.min(y_arr)), float(np.max(y_arr))
    if target_xy:
        x_min = min(x_min, target_xy[0])
        x_max = max(x_max, target_xy[0])
        y_min = min(y_min, target_xy[1])
        y_max = max(y_max, target_xy[1])

    x_range = max(x_max - x_min, 1.0) * 1.1
    y_range = max(y_max - y_min, 1.0) * 1.1
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    def to_svg_x(val: float) -> float:
        return margin_l + (val - x_center + x_range / 2) / x_range * w

    def to_svg_y(val: float) -> float:
        return margin_t + h - (val - y_center + y_range / 2) / y_range * h

    # Build path
    path_points = " ".join([f"{'M' if i == 0 else 'L'} {to_svg_x(float(px)):.1f} {to_svg_y(float(py)):.1f}" for i, (px, py) in enumerate(zip(x_arr, y_arr))])

    # Target marker
    target_marker = ""
    if target_xy:
        tx, ty = to_svg_x(target_xy[0]), to_svg_y(target_xy[1])
        target_marker = f'''
        <circle cx="{tx:.1f}" cy="{ty:.1f}" r="8" fill="none" stroke="#28a745" stroke-width="2"/>
        <line x1="{tx - 12}" y1="{ty}" x2="{tx + 12}" y2="{ty}" stroke="#28a745" stroke-width="2"/>
        <line x1="{tx}" y1="{ty - 12}" x2="{tx}" y2="{ty + 12}" stroke="#28a745" stroke-width="2"/>
        <text x="{tx + 12}" y="{ty - 8}" font-size="10" fill="#28a745">Target</text>
        '''

    # Start marker
    start = ""
    if start_marker and x_arr.size > 0:
        sx, sy = to_svg_x(float(x_arr[0])), to_svg_y(float(y_arr[0]))
        start = f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="5" fill="#E45756"/><text x="{sx + 8}" y="{sy - 4}" font-size="9" fill="#E45756">Start</text>'

    # End marker
    end_marker = ""
    if x_arr.size > 0:
        ex, ey = to_svg_x(float(x_arr[-1])), to_svg_y(float(y_arr[-1]))
        end_marker = f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="4" fill="{color}"/>'

    svg = f"""
<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">
  <path d="{path_points}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>
  {target_marker}
  {start}
  {end_marker}
  <line x1="{margin_l}" y1="{margin_t + h}" x2="{margin_l + w}" y2="{margin_t + h}" stroke="#444" stroke-width="1"/>
  <line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + h}" stroke="#444" stroke-width="1"/>
  <text x="{margin_l + w / 2}" y="{height - 8}" font-size="11" fill="#555" text-anchor="middle">{escape(x_label)}</text>
  <text x="12" y="{margin_t + h / 2}" font-size="11" fill="#555" text-anchor="middle" transform="rotate(-90 12 {margin_t + h / 2})">{escape(y_label)}</text>
  <text x="{margin_l}" y="{height - 24}" font-size="9" fill="#888">{x_center - x_range / 2:.0f}</text>
  <text x="{margin_l + w}" y="{height - 24}" font-size="9" fill="#888" text-anchor="end">{x_center + x_range / 2:.0f}</text>
</svg>
"""
    return svg.strip()


def timeseries_svg(
    t: Sequence[float],
    values: Sequence[float],
    width: int = 600,
    height: int = 200,
    color: str = "#4a90d9",
    y_label: str = "",
    y_min: float | None = None,
    y_max: float | None = None,
) -> str:
    """Generate SVG for time series plot (control, altitude, etc.)."""
    t_arr = np.asarray(list(t), dtype=float).reshape(-1)
    v_arr = np.asarray(list(values), dtype=float).reshape(-1)
    if t_arr.size < 2:
        return '<div class="empty">Insufficient data</div>'

    margin_l, margin_r, margin_t, margin_b = 50, 20, 15, 30
    w = width - margin_l - margin_r
    h = height - margin_t - margin_b

    t_min, t_max = float(np.min(t_arr)), float(np.max(t_arr))
    v_min_data = float(np.min(v_arr))
    v_max_data = float(np.max(v_arr))
    v_min_plot = y_min if y_min is not None else v_min_data - 0.05 * max(abs(v_max_data - v_min_data), 0.1)
    v_max_plot = y_max if y_max is not None else v_max_data + 0.05 * max(abs(v_max_data - v_min_data), 0.1)
    v_range = max(v_max_plot - v_min_plot, 0.01)
    t_range = max(t_max - t_min, 0.01)

    def to_svg_x(val: float) -> float:
        return margin_l + (val - t_min) / t_range * w

    def to_svg_y(val: float) -> float:
        return margin_t + h - (val - v_min_plot) / v_range * h

    path_points = " ".join([f"{'M' if i == 0 else 'L'} {to_svg_x(float(ti)):.1f} {to_svg_y(float(vi)):.1f}" for i, (ti, vi) in enumerate(zip(t_arr, v_arr))])

    svg = f"""
<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">
  <path d="{path_points}" fill="none" stroke="{color}" stroke-width="1.5"/>
  <line x1="{margin_l}" y1="{margin_t + h}" x2="{margin_l + w}" y2="{margin_t + h}" stroke="#444" stroke-width="1"/>
  <line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + h}" stroke="#444" stroke-width="1"/>
  <text x="{margin_l + w / 2}" y="{height - 6}" font-size="10" fill="#555" text-anchor="middle">Time (s)</text>
  <text x="12" y="{margin_t + h / 2}" font-size="10" fill="#555" text-anchor="middle" transform="rotate(-90 12 {margin_t + h / 2})">{escape(y_label)}</text>
  <text x="{margin_l}" y="{height - 16}" font-size="9" fill="#888">{t_min:.1f}</text>
  <text x="{margin_l + w}" y="{height - 16}" font-size="9" fill="#888" text-anchor="end">{t_max:.1f}</text>
  <text x="{margin_l - 4}" y="{margin_t + 4}" font-size="9" fill="#888" text-anchor="end">{v_max_plot:.2g}</text>
  <text x="{margin_l - 4}" y="{margin_t + h}" font-size="9" fill="#888" text-anchor="end">{v_min_plot:.2g}</text>
</svg>
"""
    return svg.strip()


def altitude_profile_svg(
    distance: Sequence[float],
    altitude: Sequence[float],
    width: int = 600,
    height: int = 250,
    color: str = "#59A14F",
    target_altitude: float = 0.0,
) -> str:
    """Generate SVG for altitude vs distance profile."""
    d_arr = np.asarray(list(distance), dtype=float).reshape(-1)
    a_arr = np.asarray(list(altitude), dtype=float).reshape(-1)
    if d_arr.size < 2:
        return '<div class="empty">Insufficient data</div>'

    margin_l, margin_r, margin_t, margin_b = 50, 20, 15, 35
    w = width - margin_l - margin_r
    h = height - margin_t - margin_b

    d_min, d_max = float(np.min(d_arr)), float(np.max(d_arr))
    a_min = min(float(np.min(a_arr)), target_altitude - 5)
    a_max = float(np.max(a_arr)) * 1.05
    d_range = max(d_max - d_min, 1.0)
    a_range = max(a_max - a_min, 1.0)

    def to_svg_x(val: float) -> float:
        return margin_l + (val - d_min) / d_range * w

    def to_svg_y(val: float) -> float:
        return margin_t + h - (val - a_min) / a_range * h

    path_points = " ".join([f"{'M' if i == 0 else 'L'} {to_svg_x(float(di)):.1f} {to_svg_y(float(ai)):.1f}" for i, (di, ai) in enumerate(zip(d_arr, a_arr))])

    # Target line
    target_y = to_svg_y(target_altitude)
    target_line = f'<line x1="{margin_l}" y1="{target_y:.1f}" x2="{margin_l + w}" y2="{target_y:.1f}" stroke="#28a745" stroke-width="1" stroke-dasharray="4,2"/>'

    # Fill area under curve
    fill_path = path_points + f" L {to_svg_x(d_max):.1f} {to_svg_y(a_min):.1f} L {to_svg_x(d_min):.1f} {to_svg_y(a_min):.1f} Z"

    svg = f"""
<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">
  <path d="{fill_path}" fill="{color}" fill-opacity="0.2"/>
  <path d="{path_points}" fill="none" stroke="{color}" stroke-width="2"/>
  {target_line}
  <line x1="{margin_l}" y1="{margin_t + h}" x2="{margin_l + w}" y2="{margin_t + h}" stroke="#444" stroke-width="1"/>
  <line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + h}" stroke="#444" stroke-width="1"/>
  <text x="{margin_l + w / 2}" y="{height - 8}" font-size="11" fill="#555" text-anchor="middle">Ground Distance (m)</text>
  <text x="12" y="{margin_t + h / 2}" font-size="11" fill="#555" text-anchor="middle" transform="rotate(-90 12 {margin_t + h / 2})">Altitude (m)</text>
  <text x="{margin_l}" y="{height - 20}" font-size="9" fill="#888">{d_min:.0f}</text>
  <text x="{margin_l + w}" y="{height - 20}" font-size="9" fill="#888" text-anchor="end">{d_max:.0f}</text>
  <text x="{margin_l - 4}" y="{margin_t + 4}" font-size="9" fill="#888" text-anchor="end">{a_max:.0f}</text>
  <text x="{margin_l - 4}" y="{margin_t + h}" font-size="9" fill="#888" text-anchor="end">{a_min:.0f}</text>
  <text x="{margin_l + w - 4}" y="{target_y - 4}" font-size="9" fill="#28a745" text-anchor="end">Target</text>
</svg>
"""
    return svg.strip()


def bar_svg(
    labels: Sequence[str],
    values: Sequence[float],
    width: int = 480,
    height: int = 220,
    color: str = "#59A14F",
) -> str:
    if not labels:
        return '<div class="empty">No data</div>'
    vals = np.asarray(list(values), dtype=float).reshape(-1)
    max_val = float(np.max(vals)) if vals.size else 0.0
    if max_val <= 0.0:
        return '<div class="empty">No data</div>'

    margin_l, margin_r, margin_t, margin_b = 60, 10, 10, 24
    w = width - margin_l - margin_r
    h = height - margin_t - margin_b
    bar_h = h / max(len(labels), 1)

    rects = []
    texts = []
    for i, (lab, v) in enumerate(zip(labels, vals)):
        bw = 0.0 if max_val == 0 else w * (float(v) / max_val)
        x = margin_l
        y = margin_t + i * bar_h + bar_h * 0.15
        rects.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bw:.2f}" height="{bar_h*0.7:.2f}" fill="{color}">'
            f'<title>{escape(str(lab))}: {float(v):.3g}</title></rect>'
        )
        texts.append(
            f'<text x="{x - 6}" y="{y + bar_h*0.45:.2f}" font-size="10" fill="#555" text-anchor="end">{escape(str(lab))}</text>'
        )

    svg = f"""
<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">
  {''.join(rects)}
  {''.join(texts)}
  <line x1="{margin_l}" y1="{margin_t + h}" x2="{margin_l + w}" y2="{margin_t + h}" stroke="#444" stroke-width="1" />
</svg>
"""
    return svg.strip()


def _bounds(vals: Sequence[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 1.0
    vmin = float(min(vals))
    vmax = float(max(vals))
    if abs(vmax - vmin) < 1e-9:
        return vmin - 1.0, vmax + 1.0
    return vmin, vmax


def line_svg(
    x: Sequence[float],
    y: Sequence[float],
    width: int = 480,
    height: int = 220,
    color: str = "#4C78A8",
    x_label: str = "",
    y_label: str = "",
) -> str:
    if not x or not y:
        return '<div class="empty">No data</div>'
    xs = list(map(float, x))
    ys = list(map(float, y))
    xmin, xmax = _bounds(xs)
    ymin, ymax = _bounds(ys)
    margin_l, margin_r, margin_t, margin_b = 36, 10, 10, 28
    w = width - margin_l - margin_r
    h = height - margin_t - margin_b
    pts = []
    for xi, yi in zip(xs, ys):
        px = margin_l + (xi - xmin) / max(xmax - xmin, 1e-9) * w
        py = margin_t + (ymax - yi) / max(ymax - ymin, 1e-9) * h
        pts.append(f"{px:.2f},{py:.2f}")
    poly = " ".join(pts)
    svg = f"""
<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">
  <polyline fill="none" stroke="{color}" stroke-width="1.5" points="{poly}" />
  <line x1="{margin_l}" y1="{margin_t + h}" x2="{margin_l + w}" y2="{margin_t + h}" stroke="#444" stroke-width="1" />
  <line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + h}" stroke="#444" stroke-width="1" />
  <text x="{margin_l + w/2}" y="{height - 6}" font-size="10" fill="#555" text-anchor="middle">{escape(x_label)}</text>
  <text x="6" y="{margin_t + h/2}" font-size="10" fill="#555" text-anchor="start">{escape(y_label)}</text>
</svg>
"""
    return svg.strip()


def multi_line_svg(
    series: Sequence[dict],
    width: int = 480,
    height: int = 220,
    x_label: str = "",
    y_label: str = "",
) -> str:
    if not series:
        return '<div class="empty">No data</div>'
    all_x = [float(x) for s in series for x in s.get("x", [])]
    all_y = [float(y) for s in series for y in s.get("y", [])]
    if not all_x or not all_y:
        return '<div class="empty">No data</div>'
    xmin, xmax = _bounds(all_x)
    ymin, ymax = _bounds(all_y)
    margin_l, margin_r, margin_t, margin_b = 36, 10, 10, 28
    w = width - margin_l - margin_r
    h = height - margin_t - margin_b
    polys = []
    legends = []
    for i, s in enumerate(series):
        xs = list(map(float, s.get("x", [])))
        ys = list(map(float, s.get("y", [])))
        if not xs or not ys:
            continue
        pts = []
        for xi, yi in zip(xs, ys):
            px = margin_l + (xi - xmin) / max(xmax - xmin, 1e-9) * w
            py = margin_t + (ymax - yi) / max(ymax - ymin, 1e-9) * h
            pts.append(f"{px:.2f},{py:.2f}")
        color = s.get("color", ["#4C78A8", "#F28E2B", "#59A14F", "#E15759"][i % 4])
        polys.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{" ".join(pts)}" />')
        label = s.get("label")
        if label:
            legends.append((label, color))

    legend_html = ""
    if legends:
        items = []
        x0 = margin_l + 6
        y0 = margin_t + 12
        for i, (lab, col) in enumerate(legends):
            yy = y0 + i * 12
            items.append(f'<rect x="{x0}" y="{yy-8}" width="8" height="8" fill="{col}" />')
            items.append(f'<text x="{x0+12}" y="{yy-1}" font-size="10" fill="#555">{escape(str(lab))}</text>')
        legend_html = "\n".join(items)

    svg = f"""
<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">
  {''.join(polys)}
  {legend_html}
  <line x1="{margin_l}" y1="{margin_t + h}" x2="{margin_l + w}" y2="{margin_t + h}" stroke="#444" stroke-width="1" />
  <line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + h}" stroke="#444" stroke-width="1" />
  <text x="{margin_l + w/2}" y="{height - 6}" font-size="10" fill="#555" text-anchor="middle">{escape(x_label)}</text>
  <text x="6" y="{margin_t + h/2}" font-size="10" fill="#555" text-anchor="start">{escape(y_label)}</text>
</svg>
"""
    return svg.strip()


def xy_path_svg(
    paths: Sequence[dict],
    width: int = 480,
    height: int = 240,
    show_target: bool = True,
    target_xy: tuple[float, float] = (0.0, 0.0),
) -> str:
    if not paths:
        return '<div class="empty">No data</div>'
    all_x = [float(p) for s in paths for p in [pt[0] for pt in s.get("xy", [])]]
    all_y = [float(p) for s in paths for p in [pt[1] for pt in s.get("xy", [])]]
    if not all_x or not all_y:
        return '<div class="empty">No data</div>'
    xmin, xmax = _bounds(all_x)
    ymin, ymax = _bounds(all_y)
    margin_l, margin_r, margin_t, margin_b = 30, 10, 10, 20
    w = width - margin_l - margin_r
    h = height - margin_t - margin_b
    # Keep aspect ratio
    sx = w / max(xmax - xmin, 1e-9)
    sy = h / max(ymax - ymin, 1e-9)
    s = min(sx, sy)
    x_mid = 0.5 * (xmin + xmax)
    y_mid = 0.5 * (ymin + ymax)

    def _map(pt):
        px = margin_l + (pt[0] - x_mid) * s + w / 2.0
        py = margin_t + (y_mid - pt[1]) * s + h / 2.0
        return px, py

    polys = []
    legends = []
    for i, sdata in enumerate(paths):
        xy = sdata.get("xy", [])
        if not xy:
            continue
        pts = []
        for pt in xy:
            px, py = _map(pt)
            pts.append(f"{px:.2f},{py:.2f}")
        color = sdata.get("color", ["#4C78A8", "#F28E2B", "#59A14F", "#E15759"][i % 4])
        polys.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{" ".join(pts)}" />')
        label = sdata.get("label")
        if label:
            legends.append((label, color))

    markers = ""
    if show_target:
        tx, ty = _map(target_xy)
        markers = f'<circle cx="{tx:.2f}" cy="{ty:.2f}" r="4" fill="#000"><title>Target</title></circle>'

    legend_html = ""
    if legends:
        items = []
        x0 = margin_l + 6
        y0 = margin_t + 12
        for i, (lab, col) in enumerate(legends):
            yy = y0 + i * 12
            items.append(f'<rect x="{x0}" y="{yy-8}" width="8" height="8" fill="{col}" />')
            items.append(f'<text x="{x0+12}" y="{yy-1}" font-size="10" fill="#555">{escape(str(lab))}</text>')
        legend_html = "\n".join(items)

    svg = f"""
<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">
  {''.join(polys)}
  {markers}
  {legend_html}
</svg>
"""
    return svg.strip()


def scatter_svg(
    points: Sequence[tuple[float, float]],
    width: int = 480,
    height: int = 240,
    color: str = "#59A14F",
    target_xy: tuple[float, float] | None = (0.0, 0.0),
) -> str:
    if not points:
        return '<div class="empty">No data</div>'
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    xmin, xmax = _bounds(xs)
    ymin, ymax = _bounds(ys)
    margin_l, margin_r, margin_t, margin_b = 30, 10, 10, 20
    w = width - margin_l - margin_r
    h = height - margin_t - margin_b
    sx = w / max(xmax - xmin, 1e-9)
    sy = h / max(ymax - ymin, 1e-9)
    s = min(sx, sy)
    x_mid = 0.5 * (xmin + xmax)
    y_mid = 0.5 * (ymin + ymax)

    def _map(pt):
        px = margin_l + (pt[0] - x_mid) * s + w / 2.0
        py = margin_t + (y_mid - pt[1]) * s + h / 2.0
        return px, py

    dots = []
    for pt in points:
        px, py = _map(pt)
        dots.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="2.2" fill="{color}" />')

    target = ""
    if target_xy is not None:
        tx, ty = _map(target_xy)
        target = f'<circle cx="{tx:.2f}" cy="{ty:.2f}" r="4" fill="#000"><title>Target</title></circle>'

    svg = f"""
<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">
  {''.join(dots)}
  {target}
</svg>
"""
    return svg.strip()


def render_report(
    title: str,
    summary_rows: Sequence[tuple[str, object]],
    charts: Sequence[dict],
    payload: dict,
    subtitle: str | None = None,
) -> str:
    chart_html = ""
    for c in charts:
        c_title = escape(str(c.get("title", "Chart")))
        c_svg = c.get("svg", "")
        c_caption = escape(str(c.get("caption", ""))) if c.get("caption") else ""
        chart_html += f"""
        <div class="card">
          <div class="card-title">{c_title}</div>
          <div class="chart">{c_svg}</div>
          {f'<div class="caption">{c_caption}</div>' if c_caption else ''}
        </div>
        """

    raw = escape(json.dumps(payload, indent=2, ensure_ascii=False))
    subtitle_html = f"<p class='subtitle'>{escape(subtitle)}</p>" if subtitle else ""
    return f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>{escape(title)}</title>
    <style>
      body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; color: #111; }}
      h1 {{ margin-bottom: 4px; }}
      .subtitle {{ color: #555; margin-top: 0; }}
      table {{ border-collapse: collapse; margin: 12px 0 18px; }}
      td, th {{ border: 1px solid #ddd; padding: 8px 10px; }}
      th {{ background: #f5f5f5; text-align: left; }}
      .grid {{ display: flex; flex-wrap: wrap; gap: 12px; }}
      .card {{ border: 1px solid #eee; border-radius: 8px; padding: 10px 12px; background: #fff; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }}
      .card-title {{ font-weight: 600; margin-bottom: 8px; }}
      .chart {{ overflow-x: auto; }}
      .caption {{ color: #666; font-size: 12px; margin-top: 6px; }}
      details {{ margin-top: 16px; }}
      pre {{ background: #f7f7f7; padding: 12px; overflow: auto; }}
      .empty {{ color: #777; font-size: 12px; }}
    </style>
  </head>
  <body>
    <h1>{escape(title)}</h1>
    {subtitle_html}
    <h2>Summary</h2>
    {_summary_table(summary_rows)}
    <h2>Charts</h2>
    <div class="grid">
      {chart_html}
    </div>
    <details>
      <summary>Raw JSON</summary>
      <pre>{raw}</pre>
    </details>
  </body>
</html>
"""
