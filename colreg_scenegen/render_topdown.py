from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

from .sim_kinematics import VesselState
from .spec import SceneSpec


@dataclass(frozen=True)
class TopDownRenderConfig:
    size_px: int = 768
    dpi: int = 120
    frame_offsets: Tuple[int, int, int, int] = (-90, -60, -30, -1)
    range_m: float = 2500.0
    show_trails: bool = True
    trail_len_s: float = 60.0
    show_speed_vector: bool = True
    default_length_m: float = 120.0
    default_width_ratio: float = 0.22
    display_scale: float = 5.0


def _heading_to_unit(heading_deg: float) -> np.ndarray:
    rad = np.deg2rad(heading_deg)
    return np.array([np.sin(rad), np.cos(rad)], dtype=float)


def _pick_frame_indices(n: int, dt_s: float, offsets: Tuple[int, int, int, int]) -> List[int]:
    last = n - 1
    idxs = []
    for off in offsets:
        if off == -1:
            idxs.append(last)
        else:
            k = last + int(off / dt_s)
            k = max(0, min(last, k))
            idxs.append(k)
    return idxs


def _ship_polygon(x: float, y: float, heading_deg: float, length_m: float, width_m: float):
    L = float(length_m)
    W = float(width_m)

    rect = np.array(
        [
            [ W / 2,  L / 2],
            [-W / 2,  L / 2],
            [-W / 2, -L / 2],
            [ W / 2, -L / 2],
        ],
        dtype=float,
    )
    tri = np.array(
        [
            [0.0, L / 2 + 0.35 * L],
            [ W / 2, L / 2],
            [-W / 2, L / 2],
        ],
        dtype=float,
    )

    ang = np.deg2rad(heading_deg)
    u = np.array([np.sin(ang), np.cos(ang)], dtype=float)
    r = np.array([u[1], -u[0]], dtype=float)
    R = np.stack([r, u], axis=1)

    rect_w = rect @ R.T + np.array([x, y])
    tri_w = tri @ R.T + np.array([x, y])
    return rect_w, tri_w


def _length_lookup(scene: Optional[SceneSpec], vessel_id: str, default_length_m: float) -> float:
    if scene is None:
        return default_length_m
    if vessel_id == scene.ownship.vessel_id:
        return float(scene.ownship.length_m)
    for t in scene.targets:
        if t.vessel_id == vessel_id:
            return float(t.length_m)
    return default_length_m


def _label_offset(p: np.ndarray, own_center: np.ndarray, base: float = 55.0) -> tuple[float, float]:
    """
    Put label on the side farther from ownship to reduce overlap.
    """
    rel = p - own_center
    dx = base if rel[0] >= 0 else -base
    dy = base if rel[1] >= 0 else -base

    # avoid tiny offsets when very close
    if abs(rel[0]) < 120.0:
        dx *= 1.4
    if abs(rel[1]) < 120.0:
        dy *= 1.4

    return dx, dy


def render_topdown_grid(
    tracks: Dict[str, List[VesselState]],
    ownship_id: str,
    cfg: Optional[TopDownRenderConfig] = None,
    scene: Optional[SceneSpec] = None,
) -> np.ndarray:
    cfg = cfg or TopDownRenderConfig()
    own_track = tracks[ownship_id]
    n = len(own_track)
    dt_s = own_track[1].t_s - own_track[0].t_s if n >= 2 else 1.0
    frame_idxs = _pick_frame_indices(n, dt_s, cfg.frame_offsets)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(cfg.size_px / cfg.dpi, cfg.size_px / cfg.dpi),
        dpi=cfg.dpi,
    )
    axes = axes.flatten()

    for ax, k in zip(axes, frame_idxs):
        own = own_track[k]
        center = np.array([own.x_m, own.y_m], dtype=float)

        ax.set_xlim(center[0] - cfg.range_m, center[0] + cfg.range_m)
        ax.set_ylim(center[1] - cfg.range_m, center[1] + cfg.range_m)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.5, alpha=0.35)

        for vid, tr in tracks.items():
            st = tr[k]
            p = np.array([st.x_m, st.y_m], dtype=float)

            L_real = _length_lookup(scene, vid, cfg.default_length_m)
            L = L_real * cfg.display_scale
            W = max(24.0, cfg.default_width_ratio * L)

            rect_w, tri_w = _ship_polygon(p[0], p[1], st.heading_deg, length_m=L, width_m=W)

            if vid == ownship_id:
                rect_alpha = 0.60
                tri_alpha = 0.90
                lw = 1.8
                z = 4
            else:
                rect_alpha = 0.48
                tri_alpha = 0.72
                lw = 1.4
                z = 3

            ax.fill(
                rect_w[:, 0], rect_w[:, 1],
                alpha=rect_alpha,
                edgecolor="black",
                linewidth=lw,
                zorder=z,
            )
            ax.fill(
                tri_w[:, 0], tri_w[:, 1],
                alpha=tri_alpha,
                edgecolor="black",
                linewidth=lw,
                zorder=z + 0.1,
            )

            if cfg.show_speed_vector:
                u = _heading_to_unit(st.heading_deg)
                v = u * st.speed_mps
                ax.arrow(
                    p[0], p[1],
                    v[0] * 35, v[1] * 35,
                    head_width=max(20.0, 0.10 * W),
                    length_includes_head=True,
                    alpha=0.5,
                    zorder=z + 0.2,
                )

            if cfg.show_trails and k > 1:
                trail_len = int(cfg.trail_len_s / dt_s)
                a = max(0, k - trail_len)
                xs = [tr[i].x_m for i in range(a, k + 1)]
                ys = [tr[i].y_m for i in range(a, k + 1)]
                ax.plot(xs, ys, linewidth=1.0, alpha=0.55, zorder=1)

            if vid == ownship_id:
                dx, dy = 70.0, 70.0
            else:
                dx, dy = _label_offset(p, center, base=60.0)

            ax.text(p[0] + dx, p[1] + dy, vid, fontsize=8, zorder=z + 0.3)

        ax.set_title(f"t={own.t_s:.0f}s")
        ax.text(
            0.98, 0.98,
            f"ENU frame (unit: m)\ndisplay_scale={cfg.display_scale:.1f}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", alpha=0.35),
        )

    fig.tight_layout(pad=0.5)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    if hasattr(fig.canvas, "buffer_rgba"):
        buf = np.asarray(fig.canvas.buffer_rgba())
        img = buf[:, :, :3].copy()
    else:
        argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
        img = argb[:, :, 1:4].copy()

    plt.close(fig)
    return img