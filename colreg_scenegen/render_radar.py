from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

from .sim_kinematics import VesselState
from .geometry import _relative_bearing


@dataclass(frozen=True)
class RadarRenderConfig:
    size_px: int = 768
    dpi: int = 120
    frame_offsets: Tuple[int, int, int, int] = (-90, -60, -30, -1)
    rmax_m: float = 3500.0
    # measurement noise
    range_sigma_m: float = 15.0
    bearing_sigma_deg: float = 1.0
    # clutter
    clutter_points: int = 60
    seed: int = 0
    # blob appearance
    target_sigma_m: float = 30.0
    clutter_sigma_m: float = 45.0
    target_amp: float = 1.0
    clutter_amp: float = 0.25
    grid_n: int = 512  # intensity map resolution


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


def render_radar_ppi_grid(
    tracks: Dict[str, List[VesselState]],
    ownship_id: str,
    cfg: Optional[RadarRenderConfig] = None,
) -> np.ndarray:
    cfg = cfg or RadarRenderConfig()
    rng = np.random.default_rng(cfg.seed)

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

    grid_n = int(cfg.grid_n)
    extent = (-cfg.rmax_m, cfg.rmax_m, -cfg.rmax_m, cfg.rmax_m)

    def add_gaussian(img: np.ndarray, x0: float, y0: float, amp: float, sigma_m: float) -> None:
        sx = sigma_m / (2 * cfg.rmax_m) * grid_n
        sx = max(2.0, sx)
        ix = int((x0 - extent[0]) / (extent[1] - extent[0]) * (grid_n - 1))
        iy = int((y0 - extent[2]) / (extent[3] - extent[2]) * (grid_n - 1))
        rad = int(max(6, 3 * sx))

        x1, x2 = max(0, ix - rad), min(grid_n - 1, ix + rad)
        y1, y2 = max(0, iy - rad), min(grid_n - 1, iy + rad)
        if x1 >= x2 or y1 >= y2:
            return

        xx = np.arange(x1, x2 + 1) - ix
        yy = np.arange(y1, y2 + 1) - iy
        X, Y = np.meshgrid(xx, yy)
        g = amp * np.exp(-(X * X + Y * Y) / (2 * sx * sx))
        img[y1 : y2 + 1, x1 : x2 + 1] += g

    for ax, k in zip(axes, frame_idxs):
        own = own_track[k]
        img = np.zeros((grid_n, grid_n), dtype=np.float32)

        # clutter as weak blobs
        for _ in range(cfg.clutter_points):
            cr = cfg.rmax_m * np.sqrt(rng.uniform(0, 1))
            ct = rng.uniform(0, 2 * np.pi)
            cx = cr * np.cos(ct)
            cy = cr * np.sin(ct)
            add_gaussian(img, cx, cy, amp=cfg.clutter_amp, sigma_m=cfg.clutter_sigma_m)

        # targets as stronger blobs
        for vid, tr in tracks.items():
            if vid == ownship_id:
                continue
            tgt = tr[k]
            rel = np.array([tgt.x_m - own.x_m, tgt.y_m - own.y_m], dtype=float)
            r = float(np.linalg.norm(rel))
            if r > cfg.rmax_m:
                continue

            bearing_deg = _relative_bearing(own.heading_deg, rel)  # 0 ahead, 90 starboard
            ang = np.deg2rad(bearing_deg)

            # noise
            r_n = max(0.0, r + rng.normal(0, cfg.range_sigma_m))
            b_n = ang + np.deg2rad(rng.normal(0, cfg.bearing_sigma_deg))

            # radar coords: x right (starboard), y up (ahead)
            x = r_n * np.sin(b_n)
            y = r_n * np.cos(b_n)

            add_gaussian(img, x, y, amp=cfg.target_amp, sigma_m=cfg.target_sigma_m)
            ax.text(x + 60, y + 60, vid, fontsize=8)

        # display intensity
        vmax = np.percentile(img, 99.5) if img.max() > 0 else 1.0
        ax.imshow(np.clip(img, 0, vmax), extent=extent, origin="lower", alpha=0.95)

        # rings/grid
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-cfg.rmax_m, cfg.rmax_m)
        ax.set_ylim(-cfg.rmax_m, cfg.rmax_m)
        ax.grid(True, linewidth=0.5, alpha=0.35)
        for rr in [cfg.rmax_m * 0.25, cfg.rmax_m * 0.5, cfg.rmax_m * 0.75, cfg.rmax_m]:
            ax.add_patch(plt.Circle((0, 0), rr, fill=False, linewidth=0.8, alpha=0.6))

        ax.set_title(f"PPI t={own.t_s:.0f}s")
        ax.text(
            0.98, 0.98,
            "Ahead ↑  Starboard →\nunit: m",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", alpha=0.35),
        )

    fig.tight_layout(pad=0.5)

    # backend-agnostic canvas -> RGB
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    if hasattr(fig.canvas, "buffer_rgba"):
        buf = np.asarray(fig.canvas.buffer_rgba())
        out = buf[:, :, :3].copy()
    else:
        argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
        out = argb[:, :, 1:4].copy()

    plt.close(fig)
    return out