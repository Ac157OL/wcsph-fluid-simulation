"""Realtime point visualization, PLY serialization, and report snapshots."""
from pathlib import Path

import numpy as np


def export_ply(positions, filename):
    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(positions)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(stream, positions, fmt="%.7f %.7f %.7f")


def _style_axis(ax):
    ax.set(xlim=(-3, 3), ylim=(-3, 3), zlim=(-3, 3), xlabel="x", ylabel="z", zlabel="y")
    ax.set_box_aspect((1, 1, 0.85))
    ax.view_init(elev=22, azim=-58)
    ax.set_facecolor("#071425")
    ax.figure.patch.set_facecolor("#071425")
    ax.tick_params(colors="#9bb8d6", labelsize=7)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color("#b7d5ef")


def save_particle_frame(positions, filename, sim_time=0.0):
    import matplotlib.pyplot as plt

    p = np.asarray(positions)
    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    _style_axis(ax)
    ax.scatter(p[:, 0], p[:, 2], p[:, 1], s=9, c=p[:, 1], cmap="Blues_r",
               vmin=-2.5, vmax=2.5, alpha=0.9, linewidths=0)
    ax.set_title(f"WCSPH particle view  |  t = {sim_time:.3f} s", color="white", pad=14)
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


class LiveParticleRenderer:
    """Small interactive Matplotlib renderer fed directly from Particle.position."""
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.plt = self.ax = self.scatter = None
        if enabled:
            import matplotlib.pyplot as plt
            plt.ion()
            self.plt = plt
            self.fig = plt.figure(figsize=(8, 6))
            self.ax = self.fig.add_subplot(111, projection="3d")
            _style_axis(self.ax)

    def update(self, positions, sim_time):
        if not self.enabled:
            return
        p = np.asarray(positions)
        if self.scatter is not None:
            self.scatter.remove()
        self.scatter = self.ax.scatter(p[:, 0], p[:, 2], p[:, 1], s=8, c=p[:, 1],
                                      cmap="Blues_r", alpha=0.9, linewidths=0)
        self.ax.set_title(f"WCSPH realtime  |  t = {sim_time:.3f} s", color="white")
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def close(self):
        if self.enabled:
            self.plt.ioff()
            self.plt.close(self.fig)
