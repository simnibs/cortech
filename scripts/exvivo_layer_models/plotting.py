"""Plotting of fsaverage (lh) overlays, used by plot_fsaverage_results.py and
compute_thickness_cv.py. Requires pyvista and matplotlib.
"""

import functools
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import numpy.typing as npt
import pyvista as pv
from matplotlib import colors

from common import fsaverage_dir


@functools.cache
def _inflated_mesh() -> pv.PolyData:
    v, f = nib.freesurfer.read_geometry(fsaverage_dir() / "surf" / "lh.inflated")
    return pv.make_tri_mesh(v, f)


def percentile_clim(data: npt.NDArray, lower: float, upper: float) -> tuple[float, float]:
    """Color limits at the given percentiles of the finite values in `data`."""
    lo, hi = np.nanpercentile(data[np.isfinite(data)], [lower, upper])
    return float(lo), float(hi)


def plot_overlay(
    data: npt.NDArray,
    name: str,
    filename: os.PathLike,
    clim: tuple[float, float],
    hist_data: npt.NDArray | None = None,
    hist_label: str | None = None,
    cmap: str | None = None,
):
    """Plot an overlay on the lateral view of the inflated lh of fsaverage,
    optionally next to a histogram of `hist_data` colored like the surface.
    Non-finite values in `hist_data` are ignored.

    `cmap` is the name of a matplotlib colormap, which is used reversed
    (default: viridis, not reversed).
    """
    cm = matplotlib.colormaps["viridis" if cmap is None else cmap]
    if cmap is not None:
        cm = cm.reversed()

    mesh = _inflated_mesh().copy()
    mesh[name] = data
    plotter = pv.Plotter(shape="1|2", border=False, window_size=(1500, 1000), off_screen=True)
    plotter.subplot(0)
    plotter.add_mesh(mesh, clim=list(clim), cmap=cm, show_scalar_bar=False)
    plotter.camera.zoom(1.5)
    plotter.camera.position = (-600, 0, 0)

    if hist_data is not None:
        hist_data = np.ravel(hist_data)
        hist_data = hist_data[np.isfinite(hist_data)]
        fig, ax = plt.subplots(tight_layout=True, figsize=(20, 5))
        if hist_label is not None:
            ax.set_xlabel(hist_label, fontsize=38)
        else:
            ax.set_xlabel(name, fontsize=20)

        _, bins, patches = ax.hist(hist_data, bins=50)
        norm = colors.Normalize(*clim)
        for left_edge, patch in zip(bins, patches):
            patch.set_facecolor(cm(norm(left_edge)))

        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("white")
        ax.xaxis.label.set_color("white")
        ax.tick_params(axis="x", colors="white", labelsize=38)
        plotter.subplot(1)
        plotter.add_chart(pv.ChartMPL(fig))
        plt.close(fig)

    plotter.show(auto_close=False)
    plotter.screenshot(filename, transparent_background=True)
    plotter.close()
