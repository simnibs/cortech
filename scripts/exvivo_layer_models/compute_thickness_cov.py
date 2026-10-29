"""Coefficient of variation (CoV) across hemispheres of the cortical, infra,
and supra thickness, with and without curvature correction.

For every hemisphere processed by process_exvivo_data.py (with the same
surface smoothing and registration) this computes the FreeSurfer-style
thickness white-pial, white-inf, and inf-pial, and the curvature-corrected
version of each (Glasser-style: the linear effect of the mean curvature of
inf is regressed out within the hemisphere). All are mapped to lh of
fsaverage, where the mean and CoV over hemispheres are computed, saved, and
plotted. See README.md for details.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import numpy.typing as npt

from common import (
    freesurfer_thickness,
    fsaverage_dir,
    load_subject_hemisphere,
    map_hemi_array,
    prepare_dir,
    read_config,
    save_overlay,
    write_config,
)
from plotting import percentile_clim, plot_overlay

COMPARTMENTS = {
    # name: (label, inner surface, outer surface)
    "cortical_thickness": ("Total", "white", "pial"),
    "infra_thickness": ("Infra-granular", "white", "inf"),
    "supra_thickness": ("Supra-granular", "inf", "pial"),
}
VERSIONS = {"": "Thickness", "_corrected": "Intrinsic Thickness"}


def curvature_corrected_thickness(thickness: npt.NDArray, mean_curv: npt.NDArray) -> npt.NDArray:
    """Remove the linear effect of mean curvature from thickness.

    Fits t = k * H + b over all vertices and returns t - k * H (the intercept
    is kept so the result is still in mm).
    """
    valid = np.isfinite(thickness) & np.isfinite(mean_curv)
    A = np.column_stack([mean_curv[valid], np.ones(valid.sum())])
    k, _ = np.linalg.lstsq(A, thickness[valid], rcond=None)[0]
    corrected = thickness.copy()
    corrected[valid] -= k * mean_curv[valid]
    return corrected


def hemisphere_thicknesses(sub_dir: Path, hemi: str, config: dict, args) -> dict[str, npt.NDArray]:
    """Raw and curvature-corrected thicknesses of a hemisphere on lh of fsaverage."""
    surf, _ = load_subject_hemisphere(
        sub_dir, hemi, config["sphere_reg"], config["surf_smooth"], fsaverage_reg=config["sphere_reg"]
    )
    mean_curv = getattr(surf, args.curvature_surface).curvature(smooth_iter=args.curv_smooth).H

    native = {}
    for name, (_, inner, outer) in COMPARTMENTS.items():
        t = freesurfer_thickness(surf, getattr(surf, inner).vertices, getattr(surf, outer).vertices)
        native[name] = t
        native[f"{name}_corrected"] = curvature_corrected_thickness(t, mean_curv)
        print(f"  {name}: mean {t.mean():.2f} mm")

    data = np.stack([surf.registration.resample(t) for t in native.values()], axis=1)
    if hemi == "rh":
        data = map_hemi_array(data, "lh")
    return dict(zip(native, data.T))


def plot_maps(out_dir: Path, stats: dict, mask: npt.NDArray):
    """Mean and CoV on fsaverage. Color limits are set from the raw thickness
    so raw and corrected maps are comparable."""
    for name, (label, *_) in COMPARTMENTS.items():
        for stat, hist_label in (("mean", "Thickness (mm)"), ("cov", "CoV")):
            clim = None
            for version, title in VERSIONS.items():
                data = stats[f"{stat}_{name}{version}"]
                lo, hi = percentile_clim(data[mask], 0.1, 99.9)
                trimmed = data[mask & (data > lo) & (data < hi)]
                if clim is None:
                    clim = percentile_clim(trimmed, 1, 99)
                plot_overlay(
                    data,
                    f"{'Mean' if stat == 'mean' else 'CoV'} {label} {title}",
                    out_dir / f"{stat}_{name}{version}.png",
                    clim=clim,
                    hist_data=trimmed,
                    hist_label=hist_label,
                )


def plot_cov_histograms(out_dir: Path, stats: dict, mask: npt.NDArray, n_obs: int):
    fig, axes = plt.subplots(2, len(COMPARTMENTS), figsize=(6 * len(COMPARTMENTS), 10), sharey="row")
    fig.suptitle(f"Vertex-wise CoV across {n_obs} hemisphere observations", fontsize=13, fontweight="bold")
    rows = (("", "Raw thickness", "steelblue"), ("_corrected", "Curvature-corrected thickness", "darkorange"))
    for col, (name, (label, *_)) in enumerate(COMPARTMENTS.items()):
        for row, (version, row_label, color) in enumerate(rows):
            ax = axes[row, col]
            cov = stats[f"cov_{name}{version}"][mask]
            cov = cov[np.isfinite(cov)]
            ax.hist(cov, bins=120, color=color, alpha=0.8, edgecolor="white", linewidth=0.3)
            median, mean = np.median(cov), np.mean(cov)
            ax.axvline(median, color="black", linestyle="--", linewidth=1.5, label=f"Median = {median:.3f}")
            ax.axvline(mean, color="red", linestyle=":", linewidth=1.5, label=f"Mean   = {mean:.3f}")
            ax.set_xlim([0, 0.5])
            ax.set_xlabel("CoV", fontsize=11)
            ax.set_title(f"{label}\n{row_label}", fontsize=11)
            ax.legend(fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_ylabel("Number of vertices", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "cov_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(args):
    processed_dir = args.processed_dir.resolve()
    config = read_config(processed_dir)
    samples = [(s["subject"], s["hemi"]) for s in config["samples"]]
    if args.subjects:
        samples = [s for s in samples if s[0] in args.subjects]
    if len(samples) < 2:
        raise RuntimeError("Need at least two hemispheres to compute the CoV")

    out_dir = prepare_dir(args.out_dir or processed_dir / "thickness_cov", args.overwrite)
    write_config(
        out_dir / "cov_config.json",
        dict(
            processed_dir=str(processed_dir),
            processing=config,
            samples=[{"subject": s, "hemi": h} for s, h in samples],
            curvature_surface=args.curvature_surface,
            curv_smooth=args.curv_smooth,
        ),
    )

    per_hemi = []
    for sub, hemi in samples:
        print(f"Processing {sub} ({hemi})")
        per_hemi.append(hemisphere_thicknesses(Path(config["subjects_dir"]) / sub, hemi, config, args))

    stats = {}
    for name in per_hemi[0]:
        data = np.stack([h[name] for h in per_hemi], axis=1)
        mean = data.mean(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cov = data.std(1, ddof=1) / mean
        cov[~np.isfinite(cov)] = np.nan
        stats[f"mean_{name}"], stats[f"cov_{name}"] = mean, cov
    for name, values in stats.items():
        save_overlay(values, out_dir / f"lh.{name}.mgh")

    mask = np.zeros(len(mean), dtype=bool)
    mask[nib.freesurfer.read_label(fsaverage_dir() / "label" / "lh.cortex.label")] = True

    summary = {name: np.nanmean(values[mask]) for name, values in stats.items()}
    summary |= {f"median_{name}": np.nanmedian(values[mask]) for name, values in stats.items() if name.startswith("cov")}
    with open(out_dir / "summary.csv", "w") as f:
        f.write("measure,value\n")
        for name, value in summary.items():
            f.write(f"{name},{value}\n")
    print("Averages over cortex vertices of fsaverage:")
    for name, value in summary.items():
        print(f"  {name:45s} {value:.4f}")

    print("Plotting")
    plot_maps(out_dir, stats, mask)
    plot_cov_histograms(out_dir, stats, mask, len(samples))
    print(f"Done. Results are in {out_dir}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("processed_dir", type=Path, help="Output folder of process_exvivo_data.py.")
    parser.add_argument("--out-dir", type=Path, help="Output folder (default: <processed_dir>/thickness_cov).")
    parser.add_argument("--subjects", nargs="+", help="Only use these subjects (default: all processed).")
    parser.add_argument(
        "--curvature-surface",
        default="inf",
        choices=("white", "pial", "inf"),
        help="Surface whose mean curvature is regressed out (default: %(default)s).",
    )
    parser.add_argument(
        "--curv-smooth",
        type=int,
        default=10,
        help="Smoothing iterations of the mean curvature (default: %(default)s).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace the output folder if it exists.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
