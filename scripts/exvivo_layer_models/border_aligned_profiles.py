"""Compare infra-supragranular border placement methods with intracortical
intensity profiles.

An intensity volume (e.g., qT1, R1, T2*) is sampled along trajectories from
the white matter (WM) surface through the estimated border to the pial
surface, for each border placement method. A border that follows the
anatomical layers should keep laminar intensity features at a fixed position
relative to the border regardless of cortical folding.

Methods
-------
Surfaces are read with cortech and the white, border, and pial vertices of a
vertex (which share vertex correspondence) define a trajectory consisting of
two straight segments, WM-border (infra) and border-pial (supra). The volume
is sampled with trilinear interpolation (`--order`) in scanner RAS
coordinates on two depth axes:

- normalised: u in [-1, 1] with WM at -1, the border at 0, and pial at +1.
  Each segment is sampled separately (`--n-infra` points from WM to, but not
  including, the border and `--n-supra` + 1 from the border to pial) so the
  border is always the same sample.
- distance: signed distance (mm) from the border along the trajectory,
  negative towards WM, from -`--dist-max` to `--dist-max` in steps of
  `--dist-step`. Distances beyond the WM or pial end are undefined (NaN).

On both axes, the outer `--trim` fraction of each segment (next to WM and
pial) is excluded to reduce partial voluming. Intensities are z-scored per
hemisphere with the mean and standard deviation of samples along the straight
WM-pial line, which do not depend on the border, so all methods share the
same normalisation.

Vertices are included if they are in the cortex label (if available) and the
cortical thickness is at least `--min-thickness`. Mean curvature, H, of the
white surface (positive in sulci) is smoothed with `--curv-smooth`
iterations of Laplacian smoothing and vertices are classified as gyral (H at
or below the `--curv-quantiles` lower quantile) or sulcal (at or above the
upper quantile) using quantiles over the whole cortex. Vertex areas are the
average of the Voronoi areas on the white and pial surfaces. All averages
over vertices are weighted by vertex area and ignore depths where fewer than
`--min-bin` percent of the vertices have a valid sample.

For each method, axis, and region (whole cortex and every parcel of
`--annot` with at least `--min-vertices` vertices) the following metrics are
computed:

- gyral_sulcal_shift: the shift (axis units) of the mean sulcal relative to
  the mean gyral profile that maximises their Pearson correlation (searched
  over +/- half the profile length and refined with parabolic interpolation
  around the peak). Positive values mean that features are deeper (closer to
  WM) in gyri than in sulci relative to the border. A border that follows the
  layers has a shift of zero.
  gyral_sulcal_corr: the correlation at the peak.
- gyral_sulcal_rms: root mean square over depth of the difference between
  the mean gyral and mean sulcal profiles.
- within_sd: across-vertex standard deviation at each depth, averaged over
  depth.
- border_curv_slope: absolute slope of the weighted least squares fit of the
  z-scored intensity at the border against H.
- profile_curv_slope: root mean square over depth of the (signed) slopes of
  intensity against H.

Finally, for each method the border is projected onto the straight WM-pial
line and the median projected fraction and the median and 95th percentile of
the distance from the line are reported.

Borders
-------
By default, borders are read from surf/?h.inf.<method> in the subject folder.
`inf` refers to surf/?h.inf (e.g., the manually placed border).

With `--models-dir` (output of fit_layer_models.py) the borders are instead
predicted with the final fitted models, in the same way as in the
cross-validation of fit_layer_models.py, and the surfaces are smoothed as in
process_exvivo_data.py. Methods are then models, i.e., `linear_model`,
`equivolume`, `equidistance` (local models), `equivolume.global`,
`equidistance.global`, or `inf` (surf/?h.inf). Note that a subject that was
used to fit the models is evaluated in-sample.

See README.md for the outputs.
"""

import argparse
import csv
import json
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import numpy.typing as npt
from scipy.ndimage import map_coordinates

import cortech
from cortech import Hemisphere
from common import (
    load_fitted_model,
    load_subject_hemisphere,
    predict_border,
    prepare_dir,
    save_overlay,
    write_config,
)

DEFAULT_METHODS = ["equidistance", "equivolume", "linear_model"]
METRICS = [
    "gyral_sulcal_shift",
    "gyral_sulcal_corr",
    "gyral_sulcal_rms",
    "within_sd",
    "border_curv_slope",
    "profile_curv_slope",
]
CHUNK_SIZE = 20_000  # vertices sampled at a time


def _nanmean(x: npt.NDArray) -> float:
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else np.nan


# ---------------------------------------------------------------------------
# Surfaces
# ---------------------------------------------------------------------------


def scanner_ras(surf: cortech.Surface) -> npt.NDArray:
    """Vertices of `surf` in scanner RAS."""
    if surf.is_surface_ras() and not surf.geometry.valid:
        raise ValueError(
            "Surface is in surface RAS but has no valid volume geometry, so it "
            "cannot be converted to scanner RAS."
        )
    return surf.to_scanner_ras(inplace=False)


def border_surface_name(hemi: str, method: str) -> str:
    return f"{hemi}.inf" if method == "inf" else f"{hemi}.inf.{method}"


def load_hemisphere(subject_dir: Path, hemi: str, methods: list[str], models: dict | None):
    """Load white and pial surfaces and the border of each method.

    Returns the hemisphere and a dict of border surfaces.
    """
    if models is None:
        hemisphere = Hemisphere.from_freesurfer_subject_dir(subject_dir, hemi)
        borders = {
            m: cortech.Surface.from_freesurfer_subject_dir(subject_dir, border_surface_name(hemi, m))
            for m in methods
        }
    else:
        config = models["config"]
        hemisphere, fsavg = load_subject_hemisphere(
            subject_dir,
            hemi,
            config["sphere_reg"],
            config["surf_smooth"],
            fsaverage_reg="sphere.reg",
            inf="inf" if "inf" in methods else None,
        )
        borders = {}
        for m in methods:
            if m == "inf":
                borders[m] = hemisphere.inf
            else:
                params = load_fitted_model(models["dir"], m, hemi)
                vertices = predict_border(hemisphere, fsavg, params, m, config["curv_smooth"])
                borders[m] = hemisphere.white.new(vertices)

    for m, b in borders.items():
        if b.n_vertices != hemisphere.white.n_vertices:
            raise ValueError(f"Border {m} does not have the same number of vertices as white")
    return hemisphere, borders


def cortex_mask(subject_dir: Path, hemi: str, hemisphere: Hemisphere, min_thickness: float):
    mask = hemisphere.compute_thickness() >= min_thickness
    label = subject_dir / "label" / f"{hemi}.cortex.label"
    if label.exists():
        in_label = np.zeros_like(mask)
        in_label[nib.freesurfer.read_label(label)] = True
        mask &= in_label
    else:
        print(f"  {label} not found, using all vertices")
    return mask


def read_parcels(subject_dir: Path, hemi: str, annot: str | None):
    """Return the parcel label per vertex and the names of the parcels."""
    if annot is None:
        return None, {}
    filename = subject_dir / "label" / f"{hemi}.{annot}.annot"
    if not filename.exists():
        print(f"  {filename} not found, only computing metrics for the whole cortex")
        return None, {}
    labels, _, names = nib.freesurfer.read_annot(filename)
    return labels, {i: n.decode() for i, n in enumerate(names)}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_volume(vol: npt.NDArray, vox2ras: npt.NDArray, points: npt.NDArray, order: int) -> npt.NDArray:
    """Interpolate `vol` at `points` (..., 3) in scanner RAS. Points that are
    NaN or outside the volume are NaN."""
    shape = points.shape[:-1]
    points = points.reshape(-1, 3)
    values = np.full(len(points), np.nan, dtype=np.float32)
    valid = np.isfinite(points).all(1)
    ras2vox = np.linalg.inv(vox2ras)
    ijk = points[valid] @ ras2vox[:3, :3].T + ras2vox[:3, 3]
    values[valid] = map_coordinates(
        vol, ijk.T, order=order, mode="constant", cval=np.nan, prefilter=order > 1
    )
    return values.reshape(shape)


def sample_profiles(vol, vox2ras, order, positions, n_vertices) -> npt.NDArray:
    """Sample (n_vertices, n_depths) profiles in chunks of vertices.
    `positions(slice)` returns the sampling points of those vertices."""
    return np.concatenate(
        [
            sample_volume(vol, vox2ras, positions(slice(i, i + CHUNK_SIZE)), order)
            for i in range(0, n_vertices, CHUNK_SIZE)
        ]
    )


def normalised_axis(n_infra: int, n_supra: int, trim: float) -> npt.NDArray:
    u = np.concatenate([np.linspace(-1, 0, n_infra, endpoint=False), np.linspace(0, 1, n_supra + 1)])
    return u[(u >= -1 + trim - 1e-9) & (u <= 1 - trim + 1e-9)]


def normalised_positions(white, border, pial, u):
    """Points at u in [-1, 0) between border and WM and u in [0, 1] between
    border and pial."""
    end = np.where(u[None, :, None] < 0, white[:, None], pial[:, None])
    return border[:, None] + np.abs(u)[None, :, None] * (end - border[:, None])


def distance_axis(dist_max: float, step: float) -> npt.NDArray:
    n = int(round(dist_max / step))
    return np.arange(-n, n + 1) * step


def distance_positions(white, border, pial, d, trim):
    """Points at signed distance d from the border (negative towards WM).
    Points beyond the trimmed segments are NaN."""
    end = np.where(d[None, :, None] < 0, white[:, None], pial[:, None])
    direction = end - border[:, None]
    length = np.linalg.norm(direction, axis=-1)
    dist = np.abs(d)[None]
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(length > 0, dist / length, 0.0)
    points = border[:, None] + t[..., None] * direction
    points[dist > (1 - trim) * length] = np.nan
    return points


def line_positions(white, pial, t):
    return white[:, None] + t[None, :, None] * (pial - white)[:, None]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def weighted_profile_stats(z, weights, min_frac):
    """Weighted mean and standard deviation over vertices (rows) at each
    depth. Depths where less than `min_frac` of the vertices are valid are NaN."""
    valid = np.isfinite(z)
    W = np.where(valid, weights[:, None], 0.0)
    zz = np.where(valid, z, 0.0)
    sum_w = W.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (W * zz).sum(0) / sum_w
        sd = np.sqrt((W * (zz - mean) ** 2).sum(0) / sum_w)
    bad = (sum_w == 0) | (valid.mean(0) < min_frac) if len(z) else np.ones(z.shape[1], bool)
    mean[bad] = np.nan
    sd[bad] = np.nan
    return mean, sd


def weighted_slopes(z, curv, weights, min_frac):
    """Weighted least squares slope of z against curvature at each depth."""
    valid = np.isfinite(z)
    W = np.where(valid, weights[:, None], 0.0)
    zz = np.where(valid, z, 0.0)
    sum_w = W.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = curv[:, None] - (W * curv[:, None]).sum(0) / sum_w
        zc = zz - (W * zz).sum(0) / sum_w
        slope = (W * c * zc).sum(0) / (W * c**2).sum(0)
    slope[(sum_w == 0) | (valid.mean(0) < min_frac)] = np.nan
    return slope


def _correlation(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan
    x, y = x[valid] - x[valid].mean(), y[valid] - y[valid].mean()
    den = np.sqrt((x**2).sum() * (y**2).sum())
    return (x * y).sum() / den if den > 0 else np.nan


def profile_shift(axis, gyral, sulcal):
    """Shift of `sulcal` relative to `gyral` (axis units) that maximises
    their correlation, and the correlation at the peak.

    The profiles are resampled to a uniform axis if needed. If the sulcal
    profile is the gyral profile shifted by s, i.e., sulcal(x + s) = gyral(x),
    the result is s.
    """
    step = np.diff(axis).min()
    grid = np.arange(axis[0], axis[-1] + step / 2, step)
    if len(grid) != len(axis) or not np.allclose(grid, axis):

        def resample(y):
            v = np.isfinite(y)
            return np.interp(grid, axis[v], y[v], left=np.nan, right=np.nan) if v.any() else np.full_like(grid, np.nan)

        gyral, sulcal = resample(gyral), resample(sulcal)

    n = len(grid)
    lags = np.arange(-(n // 2), n // 2 + 1)
    corr = np.array(
        [
            _correlation(gyral[max(0, -k) : n - max(0, k)], sulcal[max(0, k) : n - max(0, -k)])
            for k in lags
        ]
    )
    if not np.isfinite(corr).any():
        return np.nan, np.nan
    i = np.nanargmax(corr)
    delta = 0.0
    if 0 < i < len(lags) - 1 and np.isfinite(corr[i - 1 : i + 2]).all():
        c0, c1, c2 = corr[i - 1 : i + 2]
        den = c0 - 2 * c1 + c2
        if den < 0:
            delta = 0.5 * (c0 - c2) / den
    return (lags[i] + delta) * step, corr[i]


def region_metrics(axis, z, curv, area, gyral, sulcal, border_index, min_frac):
    """Metrics of z-scored profiles (n_vertices, n_depths) of one region."""
    mean_g, _ = weighted_profile_stats(z[gyral], area[gyral], min_frac)
    mean_s, _ = weighted_profile_stats(z[sulcal], area[sulcal], min_frac)
    _, sd = weighted_profile_stats(z, area, min_frac)
    slopes = weighted_slopes(z, curv, area, min_frac)
    shift, corr = profile_shift(axis, mean_g, mean_s)
    return dict(
        gyral_sulcal_shift=shift,
        gyral_sulcal_corr=corr,
        gyral_sulcal_rms=np.sqrt(_nanmean((mean_g - mean_s) ** 2)),
        within_sd=_nanmean(sd),
        border_curv_slope=abs(slopes[border_index]),
        profile_curv_slope=np.sqrt(_nanmean(slopes**2)),
        n_vertices=len(z),
    )


def border_diagnostics(white, border, pial):
    """Fraction of the border projected onto the WM-pial line and the
    distance of the border from the line."""
    line = pial - white
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = ((border - white) * line).sum(1) / (line**2).sum(1)
    off_line = np.linalg.norm(border - (white + frac[:, None] * line), axis=1)
    return dict(
        median_border_fraction=np.nanmedian(frac),
        median_off_line_distance=np.nanmedian(off_line),
        p95_off_line_distance=np.nanpercentile(off_line, 95),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def process_hemisphere(subject_dir, hemi, vol, vox2ras, args, models):
    """Sample profiles of all methods and compute their metrics.

    Returns metric rows, diagnostics rows, and a dict of results for plotting
    and saving.
    """
    hemisphere, borders = load_hemisphere(subject_dir, hemi, args.methods, models)
    mask = cortex_mask(subject_dir, hemi, hemisphere, args.min_thickness)
    curv = hemisphere.white.curvature(smooth_iter=args.curv_smooth).H
    mask &= np.isfinite(curv)
    area = 0.5 * (hemisphere.white.voronoi_area() + hemisphere.pial.voronoi_area())
    q_gyral, q_sulcal = np.quantile(curv[mask], args.curv_quantiles)
    gyral, sulcal = curv <= q_gyral, curv >= q_sulcal
    print(f"  {mask.sum()} of {len(mask)} vertices in mask")

    parcels, parcel_names = read_parcels(subject_dir, hemi, args.annot)
    regions = {"cortex": mask}
    if parcels is not None:
        for k, name in parcel_names.items():
            region = mask & (parcels == k)
            if region.sum() >= args.min_vertices:
                regions[name] = region

    white = scanner_ras(hemisphere.white)[mask]
    pial = scanner_ras(hemisphere.pial)[mask]
    curv, area, gyral, sulcal = curv[mask], area[mask], gyral[mask], sulcal[mask]
    regions = {name: r[mask] for name, r in regions.items()}
    n = len(white)

    # z-score parameters from the straight WM-pial line (same for all methods)
    t = np.linspace(args.trim, 1 - args.trim, args.n_infra + args.n_supra + 1)
    line = sample_profiles(vol, vox2ras, args.order, lambda s: line_positions(white[s], pial[s], t), n)
    z_mean, z_sd = np.nanmean(line), np.nanstd(line)

    axes = {
        "normalised": normalised_axis(args.n_infra, args.n_supra, args.trim),
        "distance": distance_axis(args.dist_max, args.dist_step),
    }
    border_index = {name: int(np.argmin(np.abs(a))) for name, a in axes.items()}
    min_frac = args.min_bin / 100

    metric_rows, diag_rows = [], []
    results = dict(axes=axes, curv=curv, area=area, gyral=gyral, sulcal=sulcal, mask=mask, profiles={})
    for method, border_surf in borders.items():
        print(f"  Sampling {method}")
        border_all = scanner_ras(border_surf)
        save_overlay(
            sample_volume(vol, vox2ras, border_all, args.order),
            args.out / f"{hemi}.border_intensity.{method}.mgh",
        )
        border = border_all[mask]
        diag_rows.append(dict(hemi=hemi, method=method, **border_diagnostics(white, border, pial)))

        positions = {
            "normalised": lambda s: normalised_positions(white[s], border[s], pial[s], axes["normalised"]),
            "distance": lambda s: distance_positions(white[s], border[s], pial[s], axes["distance"], args.trim),
        }
        for axis_name, axis in axes.items():
            z = (sample_profiles(vol, vox2ras, args.order, positions[axis_name], n) - z_mean) / z_sd
            results["profiles"][(method, axis_name)] = z
            for region_name, r in regions.items():
                metrics = region_metrics(
                    axis, z[r], curv[r], area[r], gyral[r], sulcal[r], border_index[axis_name], min_frac
                )
                metric_rows += [
                    dict(hemi=hemi, method=method, axis=axis_name, region=region_name, metric=k, value=v)
                    for k, v in metrics.items()
                ]
    results.update(z_mean=z_mean, z_sd=z_sd)
    return metric_rows, diag_rows, results


def plot_profiles(filename, results, methods, min_frac):
    axes, area, gyral, sulcal = (results[k] for k in ("axes", "area", "gyral", "sulcal"))
    fig, ax = plt.subplots(3, len(methods), figsize=(4 * len(methods), 10), squeeze=False, sharey="row")
    for col, method in enumerate(methods):
        for row, axis_name in enumerate(("normalised", "distance")):
            z = results["profiles"][(method, axis_name)]
            a = ax[row, col]
            for group, label, color in ((gyral, "gyral", "tab:blue"), (sulcal, "sulcal", "tab:red")):
                mean, _ = weighted_profile_stats(z[group], area[group], min_frac)
                a.plot(axes[axis_name], mean, color=color, label=label)
            a.axvline(0, color="k", linestyle="--", linewidth=0.8)
            a.set_xlim(axes[axis_name][0], axes[axis_name][-1])
            a.set_xlabel("normalised depth (WM -1, border 0, pial 1)" if row == 0 else "distance from border (mm)")
        _, sd = weighted_profile_stats(results["profiles"][(method, "distance")], area, min_frac)
        ax[2, col].plot(axes["distance"], sd, color="k")
        ax[2, col].axvline(0, color="k", linestyle="--", linewidth=0.8)
        ax[2, col].set_xlim(axes["distance"][0], axes["distance"][-1])
        ax[2, col].set_xlabel("distance from border (mm)")
        ax[0, col].set_title(method)
    ax[0, 0].set_ylabel("z-scored intensity")
    ax[1, 0].set_ylabel("z-scored intensity")
    ax[2, 0].set_ylabel("across-vertex SD")
    ax[0, 0].legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def write_rows(filename, rows):
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(metric_rows):
    table = {}
    for r in metric_rows:
        if r["region"] == "cortex" and r["metric"] in METRICS:
            table.setdefault((r["hemi"], r["method"], r["axis"]), {})[r["metric"]] = r["value"]
    header = ["hemi", "method", "axis"] + METRICS
    widths = [4, 20, 10] + [max(len(m), 8) for m in METRICS]
    print("Whole cortex:")
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    for key, metrics in table.items():
        values = [f"{metrics[m]:.4f}".rjust(w) for m, w in zip(METRICS, widths[3:])]
        print("  ".join([k.ljust(w) for k, w in zip(key, widths)] + values))


def main(args):
    models = None
    if args.models_dir is not None:
        with open(args.models_dir / "fit_config.json") as f:
            models = dict(dir=args.models_dir.resolve(), config=json.load(f)["processing"])
    subjects_dir = args.subjects_dir
    if subjects_dir is None:
        subjects_dir = models["config"]["subjects_dir"] if models else os.environ.get("SUBJECTS_DIR")
    if subjects_dir is None:
        raise ValueError("Specify --subjects-dir or set SUBJECTS_DIR")
    subject_dir = Path(subjects_dir) / args.subject

    image = nib.load(args.volume)
    vol = np.asarray(image.dataobj, dtype=np.float32)
    if vol.ndim == 4 and vol.shape[3] == 1:
        vol = vol[..., 0]
    if vol.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {vol.shape}")

    hemis = [h for h in args.hemis if (subject_dir / "surf" / f"{h}.white").exists()]
    if not hemis:
        raise FileNotFoundError(f"No white surfaces of {args.hemis} in {subject_dir / 'surf'}")

    args.out = prepare_dir(args.out, args.overwrite)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    write_config(
        args.out / "profiles_config.json",
        dict(config, subjects_dir=str(subjects_dir), hemis=hemis, processing=models and models["config"]),
    )

    metric_rows, diag_rows = [], []
    for hemi in hemis:
        print(f"Processing {args.subject} ({hemi})")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN depths
            m, d, results = process_hemisphere(subject_dir, hemi, vol, image.affine, args, models)
        metric_rows += m
        diag_rows += d
        plot_profiles(args.out / f"{hemi}.profiles.png", results, args.methods, args.min_bin / 100)
        if args.save_profiles:
            np.savez_compressed(
                args.out / f"{hemi}.profiles.npz",
                **{f"{axis}_axis": a for axis, a in results["axes"].items()},
                **{f"{m}.{axis}": z for (m, axis), z in results["profiles"].items()},
                **{k: results[k] for k in ("curv", "area", "gyral", "sulcal", "mask", "z_mean", "z_sd")},
            )

    write_rows(args.out / "metrics.csv", metric_rows)
    write_rows(args.out / "border_diagnostics.csv", diag_rows)
    print_summary(metric_rows)
    print(f"Done. Results are in {args.out}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--subjects-dir", type=Path, help="FreeSurfer subjects folder (default: $SUBJECTS_DIR, or that of --models-dir).")
    parser.add_argument("--subject", required=True, help="Subject name.")
    parser.add_argument("--volume", required=True, type=Path, help="Intensity volume in the scanner space of the surfaces.")
    parser.add_argument("--out", required=True, type=Path, help="Output folder.")
    parser.add_argument("--hemis", nargs="+", default=["lh", "rh"], choices=("lh", "rh"), help="Hemispheres (default: those of %(default)s that exist).")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help="Border methods: surf/?h.inf.<method>, or models with --models-dir; `inf` is surf/?h.inf (default: %(default)s).",
    )
    parser.add_argument("--models-dir", type=Path, help="Output folder of fit_layer_models.py. Predict the borders with these models.")
    parser.add_argument("--annot", default="aparc", help="Parcellation (label/?h.<annot>.annot) for regional metrics (default: %(default)s).")
    parser.add_argument("--n-infra", type=int, default=20, help="Samples from WM to the border (default: %(default)s).")
    parser.add_argument("--n-supra", type=int, default=20, help="Samples from the border to pial, excluding the border (default: %(default)s).")
    parser.add_argument("--trim", type=float, default=0.1, help="Fraction of each segment to exclude at WM and pial (default: %(default)s).")
    parser.add_argument("--dist-max", type=float, default=1.5, help="Maximum distance from the border in mm (default: %(default)s).")
    parser.add_argument("--dist-step", type=float, default=0.05, help="Distance axis step in mm (default: %(default)s).")
    parser.add_argument("--curv-smooth", type=int, default=10, help="Smoothing iterations of the white mean curvature (default: %(default)s).")
    parser.add_argument("--curv-quantiles", type=float, nargs=2, default=[0.25, 0.75], help="Curvature quantiles defining gyral and sulcal vertices (default: %(default)s).")
    parser.add_argument("--min-thickness", type=float, default=0.5, help="Minimum cortical thickness in mm (default: %(default)s).")
    parser.add_argument("--min-vertices", type=int, default=50, help="Minimum number of vertices in a parcel (default: %(default)s).")
    parser.add_argument("--min-bin", type=float, default=20, help="Minimum percentage of vertices with a valid sample at a depth (default: %(default)s).")
    parser.add_argument("--order", type=int, default=1, help="Interpolation order (default: %(default)s).")
    parser.add_argument("--save-profiles", action="store_true", help="Save the z-scored profiles to ?h.profiles.npz.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the output folder if it exists.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
