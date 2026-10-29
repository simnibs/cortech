"""Plot group results of process_exvivo_data.py and fit_layer_models.py on
the inflated lh of fsaverage (right hemispheres are mapped to lh).

Plots the mean and standard deviation over hemispheres of

1. the infra, supra, and total cortical thickness (FreeSurfer morphometry),
2. the curvatures of inf and of white/pial (average), and the correlation
   between layer thickness and curvature,
3. the surface gradients (if they were computed),

and the cross-validation errors and parameters of the fitted models, and the
average sulc/curv of fsaverage. See README.md for details.
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.typing as npt

from common import fsaverage_dir, lh_fsaverage_name, load_overlay, prepare_dir, read_config, write_config
from plotting import percentile_clim, plot_overlay

THICKNESSES = {
    # name in plots: (morphometry file, histogram threshold, color limits of the mean)
    "cortical_thickness": ("thickness", 1.0, (1.0, 3.0)),
    "infra_thickness": ("thickness.wm.inf", 0.5, (0.5, 1.5)),
    "supra_thickness": ("thickness.inf.pial", 0.5, (0.5, 1.5)),
}
CURVATURES = {
    # curvature: (label, short label)
    "H": ("Avg. curvature", "avg."),
    "K": ("Gaussian curvature", "gaussian"),
    "k1": ("K1 curvature", "k1"),
    "k2": ("K2 curvature", "k2"),
}
CURVATURE_SURFACES = {
    # surface: (name in plots, label, percentiles for color limits)
    "avg": ("wm_pial", "wm + pial", (5, 95)),
    "inf": ("inf", "inf", (1, 99)),
}
GRADIENTS = {
    # name in plots: (overlay, label)
    "cortical_thickness": ("thickness.gradient.magnitude", "thk gradient magnitude (cortex)"),
    "infra_thickness": ("thickness.wm.inf.gradient.magnitude", "thk gradient magnitude (infra)"),
    "supra_thickness": ("thickness.inf.pial.gradient.magnitude", "thk gradient magnitude (supra)"),
    "H_inf": ("H.gradient.magnitude.inf", "H gradient magnitude (inf)"),
    "k1_inf": ("k1.gradient.magnitude.inf", "K1 gradient magnitude (inf)"),
    "k2_inf": ("k2.gradient.magnitude.inf", "K2 gradient magnitude (inf)"),
}
FRACTION_MODELS = {"equidistance": ("Isodistance", "Isodist."), "equivolume": ("Isovolume", "Isovol.")}
LINEAR_MODEL_PARAMS = {
    # parameter: (label, color limits)
    "intercept": ("Intercept", (0.0, 0.6)),
    "beta_k1": ("Weight k1", (-0.1, 0.1)),
    "beta_k2": ("Weight k2", (-0.1, 0.1)),
    "beta_k1k2": ("Weight k1*k2", (-0.1, 0.1)),
}


def row_correlation(x: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
    """Pearson correlation between corresponding rows of x and y."""
    x = x - x.mean(1, keepdims=True)
    y = y - y.mean(1, keepdims=True)
    den = np.sqrt((x**2).sum(1) * (y**2).sum(1))
    return (x * y).sum(1) / (den + np.finfo(float).eps)


def shape_index(k1: npt.NDArray, k2: npt.NDArray) -> npt.NDArray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return (2 / np.pi) * np.arctan((k1 + k2) / (k2 - k1))


class Plotter:
    def __init__(self, processed_dir: Path, samples, out_dir: Path):
        self.processed_dir = processed_dir
        self.samples = samples
        self.out_dir = out_dir

    def load(self, name: str) -> npt.NDArray:
        """Load an overlay of all hemispheres on lh of fsaverage as an
        (n_vertices, n_hemispheres) array."""
        return np.stack(
            [
                load_overlay(self.processed_dir / sub / lh_fsaverage_name(hemi, name))
                for sub, hemi in self.samples
            ],
            axis=1,
        )

    def plot(self, data, filename, label, clim, hist_data=None, cmap=None):
        """Plot `data` with a histogram of `hist_data` (default: all of `data`)."""
        plot_overlay(
            data,
            label,
            self.out_dir / f"{filename}.png",
            clim=clim,
            hist_data=data if hist_data is None else hist_data,
            hist_label=label,
            cmap=cmap,
        )

    def plot_percentile(self, data, filename, label, percentiles):
        """Plot with color limits at percentiles of the data."""
        self.plot(data, filename, label, percentile_clim(data, *percentiles))


def plot_thickness(p: Plotter):
    thk = {name: p.load(f) for name, (f, *_) in THICKNESSES.items()}
    for name, (_, threshold, clim) in THICKNESSES.items():
        mean, std = thk[name].mean(1), thk[name].std(1)
        p.plot(mean, f"mean_{name}", "Thickness (mm)", clim, hist_data=mean[mean > threshold])
        p.plot(std, f"std_{name}", r"$\sigma$ thickness (mm)", (0.15, 0.5), hist_data=std[std > 0])

    for layer in ("infra", "supra"):
        frac = (thk[f"{layer}_thickness"] / (thk["cortical_thickness"] + 1e-5)).mean(1)
        p.plot(frac, f"mean_{layer}_fraction", "Thickness fraction", (0.3, 0.6), hist_data=frac[frac > 0.2])

    diff = (thk["infra_thickness"] - thk["supra_thickness"]).mean(1)
    p.plot(diff, "mean_infra_supra_thickness_difference", "Thickness difference", (-0.5, 0.5), cmap="coolwarm")
    return thk


def plot_curvature(p: Plotter, thk):
    for surf, (surf_name, surf_label, percentiles) in CURVATURE_SURFACES.items():
        curv = {c: p.load(f"{c}.{surf}") for c in CURVATURES}
        curv["shape_index"] = shape_index(curv["k1"], curv["k2"])
        for c, values in curv.items():
            if c == "shape_index":
                label, pct = f"Shape index ({surf_label})", (5, 95)
            else:
                label, pct = f"{CURVATURES[c][0]} ({surf_label})", percentiles
            p.plot_percentile(np.nanmean(values, 1), f"mean_{c}_{surf_name}", label, pct)
            p.plot_percentile(np.nanstd(values, 1), f"std_{c}_{surf_name}", f"SD of {label}", pct)

        if surf == "inf":
            for layer in ("infra", "supra"):
                for c, (_, short) in CURVATURES.items():
                    r = row_correlation(thk[f"{layer}_thickness"], np.abs(curv[c]))
                    label = f"Corr. {layer} thk vs {short} curv"
                    p.plot(r, f"corr_{layer}_thickness_{c}_inf", label, (-0.7, 0.7))


def plot_gradients(p: Plotter):
    for name, (overlay, label) in GRADIENTS.items():
        gradient = p.load(overlay)
        for stat, values, prefix in (("mean", gradient.mean(1), "Mu"), ("std", gradient.std(1), "SD")):
            p.plot(
                values,
                f"{stat}_{name}_gradient_magnitude",
                f"{prefix} {label}",
                percentile_clim(values, 1, 99),
                hist_data=values.clip(*percentile_clim(values, 0.1, 99.9)),
            )


def plot_models(p: Plotter, models_dir: Path):
    def load(model, name):
        return load_overlay(models_dir / model / f"lh.{name}.mgh")

    for method, (title, short) in FRACTION_MODELS.items():
        if not (models_dir / method).exists():
            print(f"No {method} model in {models_dir}, skipping.")
            continue
        for variant, desc in (("global", "global"), ("neighbors", "local")):
            error = load(method, f"abs.average.error.{method}.{variant}")
            p.plot(error, f"{method}_{variant}_abs_error", f"{short} ({desc}) abs. error (mm)", (0, 0.3))
            print(f"{title} model ({variant}) mean absolute error: {error.mean():.4f} mm")
        frac = load(method, f"{method}.neighbor.frac")
        p.plot(frac, f"{method}_neighbor_fraction", f"{short} (local) fractions", (0.2, 0.6))

    if not (models_dir / "linear_model").exists():
        print(f"No linear model in {models_dir}, skipping.")
        return
    error = load("linear_model", "abs.average.error.linear_model")
    p.plot(error, "linear_model_abs_error", "Linear model abs. error (mm)", (0, 0.3))
    print(f"Linear model mean absolute error: {error.mean():.4f} mm")
    for param, (label, clim) in LINEAR_MODEL_PARAMS.items():
        p.plot(load("linear_model", f"{param}.linear_model"), f"linear_model_{param}", label, clim)


def plot_fsaverage(p: Plotter):
    surf_dir = fsaverage_dir() / "surf"
    sulc = nib.freesurfer.read_morph_data(surf_dir / "lh.avg_sulc").astype(float)
    curv = nib.freesurfer.read_morph_data(surf_dir / "lh.avg_curv").astype(float)
    p.plot(sulc, "fsaverage_avg_sulc", "fsaverage sulc", (-1, 1), cmap="gray")
    p.plot(curv, "fsaverage_avg_curv", "fsaverage curv", (-0.1, 0.1), cmap="RdYlGn")


def main(args):
    processed_dir = args.processed_dir.resolve()
    config = read_config(processed_dir)
    samples = [(s["subject"], s["hemi"]) for s in config["samples"]]
    if args.subjects:
        samples = [s for s in samples if s[0] in args.subjects]
    models_dir = (args.models_dir or processed_dir / "models_neighborhood_1").resolve()

    out_dir = prepare_dir(args.out_dir or processed_dir / "plots", args.overwrite)
    write_config(
        out_dir / "plot_config.json",
        dict(
            processed_dir=str(processed_dir),
            models_dir=str(models_dir),
            samples=[{"subject": s, "hemi": h} for s, h in samples],
        ),
    )
    p = Plotter(processed_dir, samples, out_dir)

    print(f"Plotting results of {len(samples)} hemispheres to {out_dir}")
    thk = plot_thickness(p)
    plot_curvature(p, thk)
    if config["gradients"]:
        plot_gradients(p)
    if models_dir.exists():
        plot_models(p, models_dir)
    else:
        print(f"{models_dir} not found, not plotting models.")
    plot_fsaverage(p)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("processed_dir", type=Path, help="Output folder of process_exvivo_data.py.")
    parser.add_argument(
        "--models-dir",
        type=Path,
        help="Output folder of fit_layer_models.py (default: <processed_dir>/models_neighborhood_1).",
    )
    parser.add_argument("--out-dir", type=Path, help="Output folder (default: <processed_dir>/plots).")
    parser.add_argument("--subjects", nargs="+", help="Only use these subjects (default: all processed).")
    parser.add_argument("--overwrite", action="store_true", help="Replace the output folder if it exists.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
