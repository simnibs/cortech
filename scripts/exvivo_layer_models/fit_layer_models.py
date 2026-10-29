"""Fit and cross-validate models of the infra-supragranular border.

Reads the output of process_exvivo_data.py and fits three kinds of models on
lh of fsaverage (right hemispheres are mapped to lh):

- linear_model: the fraction of cortical thickness below the border is
  predicted from the principal curvatures, [1, k1, k2, k1*k2], with a separate
  linear model per vertex fitted on the vertex and its neighborhood.
- equivolume / equidistance: the best layer fraction of the sweep, chosen per
  vertex neighborhood (majority vote), per vertex, per aparc region, or
  globally.

Each model is evaluated with leave-one-out cross-validation: the model is
fitted on all but one hemisphere and used to place the border on the left out
hemisphere, which is compared to its true border. Finally, each model is
fitted on all hemispheres and saved in the layout of cortech/models. See
README.md for details.
"""

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt

import cortech
from common import (
    LINEAR_MODEL_PARAMETERS,
    SWEEP_METHODS,
    freesurfer_thickness,
    fsaverage_dir,
    frac_grid,
    frac_label,
    lh_fsaverage_name,
    load_overlay,
    load_subject_hemisphere,
    map_hemi_array,
    predict_border,
    prepare_dir,
    read_aparc,
    read_config,
    save_overlay,
    write_config,
)

ALL_MODELS = ("linear_model",) + SWEEP_METHODS
PREDICTOR_CLIP_RANGE = (0.1, 99.9)


class Evaluator:
    """Place the infra-supra border on a (left out) hemisphere using a model
    defined on lh of fsaverage and compute the error in infra thickness."""

    def __init__(self, config: dict, out_dir: Path):
        self.config = config
        self.out_dir = out_dir
        self.abs_errors = {}

    def __call__(self, sample, params: npt.NDArray, method: str):
        sub, hemi = sample
        surf, fsavg = load_subject_hemisphere(
            Path(self.config["subjects_dir"]) / sub,
            hemi,
            self.config["sphere_reg"],
            self.config["surf_smooth"],
            fsaverage_reg="sphere.reg",
        )
        params = np.asarray(params)
        if params.size > 1 and hemi == "rh":  # parameters on lh of fsaverage
            params = map_hemi_array(params, "rh")
        predicted = predict_border(surf, fsavg, params, method, self.config["curv_smooth"])

        w = surf.white.vertices
        error = freesurfer_thickness(surf, w, surf.inf.vertices) - freesurfer_thickness(
            surf, w, predicted
        )
        error_fsav = surf.registration.resample(error)
        if hemi == "rh":
            error_fsav = map_hemi_array(error_fsav, "lh")
        return error, error_fsav

    def cross_validate(self, samples, fit, method: str, name: str):
        """Leave-one-out cross-validation. `fit(train_indices)` returns the
        model parameters. Saves the errors per left out subject and the mean
        absolute error over subjects.
        """
        print(f"Cross-validating {name}")
        errors = []
        for s, sample in enumerate(samples):
            train = [i for i in range(len(samples)) if i != s]
            error, error_fsav = self(sample, fit(train), method)
            sub, hemi = sample
            save_overlay(error, self.out_dir / f"{hemi}.error.{sub}.{name}.mgh")
            save_overlay(error_fsav, self.out_dir / f"lh.error.{sub}.{name}.fsaverage.mgh")
            errors.append(error_fsav)
        abs_error = np.mean(np.abs(errors), axis=0)
        save_overlay(abs_error, self.out_dir / f"lh.abs.average.error.{name}.mgh")
        self.abs_errors[name] = abs_error.mean()


def neighborhoods(surf, size: int) -> list[npt.NDArray]:
    """Indices of each vertex and its neighbors up to ring `size`."""
    if size == 0:
        return [np.array([i]) for i in range(surf.n_vertices)]
    return surf.k_ring_neighbors(k=size)[0]


def group_by_size(knn: list[npt.NDArray]):
    """Yield (vertex indices, neighborhood array) for vertices with the same
    number of neighbors."""
    m = np.array([x.size for x in knn])
    for mm in np.unique(m):
        i = np.where(m == mm)[0]
        yield i, np.array([knn[j] for j in i])


def save_lh_and_rh(data: npt.NDArray, out_dir: Path, name: str):
    """Save a model defined on lh of fsaverage, and its mapping to rh."""
    save_overlay(data, out_dir / f"lh.{name}.mgh")
    save_overlay(map_hemi_array(data, "rh"), out_dir / f"rh.{name}.mgh")


# ---------------------------------------------------------------------------
# Linear model
# ---------------------------------------------------------------------------


def load_linear_model_data(processed_dir, samples, target, thickness, k1, k2):
    """Load and prepare the predictors and targets of the linear model.

    The target is the fraction of `thickness` below the border, `target` /
    `thickness`. The curvatures are clipped at percentiles PREDICTOR_CLIP_RANGE
    and centered (per subject).

    Returns
    -------
    predictors : (n_subjects, n_vertices, 4) [1, k1, k2, k1*k2]
    target_fraction : (n_subjects, n_vertices)
    target_thickness, total_thickness : (n_subjects, n_vertices)
    """

    def load(name):
        return np.array(
            [load_overlay(processed_dir / sub / lh_fsaverage_name(hemi, name)) for sub, hemi in samples]
        )

    target_thickness = load(target)
    total_thickness = load(thickness)
    target_fraction = np.clip(target_thickness / (total_thickness + np.finfo(float).eps), 0, 1)

    curvatures = []
    for x in (load(k1), load(k2)):
        lo, hi = np.percentile(x, PREDICTOR_CLIP_RANGE, axis=1)
        x = np.clip(x, lo[:, None], hi[:, None])
        curvatures.append(x - x.mean(axis=1, keepdims=True))
    k1, k2 = curvatures

    predictors = np.stack([np.ones_like(k1), k1, k2, k1 * k2], axis=-1)
    return predictors, target_fraction, target_thickness, total_thickness


def fit_linear_model(predictors, targets, knn, n_vertices):
    """Least squares fit of a linear model per vertex using the data from the
    vertex and its neighbors (in `knn`) from all subjects.

    predictors : (n_subjects, n_vertices, n_predictors)
    targets : (n_subjects, n_vertices)

    Returns betas : (n_vertices, n_predictors)
    """
    n_predictors = predictors.shape[-1]
    betas = np.zeros((n_vertices, n_predictors))

    # vertices first
    predictors = predictors.swapaxes(0, 1)
    targets = targets.swapaxes(0, 1)

    for i, nid in group_by_size(knn):
        # (vertices, neighbors * subjects, predictors)
        X = predictors[nid].reshape(nid.shape[0], -1, n_predictors)
        y = targets[nid].reshape(nid.shape[0], -1)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        betas[i] = np.squeeze(
            Vt.swapaxes(1, 2) @ (U.swapaxes(1, 2) @ y[..., None] / S[..., None])
        )
    return betas


def run_linear_model(processed_dir, out_dir, samples, knn, n_vertices, config, args):
    predictors, target_fraction, target_thickness, total_thickness = load_linear_model_data(
        processed_dir,
        samples,
        args.target,
        args.thickness,
        f"k1.{args.curvature_surface}",
        f"k2.{args.curvature_surface}",
    )
    evaluator = Evaluator(config, out_dir)
    fsav_only_errors = []

    def fit(train):
        betas = fit_linear_model(predictors[train], target_fraction[train], knn, n_vertices)
        s = next(i for i in range(len(samples)) if i not in train)
        sub = samples[s][0]
        for name, b in zip(LINEAR_MODEL_PARAMETERS, betas.T):
            save_overlay(b, out_dir / f"lh.{name}.loo.{sub}.mgh")
        # Error when predicting directly on fsaverage (no surface placement)
        frac = np.clip(np.einsum("ij,ij->i", predictors[s], betas), 0, 1)
        fsav_only_errors.append(target_thickness[s] - frac * total_thickness[s])
        return betas

    evaluator.cross_validate(samples, fit, "linear_model", "linear_model")
    save_overlay(
        np.mean(np.abs(fsav_only_errors), axis=0),
        out_dir / "lh.abs.average.error.linear_model.fsaverage_prediction.mgh",
    )

    betas = fit_linear_model(predictors, target_fraction, knn, n_vertices)
    for name, b in zip(LINEAR_MODEL_PARAMETERS, betas.T):
        save_lh_and_rh(b, out_dir, f"{name}.linear_model")
    return evaluator.abs_errors


# ---------------------------------------------------------------------------
# Equivolume/equidistance models
# ---------------------------------------------------------------------------


def best_fraction_neighborhood(errors, knn):
    """For each vertex, the fraction that minimizes the (subject averaged)
    error at the most vertices in its neighborhood (ties: smallest fraction).

    errors : (n_vertices, n_fracs, n_subjects)
    """
    av_error = errors.mean(-1)
    n_fracs = av_error.shape[1]
    best = np.zeros(av_error.shape[0], dtype=int)
    for i, nid in group_by_size(knn):
        votes = np.argmin(av_error[nid], -1)
        counts = np.zeros((len(i), n_fracs), dtype=int)
        np.add.at(counts, (np.arange(len(i))[:, None], votes), 1)
        best[i] = counts.argmax(1)
    return best


def best_fraction_per_vertex(errors):
    return np.argmin(errors.mean(-1), -1)


def best_fraction_per_region(errors, labels, regions):
    """Fraction that minimizes the error at the most vertices of each region.
    Vertices outside `regions` get the first fraction."""
    av_error = errors.mean(-1)
    best = np.zeros(av_error.shape[0], dtype=int)
    for r in regions:
        inds = labels == r
        best[inds] = np.bincount(np.argmin(av_error[inds], -1)).argmax()
    return best


def best_fraction_global(errors):
    """Fraction that minimizes the error averaged over vertices and subjects."""
    return np.argmin(errors.mean((0, -1)))


def load_sweep_errors(processed_dir, samples, method, fracs):
    """Distance between the layers of the sweep and the true border.

    Returns (n_vertices, n_fracs, n_subjects)
    """
    return np.stack(
        [
            np.stack(
                [
                    load_overlay(
                        processed_dir / sub / method / lh_fsaverage_name(
                            hemi, f"infsup_distance_error.{method}.{frac_label(f)}"
                        )
                    )
                    for f in fracs
                ],
                axis=1,
            )
            for sub, hemi in samples
        ],
        axis=-1,
    )  # fmt: skip


def run_fraction_model(processed_dir, out_dir, samples, knn, method, config):
    fracs = frac_grid(config)
    errors = load_sweep_errors(processed_dir, samples, method, fracs)
    labels, regions = read_aparc(fsaverage_dir() / "label" / "lh.aparc.annot")
    evaluator = Evaluator(config, out_dir)

    local_fits = {
        "neighbors": lambda e: best_fraction_neighborhood(e, knn),
        "per.vertex": best_fraction_per_vertex,
        "regions": lambda e: best_fraction_per_region(e, labels, regions),
    }
    for variant, fit in local_fits.items():
        evaluator.cross_validate(
            samples,
            lambda train: fracs[fit(errors[..., train])],
            f"{method}_local",
            f"{method}.{variant}",
        )

    loo_global = []

    def fit_global(train):
        loo_global.append(fracs[best_fraction_global(errors[..., train])])
        return np.asarray(loo_global[-1])

    evaluator.cross_validate(samples, fit_global, f"{method}_global", f"{method}.global")
    with open(out_dir / "global_fraction_loo.csv", "w") as f:
        f.write("left_out_subject,hemi,fraction\n")
        for (sub, hemi), frac in zip(samples, loo_global):
            f.write(f"{sub},{hemi},{frac}\n")

    # Fit on all subjects
    save_lh_and_rh(
        fracs[best_fraction_neighborhood(errors, knn)], out_dir, f"{method}.neighbor.frac"
    )
    np.savetxt(out_dir / "global_fraction.csv", np.atleast_1d(fracs[best_fraction_global(errors)]))
    return evaluator.abs_errors


# ---------------------------------------------------------------------------


def main(args):
    processed_dir = args.processed_dir.resolve()
    config = read_config(processed_dir)
    samples = [(s["subject"], s["hemi"]) for s in config["samples"]]
    if args.subjects:
        samples = [s for s in samples if s[0] in args.subjects]
    if len(samples) < 2:
        raise RuntimeError("Need at least two hemispheres for cross-validation")

    out_dir = args.out_dir or processed_dir / f"models_neighborhood_{args.neighborhood_size}"
    out_dir = prepare_dir(out_dir, args.overwrite)
    fit_config = dict(
        processed_dir=str(processed_dir),
        processing=config,
        samples=[{"subject": s, "hemi": h} for s, h in samples],
        models=args.models,
        neighborhood_size=args.neighborhood_size,
        target=args.target,
        thickness=args.thickness,
        curvature_surface=args.curvature_surface,
    )
    write_config(out_dir / "fit_config.json", fit_config)

    fsav_white = cortech.ManifoldSurface.from_freesurfer_subject_dir(fsaverage_dir(), "lh.white")
    knn = neighborhoods(fsav_white, args.neighborhood_size)

    summary = {}
    for model in args.models:
        model_dir = out_dir / model
        model_dir.mkdir()
        if model == "linear_model":
            summary |= run_linear_model(
                processed_dir, model_dir, samples, knn, fsav_white.n_vertices, config, args
            )
        else:
            summary |= run_fraction_model(processed_dir, model_dir, samples, knn, model, config)

    with open(out_dir / "cv_summary.csv", "w") as f:
        f.write("model,mean_abs_error\n")
        for name, err in summary.items():
            f.write(f"{name},{err}\n")
    print("Mean absolute cross-validation error (mm):")
    for name, err in summary.items():
        print(f"  {name:30s} {err:.4f}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("processed_dir", type=Path, help="Output folder of process_exvivo_data.py.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output folder (default: <processed_dir>/models_neighborhood_<neighborhood-size>).",
    )
    parser.add_argument(
        "--models", nargs="+", choices=ALL_MODELS, default=list(ALL_MODELS), help="Models to fit (default: all)."
    )
    parser.add_argument("--subjects", nargs="+", help="Only use these subjects (default: all processed).")
    parser.add_argument(
        "--neighborhood-size",
        type=int,
        default=1,
        help="Neighborhood (k-ring) used when fitting local models; 0 is the vertex only (default: %(default)s).",
    )
    parser.add_argument(
        "--target",
        default="thickness.wm.inf",
        help="Linear model: thickness below the border, as named by process_exvivo_data.py (default: %(default)s).",
    )
    parser.add_argument(
        "--thickness",
        default="thickness_cortex",
        help="Linear model: cortical thickness the target is a fraction of (default: %(default)s).",
    )
    parser.add_argument(
        "--curvature-surface",
        default="avg",
        choices=("avg", "white", "pial", "inf"),
        help="Linear model: surface whose curvatures are the predictors. Note that cortech predicts "
        "with the average of white and pial curvature (default: %(default)s).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace the output folder if it exists.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
