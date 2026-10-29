"""Process, fit, and cross-validate the border models for several amounts of
smoothing, and select the best setting of each model.

For every combination of surface smoothing (`--surf-smooth`) and curvature
smoothing (`--curv-smooth`) this runs process_exvivo_data.py into
<out>/surf<S>_curv<C>, and for every `--neighborhood-sizes` it runs
fit_layer_models.py into <out>/surf<S>_curv<C>/models_neighborhood_<k>.
Finished runs are skipped (processing_config.json or cv_summary.csv exists),
so an interrupted sweep can be resumed by running the same command again;
unfinished runs are redone.

Finally, the cross-validation errors of all runs are collected in
<out>/sweep_summary.csv and the setting with the lowest mean absolute error
is selected for each model:

- linear_model
- equivolume local (`equivolume.neighbors`) and global (`equivolume.global`)
- equidistance local (`equidistance.neighbors`) and global
  (`equidistance.global`)

The selection is written to <out>/best_models.csv and the final models of
the selected runs are copied to <out>/best_models/<method>/ (the layout of
cortech/models/<method>/<registration>/). Note that a model must be applied
with the curvature smoothing it was fitted with (`curv_smooth` in
best_models.csv), and that the reported errors of the selected models are
slightly optimistic as the same cross-validation is used for selection.
"""

import argparse
import csv
import itertools
import shlex
import shutil
from pathlib import Path

import fit_layer_models
import process_exvivo_data
from common import CONFIG_NAME, LINEAR_MODEL_PARAMETERS, SWEEP_METHODS, write_config

# model name in cv_summary.csv: (name in best_models, method folder, final model files)
SELECTED_MODELS = {
    "linear_model": (
        "linear_model",
        "linear_model",
        [f"?h.{p}.linear_model.mgh" for p in LINEAR_MODEL_PARAMETERS],
    ),
    **{
        f"{m}.{variant}": (f"{m}.{label}", m, files)
        for m in SWEEP_METHODS
        for variant, label, files in (
            ("neighbors", "local", [f"?h.{m}.neighbor.frac.mgh"]),
            ("global", "global", ["global_fraction.csv"]),
        )
    },
}


def run_dir(out: Path, surf: int, curv: int) -> Path:
    return out / f"surf{surf}_curv{curv}"


def run_processing(args, surf: int, curv: int):
    d = run_dir(args.out, surf, curv)
    if (d / CONFIG_NAME).exists() and not args.overwrite:
        print(f"Skipping processing of {d.name} (done)")
        return
    print(f"=== Processing {d.name} ===")
    argv = [str(args.subjects_dir), str(d), "--surf-smooth", str(surf), "--curv-smooth", str(curv)]
    argv += ["--sphere-reg", args.sphere_reg, "--overwrite"]
    argv += ["--skip-gradients"] if not args.gradients else []
    argv += ["--subjects", *args.subjects] if args.subjects else []
    process_exvivo_data.main(process_exvivo_data.parse_args(argv + shlex.split(args.process_args)))


def run_fit(args, surf: int, curv: int, k: int):
    d = run_dir(args.out, surf, curv) / f"models_neighborhood_{k}"
    if (d / "cv_summary.csv").exists() and not args.overwrite:
        print(f"Skipping fit of {d.parent.name}/{d.name} (done)")
        return
    print(f"=== Fitting {d.parent.name}/{d.name} ===")
    argv = [str(d.parent), "--out-dir", str(d), "--neighborhood-size", str(k), "--overwrite"]
    fit_layer_models.main(fit_layer_models.parse_args(argv + shlex.split(args.fit_args)))


def collect_results(args) -> list[dict]:
    rows = []
    for surf, curv, k in itertools.product(args.surf_smooth, args.curv_smooth, args.neighborhood_sizes):
        d = run_dir(args.out, surf, curv) / f"models_neighborhood_{k}"
        with open(d / "cv_summary.csv") as f:
            for r in csv.DictReader(f):
                rows.append(
                    dict(
                        surf_smooth=surf,
                        curv_smooth=curv,
                        neighborhood_size=k,
                        model=r["model"],
                        mean_abs_error=float(r["mean_abs_error"]),
                        models_dir=str(d),
                    )
                )
    return rows


def select_best(rows: list[dict]) -> list[dict]:
    best = []
    for model, (name, *_) in SELECTED_MODELS.items():
        candidates = [r for r in rows if r["model"] == model]
        if not candidates:
            print(f"No results for {model}, skipping")
            continue
        best.append(dict(min(candidates, key=lambda r: r["mean_abs_error"]), model=name, cv_model=model))
    return best


def copy_best_models(best: list[dict], out: Path):
    lookup = {name: (method, files) for name, method, files in SELECTED_MODELS.values()}
    for b in best:
        method, patterns = lookup[b["model"]]
        dst = out / method
        dst.mkdir(parents=True, exist_ok=True)
        for pattern in patterns:
            for f in (Path(b["models_dir"]) / method).glob(pattern):
                shutil.copy2(f, dst / f.name)


def write_rows(filename: Path, rows: list[dict]):
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    write_config(
        args.out / "sweep_config.json",
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    )
    for surf, curv in itertools.product(args.surf_smooth, args.curv_smooth):
        run_processing(args, surf, curv)
        for k in args.neighborhood_sizes:
            run_fit(args, surf, curv, k)

    rows = collect_results(args)
    write_rows(args.out / "sweep_summary.csv", rows)
    best = select_best(rows)
    write_rows(args.out / "best_models.csv", best)

    best_dir = args.out / "best_models"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    copy_best_models(best, best_dir)

    print("\nBest settings (mean absolute cross-validation error, mm):")
    print(f"  {'model':20s} {'surf':>5s} {'curv':>5s} {'nbhd':>5s} {'error':>8s}")
    for b in best:
        print(
            f"  {b['model']:20s} {b['surf_smooth']:5d} {b['curv_smooth']:5d} "
            f"{b['neighborhood_size']:5d} {b['mean_abs_error']:8.4f}"
        )
    print(f"Final models of the best settings copied to {best_dir}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("subjects_dir", type=Path, help="Folder with FreeSurfer subject folders (see process_exvivo_data.py).")
    parser.add_argument("out", type=Path, help="Output folder of the sweep.")
    parser.add_argument("--surf-smooth", type=int, nargs="+", default=[5], help="Surface smoothing iterations (default: %(default)s).")
    parser.add_argument(
        "--curv-smooth", type=int, nargs="+", default=[0, 5, 10, 15, 20], help="Curvature smoothing iterations (default: %(default)s)."
    )
    parser.add_argument(
        "--neighborhood-sizes", type=int, nargs="+", default=[1], help="Neighborhood sizes of the local fits (default: %(default)s)."
    )
    parser.add_argument("--sphere-reg", default="sphere.reg", help="Registration to fsaverage (default: %(default)s).")
    parser.add_argument("--subjects", nargs="+", help="Subjects to use (default: all in subjects_dir).")
    parser.add_argument(
        "--gradients", action="store_true", help="Also compute surface gradients (slow, not needed for the fits)."
    )
    parser.add_argument("--process-args", default="", help='Extra arguments for process_exvivo_data.py, e.g., "--n-fracs 31".')
    parser.add_argument("--fit-args", default="", help='Extra arguments for fit_layer_models.py, e.g., "--models linear_model".')
    parser.add_argument("--overwrite", action="store_true", help="Redo runs that are already done.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
