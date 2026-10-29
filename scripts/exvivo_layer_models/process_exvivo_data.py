"""Process ex vivo FreeSurfer runs (with an infra-supragranular border surface)
for fitting layer models.

For every subject and hemisphere this

1. maps morphometry (e.g., thickness), curvatures, and FreeSurfer-style
   thicknesses (white-pial, white-inf, inf-pial) to fsaverage,
2. sweeps equivolume and equidistance layer fractions and saves the distance
   between each predicted layer and the true infra-supra border,
3. computes surface gradients of curvature and morphometry (optional),

and finally computes group statistics on fsaverage. The parameters of the run
are saved to processing_config.json in the output directory, which
fit_layer_models.py reads. See README.md for details.
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from common import (
    CONFIG_NAME,
    SWEEP_METHODS,
    FsaverageOverlayWriter,
    aseg_label_codes,
    frac_grid,
    frac_label,
    freesurfer_thickness,
    fsaverage_name,
    lh_fsaverage_name,
    load_overlay,
    load_subject_hemisphere,
    prepare_dir,
    read_aparc,
    save_overlay,
    write_config,
    write_csv,
)

DEFAULT_QUANTITIES = ["thickness", "thickness.inf.pial", "thickness.wm.inf"]


def find_subjects(subjects_dir: Path, subjects: list[str] | None):
    """Return sorted (subject, hemi) pairs for subjects with surfaces."""
    if subjects is None:
        subjects = sorted(
            p.name
            for p in subjects_dir.iterdir()
            if p.is_dir() and "fsaverage" not in p.name
        )
    samples = []
    for sub in subjects:
        hemis = sorted(f.stem for f in (subjects_dir / sub / "surf").glob("?h.white"))
        if not hemis:
            print(f"No surfaces found for {sub}, skipping.")
        samples += [(sub, h) for h in hemis]
    return samples


def map_morphometry(surf, sub_dir, out_dir, writer, hemi, quantities, labels, regions):
    """Map FreeSurfer morphometry files to fsaverage and save their mean and
    standard deviation per aparc region."""
    stats = {"aparc_label": aseg_label_codes(regions, hemi)}
    for q in quantities:
        data = nib.freesurfer.io.read_morph_data(sub_dir / "surf" / f"{hemi}.{q}")
        stats[f"{q}_avg"] = [data[labels == r].mean() for r in regions]
        stats[f"{q}_std"] = [data[labels == r].std() for r in regions]
        writer.save(surf.registration.resample(data), out_dir, q)
    write_csv(out_dir / f"{hemi}.cortical_measures.csv", stats)


def map_curvatures(surf, out_dir, writer, curvatures):
    for surf_name, curv in curvatures.items():
        for field, values in zip(curv._fields, curv):
            writer.save(surf.registration.resample(values), out_dir, f"{field}.{surf_name}")


def map_thicknesses(surf, out_dir, writer):
    w, p, i = surf.white.vertices, surf.pial.vertices, surf.inf.vertices
    thicknesses = {
        "thickness_cortex": freesurfer_thickness(surf, w, p),
        "thickness_inf": freesurfer_thickness(surf, w, i),
        "thickness_sup": freesurfer_thickness(surf, i, p),
    }
    for name, t in thicknesses.items():
        writer.save(surf.registration.resample(t), out_dir, name)


def sweep_layer_fractions(surf, out_dir, writer, hemi, fracs, curv, labels, regions):
    """Place equivolume and equidistance layers at each fraction in `fracs`
    and compare them to the true infra-supra border (inf).

    Saves, per method and fraction, the predicted surface, the distance to
    inf, and the infra/supra thicknesses implied by the prediction. Also saves
    the fraction with the lowest mean distance error in each aparc region.
    """
    w, p, i = surf.white.vertices, surf.pial.vertices, surf.inf.vertices
    for method in SWEEP_METHODS:
        method_dir = out_dir / method
        method_dir.mkdir()
        region_err_avg = np.zeros((len(regions), len(fracs)))
        region_err_std = np.zeros((len(regions), len(fracs)))

        for j, frac in enumerate(fracs):
            fl = frac_label(frac)
            layer = surf.estimate_layers(
                method=method, frac=frac, curv=curv.H, return_surface=True
            )
            layer.save(method_dir / f"{hemi}.inf.{method}.{fl}")

            measures = {
                "infsup_distance_error": np.linalg.norm(i - layer.vertices, axis=1),
                "inf_thickness": freesurfer_thickness(surf, w, layer.vertices),
                "sup_thickness": freesurfer_thickness(surf, layer.vertices, p),
            }
            for name, values in measures.items():
                name = f"{name}.{method}.{fl}"
                save_overlay(values, method_dir / f"{hemi}.{name}.mgh")
                writer.save(surf.registration.resample(values), method_dir, name)

            dist = measures["infsup_distance_error"]
            for k, r in enumerate(regions):
                region_err_avg[k, j] = dist[labels == r].mean()
                region_err_std[k, j] = dist[labels == r].std()

        best = np.argmin(region_err_avg, axis=1)
        rows = np.arange(len(regions))
        write_csv(
            method_dir / f"{hemi}.{method}_errors.csv",
            {
                "aparc_label": aseg_label_codes(regions, hemi),
                "best_fraction": fracs[best],
                "dist_error_avg": region_err_avg[rows, best],
                "dist_error_std": region_err_std[rows, best],
            },
        )


def map_gradients(surf, sub_dir, out_dir, writer, hemi, curvatures, quantities):
    """Compute surface gradients (in the principal curvature directions) of
    the curvatures (on their own surface) and of morphometry (on inf)."""

    def save(coeffs, name, suffix=""):
        overlays = {f"{name}.gradient.magnitude{suffix}": np.linalg.norm(coeffs, axis=1)}
        for c in range(coeffs.shape[1]):
            overlays[f"{name}.gradient.{c}{suffix}"] = coeffs[:, c]
        for n, values in overlays.items():
            save_overlay(values, out_dir / f"{hemi}.{n}.mgh")
            writer.save(surf.registration.resample(values), out_dir, n)

    for surf_name, curv in curvatures.items():
        if surf_name == "avg":
            continue
        for field, values in zip(curv._fields, curv):
            coeffs = surf.compute_surface_gradients(values, surface_name=surf_name)
            save(coeffs, field, f".{surf_name}")

    for q in quantities:
        data = nib.freesurfer.io.read_morph_data(sub_dir / "surf" / f"{hemi}.{q}")
        save(surf.compute_surface_gradients(data, surface_name="inf"), q)


def process_hemisphere(sub_dir: Path, out_dir: Path, hemi: str, config: dict):
    surf, _ = load_subject_hemisphere(
        sub_dir,
        hemi,
        config["sphere_reg"],
        config["surf_smooth"],
        fsaverage_reg=config["sphere_reg"],
    )
    labels, regions = read_aparc(sub_dir / "label" / f"{hemi}.aparc.annot")
    writer = FsaverageOverlayWriter(hemi)

    smooth = config["curv_smooth"]
    curvatures = {
        "white": surf.white.curvature(smooth_iter=smooth),
        "pial": surf.pial.curvature(smooth_iter=smooth),
        "inf": surf.inf.curvature(smooth_iter=smooth),
    }
    curvatures["avg"] = surf.compute_average_curvature(
        curvatures["white"], curvatures["pial"]
    )
    quantities = config["quantities"]

    print("  Mapping morphometry, curvature and thickness to fsaverage")
    map_morphometry(surf, sub_dir, out_dir, writer, hemi, quantities, labels, regions)
    map_curvatures(surf, out_dir, writer, curvatures)
    map_thicknesses(surf, out_dir, writer)

    print("  Sweeping equivolume and equidistance fractions")
    sweep_layer_fractions(
        surf, out_dir, writer, hemi, frac_grid(config), curvatures["avg"], labels, regions
    )

    if config["gradients"]:
        print("  Computing surface gradients")
        map_gradients(surf, sub_dir, out_dir, writer, hemi, curvatures, quantities)

    writer.finalize()


def compute_group_stats(out_dir: Path, samples: list[tuple[str, str]]):
    """Mean and standard deviation across subjects of the fsaverage overlays
    in the subject folders, per hemisphere and for all subjects on lh."""
    group_dir = out_dir / "group"
    group_dir.mkdir()

    suffix = ".fsaverage.mgh"
    names = sorted(
        {
            f.name[3 : -len(suffix)]  # strip "?h." and suffix
            for sub, hemi in samples
            for f in (out_dir / sub).glob(fsaverage_name(hemi, "*"))
        }
    )
    groups = {
        "lh": [s for s in samples if s[1] == "lh"],
        "rh": [s for s in samples if s[1] == "rh"],
        "all.on.lh": samples,
    }
    for name in names:
        for group, members in groups.items():
            files = [
                out_dir / sub / (
                    lh_fsaverage_name(hemi, name)
                    if group == "all.on.lh"
                    else fsaverage_name(hemi, name)
                )
                for sub, hemi in members
            ]  # fmt: skip
            files = [f for f in files if f.exists()]
            if not files:
                continue
            data = np.stack([load_overlay(f) for f in files], axis=1)
            save_overlay(data.mean(1), group_dir / f"{group}.{name}.mean.mgh")
            save_overlay(data.std(1), group_dir / f"{group}.{name}.std.mgh")


def main(args):
    config = dict(
        subjects_dir=str(args.subjects_dir.resolve()),
        quantities=args.quantities,
        surf_smooth=args.surf_smooth,
        curv_smooth=args.curv_smooth,
        sphere_reg=args.sphere_reg,
        frac_min=args.frac_min,
        frac_max=args.frac_max,
        n_fracs=args.n_fracs,
        gradients=not args.skip_gradients,
    )
    out_dir = prepare_dir(args.out_dir, args.overwrite)
    samples = find_subjects(args.subjects_dir, args.subjects)
    if not samples:
        raise RuntimeError(f"No subjects with surfaces found in {args.subjects_dir}")

    for sub, hemi in samples:
        print(f"Processing {sub} ({hemi})")
        sub_out = out_dir / sub
        sub_out.mkdir(exist_ok=True)
        process_hemisphere(args.subjects_dir / sub, sub_out, hemi, config)

    print("Computing group statistics")
    compute_group_stats(out_dir, samples)

    config["samples"] = [{"subject": s, "hemi": h} for s, h in samples]
    write_config(out_dir / CONFIG_NAME, config)
    print(f"Done. Wrote {out_dir / CONFIG_NAME}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "subjects_dir",
        type=Path,
        help="Folder with FreeSurfer subject folders. Each needs surf/?h.{white,pial,inf,<sphere-reg>}, "
        "label/?h.aparc.annot and the files in --quantities.",
    )
    parser.add_argument("out_dir", type=Path, help="Output folder.")
    parser.add_argument(
        "--subjects", nargs="+", help="Subjects to process (default: all in subjects_dir)."
    )
    parser.add_argument(
        "--quantities",
        nargs="+",
        default=DEFAULT_QUANTITIES,
        help="FreeSurfer morphometry files (surf/?h.<name>) to map to fsaverage (default: %(default)s).",
    )
    parser.add_argument(
        "--surf-smooth", type=int, default=5, help="Taubin smoothing iterations for the surfaces (default: %(default)s)."
    )
    parser.add_argument(
        "--curv-smooth", type=int, default=20, help="Smoothing iterations for the curvatures (default: %(default)s)."
    )
    parser.add_argument(
        "--sphere-reg",
        default="sphere.reg",
        help="Spherical registration to fsaverage, e.g., sphere.reg or josa.sphere.reg (default: %(default)s).",
    )
    parser.add_argument("--frac-min", type=float, default=0.2, help="Smallest layer fraction in the sweep (default: %(default)s).")
    parser.add_argument("--frac-max", type=float, default=0.8, help="Largest layer fraction in the sweep (default: %(default)s).")
    parser.add_argument("--n-fracs", type=int, default=61, help="Number of layer fractions in the sweep (default: %(default)s).")
    parser.add_argument("--skip-gradients", action="store_true", help="Do not compute surface gradients.")
    parser.add_argument("--overwrite", action="store_true", help="Replace out_dir if it exists.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
