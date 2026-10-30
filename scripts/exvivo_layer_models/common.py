"""Helpers shared by the scripts in this folder."""

import datetime
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.typing as npt

from cortech.cortex import Hemisphere

CONFIG_NAME = "processing_config.json"
SWEEP_METHODS = ("equivolume", "equidistance")

# Taubin step sizes used for all surface smoothing in these scripts (these
# were the defaults of `smooth_taubin` when the models were first fitted).
TAUBIN_A = 0.6
TAUBIN_B = -0.61

# Prefix of fsaverage overlays from right hemispheres mapped to the left
# hemisphere (all model fitting happens on lh of fsaverage).
RH_ON_LH = "rh-on-lh"

# Parameters of the linear border model (in the order of its predictors
# [1, k1, k2, k1*k2])
LINEAR_MODEL_PARAMETERS = ["intercept", "beta_k1", "beta_k2", "beta_k1k2"]


# ---------------------------------------------------------------------------
# Files and naming
# ---------------------------------------------------------------------------


def fsaverage_dir() -> Path:
    home = os.environ.get("FREESURFER_HOME")
    if not home:
        raise RuntimeError("FREESURFER_HOME is not set")
    return Path(home) / "subjects" / "fsaverage"


def frac_label(frac: float) -> str:
    """String used for a layer fraction in filenames, e.g., 0.21 -> '0.21'."""
    return f"{round(float(frac), 6):g}"


def frac_grid(config: dict) -> npt.NDArray:
    return np.linspace(config["frac_min"], config["frac_max"], config["n_fracs"])


def fsaverage_name(hemi: str, name: str) -> str:
    """Filename of an overlay on fsaverage `hemi`."""
    return f"{hemi}.{name}.fsaverage.mgh"


def lh_fsaverage_name(hemi: str, name: str) -> str:
    """Filename of the version of an overlay that lives on lh of fsaverage."""
    return fsaverage_name("lh" if hemi == "lh" else RH_ON_LH, name)


def save_overlay(data: npt.NDArray, filename: os.PathLike) -> None:
    image = nib.freesurfer.mghformat.MGHImage(
        np.asarray(data, dtype="float32"), np.eye(4)
    )
    nib.save(image, filename)


def load_overlay(filename: os.PathLike) -> npt.NDArray:
    return nib.load(filename).get_fdata().squeeze()


def read_aparc(annot_file: os.PathLike):
    """Return the per-vertex aparc label and the unique (valid) labels."""
    labels, _, _ = nib.freesurfer.io.read_annot(annot_file)
    regions = np.unique(labels)
    return labels, regions[regions > 0]


def aseg_label_codes(regions: npt.NDArray, hemi: str) -> npt.NDArray:
    """Map aparc annotation labels to the aparc+aseg label codes."""
    return regions + (1000 if hemi == "lh" else 2000)


def write_csv(filename: os.PathLike, columns: dict[str, npt.NDArray]) -> None:
    names = list(columns)
    data = np.column_stack([np.asarray(columns[n], dtype=float) for n in names])
    np.savetxt(filename, data, delimiter=",", header=",".join(names), comments="")


def prepare_dir(path: Path, overwrite: bool) -> Path:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists (use --overwrite to replace it)")
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


# ---------------------------------------------------------------------------
# Mapping between hemispheres (requires FreeSurfer's mris_apply_reg)
# ---------------------------------------------------------------------------


def map_hemi_file(src: os.PathLike, trg: os.PathLike, to_hemi: str) -> None:
    """Map an fsaverage overlay to the other hemisphere via fsaverage_sym.
    Multi-frame overlays are mapped frame by frame.
    """
    fsav = fsaverage_dir()
    cmd = [
        "mris_apply_reg",
        "--src", str(src),
        "--trg", str(trg),
        "--streg",
        str(fsav / "xhemi" / "surf" / f"{to_hemi}.fsaverage_sym.sphere.reg"),
        str(fsav / "surf" / f"{to_hemi}.fsaverage_sym.sphere.reg"),
    ]  # fmt: skip
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"mris_apply_reg failed for {src}:\n{result.stdout}\n{result.stderr}"
        )


def map_hemi_array(data: npt.NDArray, to_hemi: str) -> npt.NDArray:
    """Map an fsaverage overlay array, (n_vertices,) or (n_vertices, n_frames),
    to the other hemisphere.
    """
    data = np.asarray(data)
    with tempfile.TemporaryDirectory() as d:
        src, trg = Path(d) / "src.mgh", Path(d) / "trg.mgh"
        frames = data.reshape(data.shape[0], 1, 1, -1) if data.ndim > 1 else data
        save_overlay(frames, src)
        map_hemi_file(src, trg, to_hemi)
        return nib.load(trg).get_fdata().reshape(data.shape)


class FsaverageOverlayWriter:
    """Save overlays that live on fsaverage of `hemi`. For right hemispheres,
    `finalize` additionally maps every saved overlay to the left hemisphere
    (as RH_ON_LH files, see `lh_fsaverage_name`).
    """

    def __init__(self, hemi: str, chunk_size: int = 50):
        self.hemi = hemi
        self.chunk_size = chunk_size
        self._saved = []

    def save(self, data: npt.NDArray, out_dir: Path, name: str) -> None:
        save_overlay(data, out_dir / fsaverage_name(self.hemi, name))
        self._saved.append((out_dir, name))

    def finalize(self) -> None:
        if self.hemi == "lh":
            return
        print(f"  Mapping {len(self._saved)} overlays from rh to lh")
        for i in range(0, len(self._saved), self.chunk_size):
            chunk = self._saved[i : i + self.chunk_size]
            data = np.stack(
                [load_overlay(d / fsaverage_name(self.hemi, n)) for d, n in chunk], 1
            )
            data = map_hemi_array(data, "lh")
            for (d, n), x in zip(chunk, data.T):
                save_overlay(x, d / lh_fsaverage_name(self.hemi, n))
        self._saved = []


# ---------------------------------------------------------------------------
# Surfaces
# ---------------------------------------------------------------------------


def load_subject_hemisphere(
    subject_dir: os.PathLike,
    hemi: str,
    sphere_reg: str,
    surf_smooth: int,
    fsaverage_reg: str,
    inf: str | None = "inf",
) -> tuple[Hemisphere, Hemisphere]:
    """Load white, pial and infra-supra (`inf`, if not None) surfaces of a
    subject, Taubin smooth them, and set up resampling from subject to
    fsaverage (`surf.registration.resample`).

    Returns the subject and fsaverage hemispheres.
    """
    surf = Hemisphere.from_freesurfer_subject_dir(
        subject_dir, hemi, inf=inf, registration=sphere_reg
    )
    for s in (surf.white, surf.pial, surf.inf):
        if s is None:
            continue
        s.smooth_taubin(a=TAUBIN_A, b=TAUBIN_B, n_iter=surf_smooth, inplace=True)

    fsavg = Hemisphere.from_freesurfer_subject_dir(
        "fsaverage", hemi, registration=fsaverage_reg
    )
    surf.registration.set_projection(fsavg.registration.vertices)
    return surf, fsavg


def predict_border(
    surf: Hemisphere, fsavg: Hemisphere, params: npt.NDArray, model: str, curv_smooth: int
) -> npt.NDArray:
    """Place the infra-supra border on `surf` using a border model.

    `params` is either a scalar (global fraction) or defined per vertex of
    `fsavg` (the fsaverage hemisphere with the same `name` as `surf`), in
    which case it is resampled to `surf` via the spherical registrations.
    `model` is a cortech infra-supra model name, i.e., it contains
    `equivolume`, `equidistance` or `linear`. Returns the border vertices.
    """
    fsavg.registration.set_projection(surf.registration.vertices)
    params = np.asarray(params)
    if params.size > 1:  # parameters per fsaverage vertex
        params = fsavg.registration.resample(params).squeeze()
    surf.infra_supra_model = {model: params}
    return surf.fit_infra_supra_border(curv_args={"smooth_iter": curv_smooth})[model]


def load_fitted_model(models_dir: Path, model: str, hemi: str) -> npt.NDArray:
    """Load a final model written by fit_layer_models.py on fsaverage `hemi`.

    `model` is `linear_model`, `<equivolume|equidistance>` (the local,
    neighborhood model), or `<equivolume|equidistance>.global`.
    """
    method, _, variant = model.partition(".")
    d = models_dir / method
    if method == "linear_model" and not variant:
        return np.stack(
            [load_overlay(d / f"{hemi}.{p}.linear_model.mgh") for p in LINEAR_MODEL_PARAMETERS], 1
        )
    if method in SWEEP_METHODS and not variant:
        return load_overlay(d / f"{hemi}.{method}.neighbor.frac.mgh")
    if method in SWEEP_METHODS and variant == "global":
        return np.loadtxt(d / "global_fraction.csv")
    raise ValueError(f"Unknown model {model}")


def freesurfer_thickness(
    surf: Hemisphere, inner: npt.NDArray, outer: npt.NDArray
) -> npt.NDArray:
    """FreeSurfer-style thickness between two sets of vertices sharing the
    topology of `surf` (e.g., white -> inf)."""
    h = Hemisphere(surf.name, surf.white.new(inner), surf.pial.new(outer))
    return h.compute_freesurfer_thickness()


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------


def cortech_version() -> str:
    """Git commit of the cortech checkout these scripts live in."""
    repo = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + ("-dirty" if dirty else "")


def write_config(filename: os.PathLike, config: dict) -> None:
    config = dict(
        config,
        cortech_version=cortech_version(),
        created=datetime.datetime.now().isoformat(timespec="seconds"),
    )
    with open(filename, "w") as f:
        json.dump(config, f, indent=2)


def read_config(processed_dir: Path) -> dict:
    filename = processed_dir / CONFIG_NAME
    if not filename.exists():
        raise FileNotFoundError(
            f"{filename} not found. Is {processed_dir} the output of "
            "process_exvivo_data.py (and did it finish)?"
        )
    with open(filename) as f:
        return json.load(f)
