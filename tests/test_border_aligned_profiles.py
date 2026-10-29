"""Test scripts/exvivo_layer_models/border_aligned_profiles.py on a synthetic
folded sheet with a bright band at a known, curvature-dependent depth."""

import csv
import importlib
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from scipy.spatial import cKDTree

try:
    import cortech.cgal.polygon_mesh_processing  # noqa: F401
    import cortech.utils_cpp  # noqa: F401
    from cortech.freesurfer import write_geometry
except ImportError as e:
    pytest.skip(f"cortech's compiled extensions are not available ({e})", allow_module_level=True)

SCRIPT_DIR = Path(__file__).parents[1] / "scripts" / "exvivo_layer_models"

# Sheet: white z = A sin(2 pi x / L), pial offset by T along the normal
A, L, T = 2.5, 20.0, 2.5
X_MAX, Y_MAX = 2 * L, 10.0
F_AMP = 0.12  # band at fraction 0.5 + F_AMP * sin(2 pi x / L) of T from white
SIGMA = 0.2  # band width (mm)


def white_curve(x):
    """Height and unit normal (pointing to pial) of the white surface."""
    w = 2 * np.pi / L
    z = A * np.sin(w * x)
    dz = A * w * np.cos(w * x)
    n = np.stack([-dz, np.ones_like(x)], -1) / np.sqrt(1 + dz**2)[:, None]
    return z, n


def f_true(x):
    return 0.5 + F_AMP * np.sin(2 * np.pi * x / L)


def make_subject(root: Path, real_ras: bool = True):
    """Write surfaces, labels, and an intensity volume. Returns the subject
    folder, the volume filename, and the white vertex x coordinates."""
    rng = np.random.default_rng(0)
    sub = root / "sub"
    (sub / "surf").mkdir(parents=True)
    (sub / "label").mkdir()

    xs, ys = np.arange(0, X_MAX + 1e-6, 0.25), np.arange(0, Y_MAX + 1e-6, 0.5)
    nx, ny = len(xs), len(ys)
    x, y = (a.ravel() for a in np.meshgrid(xs, ys, indexing="ij"))
    z, n2 = white_curve(x)
    normal = np.stack([n2[:, 0], np.zeros_like(x), n2[:, 1]], 1)
    white = np.stack([x, y, z], 1)

    idx = np.arange(nx * ny).reshape(nx, ny)
    v00, v10, v01, v11 = idx[:-1, :-1], idx[1:, :-1], idx[:-1, 1:], idx[1:, 1:]
    faces = np.concatenate(
        [np.stack([v00, v10, v11], -1).reshape(-1, 3), np.stack([v00, v11, v01], -1).reshape(-1, 3)]
    )

    surfaces = {
        "white": white,
        "pial": white + T * normal,
        "inf.equidistance": white + 0.5 * T * normal,
        "inf.noisy": white + np.clip(f_true(x) + 0.05 * rng.standard_normal(len(x)), 0.1, 0.9)[:, None] * T * normal,
        "inf.true": white + f_true(x)[:, None] * T * normal,
    }
    for name, v in surfaces.items():
        write_geometry(sub / "surf" / f"lh.{name}", v, faces, real_ras=real_ras)

    # Cortex label away from the edges and two parcels
    interior = np.where((x > 2) & (x < X_MAX - 2) & (y > 1) & (y < Y_MAX - 1))[0]
    with open(sub / "label" / "lh.cortex.label", "w") as f:
        f.write(f"#!ascii label\n{len(interior)}\n")
        f.writelines(f"{i} {x[i]:.3f} {y[i]:.3f} {z[i]:.3f} 0.0\n" for i in interior)
    ctab = np.array([[255, 0, 0, 0, 0], [0, 0, 255, 0, 0]])
    nib.freesurfer.write_annot(
        sub / "label" / "lh.parc.annot", (x >= L).astype(int), ctab, [b"first", b"second"], fill_ctab=True
    )

    # Volume: background 1 and a band of intensity 3 at f_true(x0) * T from
    # the white surface, where x0 is the closest point on the white surface
    voxel = 0.1
    origin = np.array([-1.0, -1.0, -A - 1.0])
    shape = (int((X_MAX + 2) / voxel), int((Y_MAX + 2) / voxel), int((2 * A + T + 2) / voxel))
    gx = origin[0] + voxel * np.arange(shape[0])
    gz = origin[2] + voxel * np.arange(shape[2])
    px, pz = (a.ravel() for a in np.meshgrid(gx, gz, indexing="ij"))
    x0 = np.arange(-5, X_MAX + 5, 0.005)
    z0, n0 = white_curve(x0)
    _, i = cKDTree(np.stack([x0, z0], 1)).query(np.stack([px, pz], 1))
    depth = (px - x0[i]) * n0[i, 0] + (pz - z0[i]) * n0[i, 1]
    band = 1 + 2 * np.exp(-((depth - f_true(x0[i]) * T) ** 2) / (2 * SIGMA**2))
    vol = np.repeat(band.reshape(shape[0], 1, shape[2]), shape[1], axis=1).astype(np.float32)
    affine = np.diag([voxel, voxel, voxel, 1.0])
    affine[:3, 3] = origin
    volume = root / "volume.nii.gz"
    nib.save(nib.Nifti1Image(vol, affine), volume)
    return sub, volume, x


@pytest.fixture(scope="module")
def bap():
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        yield importlib.import_module("border_aligned_profiles")
    finally:
        sys.path.remove(str(SCRIPT_DIR))


def run(bap, sub, volume, out):
    args = bap.parse_args(
        [
            "--subjects-dir", str(sub.parent),
            "--subject", sub.name,
            "--volume", str(volume),
            "--out", str(out),
            "--hemis", "lh",
            "--methods", "equidistance", "noisy", "true",
            "--annot", "parc",
            "--save-profiles",
        ]
    )  # fmt: skip
    bap.main(args)


def read_metrics(filename):
    with open(filename) as f:
        return {
            (r["method"], r["axis"], r["region"], r["metric"]): float(r["value"])
            for r in csv.DictReader(f)
        }


def test_border_aligned_profiles(bap, tmp_path):
    sub, volume, x = make_subject(tmp_path)
    out = tmp_path / "out"
    run(bap, sub, volume, out)

    for f in (
        "metrics.csv",
        "border_diagnostics.csv",
        "lh.profiles.png",
        "lh.profiles.npz",
        "lh.border_intensity.true.mgh",
    ):
        assert (out / f).exists()

    m = read_metrics(out / "metrics.csv")
    regions = {k[2] for k in m}
    assert regions == {"cortex", "first", "second"}

    # Curvature is positive in sulci: gyral vertices (crowns, large z) have
    # the most negative curvature
    p = np.load(out / "lh.profiles.npz")
    mask = p["mask"]
    sin = np.sin(2 * np.pi * x[mask] / L)
    assert sin[p["gyral"]].mean() > 0.8
    assert sin[p["sulcal"]].mean() < -0.8

    # The band is at the true border in gyri and sulci
    shift = {method: m[(method, "distance", "cortex", "gyral_sulcal_shift")] for method in ("equidistance", "noisy", "true")}
    assert abs(shift["true"]) < 0.05

    # With equidistance, the band is (f_sulcal - f_gyral) * T further from WM
    # in sulci than in gyri (i.e., deeper in sulci, which is a negative shift)
    expected = T * (f_true(x[mask][p["sulcal"]]).mean() - f_true(x[mask][p["gyral"]]).mean())
    assert expected < -0.4
    assert shift["equidistance"] == pytest.approx(expected, abs=0.1)

    for metric in ("within_sd", "border_curv_slope"):
        values = {method: m[(method, "distance", "cortex", metric)] for method in ("equidistance", "noisy", "true")}
        assert values["true"] == min(values.values()), metric

    with open(out / "border_diagnostics.csv") as f:
        diag = {r["method"]: r for r in csv.DictReader(f)}
    assert float(diag["equidistance"]["median_border_fraction"]) == pytest.approx(0.5, abs=1e-3)
    assert float(diag["true"]["median_off_line_distance"]) < 1e-3


def test_surface_ras_without_geometry(bap, tmp_path):
    sub, volume, _ = make_subject(tmp_path, real_ras=False)
    with pytest.raises(ValueError, match="volume geometry"):
        run(bap, sub, volume, tmp_path / "out")
