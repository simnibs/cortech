from pathlib import Path

import numpy as np
import pytest

from cortech.sphere import fibonacci_points, fibonacci_sphere


@pytest.fixture(name="BERT_DIR", scope="class")
def bert_dir():
    return Path(__file__).parent / "data" / "bert_resampled"


@pytest.fixture(name="VOL_GEOM", scope="class")
def volume_geometry():
    """Volume geometry extracted for lh.white"""
    xras = np.array([-1.0000, 0.0000, 0.0000])
    yras = np.array([0.0000, 0.0000, -1.0000])
    zras = np.array([0.0000, 1.0000, 0.0000])
    return dict(
        valid=True,
        filename="/mnt/depot64/freesurfer/freesurfer.7.4.0/subjects/bert/mri/wm.mgz",
        volume=np.array([256, 256, 256]),
        voxelsize=np.array([1.0000, 1.0000, 1.0000]),
        xras=xras,
        yras=yras,
        zras=zras,
        cras=np.array([5.3997, 18.0000, 0.0000]),
        cosines=np.column_stack((xras, yras, zras)),
        vox2ras=np.array(
            [
                [-1.00000, 0.00000, 0.00000, 133.39972],
                [0.00000, 0.00000, 1.00000, -110.00000],
                [0.00000, -1.00000, 0.00000, 128.00000],
                [0.00000, 0.00000, 0.00000, 1.00000],
            ]
        ),
        vox2ras_tkr=np.array(
            [
                [-1.00000, 0.00000, 0.00000, 128.00000],
                [0.00000, 0.00000, 1.00000, -128.00000],
                [0.00000, -1.00000, 0.00000, 128.00000],
                [0.00000, 0.00000, 0.00000, 1.00000],
            ]
        ),
    )


@pytest.fixture(scope="module")
def sphere_points(n=100, r=1.0):
    return fibonacci_points(n, r)


@pytest.fixture(scope="module")
def sphere_tuple(n=100, r=1.0):
    return fibonacci_sphere(n, r)


@pytest.fixture(scope="module")
def diamond_vertices():
    # axis-aligned diamond shape
    return np.array(
        [[0, 0, 1], [-1, 0, 0], [0, -1, 0], [1, 0, 0], [0, 1, 0], [0, 0, -1]],
        dtype=float,
    )


@pytest.fixture(scope="module")
def diamond_faces():
    return np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 1],
            [1, 5, 2],
            [2, 5, 3],
            [3, 5, 4],
            [4, 5, 1],
        ]
    )


@pytest.fixture(scope="module")
def diamond_barycenters():
    return np.array(
        [
            [-0.33333333, -0.33333333, 0.33333333],
            [0.33333333, -0.33333333, 0.33333333],
            [0.33333333, 0.33333333, 0.33333333],
            [-0.33333333, 0.33333333, 0.33333333],
            [-0.33333333, -0.33333333, -0.33333333],
            [0.33333333, -0.33333333, -0.33333333],
            [0.33333333, 0.33333333, -0.33333333],
            [-0.33333333, 0.33333333, -0.33333333],
        ]
    )


@pytest.fixture(scope="module")
def diamond_adjacency_matrix():
    return np.array(
        [
            [0, 1, 1, 1, 1, 0],
            [1, 0, 1, 0, 1, 1],
            [1, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 1],
            [1, 1, 0, 1, 0, 1],
            [0, 1, 1, 1, 1, 0],
        ],
        dtype=float,
    )
