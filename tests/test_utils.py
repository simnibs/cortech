import numpy as np
import pytest
import scipy.sparse as sp

import cortech.utils


@pytest.mark.parametrize("arr", [np.array(1.0), np.array([[1.0]])])
@pytest.mark.parametrize("n", [2,4])
def test_atleast_nd(arr, n):
    x = cortech.utils.atleast_nd(arr, n)
    assert x.ndim == n


def test_sliced_argmin():
    arr = np.array([0,1,2,2,1,0,2,2,2,3,1])
    indptr = [0, 3, 6, 8, len(arr)]
    x = cortech.utils.sliced_argmin(arr, indptr)
    np.testing.assert_allclose(x, [0, 5, 6, 10])


def test_normalize():
    np.random.seed(0)
    a = np.random.randn(10,3)

    b = cortech.utils.normalize(a, axis=0)
    c = cortech.utils.normalize(a, axis=1)

    np.testing.assert_allclose(np.linalg.norm(b, axis=0), 1.0)
    np.testing.assert_allclose(np.linalg.norm(c, axis=1), 1.0)


@pytest.mark.parametrize("frac", ["float", "array"])
def test_compute_sphere_radius(frac):
    match frac:
        case "float":
            frac = 0.4
        case "array":
            frac = np.full(10, 0.4)

    radius = np.full(10, 1.0)
    radius_cubed = radius**3
    thickness = np.full(10, 1.0)

    radius_frac = cortech.utils.compute_sphere_radius(frac, thickness, radius, radius_cubed)

    # Test that the following two volumes are equal:
    # - (1-frac)*volume in between the sphere with radius `radius` and the
    #   sphere with radius `radius_frac`.
    # - frac*volume in between the sphere with radius `radius_frac` and the
    #   sphere with volume `radius + thickness`.
    v_inner = 4/3 * np.pi * radius_cubed
    v_outer = 4/3 * np.pi * (radius + thickness) ** 3
    v_frac =  4/3 * np.pi * radius_frac ** 3

    np.testing.assert_allclose((1-frac) * (v_frac - v_inner), frac * (v_outer - v_frac))


def test_compute_tangent_vectors():
    v = np.array([[1,0,0], [0,1,0], [0,0,1]])
    u = cortech.utils.compute_tangent_vectors(v)

    # Test that u0 and u1 are both orthogonal to v
    np.testing.assert_allclose(u @ v[:, None].swapaxes(1,2), 0)
    # Test that u0 is orthogonal to u1
    np.testing.assert_allclose(np.sum(u[:,0] * u[:,1], axis=1), 0)


@pytest.mark.parametrize("k", [1, 2])
def test_k_ring_neighbors_single(k, diamond_vertices, diamond_adjacency_matrix):
    """Find vertices that are neighbors of a single vertex."""
    indices = np.array([[0]])
    A = sp.csr_array(diamond_adjacency_matrix)
    knn, kr = cortech.utils.k_ring_neighbors(k, indices, len(diamond_vertices), A.indices, A.indptr)

    match k:
        case 1:
            np.testing.assert_allclose(knn, [[0, 1, 2, 3, 4]])
            np.testing.assert_allclose(kr, [[0, 1, 5]])
        case 2:
            np.testing.assert_allclose(knn, [[0, 1, 2, 3, 4, 5]])
            np.testing.assert_allclose(kr, [[0, 1, 5, 6]])

@pytest.mark.parametrize("k", [1, 2])
def test_k_ring_neighbors_two_separate(k, diamond_vertices, diamond_adjacency_matrix):
    """Find vertices that are neighbors of each vertex separately."""
    indices = np.array([[0],[3]])
    A = sp.csr_array(diamond_adjacency_matrix)
    knn, kr = cortech.utils.k_ring_neighbors(k, indices, len(diamond_vertices), A.indices, A.indptr)

    match k:
        case 1:
            np.testing.assert_allclose(knn, [[0, 1, 2, 3, 4], [3,0,2,4,5]])
            np.testing.assert_allclose(kr, [[0, 1, 5], [0,1,5]])
        case 2:
            np.testing.assert_allclose(knn, [[0, 1, 2, 3, 4, 5], [3,0,2,4,5,1]])
            np.testing.assert_allclose(kr, [[0, 1, 5, 6],[ 0, 1, 5, 6]])


def test_k_ring_neighbors_two_simultaneous(diamond_vertices, diamond_adjacency_matrix):
    """Find vertices that are neighbors of either of two vertices."""
    k = 1
    indices = np.array([[0,3]])
    A = sp.csr_array(diamond_adjacency_matrix)
    knn, kr = cortech.utils.k_ring_neighbors(k, indices, len(diamond_vertices), A.indices, A.indptr)

    np.testing.assert_allclose(knn, [[0, 3, 1, 2, 4, 5]])
    np.testing.assert_allclose(kr, [[0, 2, 6]])
