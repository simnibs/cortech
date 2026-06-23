import math

import numba
import numba.typed
import numpy as np
import numpy.typing as npt

# from cortech.utils_cpp import map_values


def atleast_nd_append(arr, n):
    if arr.ndim == n:
        return arr
    else:
        return atleast_nd_append(arr[..., None], n)


def atleast_nd_prepend(arr, n):
    if arr.ndim == n:
        return arr
    else:
        return atleast_nd_prepend(arr[None], n)


def sliced_argmin(x: npt.NDArray, indptr: npt.NDArray):
    """Perform argmin on slices of x.

    PARAMETERS
    ----------
    x : 1-d array
        The array to perform argmin on.
    indptr : 1-d array-like
        The indices of the slices. The ith slice is indptr[i]:indptr[i+1].

    RETURNS
    -------
    res : 1-d array
        The indices (into x) corresponding to the minimum values in each chunk.
    """
    assert x.ndim == 1
    return np.array([x[i:j].argmin() + i for i, j in zip(indptr[:-1], indptr[1:])])


def normalize(arr: npt.NDArray, axis=None, inplace: bool = False):
    """Normalize along a particular axis (or axes) of `v` avoiding
    RuntimeWarning due to division by zero.

    PARAMETERS
    ----------
    v : ndarray
        Array to nomalize.
    axis:
        The axis to normalize along, e.g., `axis=1` will normalize rows).
        Like axis argument of numpy.linalg.norm.

    RETURNS
    -------
    v (normalized)
    """
    size = np.linalg.norm(arr, axis=axis, keepdims=True)
    if inplace:
        np.divide(
            arr,
            size,
            out=arr,
            where=size != 0,
        )
    else:
        return np.divide(arr, size, out=None, where=size != 0)


def compute_sphere_radius(
    frac: float | npt.NDArray,
    T: npt.NDArray,
    R: npt.NDArray,
    R3: None | npt.NDArray = None,
):
    """

    frac
        If float, will be broadcast against all T/R. If array, the last
        dimension should match the dimension of T/R.
    T
        Thickness
    R
        Radius
    R3
        Radius cubed.

    """
    frac_nd = np.array(frac)
    R3 = R**3 if R3 is None else R3
    return np.cbrt(frac_nd * ((R + T) ** 3 - R3) + R3)


def compute_tangent_vectors(vectors: npt.NDArray) -> npt.NDArray:
    """Orthogonalize the identity matrix wrt. v and compute SVD to get basis
    for tangent plane.

    Parameters
    ----------
    v : npt.NDArray
        A single vector or an array of vectors (in rows).

    Returns
    -------
    V : npt.NDArray
        Vectors spanning the plane orthogonal to each vector in `v`. If v has
        shape (n, 3) then V has shape (n, 2, 3).
    """
    v = np.atleast_2d(vectors)[..., None]
    I = np.identity(v.shape[1])[None]
    _, S, V = np.linalg.svd(
        I - I @ v / (np.sum(v**2, axis=1)[:, None]) @ v.transpose(0, 2, 1)
    )
    assert np.allclose(S[:, -1], 0, atol=1e-6)
    assert np.allclose(S[:, :-1], 1, atol=1e-6), "Degenerate elements encountered"

    return V[:, :2].squeeze()


def k_ring_neighbors(
    k: int,
    indices: npt.NDArray[int],
    n: int,
    conn_indices: npt.NDArray,
    conn_indptr: npt.NDArray,
):
    n_out = indices.shape[0]

    is_visited = np.zeros(n, bool)
    visited_idx = np.zeros(n, int)
    visited_level = np.zeros(k + 2, int)
    out_knn = numba.typed.List.empty_list(numba.int64[::1])
    out_kr = np.zeros((n_out, k + 2), int)

    multi_bfs(
        k,
        indices,
        conn_indices,
        conn_indptr,
        visited_idx,
        visited_level,
        is_visited,
        out_kr,
        out_knn,
    )
    out_knn = list(out_knn)
    return out_knn, out_kr


@numba.njit
def multi_bfs(
    max_depth: int,
    indices,  #: npt.NDArray[int],
    conn_indices,  #: npt.NDArray[int] ,
    conn_indptr,  #: npt.NDArray[int],
    visited_idx,  #: npt.NDArray[int],
    visited_level,  #: npt.NDArray[int],
    is_visited,  #: npt.NDArray[bool],
    out_kr,  #: npt.NDArray[int],
    out_knn,  #: numba.typed.List,
):
    """Execute multiple BFS's.

    Parameters
    ----------
    max_depth : int
        The depth
    indices : _type_
        Indices of the vertices to use as starting positions, e.g.,
            [[0], [1], ...]
        starting the search from 0, 1, ... or
            [[0,10], [30,45]]
        starting the search from 0 *and* 10, 30 *and* 45, ...
    A_indices : _type_
        _description_
    A_indptr : _type_
        _description_
    visited_idx : _type_
        _description_
    visited_level : _type_
        _description_
    is_visited : bool
        _description_
    out_kr : _type_
        _description_
    out_knn : _type_
        _description_
    """

    for i, idx in enumerate(indices):
        is_visited[idx] = True

        n_visited = len(idx)
        visited_idx[:n_visited] = idx
        visited_level[1] = n_visited

        k_idx, k_ring = bfs(
            max_depth,
            idx,
            conn_indices,
            conn_indptr,
            visited_idx,
            visited_level,
            is_visited,
            n_visited,
        )
        out_knn.append(k_idx.copy())  # copy is important!
        out_kr[i] = k_ring

        is_visited[k_idx] = False  # reset


@numba.njit
def bfs(
    max_depth,  #: int,
    start_idx,  #: npt.NDArray[int],
    conn_idx,  #: npt.NDArray[int],
    conn_indptr,  #: npt.NDArray[int],
    visited_idx,  #: npt.NDArray[int],
    visited_level,  #: npt.NDArray[int],
    is_visited,  #: npt.NDArray[bool],
    n_visited,  #: int,
    level: int = 0,
):
    """Find k-ring neighbors using breadth-first search.

    indices for a given ring is returned by

        seen_idx[ring_indptr[i]:ring_indptr[i+1]]

    (like scipy.sparse indices and indptr).

    Parameters
    ----------
    max_depth: int
        Max depth to recurse into.
    start_idx: npt.NDArray[int]
        The starting index/indices of the BFS.
    conn_idx: npt.NDArray[int]
        Vertex connectivity information. This corresponds to the information
        in the `indices` field in scipy.sparse CSR format.
    conn_indptr: npt.NDArray[int],
        Vertex connectivity information whose values define slices into
        `conn_idx`. This corresponds to the information in the `indptr` field
        in scipy.sparse CSR format.
    visited_idx: npt.NDArray[int]
        Array to hold the indices of the visited (neighboring) vertices.
    visited_level: npt.NDArray[int]
        Array to hold the information about the depth level to which the
        indices in `visited_idx` belongs. Length is `max_depth + 2`.
    is_visited: npt.NDArray[bool]
        Boolean array indicating if a vertex has already been visited.
    n_visited: int
        Number of vertices that has been visited already.
    level: int = 0
        Current level/depth.

    Returns
    -------
    visited_idx
        The indices of neighboring vertices.
    visited_level
        The

    Notes
    -----
    JIT compilation gives approximately 1000x speedup.
    """
    if level >= max_depth:
        return visited_idx[:n_visited], visited_level[:n_visited]

    level += 1
    n_visited_current = 0
    for i in start_idx:
        for j in conn_idx[conn_indptr[i] : conn_indptr[i + 1]]:
            if not is_visited[j]:
                is_visited[j] = True
                visited_idx[n_visited] = j
                n_visited_current += 1
                n_visited += 1
    visited_level[level + 1] = n_visited

    return bfs(
        max_depth,
        visited_idx[n_visited - n_visited_current : n_visited],
        conn_idx,
        conn_indptr,
        visited_idx,
        visited_level,
        is_visited,
        n_visited,
        level,
    )


@numba.njit
def _compute_freesurfer_thickness(vw, vp, np_, knn_flat, knn_offsets):
    """Two-pass FreeSurfer-style cortical thickness.

    Mirrors MRISmeasureCorticalThickness in mrisurf_metricProperties.cpp.
    Both passes use pial normals for the outward-direction consistency check.

    Parameters
    ----------
    vw : (n, 3) float64
        White matter vertex coordinates.
    vp : (n, 3) float64
        Pial vertex coordinates.
    np_ : (n, 3) float64
        Pial vertex normals.
    knn_flat : (K,) int64
        Flat array of k-ring neighbor indices (CSR values).
    knn_offsets : (n+1,) int64
        CSR-style row pointers into knn_flat.

    Returns
    -------
    w2p_dist : (n,) float64
    gw_dist : (n,) float64
    directions : (n, 3) float64
        Unit vectors from white[i] toward its closest valid pial match.
    """
    n = vw.shape[0]
    w2p_dist = np.empty(n)
    gw_dist = np.empty(n)
    directions = np.zeros((n, 3))

    for i in range(n):
        start = knn_offsets[i]
        end = knn_offsets[i + 1]

        nx = np_[i, 0]
        ny = np_[i, 1]
        nz = np_[i, 2]

        # Unconditional starting distance: same-index vertex (no normal check)
        dx0 = vp[i, 0] - vw[i, 0]
        dy0 = vp[i, 1] - vw[i, 1]
        dz0 = vp[i, 2] - vw[i, 2]
        init_dist = math.sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0)

        # Pass 1: white[i] → closest valid pial[j]
        min_w2p = init_dist
        best_dx = dx0
        best_dy = dy0
        best_dz = dz0

        for idx in range(start, end):
            j = knn_flat[idx]
            if j == i:
                continue
            dx = vp[j, 0] - vw[i, 0]
            dy = vp[j, 1] - vw[i, 1]
            dz = vp[j, 2] - vw[i, 2]
            if dx * nx + dy * ny + dz * nz < 0:
                continue
            if np_[j, 0] * nx + np_[j, 1] * ny + np_[j, 2] * nz < 0:
                continue
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            if d < min_w2p:
                min_w2p = d
                best_dx = dx
                best_dy = dy
                best_dz = dz

        w2p_dist[i] = min_w2p
        if min_w2p > 0:
            directions[i, 0] = best_dx / min_w2p
            directions[i, 1] = best_dy / min_w2p
            directions[i, 2] = best_dz / min_w2p

        # Pass 2: closest valid white[j] from pial[i]
        min_gw = init_dist

        for idx in range(start, end):
            j = knn_flat[idx]
            if j == i:
                continue
            dx = vp[i, 0] - vw[j, 0]
            dy = vp[i, 1] - vw[j, 1]
            dz = vp[i, 2] - vw[j, 2]
            if dx * nx + dy * ny + dz * nz < 0:
                continue
            if np_[j, 0] * nx + np_[j, 1] * ny + np_[j, 2] * nz < 0:
                continue
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            if d < min_gw:
                min_gw = d

        gw_dist[i] = min_gw

    return w2p_dist, gw_dist, directions


# @numba.njit
# def bfs_distance(
#     max_distance,  #: int,
#     start_idx,  #: npt.NDArray[int],
#     conn_idx,  #: npt.NDArray[int],
#     conn_indptr,  #: npt.NDArray[int],
#     visited_idx,  #: npt.NDArray[int],
#     visited_distance,  #: npt.NDArray[int],
#     is_visited,  #: npt.NDArray[bool],
#     n_visited,  #: int,
#     min_distance: int = 0,
# ):
#     """Find neighbors less than `max_distance` away using breadth-first search.

#     indices for a given ring is returned by

#         seen_idx[ring_indptr[i]:ring_indptr[i+1]]

#     (like scipy.sparse indices and indptr).

#     Parameters
#     ----------
#     max_depth: int
#         Max depth to recurse into.
#     start_idx: npt.NDArray[int]
#         The starting index/indices of the BFS.
#     conn_idx: npt.NDArray[int]
#         Vertex connectivity information. This corresponds to the information
#         in the `indices` field in scipy.sparse CSR format.
#     conn_indptr: npt.NDArray[int],
#         Vertex connectivity information whose values define slices into
#         `conn_idx`. This corresponds to the information in the `indptr` field
#         in scipy.sparse CSR format.
#     edge_lengths:

#     visited_idx: npt.NDArray[int]
#         Array to hold the indices of the visited (neighboring) vertices.
#     visited_level: npt.NDArray[int]
#         Array to hold the information about the depth level to which the
#         indices in `visited_idx` belongs. Length is `max_depth + 2`.
#     is_visited: npt.NDArray[bool]
#         Boolean array indicating if a vertex has already been visited.
#     n_visited: int
#         Number of vertices that has been visited already.
#     level: int = 0
#         Current level/depth.

#     Returns
#     -------
#     visited_idx
#         The indices of neighboring vertices.
#     visited_level
#         The

#     Notes
#     -----
#     JIT compilation gives approximately 1000x speedup.
#     """
#     if level >= max_depth:
#         return visited_idx[:n_visited], visited_distance[:n_visited]

#     level += 1
#     n_visited_current = 0
#     for i in start_idx:
#         for j in conn_idx[conn_indptr[i] : conn_indptr[i + 1]]:
#             if not is_visited[j]:
#                 is_visited[j] = True
#                 visited_idx[n_visited] = j
#                 edge_length[i, j]

#                 n_visited_current += 1
#                 n_visited += 1
#     visited_distance[level + 1] = n_visited

#     return bfs_distance(
#         max_depth,
#         visited_idx[n_visited - n_visited_current : n_visited],
#         conn_idx,
#         conn_indptr,
#         visited_idx,
#         visited_distance,
#         is_visited,
#         n_visited,
#         level,
#     )
