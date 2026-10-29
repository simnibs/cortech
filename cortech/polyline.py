import numpy as np
import numpy.typing as npt

import scipy

from cortech.utils import _ensure_array_is_index

class ClosedPolyline:
    def __init__(self, vertices: npt.NDArray):
        self.vertices = vertices
        self.indices = np.arange(self.n_vertices)

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, value):
        value = np.atleast_2d(np.asarray(value))
        assert value.ndim == 2
        self._vertices = value
        self.n_vertices, self.n_dim = value.shape


    def reverse(self):
        self.polylines = np.roll(self.indices[::-1],1)
        self.vertices = np.roll(self.vertices[::-1],1,0)

    def edges(self):
        """
        indices : 0, 1, 2, ..., n
        edges   : (0,1), (1,2), ..., (n, 0)


        """
        return np.stack((self.indices, np.roll(self.indices, -1)), -1)


    def adjacency_matrix(self, connectivity="vertex", include_self: bool = False):
        """Assemble the adjacency matrix for vertices or faces."""

        match connectivity:
            case "vertex":
                edges = self.edges()

                row_ind = edges.ravel()
                col_ind = edges[:,::-1].ravel()
                data = np.ones_like(row_ind)
                shape = (self.n_vertices, self.n_vertices)

            # case "edge":
            #     edges = self.edges(sort_axis_1=True)  # e.g., (1,0) and (0,1) -> (0,1)

            #     # ONLY WORKS FOR MANIFOLD EDGES!

            #     # # Now sort the vertex-vertex edges
            #     # # first column has 1st priority
            #     # a0 = edges[:, 0].argsort()
            #     # s0 = edges[a0]
            #     # # second column has 2nd priority (actually, we just want to sub-sort `s0`)
            #     # a1 = s0[:, 1].argsort()
            #     # s1 = s0[a1]
            #     # # "stable" keeps order of like items, hence both columns will be
            #     # # sorted after this operation
            #     # a2 = np.argsort(s1[:, 0], kind="stable")  # .argsort(stable=True)
            #     # # s2 = s1[a2] # the sorted edges

            #     # faces_enum = np.broadcast_to(
            #     #     np.arange(self.n_faces)[:, None], self.faces.shape
            #     # ).ravel()
            #     # face_edges = faces_enum[a0[a1[a2]]].reshape(-1, 2)

            #     # uc: number of occurrences of each edge
            #     # uc == 1 is a border edge
            #     # uc == 2 is a manifold edge
            #     # uc > 2 is a non-manifold edge (part of more than two faces)
            #     # ui : provides an index for each unique edge

            #     _, ui, uc = np.unique(
            #         edges, axis=0, return_inverse=True, return_counts=True
            #     )
            #     if np.any(uc > 2):
            #         raise RuntimeError(
            #             "Non-manifold topology (some edges are part of more than two triangles)"
            #         )

            #     # we can ignore border edges as these do not connect any faces
            #     manifold_edges = uc[ui] == 2

            #     # sort edges so that when applying to faces_enum, faces that
            #     # shared a particular edge will be next to each other.
            #     s = ui[manifold_edges].argsort()

            #     faces_enum = np.broadcast_to(
            #         np.arange(self.n_faces)[:, None], self.faces.shape
            #     ).ravel()

            #     face_edges = faces_enum[manifold_edges][s].reshape(-1, 2)
            #     row_ind = face_edges.ravel()
            #     col_ind = face_edges[:, ::-1].ravel()
            #     shape = (self.n_faces, self.n_faces)
            #     data = np.ones(face_edges.size)
            case _:
                raise ValueError(f"Invalid `connectivity` (got {connectivity})")
        A = scipy.sparse.csr_array((data, (row_ind, col_ind)), shape)
        if include_self:
            A = A.tolil()
            A.setdiag(1)
            A = A.tocsr()
        A.sum_duplicates()  # ensure canonical format
        A.data[:] = 1

        return A

    def _smooth_laplacian_prepare(
        self, select: npt.NDArray | None = None
    ) -> tuple[npt.NDArray, scipy.sparse.csr_array, npt.NDArray, npt.NDArray | None]:
        """Precompute a few things needed when applying Laplacian smoothing
        steps.
        """
        A = self.adjacency_matrix()
        A = A[select] if select is not None else A
        # normalize rows (full A is symmetric)
        A = scipy.sparse.diags_array(1.0 / A.sum(1)) @ A
        return A

    def _smooth_laplacian_step(
        self,
        x: npt.NDArray,
        a: float,
        A: scipy.sparse.csr_array,
        select,
    ):
        """Perform a single Laplacian smoothing steps, i.e.,

            x_i = x_i + a * sum_{j in N(i)} (w_ij * (x_j - x_i))

        where N(i) is the neighborhood of i. Here we use w_ij = 1/|N(i)| where
        |N(i)| is the number of neighbors of i.

        Parameters
        ----------
        x : npt.NDArray
            The array to smooth (can be the vertex coordinates or a function
            defined on the vertices).
        a : float
            Step size.
        A :
            Vertex adjacency matrix.
        """
        x_ = x if select is None else x[select]
        return a * (A @ x - x_)

    def smooth_laplacian(
        self,
        a: float = 0.33,
        n_iter: int = 1,
        select: npt.NDArray | None = None,
        array: npt.NDArray | None = None,
        inplace: bool = False,
    ):
        """Perform a number of Laplacian smoothing steps.

        select
            Apply smoothing only to this subset.
        protect_border_edges
            Do not modify border edges.

        """
        if array is None:
            apply_to_vertices = True
            array = self.vertices if inplace else self.vertices.copy()
        else:
            apply_to_vertices = False
            assert array.shape[0] == self.n_vertices
            array = array if inplace else array.copy()

        if select is not None:
            select = _ensure_array_is_index(select, self.n_vertices)

        A = self._smooth_laplacian_prepare(select)
        for _ in range(n_iter):
            if select is None:
                array += self._smooth_laplacian_step(array, a, A, select)
            else:
                array[select] += self._smooth_laplacian_step(array, a, A, select)

        if apply_to_vertices:
            if inplace:
                self.vertices = array
                return self
            else:
                return ClosedPolyline(array)
        else:
            return array


    def smooth_taubin(
        self,
        a: float = 0.33,
        b: float = -0.34,
        n_iter: int = 1,
        select: npt.NDArray | None = None,
        array: npt.NDArray | None = None,
        inplace: bool = False,
    ):
        """Perform Taubin smoothing, i.e., Laplacian smoothing step with
        positive weight (`a`) followed by a Laplacian step with negative weight
        (`b`).

        References
        ----------
        https://graphics.stanford.edu/courses/cs468-01-fall/Papers/taubin-smoothing.pdf
        """
        assert 0 < a < -b, "a should be between 0 and -b."
        if array is None:
            apply_to_vertices = True
            array = self.vertices if inplace else self.vertices.copy()
        else:
            apply_to_vertices = False
            assert array.shape[0] == self.n_vertices
            array = array if inplace else array.copy()

        A = self._smooth_laplacian_prepare(select)

        for _ in range(n_iter):
            if select is None:
                array += self._smooth_laplacian_step(
                    array, a, A, select
                )  # regular step
                array += self._smooth_laplacian_step(array, b, A, select)  # Taubin step
            else:
                array[select] += self._smooth_laplacian_step(array, a, A, select)
                array[select] += self._smooth_laplacian_step(array, b, A, select)

        if apply_to_vertices:
            if inplace:
                self.vertices = array
                return self
            else:
                return ClosedPolyline(array)
        else:
            return array