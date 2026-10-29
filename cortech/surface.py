from pathlib import Path
import warnings

import nibabel as nib
import numpy as np
import numpy.typing as npt
from scipy.ndimage import map_coordinates
import scipy.sparse
from scipy.spatial import cKDTree

from cortech.cgal import boost_graph
from cortech.cgal import alpha_wrap_3
import cortech.cgal.polygon_mesh_processing as pmp
import cortech.cgal.surface_mesh_segmentation as ss
import cortech.cgal.surface_mesh_simplification as sms
import cortech.cgal.surface_mesh_skeletonization as smskel
import cortech.cgal.convex_hull_3

from cortech.constants import Curvature
import cortech.freesurfer
import cortech.sphere_utils
import cortech.utils
from cortech.utils import _ensure_array_is_index


def _merge_two_surfaces(s0, s1):
    vertices = np.concat((s0.vertices, s1.vertices))
    faces = np.concat((s0.faces, s0.n_vertices + s1.faces))

    # Merge data
    face_id = np.concat((s0.face_id, s1.face_id))

    vd = {}
    keys = set(s0.data.vertex.keys()).union(set(s1.data.vertex.keys()))
    for k in keys:
        in_s0 = k in s0.data.vertex
        in_s1 = k in s1.data.vertex
        if in_s0 and in_s1:
            vd[k] = np.concat((s0.data.vertex[k], s1.data.vertex[k]))
        elif in_s0:
            v = s0.data.vertex[k]
            if v.ndim > 1:
                nans = np.full((s1.n_vertices, *v.shape[1:]), np.nan)
            else:
                nans = np.full(s1.n_vertices, np.nan)
            vd[k] = np.concat((v, nans))
        elif in_s1:
            v = s1.data.vertex[k]
            if v.ndim > 1:
                nans = np.full((s0.n_vertices, *v.shape[1:]), np.nan)
            else:
                nans = np.full(s0.n_vertices, np.nan)
            vd[k] = np.concat((nans, v))

    fd = {}
    keys = set(s0.data.face.keys()).union(set(s1.data.face.keys()))
    for k in keys:
        in_s0 = k in s0.data.face
        in_s1 = k in s1.data.face
        if in_s0 and in_s1:
            fd[k] = np.concat((s0.data.face[k], s1.data.face[k]))
        elif in_s0:
            v = s0.data.face[k]
            if v.ndim > 1:
                nans = np.full((s1.n_faces, *v.shape[1:]), np.nan)
            else:
                nans = np.full(s1.n_faces, np.nan)
            fd[k] = np.concat((v, nans))
        elif in_s1:
            v = s1.data.face[k]
            if v.ndim > 1:
                nans = np.full((s0.n_faces, *v.shape[1:]), np.nan)
            else:
                nans = np.full(s0.n_faces, np.nan)
            fd[k] = np.concat((nans, v))

    return s0.new(vertices, faces, face_id=face_id, vertex_data=vd, face_data=fd)


def merge(*surfaces):
    """Merge a number of surfaces by concatenating their vertices and faces and
    updating the latter accordingly. Vertex and face data are concatenated as
    well.
    """
    out = _merge_two_surfaces(*surfaces[:2])
    for s in surfaces[2:]:
        out = _merge_two_surfaces(out, s)
    return out


class ElementData(dict):
    def __init__(self, data=None, n: int | None = None):
        self.n = n
        if data is not None:
            for k, v in data.items():
                self[k] = v

    def __setitem__(self, k, v):
        self.n = len(v) if self.n is None else self.n
        assert (
            len(v) == self.n
        ), f"Invalid size of data (expected {self.n} got {len(v)})"
        super().__setitem__(k, np.asarray(v))

    def __str__(self):
        if len(self) > 0:
            size = min(max(len(k) for k in self) + 4, 20)
            return "\n".join(
                f"\033[1m{k:{size}s}\033[0m {v.dtype} {v.shape}"
                for k, v in self.items()
            )
        else:
            return "(empty)"

    def __repr__(self):
        if len(self) > 0:
            size = min(max(len(k) for k in self) + 4, 20)
            return "\n".join(
                f"\033[1m{k:{size}s}\033[0m {v.dtype} {v.shape}"
                for k, v in self.items()
            )
        else:
            return "(empty)"

    def full(self, k, v):
        assert self.n is not None, "Size of ElementData is unknown"
        self.__setitem__(k, np.full(self.n, v))

    def new_from_subset(self, indices):
        return ElementData({k: v[indices] for k, v in self.items()}, len(indices))


class TriangulationData:
    def __init__(
        self,
        vertex: dict | ElementData | None = None,
        face: dict | ElementData | None = None,
    ):
        match vertex:
            case None:
                self.vertex = ElementData()
            case ElementData():
                self.vertex = vertex
            case dict():
                self.vertex = ElementData(vertex)
            case _:
                raise ValueError()

        match face:
            case None:
                self.face = ElementData()
            case ElementData():
                self.face = face
            case dict():
                self.face = ElementData(face)
            case _:
                raise ValueError()

    def __str__(self):
        s = "\033[1mVertex data\033[0m\n" + str(self.vertex) + "\n\n"
        s += "\033[1mFace data\033[0m\n" + str(self.face)
        return s

    def __repr__(self):
        s = "\033[1mVertex data\033[0m\n" + repr(self.vertex) + "\n\n"
        s += "\033[1mFace data\033[0m\n" + repr(self.face)
        return s


# class Vertices(np.ndarray):
#     def __init__(self, coords, data = None):
#         super().__init__()
#         self.coords = coords
#         self.data = data

#     @property
#     def coords(self):
#         return self._coords

#     @coords.setter
#     def coords(self, value):
#         value = np.atleast_2d(value)
#         assert value.ndim == 2
#         self._coords = value
#         self.n, self.dim = value.shape

#     def __array__(self):
#         return self.coords


# class Vertices:
#     def __init__(self, coords, data = None, ):
#         self.coords = coords

#     @property
#     def coords(self):
#         return self._coords

#     @coords.setter
#     def coords(self, value):
#         value = np.atleast_2d(value)
#         assert value.ndim == 2
#         self._coords = value
#         self.n, self.dim = value.shape

#     def __array__(self):
#         return self.coords


# class Faces:


class NonManifoldSurface:
    def __init__(
        self,
        vertices: npt.NDArray,
        faces: npt.NDArray,
        *,
        space: str = "scanner",
        geometry: dict | cortech.freesurfer.VolumeGeometry | None | str = "default",
        edge_pairs: npt.NDArray | None = None,
        face_id: npt.ArrayLike | int | None = None,
        vertex_data: ElementData | dict | None = None,
        face_data: ElementData | dict | None = None,
    ):
        """Class for representing a triangulated, possibly non-manifold
        (in a topological sense) surface.
        """
        self.vertices = vertices
        self.faces = faces

        self.edge_pairs = (
            np.array([[1, 2], [2, 0], [0, 1]], dtype=self.faces.dtype)
            if edge_pairs is None
            else edge_pairs
        )
        # because we know that the vertices are(0,1,2)
        self.opposing_vertex = 3 - self.edge_pairs.sum(1)

        self.face_id = face_id

        vertex_data = ElementData(vertex_data, self.n_vertices)
        face_data = ElementData(face_data, self.n_faces)
        self.data = TriangulationData(vertex_data, face_data)

        assert space in {"scanner", "surface"}
        self.space = space

        if isinstance(geometry, dict):
            self.geometry = cortech.freesurfer.VolumeGeometry(**geometry)
        elif isinstance(geometry, cortech.freesurfer.VolumeGeometry):
            self.geometry = geometry
        elif geometry is None:
            self.geometry = cortech.freesurfer.VolumeGeometry(False)
        elif geometry == "default":
            self.geometry = cortech.freesurfer.VolumeGeometry.with_defaults()
        else:
            raise ValueError("Invalid geometry")

    # Properties

    @property
    def face_id(self):
        return self._face_id

    @face_id.setter
    def face_id(self, value):
        value = np.asarray(value) if isinstance(value, (list, tuple)) else value
        if value is None:
            self._face_id = np.full(self.n_faces, 0)
        elif isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.integer):
            assert len(value) == self.n_faces and value.ndim == 1
            self._face_id = np.asarray(value)
        elif np.issubdtype(type(value), np.integer):
            self._face_id = np.full(self.n_faces, value)
        else:
            raise ValueError(f"Invalid face_id ({value})")

    @property
    def faces(self):
        return self._faces

    @faces.setter
    def faces(self, value):
        value = np.atleast_2d(np.asarray(value)).astype(int)
        assert value.ndim == 2
        self._faces = value
        self.n_faces, self.vertices_per_face = value.shape
        self.nf = self.n_faces

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, value):
        value = np.atleast_2d(np.asarray(value))
        assert value.ndim == 2
        self._vertices = value
        self.n_vertices, self.n_dim = value.shape
        self.nv = self.n_vertices

    # Methods

    def __repr__(self):
        out = f"{str(type(self))}\n"
        out += f"\033[1m# vertices\033[0m     {self.n_vertices}\n"
        out += f"\033[1m# faces\033[0m        {self.n_faces}\n\n"
        out += repr(self.data)
        return out

    def _check_face_id(self, face_id: npt.ArrayLike | None):
        """Check validity of `face_id`."""
        face_id = self.face_id if face_id is None else np.asarray(face_id)
        assert len(face_id) == self.n_faces and face_id.ndim == 1
        return face_id

    def as_gifti(self):
        header = None
        vol_geom = self.geometry.as_gifti_dict()

        if self.is_scanner_ras():
            fs_from = "scanner"
            fs_to = "tkr"
        elif self.is_surface_ras():
            fs_from = "tkr"
            fs_to = "scanner"
        nii_from = self.geometry._fsstring_to_niistring[fs_from]
        nii_to = self.geometry._fsstring_to_niistring[fs_to]
        affine = self.geometry.get_affine(fs_to, fr=fs_from)
        coordsys = nib.gifti.GiftiCoordSystem(nii_from, nii_to, affine)

        vertices = nib.gifti.GiftiDataArray(
            self.vertices.astype(np.float32),
            intent="NIFTI_INTENT_POINTSET",
            coordsys=coordsys,
            meta=vol_geom,
        )
        faces = nib.gifti.GiftiDataArray(
            self.faces.astype(np.int32),
            intent="NIFTI_INTENT_TRIANGLE",
            coordsys=coordsys,
        )
        faces.coordsys = None

        return nib.gifti.GiftiImage(header=header, darrays=[vertices, faces])

    def mesh(self, faces: npt.NDArray | None = None):
        """For each face, index into `self.vertices` and return an array with
        vertex coordinates for each face, i.e., (n_faces, n_vertices_per_face, 3).
        """
        f = self.faces if faces is None else self.faces[faces]
        return self.vertices[f]

    # def autorefine(self, apply_iterative_snap_rounding: bool = True, n_iter: int = 5):
    #     v, f = pmp.autorefine_triangle_soup(
    #         self.vertices, self.faces, apply_iterative_snap_rounding, n_iter
    #     )
    #     return self.new(v, f)

    def bounding_box(self):
        return np.stack((self.vertices.min(0), self.vertices.max(0)))

    def circumsphere(self, indices=None, return_radius: bool = True):
        """Implements the following formulae from [1]

                |                                                           |
                | |c-a|^2 [(b-a)x(c-a)]x(b-a) + |b-a|^2 (c-a)x[(b-a)x(c-a)] |
                |                                                           |
            r = -------------------------------------------------------------
                                    2 | (b-a)x(c-a) |^2

                    |c-a|^2 [(b-a)x(c-a)]x(b-a) + |b-a|^2 (c-a)x[(b-a)x(c-a)]
            c = a + ---------------------------------------------------------
                                    2 | (b-a)x(c-a) |^2

        Returns
        -------
        center
            Center of the circumsphere.
        radius
            Radius of the circumsphere.

        References
        ----------
        [1] https://ics.uci.edu/~eppstein/junkyard/circumcenter.html
        """
        m = self.mesh()
        if indices is None:
            indices = slice(None, None)
        a = m[indices, 0]
        ba = m[indices, 1] - a
        ca = m[indices, 2] - a

        ba_ca = np.linalg.cross(ba, ca)
        num1 = np.vecdot(ca, ca, keepdims=True) * np.linalg.cross(ba_ca, ba)
        num2 = np.vecdot(ba, ba, keepdims=True) * np.linalg.cross(ca, ba_ca)
        num = num1 + num2
        denom = 2 * np.vecdot(ba_ca, ba_ca, keepdims=True)

        center = a + (num1 + num2) / denom
        if return_radius:
            radius = np.linalg.norm(num, axis=-1) / denom.squeeze()

        return (center, radius) if return_radius else center

    def clear_data(self):
        """Clear all face and vertex data."""
        self.clear_face_data()
        self.clear_vertex_data()

    def clear_face_data(self):
        """Clear all face data."""
        self.data.face = ElementData(n=self.n_faces)

    def clear_vertex_data(self):
        """Clear all vertex data."""
        self.data.vertex = ElementData(n=self.n_vertices)

    def convex_hull(self):
        """Return a triangulation of the convex hull."""
        v, f = cortech.cgal.convex_hull_3.convex_hull(self.vertices)
        return self.new(v, f)

    def copy(self):
        return self.new()

    def do_intersect(self, other: "NonManifoldSurface") -> bool:
        """Return True if there exists a face in `self` and a face in `other`
        which interesects. Otherwise, return False.

        Parameters
        ----------
        other : cortech.Surface

        Returns
        -------
        True if any intersections exist, else False.
        """
        return pmp.do_intersect(self.vertices, self.faces, other.vertices, other.faces)

    def does_not_self_intersect(self):
        """Check whether the surface does *not* contain self-intersections."""
        return not self.does_self_intersect()

    def does_self_intersect(self):
        """Check whether the surface contains self-intersections."""
        return pmp.does_triangle_soup_self_intersect(self.vertices, self.faces)

    def duplicate_non_manifold_edges(self):
        """Duplicate non-manifold edges, e.g., edges which is shared by more
        than two faces.
        """
        v, f = pmp.duplicate_non_manifold_edges_in_polygon_soup(
            self.vertices, self.faces
        )
        # new vertices are pushed back to the input array
        vid = np.arange(len(v))
        vid[self.n_vertices :] = -1
        vd = dict(original_id=vid)
        return self.new(v, f, vertex_data=vd, keep_face_id=True, keep_face_data=True)

    def duplicate_non_manifold_vertices(self):
        """Duplicate non-manifold vertices, i.e., vertices that are "pintched"."""
        v, f, vid, _ = pmp.duplicate_non_manifold_vertices(self.vertices, self.faces)
        vd = dict(original_id=vid)
        return self.new(v, f, vertex_data=vd, keep_face_id=True, keep_face_data=True)

    def edges(
        self,
        select: npt.ArrayLike | None = None,
        unique: bool = False,
        sort_axis_0: bool = False,
        sort_axis_1: bool = False,
        retain_edge_order: bool = True,
    ):
        """Return the indices of vertex pairs making up the edges of the faces.

        Parameters
        ----------
        select :
            Indices of the faces for which to return the edges (by default, all
            faces).
        unique : bool
            Return only unique edges. Implies `sort_axis_1=True`.
        sort_axis_0 : bool
            If True, sort edges along this axis. Implies `sort_axis_1=True`.
        sort_axis_1 : bool
            If True, sort edges along this axis.
        retain_edge_order : bool
            If True, unique edges are not sorted (which they would be by
            default when using np.unique). Only applies if `unique=True`.

        Returns
        -------
        edges
            An array of shape (n_edges, 2). This can be reshaped into
            (-1, 3, 2) where the ith row corresponds to the edges for the ith
            face.
        """
        # force sorting of axis 1
        sort_axis_1 = unique or sort_axis_0 or sort_axis_1

        faces = self.faces if select is None else self.faces[np.asarray(select)]
        edges = faces[:, self.edge_pairs]
        edges = edges.reshape((-1, 2))
        edges = np.sort(edges, axis=1) if sort_axis_1 else edges
        if unique:
            u, uidx, uinv = np.unique(edges, True, True, axis=0)
            if retain_edge_order:
                enumerated_edges = uidx.argsort().argsort()[uinv]
                edges = edges[np.unique(enumerated_edges, True)[1]]
            else:
                edges = u
        else:
            if sort_axis_0:
                s0 = edges[:, 1].argsort()
                s1 = edges[s0, 0].argsort(stable=True)
                edges = edges[s0[s1]]
        return edges

    def edge_vectors(self, edges: npt.ArrayLike | None = None):
        """Return the edge vectors for `edges` which is an (n, 2) array of
        vertex indices. The edge vectors point from edges[:, 0] to edges[:, 1].

        Parameters
        ----------
        edges :
            Array of shape (..., 2) where the last dimension contains vertex
            indices forming an edge (by default, all edges).

        Returns
        -------
        """
        e = self.edges() if edges is None else np.asarray(edges)
        edge_mesh = self.vertices[e]  # (n_edges, coords)
        return np.diff(edge_mesh, axis=-2).squeeze()

    def edges_norm(self, edges: npt.NDArray | None = None):  # , unique: bool = False
        """Returns the norm (length) of the vectors formed by `edges`.

        Parameters
        ----------
        edges
            Array of shape (..., 2) where the last dimension contains vertex
            indices forming an edge (by default, all edges).

        Returns
        -------
        edge norms
        """
        ev = self.edge_vectors(edges)  # ([n_edges,] 2)
        return np.linalg.norm(ev, axis=-1)

    def face_centers(self, select=None):
        """Return the center of each face.

        Parameters
        ----------
        select :
            Indices or boolean mask of faces for which to calculate centers (by
            default, all faces).

        """
        return self.mesh(select).mean(-2)

    def face_areas(self, select: npt.ArrayLike | None = None):
        """Return the area of each face.

        Parameters
        ----------
        select :
            Indices or boolean mask of faces for which to calculate area (by
            default, all faces).

        """
        return 0.5 * np.linalg.norm(self._face_normals_unnormalized(select), axis=1)

    def _face_normals_unnormalized(self, select: npt.ArrayLike | None = None):
        m = self.mesh(select)
        return np.cross(m[:, 1] - m[:, 0], m[:, 2] - m[:, 0]).astype(float)

    def face_normals(self, select: npt.NDArray | None = None):
        """Compute normal vectors for each face in the mesh.

        PARAMETERS
        ----------
        select : ndarray
            A subset of the faces for which to calculate the normals (default
            is all faces).

        RETURNS
        ----------
        face_normals : ndarray
            Normal vectors of each face.
        """
        tnormals = self._face_normals_unnormalized(select)
        tnormals /= np.linalg.norm(tnormals, axis=1, keepdims=True)
        return tnormals

    def face_range(self):
        """Return an array [0, 1, ..., self.n_faces -1]."""
        return np.arange(self.n_faces)

    def face_selection_to_vertex_selection(self, select: npt.ArrayLike):
        """Get the (unique) vertex indices corresponding to the faces in
        `select`.

        Parameters
        ----------
        select
            Indices or boolean mask of selected faces.

        """
        return np.unique(self.faces[select])

    def faces_to_edges(self, retain_edge_order: bool = True):
        """Compute a mapping of faces to unique edges."""
        # if not unique:
        #     return np.arange(3*self.n_faces).reshape(-1, 3)

        e = self.edges(sort_axis_1=True)
        _, uidx, uinv = np.unique(e, axis=0, return_index=True, return_inverse=True)
        uinv = uinv.reshape(-1, 3)
        if retain_edge_order:
            # double argsort remaps uidx to (0, ..., len(uidx)), e.g.,
            #             argsort           argsort
            #   5 7 0 2 1  ---->  2 4 3 0 1  ---->  3 4 0 2 1
            uidx_remap = uidx.argsort().argsort()
            return uidx_remap[uinv]
        else:
            return uinv

    def intersections_with(
        self, other: "NonManifoldSurface", return_unique: bool = False
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Compute intersecting pairs of triangles between `self` and `other`.

        Parameters
        ----------
        other : cortech.Surface
        return_unique : bool
            Return unique indices of intersecting faces for self and other.

        Returns
        -------
        intersect_pairs: tuple[npt.NDArray, npt.NDArray]
            Tuple of length two. `intersect_pairs[0]` contains the intersecting
            faces of self, `intersect_pairs[1]` contains the intersecting
            faces of other.
            If `return_unique = False`, then
            `len(intersect_pairs[0]) == len(intersect_pairs[1])` and each
            position correspond to a pair of intersecting faces.
            If `return_unique = True`, then each array corresponds to the
            *unique* intersecting faces of each surface.

        Notes
        -----
        CGAL/Polygon_mesh_processing/intersection.h
        PMP::internal::compute_face_face_intersection
        """
        intersect_pairs = pmp.intersecting_meshes(
            self.vertices, self.faces, other.vertices, other.faces
        ).T
        if intersect_pairs.size == 0:
            return np.array([], dtype=int), np.array([], dtype=int)
        else:
            if return_unique:
                return np.unique(intersect_pairs[0]), np.unique(intersect_pairs[1])
            else:
                return intersect_pairs[0], intersect_pairs[1]

    def is_closed(self):
        """Check if the surface is closed, i.e., if all edges are shared by
        exactly two faces.
        """
        _, c = np.unique(self.edges(sort_axis_1=True), return_counts=True, axis=0)
        return np.all(c == 2)

    def is_manifold(self, select: npt.ArrayLike | None = None):
        """Check if the surface is *topologically* manifold. The check is
        performed using

            CGAL::Polygon_mesh_processing::is_polygon_soup_a_polygon_mesh

        which checks that
            1. each edge has at most two incident faces and such an edge is
                visited in opposite direction along the two face boundaries,
            2. no polygon has twice the same vertex, and
            3. the polygon soup describes a manifold surface.

        Parameters
        ----------
        select
            Indices or boolean mask of a subset of the faces. The topology of
            this selection of faces is checked.

        Returns
        -------


        Notes
        -----

        """
        faces = self.faces if select is None else self.faces[np.asarray(select)]
        return pmp.is_polygon_soup_a_polygon_mesh(faces)

    def merge_duplicate_faces(self, keep_face_data: bool = False, cls=None):
        """Merge duplicate faces. Two faces are duplicate if they share the
        same vertices in the same order.

        Parameters
        ----------
        keep_vertex_data : bool
            If true, this will find, for each point in the merged vertex pool,
            the corresponding point in the original vertex pool. For the merged
            vertices, the vertex data that is kept is arbitrary.

        """
        v, f = pmp.merge_duplicate_polygons_in_polygon_soup(self.vertices, self.faces)
        if keep_face_data:
            tree = scipy.spatial.KDTree(self.face_centers)
            _, index = tree.query(v[f].mean(-2))
            fd = self.data.face.new_from_subset(index)
        else:
            fd = None
        return self.new(
            v, f, keep_face_id=True, keep_vertex_data=True, cls=cls, face_data=fd
        )

    def merge_duplicate_points(self, keep_vertex_data: bool = False, cls=None):
        """Merge duplicate points.

        keep_vertex_data : bool
            If true, this will find, for each point in the merged vertex pool,
            the corresponding point in the original vertex pool. For the merged
            vertices, the vertex data that is kept is arbitrary.

        """
        v, f = pmp.merge_duplicate_points_in_polygon_soup(self.vertices, self.faces)
        if keep_vertex_data:
            tree = scipy.spatial.KDTree(self.vertices)
            _, index = tree.query(v)
            vd = self.data.vertex.new_from_subset(index)
        else:
            vd = None
        return self.new(
            v, f, keep_face_id=True, keep_face_data=True, cls=cls, vertex_data=vd
        )

    def non_manifold_vertices(self):
        """Return an array vertex indices with non-manifold topology."""
        return pmp.non_manifold_vertices(self.vertices, self.faces)

    def orient_faces(self, keep_vertex_data: bool = False, cls=None):
        """If the operation succeeded without duplicating any points, then
        vertex is always kept. If the operation only succeeded by duplicating
        point (thus producing a combinatorically manifold but self-intersecting
        mesh), then vertex is *not* kept by default, unless `keep_vertex_data`
        is true, in which case the data from the closest original vertex is
        retained.

        Parameters
        ----------


        """
        kw = dict(keep_face_data=True, cls=cls)
        success, (v, f) = pmp.orient_polygon_soup(self.vertices, self.faces)
        if success:
            return self.new(v, f, keep_vertex_data=True, **kw)
        else:
            if keep_vertex_data:
                tree = scipy.spatial.KDTree(self.vertices)
                _, index = tree.query(v)
                vd = self.data.vertex.new_from_subset(index)
                return self.new(v, f, vertex_data=vd, **kw)
            else:
                return self.new(v, f, **kw)

    def astype(self, cls):
        """Return self as another surface class."""
        return self.new(
            cls=cls, keep_face_id=True, keep_vertex_data=True, keep_face_data=True
        )

    def as_manifold_surface(self):
        """Try to convert `self` to a ManifoldSurface."""
        return self.astype(ManifoldSurface)

    def new(
        self,
        vertices: npt.NDArray | None = None,
        faces: npt.NDArray | None = None,
        cls=None,
        *,
        keep_face_id: bool = False,
        keep_vertex_data: bool = False,
        keep_face_data: bool = False,
        **kwargs,
    ):
        """Return a new instance of this object which inherits certain
        properties of `self`.

        Note that vertex and face data are not automatically carried over!

        """
        vertices = self.vertices.copy() if vertices is None else vertices
        faces = self.faces.copy() if faces is None else faces
        if keep_face_id:
            kwargs["face_id"] = self.face_id.copy()
        if keep_vertex_data:
            assert "vertex_data" not in kwargs
            kwargs["vertex_data"] = self.data.vertex.copy()
        if keep_face_data:
            assert "face_data" not in kwargs
            kwargs["face_data"] = self.data.face.copy()
        cls = type(self) if cls is None else cls
        return cls(
            vertices,
            faces,
            space=self.space,
            geometry=self.geometry,
            edge_pairs=self.edge_pairs,
            **kwargs,
        )

    def plot(
        self,
        scalars: npt.ArrayLike | str | None = None,
        mesh_kwargs=None,
        plotter_kwargs=None,
        show=True,
    ):
        """

        scalars:
            If "face_id", then `self.face_id` will be used as scalar
        """

        # only works when pyvista is installed
        from cortech.visualization import plot_surface

        plotter = plot_surface(
            self, scalars, mesh_kwargs=mesh_kwargs, plotter_kwargs=plotter_kwargs
        )
        if show:
            plotter.show()
        return plotter

    def remove_unused_vertices(
        self, overwrite_original_id: bool = True, inplace: bool = False
    ):
        """Remove unused vertices and reindex the faces accordingly."""
        vertices_used = np.unique(self.faces)
        reindexer = np.zeros(self.n_vertices, dtype=self.faces.dtype)
        reindexer[vertices_used] = np.arange(vertices_used.size, dtype=self.faces.dtype)

        v = self.vertices[vertices_used]
        f = reindexer[self.faces]

        vertex_data = self.data.vertex.new_from_subset(vertices_used)
        if overwrite_original_id or ("original_id" not in vertex_data):
            vertex_data["original_id"] = vertices_used

        if inplace:
            self.vertices = v
            self.faces = f
            self.data.vertex = vertex_data
            return self
        else:
            return self.new(
                v, f, keep_face_id=True, keep_face_data=True, vertex_data=vertex_data
            )

    def remove_faces(
        self,
        select: npt.ArrayLike,
        overwrite_original_id: bool = True,
        inplace: bool = False,
    ):
        """Remove the specified faces. The surface will be remove_unused_vertices afterwards,
        removing any unused vertices.

        Parameters
        ----------
        select
            Indices or boolean mask indicating the faces to remove.
        """
        # invert selection
        select = _ensure_array_is_index(select, self.n_faces, invert=True)
        return self.select_faces(select, overwrite_original_id, inplace)

    def remove_vertices(
        self,
        select: npt.ArrayLike,
        overwrite_original_id: bool = True,
        inplace: bool = False,
    ):
        """Remove the specified vertices along with any faces that reference
        these vertices.

        Paramters
        ---------
        select :
            Indices or boolean mask indicating the vertices to remove.

        Returns
        -------

        """
        # invert selection
        select = _ensure_array_is_index(select, self.n_vertices, invert=True)
        return self.select_vertices(select, overwrite_original_id, inplace)

    def reverse_face_orientation(self, inplace: bool = False):
        """Reverse the face orientation by reversing the rows of `self.faces`.
        """
        if inplace:
            self.faces = self.faces[:, ::-1]
            return self
        else:
            return self.new(
                faces=self.faces[:, ::-1].copy(),
                keep_face_id=True,
                keep_vertex_data=True,
                keep_face_data=True,
            )

    def save(self, filename: Path | str, scalars: dict | None = None):
        filename = Path(filename)
        _warn_msg = "`scalars` were provided but {ext} format does not support data fields (ignoring them)"
        if scalars is not None and filename.suffix in {".gii", ".off", ".stl"}:
            warnings.warn(_warn_msg.format(ext=filename.suffix))

        match filename.suffix:
            case ".gii":
                self.as_gifti().to_filename(filename)
            case ".obj" | ".stl" | ".vtk":
                import pyvista as pv

                m = pv.make_tri_mesh(self.vertices, self.faces)
                m["face_id"] = self.face_id
                for k, v in self.data.vertex.items():
                    m[k] = v
                for k, v in self.data.face.items():
                    m[k] = v
                if scalars is not None:
                    for k, v in scalars.items():
                        m[k] = v
                m.save(filename)
            case ".off":
                self.save_off(filename)
            case _:
                if scalars is not None:
                    warnings.warn(_warn_msg.format(ext="FreeSurfer's surface"))
                cortech.freesurfer.write_geometry(
                    filename,
                    self.vertices,
                    self.faces,
                    real_ras=self.is_scanner_ras(),
                    vol_geom=self.geometry.as_freesurfer_dict(),
                )

    def select_faces(
        self,
        select: npt.ArrayLike,
        overwrite_original_id: bool = True,
        inplace: bool = False,
    ):
        """Return a new instance of `self` constrained to the faces in
        `select`.

        Parameters
        ----------
        select : npt.ArrayLike
            Selection specified as indices or boolean array.
        overwrite_original_id : bool
            If True, the indices in `original_id` will point to the surface
            from which the selection was made. If False and `original_id`
            exists (e.g., from a previous selection), the values will be kept
            thus referring back to the surface where the initial selection was
            made (default = True).
        inplace : bool
            Modify this instance in-place. Otherwise, return a new instance
            (default = False)

        Returns
        -------
            A new instance of `self` constrained to the selection.

        """
        select = _ensure_array_is_index(select, self.n_faces)

        selected_faces = self.faces[select]
        face_id = self.face_id[select]
        face_data = self.data.face.new_from_subset(select)
        if overwrite_original_id or ("original_id" not in face_data):
            face_data["original_id"] = select
        if inplace:
            self.faces = selected_faces
            self.face_id = face_id
            self.data.face = face_data
            out = self
        else:
            out = self.new(
                faces=selected_faces,
                face_id=face_id,
                keep_vertex_data=True,
                face_data=face_data,
            )
        out = out.remove_unused_vertices(overwrite_original_id, inplace)
        return out

    def select_vertices(
        self,
        select: npt.ArrayLike,
        overwrite_original_id: bool = True,
        inplace: bool = False,
    ):
        """Return a new instance of `self` constrained to the faces which
        reference the vertices in `select`.

        Parameters
        ----------
        select : npt.ArrayLike
            Selection specified as indices or boolean array.
        overwrite_original_id
            If True, the indices in `original_id` will point to the surface
            from which the selection was made. If False and `original_id`
            exists (e.g., from a previous selection), the values will be kept
            thus referring back to the surface where the initial selection was
            made (default = True).
        inplace : bool
            Modify this instance in-place. Otherwise, return a new instance
            (default = False)

        Returns
        -------
            A new instance of `self` constrained to the selection.
        """
        faces_selected = self.vertex_selection_to_face_selection(select)
        return self.select_faces(faces_selected, overwrite_original_id, inplace)

    def self_intersections(self):
        """Compute intersecting pairs of faces."""
        return pmp.triangle_soup_self_intersections(self.vertices, self.faces)

    def save_off(self, filename):
        """Writes mesh surfaces as an .off file

        Parameters
        -----------
        msh: Mesh
            Mesh object
        fn: str
            Name of file
        """
        with open(filename, "wb") as f:
            f.write("OFF\n".encode())
            f.write("# File created by CORTECH \n\n".encode())
            np.savetxt(
                f, np.array([self.n_vertices, self.n_faces, 0])[None, :], fmt="%u"
            )
            np.savetxt(f, self.vertices, fmt="%0.6f")
            np.savetxt(
                f,
                np.concatenate(
                    (np.repeat(self.faces.shape[1], self.n_faces)[:, None], self.faces),
                    axis=1,
                ).astype(np.uint),
                fmt="%u",
            )

    def is_surface_ras(self):
        return self.space == "surface"

    def is_scanner_ras(self):
        return self.space == "scanner"

    def to_scanner_ras(self, *, inplace: bool = True):
        """Convert the surface vertex coordinates to scanner RAS.

        Parameters
        ----------
        inplace : bool
            Modify this instance in-place. Otherwise, return a new instance
            (default = True)
        """
        if self.is_surface_ras():
            trans = self.geometry.get_affine("scanner", fr="tkr")
            v = nib.affines.apply_affine(trans, self.vertices)
            if inplace:
                self.vertices = v
                self.space = "scanner"
        else:
            v = self.vertices
        return v

    def to_surface_ras(self, *, inplace: bool = True):
        """Convert the surface vertex coordinates to surface RAS.

        Parameters
        ----------
        inplace : bool
            Modify this instance in-place. Otherwise, return a new instance
            (default = True)
        """
        if self.space == "scanner":
            trans = self.geometry.get_affine("tkr", fr="scanner")
            v = nib.affines.apply_affine(trans, self.vertices)
            if inplace:
                self.vertices = v
                self.space = "surface"
        else:
            v = self.vertices
        return v

    def vertex_range(self):
        """Return an array [0, 1, ..., self.n_vertices -1]."""
        return np.arange(self.n_vertices)

    def vertex_selection_to_face_selection(self, select: npt.ArrayLike, n: int = 3):
        """Convert a vertex selection to a face selection.

        Parameters
        ----------
        select : npt.ArrayLike
            Selection specified as indices or boolean array.
        n : int
            The number of selected vertices that a face must contain in order
            to be included in the selection (default = 3).

        Returns
        -------
        face selection
            The selection will be indices or boolean array depending on dtype
            of `selection` (e.g., boolean vertex selection will return a
            boolean face selection).
        """
        assert 0 < n <= 3
        select = np.asarray(select)
        if select.dtype == bool:
            assert len(select) == self.n_vertices
            valid_vertices = select[self.faces]
            return valid_vertices.sum(1) >= n
        else:
            valid_vertices = np.isin(self.faces, select)
            return np.flatnonzero(valid_vertices.sum(1) >= n)

    # constructors

    @classmethod
    def from_file(cls, filename: Path | str, **kwargs):
        filename = Path(filename)

        match filename.suffix:
            case ".gii":
                return cls.from_gifti(filename, **kwargs)
            case ".obj" | ".stl" | ".vtk" | ".vtp":
                return cls.from_vtk(filename, **kwargs)
            case _:
                # if it doesn't match any of the above extensions, assume
                # FreeSurfer format
                return cls.from_freesurfer(filename, **kwargs)

    @classmethod
    def from_vtk(cls, filename: Path | str, **kwargs):
        """

        Parameters
        ----------
        filename : Path | str
            File to read.

        Returns
        -------
        Surface :
            Instance of self.
        """
        import pyvista as pv

        m = pv.read(filename)
        kwargs["vertex_data"] = dict(m.point_data.items())
        kwargs["face_data"] = dict(m.cell_data.items())
        faces = m.cells if m.faces is None else m.faces
        return cls(m.points, faces.reshape(-1, 4)[:, 1:], **kwargs)

    @classmethod
    def from_freesurfer(cls, filename: Path | str, **kwargs):
        """Read default and .srf files from FreeSurfer.


        Parameters
        ----------
        filename : Path | str
            File to read.

        Returns
        -------
        Surface :
            Instance of self.

        """
        v, f, m = cortech.freesurfer.read_geometry(filename, read_metadata=True)
        space = "scanner" if m.real_ras else "surface"
        geometry = cortech.freesurfer.VolumeGeometry(**m.vol_geom)
        return cls(v, f, space=space, geometry=geometry, **kwargs)

    @classmethod
    def from_freesurfer_subject_dir(cls, subject_dir: Path | str, surface: str):
        special_subjects = ("bert", "fsaverage", "fsaverage6", "fsaverage5")
        if any(i == subject_dir for i in special_subjects):
            assert cortech.freesurfer.HAS_FREESURFER, "Could not find FREESURFER_HOME"
            subject_dir = cortech.freesurfer.HOME / "subjects" / subject_dir

        subject_dir = Path(subject_dir)
        filename = subject_dir / "surf" / surface
        if filename.exists():
            return cls.from_freesurfer(filename)
        elif (filename_gii := filename.parent / f"{filename.name}.gii").exists():
            return cls.from_gifti(filename_gii)
        else:
            raise ValueError(
                f"Unable to find {surface} in {subject_dir}. Tried {filename} and {filename_gii}."
            )

    @classmethod
    def from_gifti(cls, filename: nib.GiftiImage | Path | str, **kwargs):
        """Read surface from Gifti file. Will also read the following metadata
        from FreeSurfer if present

        - real_ras
        - volume geometry
            VolGeomWidth
            VolGeomHeight
            VolGeomDepth
            VolGeomXsize
            VolGeomYsize
            VolGeomZsize
            VolGeomX_R
            VolGeomX_A
            VolGeomX_S
            VolGeomY_R
            VolGeomY_A
            VolGeomY_S
            VolGeomZ_R
            VolGeomZ_A
            VolGeomZ_S
            VolGeomC_R
            VolGeomC_A
            VolGeomC_S

        Parameters
        ----------
        filename : Path | str
            File to read.

        Returns
        -------
        Surface :
            Instance of self.
        """
        if isinstance(filename, (Path, str)):
            gii = nib.load(filename)
        else:
            gii = filename
            assert isinstance(gii, nib.GiftiImage)
        v = gii.agg_data("NIFTI_INTENT_POINTSET").astype(float)
        f = gii.agg_data("NIFTI_INTENT_TRIANGLE")
        space, geometry = cortech.freesurfer.metadata.read_metadata_gifti(gii)
        return cls(v, f, space=space, geometry=geometry, **kwargs)


class ManifoldSurface(NonManifoldSurface):
    def __init__(
        self,
        vertices: npt.NDArray,
        faces: npt.NDArray,
        *,
        check_topology: bool = True,
        **kwargs,
    ) -> None:
        """Class for representing a triangulated manifold surface. The surface
        is required to be topologically manifold, e.g., an edge belongs to
        either one (boundary) or two (internal) triangles but can have self-
        intersections.

        Parameters
        ----------
        vertices
        faces

        Notes
        -----
        Validity check of the triangulation is implemented using

            CGAL::Polygon_mesh_processing::is_polygon_soup_a_polygon_mesh

        From the description of this function

            It checks that each edge has at most two incident faces and such an
            edge is visited in opposite direction along the two face
            boundaries, no polygon has twice the same vertex, and the polygon
            soup describes a manifold surface.

        The check is purely topological.
        """
        if check_topology:
            assert pmp.is_polygon_soup_a_polygon_mesh(
                faces
            ), "Triangulation does not define a manifold surface."
        super().__init__(vertices, faces, **kwargs)

    # def is_valid(self):
    #     return self.n_faces == self.n_vertices * 2 - 4

    def apply_affine(
        self, affine: npt.NDArray, move: bool = True, inplace=False
    ) -> npt.NDArray:
        """Apply an affine to an array of points.

        Parameters
        ----------
        vertices : npt.NDArray
            Node coordinates
        affine : npt.NDArray
            A 4x4 array defining the vox2world transformation.
        move : bool
            If True (default), apply translation.

        Returns
        -------
        out_coords : shape = (3,) | (n,
            Transformed point(s).
        """

        # apply rotation & scale
        out_coords = np.dot(self.vertices, affine[:3, :3].T)
        # apply translation
        if move:
            out_coords += affine[:3, 3]

        if inplace:
            self.vertices = out_coords
            return self
        else:
            return self.new(out_coords)

    def as_nonmanifold_surface(self):
        """Convert `self` to NonManifoldSurface."""
        return self.astype(NonManifoldSurface)

    def clip(self, p: npt.ArrayLike, d: npt.ArrayLike, inplace: bool = False):
        """Clip `self` by keeping the part that is on the negative side of a
        plane defined by a point `p` and a direction `d`. (I.e., the side
        opposite to the direction of `d` is kept.)


        Parameters
        ----------
        p : npt.ArrayLike
            Point on the plane.
        d : npt.ArrayLike
            Direction vector of the plane.
        inplace : bool, optional
            Modify the surface in place(default = False).

        Returns
        -------


        Notes
        -----
        CGAL::Polygon_mesh_processing::clip
        """
        p = np.asarray(p)
        assert p.ndim == 1 and p.size == 3
        d = np.asarray(d)
        assert d.ndim == 1 and d.size == 3

        v, f = pmp.clip(self.vertices, self.faces, p, d)
        if inplace:
            self.vertices = v
            self.faces = f
        else:
            return self.new(v, f)

    def connected_components(self, connectivity: str = "face", select=None):
        A = self.adjacency_matrix(connectivity)
        A = A if select is None else A[select][:, select]
        _, cc = scipy.sparse.csgraph.connected_components(
            A, directed=False, return_labels=True
        )
        cc_size = np.bincount(cc)
        sorter = cc_size.argsort()[::-1]  # sort: largest first
        return sorter[cc], cc_size[sorter]

    def connected_components_cgal(self, constrained_edges: npt.NDArray | None = None):
        """Compute connected components on the surface using CGAL.

        Parameters
        ----------
        constrained_edges
            Array of size (n, 2) defining constrained edges. Connected
            components cannot include constrained edges.

        Returns
        -------
        component_label : npt.NDArray
            The label associated with each face.
        component_size : npt.NDArray
            The size associated with each label.
        """
        return pmp.connected_components(self.vertices, self.faces, constrained_edges)

    def corefine(self, other: "ManifoldSurface", return_intersection: bool = False):
        """Corefine the `self` with `other`. Corefinement resolves all
        intersections between two meshes by embedding the intersection lines
        into each mesh.

        Parameters
        ----------
        return_intersection : bool
            If True, returns a tuple of size two containing the intersection
            edges of `self` and `other`, respectively.

        Returns
        -------
        _type_
            If `return_intersection` is False then a tuple of

                corefined self
                corefined_other

            else a tuple of

                corefined self
                corefined other
                intersection edges of self
                intersection edges of other

        Notes
        -----
        CGAL::Polygon_mesh_processing::corefine
        """
        co_self, co_other, e_self, e_other = pmp.corefine(
            self.vertices,
            self.faces,
            other.vertices,
            other.faces,
            self.face_id,
            other.face_id,
            return_intersection,
        )
        out = (
            self.new(
                co_self[0],
                co_self[1],
                face_id=co_self[2],
                vertex_data=dict(original_id=co_self[3]),
                face_data=dict(original_id=co_self[4]),
            ),
            other.new(
                co_other[0],
                co_other[1],
                face_id=co_other[2],
                vertex_data=dict(original_id=co_other[3]),
                face_data=dict(original_id=co_other[4]),
            ),
        )
        return out + (e_self, e_other) if return_intersection else out

    def curvature(self, percentile_clip_range=(0.1, 99.9), smooth_iter: int = 0):
        """Compute principal, mean, and Guassian curvatures.

        Parameters
        ----------
        niter : int:
            Number of smoothing iterations. Defaults to 0.

        Returns
        -------
        curvature : dict
            k1,k2 : principal curvatures, i.e., the directions of maximum and
                    minimum curvature, respectively.
            H     : mean curvature
            K     : Gaussian curvature
        """
        D, _ = self.principal_curvatures()

        if percentile_clip_range is not None:
            clip_range = np.percentile(D, percentile_clip_range, axis=0)
            for i, (low, hi) in enumerate(clip_range.T):
                D[:, i] = np.clip(D[:, i], low, hi)

        k1, k2 = np.ascontiguousarray(D.T)
        H = D.mean(1)
        K = D.prod(1)

        if smooth_iter > 0:
            k1, k2, H, K = np.ascontiguousarray(
                self.smooth_laplacian(
                    np.stack((k1, k2, H, K), axis=1), n_iter=smooth_iter
                ).T
            )

        return Curvature(k1=k1, k2=k2, H=H, K=K)
        # store the curvature directions as well
        # self.curv_vec = Curvature(k1=E[:, 0], k2=[E[:, 1]])

    def detect_sharp_edges(self, angle: float = 60.0):
        """Return edges whose face pairs form a dihedral angle larger than
        `angle.
        """
        return pmp.detect_sharp_edges(self.vertices, self.faces, angle)

    def distance(self, other: "ManifoldSurface", accelerate: bool | str = "barycenter"):
        return self.distance_query(other.vertices, accelerate)

    def distance_query(
        self, query_points: npt.NDArray, accelerate: bool | str = "barycenter"
    ):
        """Query the distance between `query_points` and the surface.

        Parameters
        ----------
        query_points:
            The points to query distances for.
        accelerate:
            If True, use the default acceleration in CGAL's AABB tree package
            (provided by `accelerate_distance_queries`). If False, explicitly
            set `do_not_accelerate_distance_queries`. It seems that the
            acceleration is unstable when many (e.g., >150K) primitive (faces)
            are used for the AABB tree construction. Therefore, using
            `barycenter` will calculate the closest barycenter on the surface
            to each query point and use this as the "query hint". This seems to
            work well so we set this as the default.
        """
        assert isinstance(accelerate, bool) or accelerate in {
            "barycenter",
        }
        if accelerate == "barycenter":
            barycenters = self.face_centers()
            tree = cKDTree(barycenters)
            _, index = tree.query(query_points)
            query_hints = barycenters[index]
        else:
            query_hints = None

        return cortech.cgal.aabb_tree.distance(
            self.vertices, self.faces, query_points, query_hints, accelerate
        )

    def euler_characteristic(self):
        n_edges = len(self.edges(unique=True))
        return self.n_vertices - n_edges + self.n_faces

    def expand_selection(
        self, select: npt.ArrayLike, k: int = 1, connectivity: str = "vertex"
    ):
        select = np.asarray(select)
        if input_is_bool := select.dtype == bool:
            select = np.flatnonzero(select)
        select = np.atleast_2d(select)

        # expand = boost_graph.expand_vertex_selection(self.vertices, self.faces, selection, k)
        expand = self.k_ring_neighbors(select, k, connectivity=connectivity)[0][0]

        if input_is_bool:
            match connectivity:
                case "vertex":
                    n = self.n_vertices
                case "face":
                    n = self.n_faces
            expand_as_bool = np.zeros(n, bool)
            expand_as_bool[expand] = True
            return expand_as_bool
        else:
            return expand

    def expand_face_selection(self, select: npt.ArrayLike, k: int = 1):
        """Expand a selection of faces by the kth ring neighborhood."""
        return self.expand_selection(select, k, "face")

    def expand_face_selection_for_removal(self, select: npt.ArrayLike):
        """Expand a face selection so that when removed, they will not create
        non-manifold vertices.
        """
        select = _ensure_array_is_index(select, self.n_faces)
        return boost_graph.expand_face_selection_for_removal(
            self.vertices, self.faces, select
        )

    def expand_vertex_selection(self, select: npt.ArrayLike, k: int = 1):
        """Expand a selection of vertices by the kth ring neighborhood."""
        return self.expand_selection(select, k, "vertex")

    def extract_boundary_cycles(self):
        return pmp.extract_boundary_cycles(self.vertices, self.faces)

    def fair(self, select, continuity: int = 1, inplace: bool = False):
        """Fair the selected vertices. Fairing smoothes the selection by
        computing an energy (e.g., spring, bending) minimizing approximation to
        the local neighborhood of vertices.

        Parameters
        ----------
        select :
            Indices or boolean mask of the vertices to fair.

        """
        select = _ensure_array_is_index(select, self.n_vertices)
        v = pmp.fair(self.vertices, self.faces, select, continuity)
        if inplace:
            self.vertices = v
        else:
            return self.new(
                v, keep_face_id=True, keep_vertex_data=True, keep_face_data=True
            )

    def find_border_edges(self):
        """Find border edges of `self`, i.e., edges which are only part of a
        single face.
        """
        return boost_graph.find_border_edges(self.vertices, self.faces)

    def find_patch_edges(
        self, patches=None, face_id: npt.ArrayLike | None = None,
    ):
        """Find the edges separating different patches. By default, patches are
        defined by `self.face_id`

        Parameters
        ----------
        face_id
            The ID of each face. If None, then the internal `face_id` map is
            used.

        """
        # if patches is not None:
        #     has_patch = True
        #     assert len(patches) == 2
        # else:
        #     has_patch = False
        has_patch = patches is not None

        face_id = self._check_face_id(face_id)

        A = self.adjacency_matrix("face")
        if has_patch:
            patch_faces = np.flatnonzero(np.isin(face_id, patches))
            A = A[patch_faces][:, patch_faces]
        A = A.tocoo()

        row = patch_faces[A.row] if has_patch else A.row
        col = patch_faces[A.col] if has_patch else A.col

        polyline_edges = face_id[row] != face_id[col]
        face_pairs = np.stack((row[polyline_edges], col[polyline_edges]), -1)
        face_pair_vertices = np.sort(self.faces[face_pairs].reshape(-1, 6), 1)
        does_occur_twice = np.diff(face_pair_vertices, axis=1) == 0
        assert np.all(does_occur_twice.sum(1) == 2)  # sanity

        return np.unique(
            face_pair_vertices[:, 1:][does_occur_twice].reshape(-1, 2), axis=0
        )

    def flip_edges(self, edges: npt.ArrayLike, allow_normal_flips: bool = False):
        """Flip the specified edges of the surface.


        Parameters
        ----------
        edges :
            Array of shape (n,2), where n is the number of edges, containing
            vertex indices.
        allow_normal_flips
            If False, a flip that results in a face inverting, will not be
            performed (default = False).

        Returns
        -------
        _type_
            _description_


        """
        edges = np.asarray(edges)
        assert edges.ndim == 2 and edges.shape[1] == 2
        v, f, fid, v_pmap, f_pmap = pmp.flip_edges(
            self.vertices, self.faces, edges, allow_normal_flips, self.face_id
        )
        vd = dict(original_id=v_pmap)
        fd = dict(original_id=f_pmap)
        return self.new(v, f, face_id=fid, vertex_data=vd, face_data=fd)

    def genus(self):
        return 1 - self.euler_characteristic() // 2

    def halfedge_to_face_matrix(self, one_based_index: bool = False):
        """Return a matrix which maps halfedges (directed edges) to face
        indices. That is A[i,j] and A[j,i] will be neighboring faces sharing
        the edge (i,j). If (i,j) is a border edge, one of these will return an
        invalid index (i.e., 0).

        Parameters
        ----------
        one_based_indexing
            If True, the matrix stores `face_index + 1` instead of
            `face_index`. This is useful if one is unsure whether subsequent
            lookups correspond to a valid halfedge. If not, `csr_array` will
            return 0 hence making it indistinguishable from "real" zero
            entries.

        Returns
        -------
        A : scipy.sparse.csr_matrix
            Sparse matrix of shape (n_vertices, n_vertices).
        """
        face_index = np.repeat(np.arange(self.n_faces), 3)
        if one_based_index:
            face_index += 1
        return scipy.sparse.csr_array((face_index, self.edges().T))

    def hole_fill_refine_fair(self, inplace: bool = False):
        """Fill, refine and fair holes in `self`.

        Notes
        -----
        CGAL::Polygon_mesh_processing::triangulate_refine_and_fair_hole
        """
        v, f = pmp.hole_fill_refine_fair(self.vertices, self.faces)
        if inplace:
            self.vertices = v
            self.faces = f
        else:
            return self.new(v, f)

    def integrate_on_vertices(self, indices, values):
        if values.ndim == 1:
            return np.bincount(indices, values, self.n_vertices)
        elif values.ndim == 2:
            return np.stack(
                [
                    np.bincount(indices, values[:, i], self.n_vertices)
                    for i in range(values.shape[1])
                ],
                axis=1,
            )
        else:
            raise ValueError(
                f"Only 1 or 2 dimensional value arrays supported (got shape {values.shape})"
            )

    def interpolated_corrected_curvatures(self):
        """Compute principal, mean, and Gaussian curvatures using CGAL."""
        k1, k2, H, K, k1_vec, k2_vec = pmp.interpolated_corrected_curvatures(
            self.vertices, self.faces
        )
        return Curvature(k1=k1, k2=k2, H=H, K=K)

    def k_ring_neighbors(
        self,
        select: None | npt.NDArray = None,
        k: int = 1,
        adj: None | scipy.sparse.csr_array = None,
        connectivity: str = "vertex",
    ):
        """Compute the k-ring neighborhood(s) of the selection.

        Parameters
        ----------
        k : int
            Find the kth ring neighbors.
        select : None | npt.NDArray
            The indices of the element (e.g., vertices) for which to do the
            neighbor search. Default is for all elements.
        adj : None | scipy.sparse.csr_array
            scipy.sparse adjacency matrix of the vertices. If None, then it is
            computed.
        connectivity : str
            The

        Returns
        -------
        knn : list[npt.NDArray]
            List of numpy arrays such that knn[i] contains the neighbors of
            vertex i (including i, the 0-ring).
        kr : np.NDArray
            Array of indices (into `knn`) of each ring of neighbors such
            that

                knn[i][kr[i,0]:kr[i,1]] gives the 0-ring (starting) vertices of i,
                knn[i][kr[i,1]:kr[i,2]] gives the 1-ring neighboring vertices of i
                ...

            The array has length k+2 and a similar interpretation as
            `scipy.sparse.csr_array.indptr`.

        """
        assert k > 0, "`k` must be a positive integer."

        match connectivity:
            case "vertex":
                adj = self.adjacency_matrix("vertex") if adj is None else adj
                n = self.n_vertices
            case "face":
                adj = self.adjacency_matrix("face") if adj is None else adj
                n = self.n_faces
            case _:
                raise ValueError

        select = np.arange(n) if select is None else select
        select = select[:, None] if select.ndim == 1 else select
        assert select.ndim == 2, "`indices` must be (n, n_start_indices)"

        knn, kr = cortech.utils.k_ring_neighbors(k, select, n, adj.indices, adj.indptr)

        return knn, kr

    def keep_largest_connected_component(
        self,
        connectivity="face",
        select: npt.ArrayLike | None = None,
        overwrite_original_id: bool = True
    ):
        return self.keep_n_largest_connected_components(
            1, connectivity, select, overwrite_original_id
        )

    def keep_n_largest_connected_components(
        self,
        n: int = 1,
        connectivity="face",
        select=None,
        overwrite_original_id: bool = True,
    ):
        cc, _ = self.connected_components(connectivity, select)
        return self.select_faces(cc < n, overwrite_original_id)

    def interpolate_to_nodes(
        self, vol: npt.NDArray, affine: npt.NDArray, order: int = 3
    ) -> npt.NDArray:
        """Interpolate values from a volume to surface node positions.

        Parameters
        ----------
        vol : npt.NDArray
            A volume array as read by e.g., nib.load(image).get_fdata()
        affine: npt.NDArray
            A 4x4 array storing the vox2world transformation of the image
        order: int
            Interpolation order (0-5)

        Returns
        -------
        values_at_coords: npt.NDArray
                        An Nx1 array of intensity values at each node

        """
        vertices = self.to_scanner_ras(inplace=False)

        # Map node coordinates to volume
        inv_affine = np.linalg.inv(affine)
        vox_coords = self.apply_affine(vertices, inv_affine)

        # Deal with edges ala simnibs
        im_shape = vol.shape
        for i, s in enumerate(im_shape):
            vox_coords[(vox_coords[:, i] > -0.5) * (vox_coords[:, i] < 0), i] = 0.0
            vox_coords[(vox_coords[:, i] > s - 1) * (vox_coords[:, i] < s - 0.5), i] = (
                s - 1
            )

        # Keeping the map_coordinates options exposed in case we want to change these
        return map_coordinates(
            vol, vox_coords.T, order=order, mode="constant", cval=0.0, prefilter=True
        )

    def refine(self, density: float = 2.0, select: npt.ArrayLike | None = None):
        """Refine `self` by subdivision of faces."""
        select = _ensure_array_is_index(select, self.n_faces)
        v, f = pmp.refine(self.vertices, self.faces, density, select)
        return self.new(v, f)

    def signed_distance_function(
        self, cone_angle: float = 60.0, number_of_rays: int = 25
    ):
        """Compute the signed distance function (SDF) at each vertex. The SDF
        approximates the thickness of `self` at each vertex. This is computed
        by ray tracing in a cone in the anti-normal direction.
        """
        return ss.signed_distance_function(
            self.vertices, self.faces, np.deg2rad(cone_angle), number_of_rays
        )

    def split(self, other: "ManifoldSurface"):
        """Corefine `self` and `other` and duplicate edges on `self` that are
        on the intersection with `other`.

        Parameters
        ----------
        other


        Returns
        -------
        _type_
            Tuple of split versions of self and other.

        Notes
        -----
        CGAL::Polygon_mesh_processing::split
        """
        split_self, split_other = pmp.split_with_surface(
            self.vertices, self.faces, other.vertices, other.faces
        )
        return self.new(*split_self), other.new(*split_other)

    def split_edges(self, edges: npt.ArrayLike):
        """Split a number of edges.

        Parameters
        ----------
        edges
            Array of size (n, 2) containing vertex pairs defining the edges to
            split.

        Returns
        -------

        """
        edges = np.asarray(edges)
        assert edges.ndim == 2 and edges.shape[1] == 2
        v, f, fid, v_pmap, f_pmap = pmp.split_edges(
            self.vertices, self.faces, edges, self.face_id
        )
        vd = dict(original_id=v_pmap)
        fd = dict(original_id=f_pmap)
        return self.new(v, f, face_id=fid, vertex_data=vd, face_data=fd)

    def split_long_edges(self, sizing: float, select=None, face_id=None):
        """

        Parameters
        ----------
        sizing :
            Edges are split so that no edge is longer than this value.
        select
            Restrict edge splitting to these edges. Array of size (n, 2)
            containing vertex indices defining edges. If
            None, all edges.

        Returns
        -------
        Updated surface with

        """
        face_id = self._check_face_id(face_id)
        v, f, fid, orig_vid, orig_fid = pmp.split_long_edges(
            self.vertices, self.faces, sizing, face_id, select
        )
        vd = dict(original_id=orig_vid)
        fd = dict(original_id=orig_fid)
        return self.new(v, f, face_id=fid, vertex_data=vd, face_data=fd)

    def split_with_plane(self, p, d, inplace: bool = False):
        """Split the surface at a plane defined by a point `p` and a direction
        `d`.

        Parameters
        ----------
        p : npt.ArrayLike
            Point on the plane.
        d : npt.ArrayLike
            Direction vector of the plane.
        inplace : bool, optional
            Modify the surface in place(default = False).
        Returns
        -------
        _type_
            _description_

        Notes
        -----
        CGAL::Polygon_mesh_processing::split
        """
        v, f = pmp.split_with_plane(self.vertices, self.faces, p, d)
        if inplace:
            self.vertices = v
            self.faces = f
        else:
            return self.new(v, f)

    # def repair(self, inplace: bool = False):
    #     """Repair the surface mesh as polygon soup."""
    #     raise NotImplementedError("This method does not currently work")
        # v, f = pmp.repair_mesh(self.vertices, self.faces)
        # if inplace:
        #     self.vertices = v
        #     self.faces = f
        # else:
        #     return self.new(v, f)

    def adaptive_remeshing(
        self,
        error_tol: float,
        edge_length_min: float,
        edge_length_max: float,
        face_is_selected: npt.ArrayLike | None = None,
        vertex_is_constrained: npt.ArrayLike | None = None,
        edge_is_constrained: npt.ArrayLike | None = None,
        *,
        n_iter: int = 1,
        relative: bool = False,
        protect_constraints: bool = False,
        collapse_constraints: bool = True,
        do_split: bool = True,
        do_collapse: bool = True,
        do_flip: bool = True,
        number_of_relaxation_steps: int = 1,
        disallow_surface_crossing: bool = False,
        inplace: bool = False,
    ):
        """Isotropic remeshing using a curvature adaptive sizing field.

        Parameters
        ----------
        error_tol : float
            _description_
        edge_length_min : float
            _description_
        edge_length_max : float
            _description_
        n_iter : int, optional
            _description_, by default 1
        protect_constraints : bool, optional
            If true, the edges set as constrained in edge_is_constrained
            (or by default the boundary edges) are neither split nor collapsed
            during remeshing.
        face_is_selected : npt.ArrayLike | None, optional
            Remesh only a subset of the faces in `self`.
        edge_is_constrained: npt.ArrayLike | None = None
            Do not remesh these edges.
            * A constrained edge can be split or collapsed, but not flipped,
                nor its endpoints moved by smoothing.
            * Sub-edges generated by splitting are set to be constrained.
            * Patch boundary edges (i.e. incident to only one face in the
                range) are always considered as constrained edges.
        relative : bool, optional
            If true, `error_tol`, `edge_length_min` and `edge_length_max` are
            considered as a factors with which to scale the mean edge length of
            the current surface rather than an absolute values.
        inplace : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_
        """
        if relative:
            edge_norm_mean = self.edges_norm().mean()
            error_tol = error_tol * edge_norm_mean
            edge_length_min = edge_length_min * edge_norm_mean
            edge_length_max = edge_length_max * edge_norm_mean

        if face_is_selected is not None:
            face_is_selected = _ensure_array_is_index(face_is_selected, self.n_faces)
        if vertex_is_constrained is not None:
            vertex_is_constrained = _ensure_array_is_index(
                vertex_is_constrained, self.n_vertices
            )

        v, f, fid, v_pmap, f_pmap = pmp.adaptive_remeshing(
            self.vertices,
            self.faces,
            error_tol,
            edge_length_min,
            edge_length_max,
            self.face_id,
            face_is_selected,
            vertex_is_constrained,
            edge_is_constrained,
            n_iter,
            protect_constraints,
            collapse_constraints,
            do_split,
            do_collapse,
            do_flip,
            number_of_relaxation_steps,
            disallow_surface_crossing,
        )
        if inplace:
            self.vertices = v
            self.faces = f
            self.face_id = fid
            self.data.vertex["original_id"] = v_pmap
            self.data.face["original_id"] = f_pmap
            return self
        else:
            vd = dict(original_id=v_pmap)
            fd = dict(original_id=f_pmap)
            return self.new(v, f, face_id=fid, vertex_data=vd, face_data=fd)

    def custom_remeshing(
        self,
        sizing: npt.ArrayLike,
        face_is_selected: npt.ArrayLike | None = None,
        vertex_is_constrained: npt.ArrayLike | None = None,
        edge_is_constrained: npt.ArrayLike | None = None,
        *,
        n_iter: int = 1,
        # relative: bool = False,
        protect_constraints: bool = False,
        collapse_constraints: bool = True,
        do_split: bool = True,
        do_collapse: bool = True,
        do_flip: bool = True,
        number_of_relaxation_steps: int = 1,
        disallow_surface_crossing: bool = False,
        inplace: bool = False,
    ):
        """Isotropic remeshing using a custom sizing field.

        Parameters
        ----------
        custom :
            Custom sizing field. One value per vertex.
        n_iter : int, optional
            _description_, by default 1
        protect_constraints : bool, optional
            If true, the edges set as constrained in edge_is_constrained
            (or by default the boundary edges) are neither split nor collapsed
            during remeshing.
        face_is_selected : npt.ArrayLike | None, optional
            Remesh only a subset of the faces in `self`.
        edge_is_constrained: npt.ArrayLike | None = None
            Do not remesh these edges.
            * A constrained edge can be split or collapsed, but not flipped,
                nor its endpoints moved by smoothing.
            * Sub-edges generated by splitting are set to be constrained.
            * Patch boundary edges (i.e. incident to only one face in the
                range) are always considered as constrained edges.
        relative : bool, optional
            If true, `error_tol`, `edge_length_min` and `edge_length_max` are
            considered as a factors with which to scale the mean edge length of
            the current surface rather than an absolute values.
        inplace : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_
        """
        assert len(sizing) == self.n_vertices
        # if relative:
        #     edge_norm_mean = self.edges_norm().mean()
        #     error_tol = error_tol * edge_norm_mean
        #     edge_length_min = edge_length_min * edge_norm_mean
        #     edge_length_max = edge_length_max * edge_norm_mean

        if face_is_selected is not None:
            face_is_selected = _ensure_array_is_index(face_is_selected, self.n_faces)
        if vertex_is_constrained is not None:
            vertex_is_constrained = _ensure_array_is_index(
                vertex_is_constrained, self.n_vertices
            )

        v, f, fid, v_pmap, f_pmap = pmp.custom_remeshing(
            self.vertices,
            self.faces,
            sizing,
            self.face_id,
            face_is_selected,
            vertex_is_constrained,
            edge_is_constrained,
            n_iter,
            protect_constraints,
            collapse_constraints,
            do_split,
            do_collapse,
            do_flip,
            number_of_relaxation_steps,
            disallow_surface_crossing,
        )
        if inplace:
            self.vertices = v
            self.faces = f
            self.face_id = fid
            self.data.vertex["original_id"] = v_pmap
            self.data.face["original_id"] = f_pmap
            return self
        else:
            vd = dict(original_id=v_pmap)
            fd = dict(original_id=f_pmap)
            return self.new(v, f, face_id=fid, vertex_data=vd, face_data=fd)

    def isotropic_remeshing(
        self,
        target_edge_length: float,
        face_is_selected: npt.ArrayLike | None = None,
        vertex_is_constrained: npt.ArrayLike | None = None,
        edge_is_constrained: npt.ArrayLike | None = None,
        *,
        n_iter: int = 1,
        relative: bool = False,
        protect_constraints: bool = False,
        collapse_constraints: bool = True,
        do_split: bool = True,
        do_collapse: bool = True,
        do_flip: bool = True,
        number_of_relaxation_steps: int = 1,
        disallow_surface_crossing: bool = False,
        inplace: bool = False,
    ):
        """Isotropic remeshing using a uniform sizing field.

        Adds `original_id` fields to vertex and face data where -1 marks a new
        vertex or face.

        Parameters
        ----------
        target_edge_length : float
            This remeshing uses a uniform sizing field where this value
            specifies the desired (global) target edges length in output.
        face_is_selected : npt.ArrayLike | None
            Remesh only this subset of faces.
        vertex_is_constrained : npt.ArrayLike | None

        edge_is_constrained: npt.ArrayLike | None
            Do not remesh these edges.
            * A constrained edge can be split or collapsed, but not flipped,
                nor its endpoints moved by smoothing.
            * Sub-edges generated by splitting are set to be constrained.
            * Patch boundary edges (i.e. incident to only one face in the
                range) are always considered as constrained edges.
        n_iter : int, optional
            Number of remeshing iterations (default = 1)
        relative : bool, optional
            If true, `target_edge_length` is considered as a factor with which
            to scale the mean edge length of the current surface rather than
            an absolute value.
        collapse_constraints : bool
            Allow collapse of constrained edges.
        do_split : bool
            Allow face splits during remeshing.
        do_collapse : bool
            Allow face collapse during remeshing.
        do_flip : bool
            Allow face flips during remeshing.
        protect_constraints : bool
            If true, the edges set as constrained in edge_is_constrained
            (or, by default, the boundary edges) are neither split nor
            collapsed during remeshing.
        number_of_relaxation_steps : 1
            Number of Laplacian smoothing
        inplace : bool, optional
            _description_, by default False

        Returns
        -------
        A remeshed instance of self.

        Notes
        -----
        """
        if relative:
            target_edge_length = target_edge_length * self.edges_norm().mean()

        if face_is_selected is not None:
            face_is_selected = _ensure_array_is_index(face_is_selected, self.n_faces)
        if vertex_is_constrained is not None:
            vertex_is_constrained = _ensure_array_is_index(
                vertex_is_constrained, self.n_vertices
            )

        v, f, fid, v_pmap, f_pmap = pmp.isotropic_remeshing(
            self.vertices,
            self.faces,
            target_edge_length,
            self.face_id,
            face_is_selected,
            vertex_is_constrained,
            edge_is_constrained,
            n_iter,
            protect_constraints,
            collapse_constraints,
            do_split,
            do_collapse,
            do_flip,
            number_of_relaxation_steps,
            disallow_surface_crossing,
        )
        if inplace:
            self.vertices = v
            self.faces = f
            self.face_id = fid
            self.data.vertex["original_id"] = v_pmap
            self.data.face["original_id"] = f_pmap
            return self
        else:
            vd = dict(original_id=v_pmap)
            fd = dict(original_id=f_pmap)
            return self.new(v, f, face_id=fid, vertex_data=vd, face_data=fd)

    def remeshing_with_projection(
        self, method="isotropic", tris=None, weights=None, data=None, **kwargs
    ):
        match method:
            case "adaptive":
                # siso = self.adaptive_remeshing(0.5, 0.5, 2.0, n_iter=3, relative_edge_length=True)
                sr = self.adaptive_remeshing(**kwargs)
            case "isotropic":
                # siso = self.isotropic_remeshing(0.5, n_iter=3, relative_edge_length=True)
                sr = self.isotropic_remeshing(**kwargs)
            case _:
                raise ValueError(f"Invalid `method` {method}.")

        if tris is not None and weights is not None:
            v = np.sum(self.mesh(tris) * weights[..., None], 1)
        else:
            v = self.vertices
        tris_on_sr, weights_on_sr, _, _ = sr.project_points(v)

        if data is None:
            return sr, tris_on_sr, weights_on_sr
        else:
            tris_on_self, w_on_self, _, _ = self.project_points(sr.vertices)
            data_proj = np.sum(data[self.faces[tris_on_self]] * w_on_self, 1)
            return sr, tris_on_sr, weights_on_sr, data_proj

    def inflate(self):
        raise NotImplementedError

    def deflate(self):
        raise NotImplementedError

    def orient(self, outward_orientation: bool = True):
        """Ensure that the surface is consistently oriented."""
        v, f = pmp.orient(self.vertices, self.faces, outward_orientation)
        return self.new(v, f, keep_vertex_data=True, keep_face_data=True)

    def points_inside(self, points: npt.ArrayLike, on_boundary_is_inside: bool = True):
        """For each point in `points`, test it is inside the surface or not."""
        points = np.atleast_2d(points)
        return pmp.points_inside_surface(
            self.vertices, self.faces, points, on_boundary_is_inside
        )

    def principal_curvatures(self):
        """Compute principal curvatures and corresponding directions. From these,
        the following curvature estimates can easily be calculated

        Mean curvature

            H = 0.5*(k1+k2)

        Gaussian curvature

            K = k1*k2


        Parameters
        ----------
        v : npt.NDArray
            Vertices
        f : npt.NDArray
            Faces

        Returns
        -------
        D : ndarray
            Principal curvatures with k1 and k2 (maximum and minimum curvature,
            respectively) in first and second column.
        E : ndarray
            Principal directions corresponding to the principal curvatures
            (E[:, 0] and E[:, 1] correspond to k1 and k2, respectively).

        Notes
        -----
        This function is similar to Freesurfer's
        `MRIScomputeSecondFundamentalForm`.
        """
        n = self.n_vertices
        adj = self.adjacency_matrix()
        vn = self.vertex_normals()
        vt = cortech.utils.compute_tangent_vectors(vn)

        m = np.array(adj.sum(1)).squeeze().astype(int)  # number of neighbors
        muq = np.unique(m)

        # Estimate the parameters of the second fundamental form at each vertex.
        # The second fundamental form is a quadratic form on the tangent plane of
        # the vertex
        # (see https://en.wikipedia.org/wiki/Second_fundamental_form)

        # We cannot solve for all vertices at the same time as the number of
        # equations in the system equals the number of neighbors. However, we can
        # solve all vertices with the same number of neighbors concurrently as this
        # is broadcastable

        H_uv = np.zeros((n, 2, 2))
        for mm in muq:
            i = np.where(m == mm)[0]
            vi = self.vertices[i]
            ni = self.vertices[adj[i].indices.reshape(-1, mm)]  # neighbors

            H_uv[i] = self._second_fundamental_form_coefficients(vi, ni, vt[i], vn[i])

            # # only needed for bad conditioning?
            # rsq = A[:,:2].suhole_fill_refine_fairm(1) # u**2 + v**2
            # k = b/rsq
            # kmin[i] = k.min()
            # kmax[i] = k.max()

        # Estimate curvature from the second fundamental form
        # (see https://en.wikipedia.org/wiki/Principal_curvature)
        # D = principal curvatures
        # E = principal directions, i.e., the directions of maximum and minimum
        #     curvatures.
        # Positive curvature means that the surface bends towards the normal (e.g.,
        # in a sulcus)
        D, E = np.linalg.eigh(H_uv)
        # sort in *descending* order
        D = D[:, ::-1]
        E = E[:, ::-1]
        # Rotate the tangent vectors so they correspond to the principal
        # curvature directions (i.e., we are back in the original space).
        E_tangent = E.swapaxes(1, 2) @ vt
        return D, E_tangent

    def remove_almost_degenerate_faces(
        self,
        face_is_selected: npt.ArrayLike | None = None,
        cap_threshold=-0.9396926207859083,
        needle_threshold=4.0,
        constrain_border_edges: bool = True,
        constrain_patch_edges: bool = True,
        vertex_is_constrained: npt.ArrayLike | None = None,
        edge_is_constrained: npt.ArrayLike | None = None,
        inplace: bool = False,
    ):
        """Remove faces that are caps or needles. Caps are faces with one angle
        close to 180 degrees; needles are faces where the longest/shortest edge
        ratio is high.

        Parameters
        ----------
        cap_threshold : float
            Specified as the cosine to the angle (i.e., -1 corresponds to 180
            degrees), this gives the threshold below which a triangle is
            considered a cap.
        needle_threshold : float
            The longest/shortest edge ratio above which a triangle is
            considered a needle.
        edge_is_constrained:
        """
        if face_is_selected is not None:
            face_is_selected = _ensure_array_is_index(face_is_selected, self.n_faces)
        if vertex_is_constrained is not None:
            vertex_is_constrained = _ensure_array_is_index(
                vertex_is_constrained, self.n_vertices
            )
        if constrain_border_edges:
            border = self.find_border_edges().reshape(-1, 2)
        else:
            border = np.empty((0, 2))
        if constrain_patch_edges:
            patch = self.find_patch_edges().reshape(-1, 2)
        else:
            patch = np.empty((0, 2))

        ec = np.concat((border, patch))

        if edge_is_constrained is not None:
            ec = np.concat((edge_is_constrained, ec))

        v, f, vid, fid = pmp.remove_almost_degenerate_faces(
            self.vertices,
            self.faces,
            face_is_selected,
            cap_threshold,
            needle_threshold,
            vertex_is_constrained,
            ec,
        )
        if inplace:
            raise NotImplementedError("need to handle face_id ...")
            self.vertices = v
            self.faces = f
            self.data.vertex["original_id"] = vid
            self.data.face["original_id"] = fid
            return self
        else:
            vd = dict(original_id=vid)
            fd = dict(original_id=fid)
            return self.new(
                v, f, face_id=self.face_id[fid], vertex_data=vd, face_data=fd
            )

    def remove_self_intersections(self, select: npt.ArrayLike | None = None):
        """Attempt to remove self-intersections by smoothing and possibly hole
        filling.

        Raises
        ------
        RuntimeError
            If unable to remove all self-intersections.
        """
        v, f, vid, fid = pmp.remove_self_intersections(
            self.vertices, self.faces, select
        )
        vd = dict(original_id=vid)
        fd = dict(original_id=fid)
        return self.new(v, f, vertex_data=vd, face_data=fd)

    @staticmethod
    def _second_fundamental_form_coefficients(vi, ni, vit, vin):
        """

        V = number of vertices
        N = number of neighbors

        vi : vertex at which to estimate curvature (V, 3)
        ni : neighbors (V, N, 3)
        vit : vertex tangent plane vectors (V, 2, 3)
        vin : vector normal (V, 3)

        """
        n_vi = vi.shape[0]

        # Fit a quadratic function centered on the current vertex using its
        # tangent vectors (say, u and v) as basis. The "function values" are
        # the distances from each neighbor to its projection on the tangent
        # plane
        nivi = ni - vi[:, None]
        # Quadratic features
        # (inner product of tangent vectors and vector from v to its neighbors)
        uv = vit[:, :, None] @ nivi[:, None].swapaxes(2, 3)
        uv = uv[:, :, 0]  # (V, 2, N)

        A = np.concatenate(
            (uv**2, 2 * np.prod(uv, axis=1, keepdims=True)), axis=1
        ).swapaxes(1, 2)
        # Function values
        # (inner product of normal vector and vector from v to its neighbors)
        b = nivi @ vin[..., None]
        b = b[..., 0]

        # Least squares solution
        U, S, Vt = np.linalg.svd(A, full_matrices=False)

        # coefficients for u**2, v**2, u*v
        x = Vt.swapaxes(1, 2) @ (U.swapaxes(1, 2) @ b[..., None] / S[..., None])
        x = x[..., 0]

        # Estimate the coefficients of the second fundamental form
        # Hessian
        H_uv = np.zeros((n_vi, 2, 2))
        H_uv[:, 0, 0] = 2 * x[:, 0]
        H_uv[:, 1, 1] = 2 * x[:, 1]
        H_uv[:, 0, 1] = H_uv[:, 1, 0] = 2 * x[:, 2]

        return H_uv.squeeze()

    def self_intersections(self):
        """Compute intersecting pairs of triangles."""
        return pmp.self_intersections(self.vertices, self.faces)

    def simplify(self, stop_face_count: int, inplace: bool = False):
        """Decimate the surface by edge collapse until `stop_face_count` number
        of faces has been reached.

        Parameters
        ----------
        stop_face_count
            Target number of faces.
        inplace : bool, optional
            Modify the surface in place(default = False).


        Returns
        -------
        surface
            New Surface or inplace update of self.

        Notes
        -----
        CGAL::Surface_mesh_simplification::edge_collapse
        """
        v, f = sms.simplify(self.vertices, self.faces, stop_face_count)
        if inplace:
            self.vertices = v
            self.faces = f
        else:
            return self.new(v, f)

    def skeletonize(self):
        """Return the skeleton (medial axis) of `self`.

        vertices :
        edges :

        """
        return smskel.skeletonize(self.vertices, self.faces)

    def smooth_angle_and_area(
        self,
        constrained_vertices: npt.ArrayLike | None = None,
        use_angle_smoothing: bool = True,
        use_area_smoothing: bool = False,
        use_delaunay_flips: bool = True,
        use_safety_constraints: bool = False,
        n_iter: int = 1,
        inplace: bool = False,
    ):
        v = pmp.smooth_angle_and_area(
            self.vertices,
            self.faces,
            constrained_vertices,
            n_iter,
            use_angle_smoothing,
            use_area_smoothing,
            use_delaunay_flips,
            use_safety_constraints,
        )
        if inplace:
            self.vertices = v
        else:
            return self.new(v, keep_vertex_data=True, keep_face_data=True)

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
        protect_border_edges: bool = True,
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

        if protect_border_edges:
            be = self.find_border_edges().ravel()
            select = np.arange(self.n_vertices) if select is None else select
            select = np.setdiff1d(select, be)

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
                return self.new(
                    array, keep_face_id=True, keep_vertex_data=True, keep_face_data=True
                )
        else:
            return array

    # def smooth_laplace(
    #     self,
    #     arr: npt.NDArray | None = None,
    #     dt: float = 1.0,
    #     n_iter: int = 1,
    #     inplace: bool = False,
    #     L=None,
    # ):
    #     """Laplacian smoothing using implicit integration.

    #     Parameters
    #     ----------
    #     arr : npt.NDArray | None, optional
    #         _description_, by default None
    #     dt : float, optional
    #         _description_, by default 1.0
    #     n_iter : int, optional
    #         _description_, by default 1
    #     inplace : bool, optional
    #         _description_, by default False
    #     L : _type_, optional
    #         _description_, by default None

    #     Returns
    #     -------
    #     _type_
    #         _description_
    #     """
    #     L = self.compute_laplacian_matrix() if L is None else L
    #     # matA = scipy.sparse.eye_array(L.shape[0]) - lmdb * dt * L
    #     mat_A = scipy.sparse.eye_array(L.shape[0]) + dt * L

    #     mat_A.data = mat_A.data.astype(np.float32)
    #     mat_A.indices = mat_A.indices.astype(np.int32)
    #     mat_A.indptr = mat_A.indptr.astype(np.int32)

    #     solver = KSPSolver(mat_A, "cg", "hypre")

    #     arr = self.vertices if arr is None else arr

    #     for _ in range(n_iter):
    #         arr = solver.solve(arr)

    #     if inplace:
    #         self.vertices = arr
    #         return self
    #     else:
    #         return self.new(arr, self.faces)

    def smooth_shape(
        self,
        time: float = 0.1,
        n_iter: int = 1,
        constrained_vertices: npt.NDArray | None = None,
        inplace: bool = False,
    ):
        """Perform shape smoothing via mean curvature flow.

        Parameters
        ----------
        constrained_vertices:
            Indices of vertices to fix (smoothing will not be applied to these
            vertices).
        time
        n_iter
        inplace : bool

        References
        ----------
        https://doc.cgal.org/latest/Polygon_mesh_processing/index.html
        """
        v = pmp.smooth_shape(
            self.vertices, self.faces, constrained_vertices, time, n_iter
        )
        if inplace:
            self.vertices = v
        else:
            return self.new(v, keep_vertex_data=True, keep_face_data=True)

    def smooth_shape_by_curvature_threshold(
        self,
        time: float = 0.1,
        n_iter: int = 1,
        curv_threshold: float = 0.0,
        apply_above_curv_threshold: bool = True,
        ball_radius: float = -1.0,
        inplace: bool = False,
    ):
        """Perform shape smoothing via mean curvature flow while constraining
        vertices whose curvature is either above or below a certain threshold.
        This allows strict shrinking or inflation of the surface whereas the
        standard mean curvature flow of vertices (as performed by
        `smooth_shape`) will shrink convex areas and inflate concave areas. The
        default settings (`curv_threshold = 0.0` and
        `apply_above_curv_threshold = True`) results in strict shrinkage.

        Parameters
        ----------
        time : float
            Amount of smoothing to apply at each iteration (higher values means
            more aggressive smoothing).
        n_iter : int
            Number of curvature estimation and smoothing steps to apply.
        curv_threshold : float
            Apply smoothing to vertices above/below `curv_threshold` (default = 0.0).
        apply_above_curv_threshold : bool
            If true, apply smoothing to vertices whose curvature is *above*
            `curv_threshold` (and vice verse if false) (default = True). If
            true (and curv_threshold = 0.0), the surface will strictly shrink.
        ball_radius : float
            Smooth curvature estimates within a ball with `ball_radius`. Must
            be > 0.0 except -1.0 which disables smoothing (default = -1.0).
        inplace : bool

        References
        ----------
        https://doc.cgal.org/latest/Polygon_mesh_processing/index.html
        """
        v = pmp.smooth_shape_by_curvature_threshold(
            self.vertices,
            self.faces,
            time,
            n_iter,
            curv_threshold,
            apply_above_curv_threshold,
            ball_radius,
        )
        if inplace:
            self.vertices = v
        else:
            return self.new(v)

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
                return self.new(array, keep_vertex_data=True, keep_face_data=True)
        else:
            return array

    def tangent_plane_vectors(self) -> npt.NDArray:
        """Compute vectors spanning the tangent plane of a vertex, i.e., the
        plane to which the vertex normal is orthogonal. The returned tangent
        vectors are orthogonal but their directions are arbitrary.

        Returns
        -------
        tangent vectors : npt
            Array of size (self.n_vertices, 2, 3).
        """
        return cortech.utils.compute_tangent_vectors(self.vertex_normals())

    def tangential_relaxation(
        self,
        constrained_vertices: npt.NDArray | None = None,
        n_iter: int = 1,
        inplace: bool = False,
    ):
        """

        Parameters
        ----------
        constrained_vertices:
            Indices of vertices.
        time
        n_iter
        inplace : bool

        References
        ----------
        https://doc.cgal.org/latest/Polygon_mesh_processing/index.html
        """
        v = pmp.tangential_relaxation(
            self.vertices, self.faces, constrained_vertices, n_iter
        )
        if inplace:
            self.vertices = v
        else:
            return self.new(v, keep_vertex_data=True, keep_face_data=True)

    def triangle_quality(self):
        """The 'volume-length' quality metric for a triangle is defined as

            Q = 4*sqrt(3) * Area / sum(EdgeLengths**2)

        in table 6, row 4 of Shewchuk (2002). Q=0 (worst), Q=1 (best)

        References
        ----------
        Shewchuk (2002). What Is a Good Linear Finite Element? Interpolation,
            Conditioning, Anisotropy, and Quality Measures.
            https://people.eecs.berkeley.edu/~jrs/papers/elemj.pdf
        """
        a = 6.928203230275509  # 4 * sqrt(3.0)
        A = self.face_areas()
        E = self.edges_norm().reshape(-1, 3)
        return a * A / np.sum(E**2, -1)  # 0=worst, 1=best

    def vertex_normals(self):
        """ """
        face_normals = self.face_normals()
        out = np.stack(
            [
                np.bincount(
                    self.faces.ravel(),
                    weights=np.broadcast_to(n[:, None], self.faces.shape).ravel(),
                    minlength=self.n_vertices,
                )
                for n in face_normals.T
            ],
            axis=1,
        )
        return out / np.linalg.norm(out, ord=2, axis=1, keepdims=True)

    def get_triangle_neighbors(self):
        """For each point get its neighboring triangles (i.e., the triangles to
        which it belongs).

        PARAMETERS
        ----------
        tris : ndarray
            Array describing a triangulation with size (n, 3) where n is the number
            of triangles.
        nr : int
            Number of points. If None, it is inferred from `tris` as tris.max()+1
            (default = None).

        RETURNS
        -------
        pttris : ndarray
            Array of arrays where pttris[i] are the neighboring triangles of the
            ith point.
        """
        rows = self.faces.ravel().astype('int32')
        cols = np.repeat(np.arange(self.n_faces), self.vertices_per_face).astype('int32')
        data = np.ones_like(rows)
        csr = scipy.sparse.coo_matrix(
            (data, (rows, cols)), shape=(self.n_vertices, self.n_faces)
        ).tocsr()
        return np.array(np.split(csr.indices, csr.indptr[1:-1]), dtype=object)

    def get_closest_triangles(
        self, points: npt.NDArray, n: int = 1, subset=None, return_index: bool = False
    ):
        """For each point in `points` get the `n` nearest nodes on `surf` and
        return the triangles to which these nodes belong.

        points : ndarray
            Points for which we want to find the candidate triangles. Shape (n, d)
            where n is the number of points and d is the dimension.
        surf : dict
            Dictionary with keys points and tris corresponding to the nodes and
            triangulation of the surface, respectively.
        n : int
            Number of nearest vertices in `self` to consider for each point in
            `points`.
        subset : array-like
            Use only a subset of the vertices in `surf`. Should be indices *not* a
            boolean mask!
        return_index : bool
            Return the index (or indices if n > 1) of the nearest vertex in `surf`
            for each point in `points`.

        RETURNS
        -------
        pttris : list
            Point to triangle mapping.
        """
        assert isinstance(n, int) and n >= 1

        surf_points = self.vertices if subset is None else self.vertices[subset]
        tree = scipy.spatial.cKDTree(surf_points)
        _, ix = tree.query(points, n)
        if subset is not None:
            ix = subset[ix]  # ensure ix indexes into surf['points']
        pttris = self.get_triangle_neighbors()[ix]
        if n > 1:
            pttris = list(map(lambda x: np.unique(np.concatenate(x)), pttris))
        return (pttris, ix) if return_index else pttris

    def project_points(
        self,
        points: npt.NDArray,
        pttris: int | list | np.ndarray = 5,
        subset=None,
        return_all_projections: bool = False,
    ):
        """Project each point in `points` to the closest point on the surface.
        `pttris` is used to restrict possible triangles (on self) to which each
        point can be projected. This is used to speed up computations.

        PARAMETERS
        ----------
        points : ndarray
            Array with shape (n, d) where n is the number of points and d is the
            dimension.
        pttris : int | list | ndarray
            If an integer, specifies the number of closest triangles on self
            against which each point is tested. If a ragged/nested array, the
            ith entry contains the triangles against which the ith point will
            be tested.
        return_all_projections : bool
            Whether to return all projection results (i.e., the projection of a
            point on each of the triangles which it was tested against) or only the
            projection on the closest triangle.

        RETURNS
        -------
        tris : ndarray
            The index of the triangle onto which a point was projected.
        weights : ndarray
            The linear interpolation weights resulting in the projection of a point
            onto a particular triangle.
        projs :
            The coordinates of the projection of a point on a triangle.
        dists :
            The distance of a point to its projection on a triangle.

        NOTES
        -----
        The cost function to be minimized is the squared distance between a point
        P and a triangle T

            Q(s,t) = |P - T(s,t)|**2 =
                = a*s**2 + 2*b*s*t + c*t**2 + 2*d*s + 2*e*t + f

        The gradient

            Q'(s,t) = 2(a*s + b*t + d, b*s + c*t + e)

        is set equal to (0,0) to find (s,t).

        REFERENCES
        ----------
        https://www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf

        """
        if isinstance(pttris, int):
            pttris = self.get_closest_triangles(points, pttris, subset)
        npttris = list(map(len, pttris))
        pttris = np.concatenate(pttris)

        m = self.mesh()
        v0 = m[:, 0]  # Origin of the triangle
        e0 = m[:, 1] - v0  # s coordinate axis
        e1 = m[:, 2] - v0  # t coordinate axis

        # Vector from point to triangle origin (if reverse, the negative
        # determinant must be used)
        rep_points = np.repeat(points, npttris, axis=0)
        w = v0[pttris] - rep_points

        a = np.sum(e0**2, 1)[pttris]
        b = np.sum(e0 * e1, 1)[pttris]
        c = np.sum(e1**2, 1)[pttris]
        d = np.sum(e0[pttris] * w, 1)
        e = np.sum(e1[pttris] * w, 1)
        # f = np.sum(w**2, 1)

        # s,t are so far unnormalized!
        s = b * e - c * d
        t = b * d - a * e
        det = a * c - b**2

        # Project points (s,t) to the closest points on the triangle (s',t')
        sp, tp = np.zeros_like(s), np.zeros_like(t)

        # We do not need to check a point against all edges/interior of a triangle.
        #
        #          t
        #     \ R2|
        #      \  |
        #       \ |
        #        \|
        #         \
        #         |\
        #         | \
        #     R3  |  \  R1
        #         |R0 \
        #    _____|____\______ s
        #         |     \
        #     R4  | R5   \  R6
        #
        # The code below is equivalent to the following if/else structure
        #
        # if s + t <= 1:
        #     if s < 0:
        #         if t < 0:
        #             region 4
        #         else:
        #             region 3
        #     elif t < 0:
        #         region 5
        #     else:
        #         region 0
        # else:
        #     if s < 0:
        #         region 2
        #     elif t < 0
        #         region 6
        #     else:
        #         region 1

        # Conditions
        st_l1 = s + t <= det
        s_l0 = s < 0
        t_l0 = t < 0

        # Region 0 (inside triangle)
        i = np.flatnonzero(st_l1 & ~s_l0 & ~t_l0)
        deti = det[i]
        sp[i] = s[i] / deti
        tp[i] = t[i] / deti

        # Region 1
        # The idea is to substitute the constraints on s and t into F(s,t) and
        # solve, e.g., here we are in region 1 and have Q(s,t) = Q(s,1-s) = F(s)
        # since in this case, for a point to be on the triangle, s+t must be 1
        # meaning that t = 1-s.
        i = np.flatnonzero(~st_l1 & ~s_l0 & ~t_l0)
        aa, bb, cc, dd, ee = a[i], b[i], c[i], d[i], e[i]
        numer = cc + ee - (bb + dd)
        denom = aa - 2 * bb + cc
        sp[i] = np.clip(numer / denom, 0, 1)
        tp[i] = 1 - sp[i]

        # Region 2
        i = np.flatnonzero(~st_l1 & s_l0)  # ~t_l0
        aa, bb, cc, dd, ee = a[i], b[i], c[i], d[i], e[i]
        tmp0 = bb + dd
        tmp1 = cc + ee
        j = tmp1 > tmp0
        j_ = ~j
        k, k_ = i[j], i[j_]
        numer = tmp1[j] - tmp0[j]
        denom = aa[j] - 2 * bb[j] + cc[j]
        sp[k] = np.clip(numer / denom, 0, 1)
        tp[k] = 1 - sp[k]
        sp[k_] = 0
        tp[k_] = np.clip(-ee[j_] / cc[j_], 0, 1)

        # Region 3
        i = np.flatnonzero(st_l1 & s_l0 & ~t_l0)
        cc, ee = c[i], e[i]
        sp[i] = 0
        tp[i] = np.clip(-ee / cc, 0, 1)

        # Region 4
        i = np.flatnonzero(st_l1 & s_l0 & t_l0)
        aa, cc, dd, ee = a[i], c[i], d[i], e[i]
        j = dd < 0
        j_ = ~j
        k, k_ = i[j], i[j_]
        sp[k] = np.clip(-dd[j] / aa[j], 0, 1)
        tp[k] = 0
        sp[k_] = 0
        tp[k_] = np.clip(-ee[j_] / cc[j_], 0, 1)

        # Region 5
        i = np.flatnonzero(st_l1 & ~s_l0 & t_l0)
        aa, dd = a[i], d[i]
        tp[i] = 0
        sp[i] = np.clip(-dd / aa, 0, 1)

        # Region 6
        i = np.flatnonzero(~st_l1 & t_l0)  # ~s_l0
        aa, bb, cc, dd, ee = a[i], b[i], c[i], d[i], e[i]
        tmp0 = bb + ee
        tmp1 = aa + dd
        j = tmp1 > tmp0
        j_ = ~j
        k, k_ = i[j], i[j_]
        numer = tmp1[j] - tmp0[j]
        denom = aa[j] - 2 * bb[j] + cc[j]
        tp[k] = np.clip(numer / denom, 0, 1)
        sp[k] = 1 - tp[k]
        tp[k_] = 0
        sp[k_] = np.clip(-dd[j_] / aa[j_], 0, 1)

        # Distance from original point to its projection on the triangle
        projs = v0[pttris] + sp[:, None] * e0[pttris] + tp[:, None] * e1[pttris]
        dists = np.linalg.norm(rep_points - projs, axis=1)
        weights = np.column_stack((1 - sp - tp, sp, tp))

        if return_all_projections:
            tris = pttris
        else:
            # Find the closest projection
            indptr = [0] + np.cumsum(npttris).tolist()
            i = cortech.utils.sliced_argmin(dists, indptr)
            tris = pttris[i]
            weights = weights[i]
            projs = projs[i]
            dists = dists[i]

        return tris, weights, projs, dists

    def stitch_borders(self, inplace: bool = False):
        v, f, vmap, fmap = pmp.stitch_borders(self.vertices, self.faces)
        vd = self.data.vertex.new_from_subset(vmap)
        fd = self.data.face.new_from_subset(fmap)
        key = "original_id"
        if key not in vd:
            vd[key] = vmap
        if key not in fd:
            fd[key] = fmap
        if inplace:
            self.vertices = v
            self.faces = f
            self.data = TriangulationData(vd, fd)
            return self
        else:
            return self.new(v, f, vertex_data=vd, face_data=fd)

    # def snap_borders(self):
    #     return pmp.snap_borders(self.vertices, self.faces)

    @classmethod
    def from_point_set(
        cls,
        points: npt.NDArray,
        alpha: float = 100.0,
        offset: float = 500.0,
        relative: bool = True,
    ):
        """Generate a surface from a point cloud by alpha wrapping.

        Parameters
        ----------
        points : _type_
            Points to be alpha wrapped.
        alpha : _type_
            Controls the size of the output triangles. If relative=True then
            larger value gives smaller triangles; if relative=False then
            smaller value gives smaller triangles.
        offset : _type_
            Controls how tight the points should be wrapped. If relative=True
            then larger value gives better approximation of object. If
            relative=False then smaller value gives better approximation of
            object.
        relative : bool, optional
            If true, `alpha` and `offset` are interpreted as relative to the
            diagonal length of the bounding box of `points`, i.e.,

                alpha = diagonal_length / alpha
                offset = diagonal_length / offset

        Returns
        -------
        _type_
            _description_
        """

        if relative:
            diag_length = np.linalg.norm(points.max(0) - points.min(0))
            alpha = diag_length / alpha
            offset = diag_length / offset

        v, f = alpha_wrap_3.alpha_wrap_3_points(points, alpha, offset)
        return cls(v, f)

    def alpha_wrap(
        self, alpha: float = 100.0, offset: float = 500.0, relative: bool = True
    ):
        """Generate a surface by alpha wrapping this surface.

        Parameters
        ----------
        alpha : _type_
            Controls the size of the output triangles. If relative=True then
            larger value gives smaller triangles; if relative=False then
            smaller value gives smaller triangles.
        offset : _type_
            Controls how tight the points should be wrapped. If relative=True
            then larger value gives better approximation of object. If
            relative=False then smaller value gives better approximation of
            object.
        relative : bool, optional
            If true, `alpha` and `offset` are interpreted as relative to the
            diagonal length of the bounding box of `points`, i.e.,

                alpha = diagonal_length / alpha
                offset = diagonal_length / offset

        Returns
        -------
        _type_
            _description_
        """
        v = self.vertices
        if relative:
            diag_length = np.linalg.norm(v.max(0) - v.min(0))
            alpha = diag_length / alpha
            offset = diag_length / offset

        v, f = alpha_wrap_3.alpha_wrap_3_surface(v, self.faces, alpha, offset)
        return self.new(v, f)

    def vertex_and_opposing_edge(self):
        """For each vertex in a face, get the three pairs

            [ v_i, (v_j, v_k) ]
            [ v_j, (v_i, v_k) ]
            [ v_k, (v_i, v_j) ]

        i.e., the vertex and its opposing edge.
        """
        vs = np.arange(self.vertices_per_face)
        vertex_opposite_edge = np.concatenate(
            [
                vs[np.isin(vs, e, assume_unique=True, invert=True)]
                for e in self.edge_pairs
            ]
        )
        # indices of edge pairs associated with vertex_opposite_edge
        vertex_edges = np.stack(
            [np.where(v == self.edge_pairs)[0] for v in vertex_opposite_edge]
        )
        return vertex_edges, vertex_opposite_edge

    def opposing_angles(
        self, faces: npt.ArrayLike | None = None, min_edge_length=1e-6, min_angle=1e-3
    ):
        """For each face, compute the angle opposite each edge. Optionally,
        integrate the total angle at each vertex

        Theta in Fig. 3 (c) of Meyer (2003).

        Parameters
        ----------
        integrate_vertex_angles
            For each vertex, integrate the corresponding angles for all faces
            which it is part of.


        """
        # EI = self.topology.edge_pairs
        # VI = self.topology.vertex_edges
        # V = self.topology.vertex_opposite_edge
        EI = self.edge_pairs
        VI, V = self.vertex_and_opposing_edge()

        m = self.mesh(faces)
        E = np.stack([m[:, i] - m[:, j] for i, j in EI])
        EN = np.maximum(np.linalg.vector_norm(E, axis=-1), min_edge_length)

        # we need to clip due to numerical inaccuracies
        face_angles = np.clip(
            np.acos(
                np.clip(
                    np.stack(
                        [
                            np.sum(
                                self._bool_to_sign(EI[ej, 0] == vi)
                                * E[ej]
                                * self._bool_to_sign(EI[ek, 0] == vi)
                                * E[ek],
                                -1,
                            )
                            / (EN[ej] * EN[ek])
                            for vi, (ej, ek) in zip(V, VI)
                        ],
                        -1,
                    ),
                    -1.0,
                    1.0,
                )
            ),
            min_angle,
            np.pi - min_angle,
        )

        return face_angles

    def collapse_halfedges(self, halfedges):
        """Collapse a number of halfedges. By halfedges (CGAL terminology), we
        mean a *directed* edge.

        A few checks are performed to verify the validity of the edge collapse
        but generally it is the responsibility of the user to ensure that the
        progressive edge collapses will give the desired outcome.

        Parameters
        ----------
        halfedges
            An array of shape (n,2) where the first and second columns are the
            sources and targets of the edges, respectively. Edges are collapsed
            by collapsing the source vertices onto the target vertices.

        Returns
        -------
        _type_
            _description_


        """
        assert halfedges.ndim == 2 and halfedges.shape[1] == 2
        v, f, fid, v_pmap, f_pmap = pmp.collapse_halfedges(
            self.vertices, self.faces, halfedges, self.face_id
        )
        vd = dict(original_id=v_pmap)
        fd = dict(original_id=f_pmap)
        return self.new(v, f, face_id=fid, vertex_data=vd, face_data=fd)

    def collapse_short_edges(
        self,
        target_edge_length: float,
        face_is_selected: npt.ArrayLike | None = None,
        vertex_is_constrained: npt.ArrayLike | None = None,
        edge_is_constrained: npt.ArrayLike | None = None,
        *,
        relative: bool = False,
        inplace: bool = False,
    ):
        """_summary_

                Parameters
        ----------
        target_edge_length : float
            _description_
        face_is_selected : npt.ArrayLike | None, optional
            Remesh only a subset of the faces in `self`.
        edge_is_constrained: npt.ArrayLike | None = None
            Do not remesh these edges.
            * A constrained edge can be split or collapsed, but not flipped,
                nor its endpoints moved by smoothing.
            * Sub-edges generated by splitting are set to be constrained.
            * Patch boundary edges (i.e. incident to only one face in the
                range) are always considered as constrained edges.
        relative : bool, optional
            If true, `target_edge_length` is considered as a factor with which
            to scale the mean edge length of the current surface rather than
            an absolute value.
        inplace : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_


        """
        if relative:
            target_edge_length = target_edge_length * self.edges_norm().mean()

        if face_is_selected is not None:
            face_is_selected = _ensure_array_is_index(face_is_selected, self.n_faces)
        if vertex_is_constrained is not None:
            vertex_is_constrained = _ensure_array_is_index(
                vertex_is_constrained, self.n_vertices
            )

        v, f, fid, v_pmap, f_pmap = pmp.collapse_short_edges(
            self.vertices,
            self.faces,
            target_edge_length,
            self.face_id,
            face_is_selected,
            vertex_is_constrained,
            edge_is_constrained,
        )
        if inplace:
            self.vertices = v
            self.faces = f
            self.face_id = fid
            self.data.vertex["original_id"] = v_pmap
            self.data.face["original_id"] = f_pmap
            return self
        else:
            vd = dict(original_id=v_pmap)
            fd = dict(original_id=f_pmap)
            return self.new(v, f, face_id=fid, vertex_data=vd, face_data=fd)

    def cotangents(self, face_angles: npt.NDArray | None = None):
        """Compute the cotangent to `face_angles`, i.e., all angles in

        Parameters
        ----------
        face_angles : npt.NDArray | None, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        face_angles = self.opposing_angles() if face_angles is None else face_angles
        return 1.0 / np.tan(face_angles)

    def voronoi_area(
        self, face_angles: npt.NDArray | None = None, apply_correction: bool = True
    ):
        """Calculate Voronoi area (eq. 7) of each vertex or, if
        `apply_correction` is True, calculcate "A_mixed" (fig. 4) from Meyer
        (2003).

        Parameters
        ----------

        Returns
        -------

        References
        ----------
        Meyer (2003). Discrete Differential-Geometry Operator for Triangulated
            2-Manifolds.
        """
        face_angles = self.opposing_angles() if face_angles is None else face_angles
        cotangents = self.cotangents(face_angles).ravel()

        edges = self.edges()
        edge_len_sq = np.ravel(self.edges_norm(edges) ** 2)

        # The two contributions to the Voronoi area
        # A = (cot_alpha_ij + cot_beta_ij) * (x_i - x_j)
        #   = cot_alpha_ij * (x_i - x_j) + cot_beta_ij * (x_i - x_j)
        cot_x_E2_0 = cotangents * edge_len_sq

        if apply_correction:
            # Each cot_ij * (x_i - x_j) is collected into both x_i and x_j,
            # however, the angle might be obtuse at x_i but not x_j (or vice
            # versa). Hence, we need to duplicate the array: one where x_i is the
            # source vertex and one where x_j is the source vertex
            cot_x_E2_1 = cot_x_E2_0.copy()

            # The Voronoi areas are not valid for obtuse triangles (i.e.,
            # triangles with any angle larger than pi/2). Apply correction
            # (fig. 4)
            is_obtuse_angle = face_angles > np.pi / 2.0
            is_obtuse_triangle = is_obtuse_angle.any(-1)

            # Get obtuse-ness at each vertex (related to source vertex in last dim)
            obtuse = is_obtuse_angle[..., self.edge_pairs].reshape(-1, 2)
            obtuse_tri = np.broadcast_to(
                is_obtuse_triangle[..., None], face_angles.shape
            ).ravel()

            face_area = np.broadcast_to(
                self.face_areas()[..., None], face_angles.shape
            ).ravel()

            # If angle is obtuse at x (vertex of interest), use face_area / 2.0
            # instead of Voronoi area contribution
            # (The factor 4.0 is to compensate for 1.0 / 8.0 later)
            cot_x_E2_0[obtuse[..., 0]] = 0.5 * face_area[obtuse[..., 0]] * 4.0
            cot_x_E2_1[obtuse[..., 1]] = 0.5 * face_area[obtuse[..., 1]] * 4.0

            # If not obtuse at x but triangle is obtuse at any vertex, use
            # face_area / 4.0 instead of Voronoi area contribution
            m = obtuse_tri & ~obtuse[..., 0]
            cot_x_E2_0[m] = 0.25 * face_area[m] * 4.0
            m = obtuse_tri & ~obtuse[..., 1]
            cot_x_E2_1[m] = 0.25 * face_area[m] * 4.0
        else:
            cot_x_E2_1 = cot_x_E2_0

        # collect all contributions per vertex
        A_mixed = np.bincount(
            edges.T.ravel(),
            weights=np.concat((cot_x_E2_1, cot_x_E2_0)),
            minlength=self.n_vertices,
        )
        A_mixed /= 8.0
        return A_mixed

    @staticmethod
    def _bool_to_sign(b: bool):
        return 1.0 if b else -1.0

    def degree(self, edges: npt.NDArray | None = None):
        edges = self.edges() if edges is None else edges
        return 0.5 * np.bincount(edges.ravel())

    # def degree_matrix(self, A: scipy.sparse.csr_array | None = None):
    #     A = self.vertex_adjacency() if A is None else A
    #     return scipy.sparse.diags_array(A.sum(1))

    # def laplacian_matrix(
    #     self,
    #     A: scipy.sparse.csr_array | None = None,
    #     D: scipy.sparse.dia_array | None = None,
    # ):
    #     A = self.vertex_adjacency() if A is None else A
    #     D = self.degree_matrix(A) if D is None else D
    #     return D - A

    def adjacency_matrix(self, connectivity="vertex", include_self: bool = False):
        """Assemble the adjacency matrix for vertices or faces."""

        match connectivity:
            case "vertex":
                row_ind = np.concat(
                    [self.faces[:, i] for p in self.edge_pairs for i in p]
                )
                col_ind = np.concat(
                    [self.faces[:, i] for p in self.edge_pairs for i in p[::-1]]
                )
                data = np.ones_like(row_ind)
                shape = (self.n_vertices, self.n_vertices)

            case "face":
                edges = self.edges(sort_axis_1=True)  # e.g., (1,0) and (0,1) -> (0,1)

                # ONLY WORKS FOR MANIFOLD EDGES!

                # # Now sort the vertex-vertex edges
                # # first column has 1st priority
                # a0 = edges[:, 0].argsort()
                # s0 = edges[a0]
                # # second column has 2nd priority (actually, we just want to sub-sort `s0`)
                # a1 = s0[:, 1].argsort()
                # s1 = s0[a1]
                # # "stable" keeps order of like items, hence both columns will be
                # # sorted after this operation
                # a2 = np.argsort(s1[:, 0], kind="stable")  # .argsort(stable=True)
                # # s2 = s1[a2] # the sorted edges

                # faces_enum = np.broadcast_to(
                #     np.arange(self.n_faces)[:, None], self.faces.shape
                # ).ravel()
                # face_edges = faces_enum[a0[a1[a2]]].reshape(-1, 2)

                # uc: number of occurrences of each edge
                # uc == 1 is a border edge
                # uc == 2 is a manifold edge
                # uc > 2 is a non-manifold edge (part of more than two faces)
                # ui : provides an index for each unique edge

                _, ui, uc = np.unique(
                    edges, axis=0, return_inverse=True, return_counts=True
                )
                if np.any(uc > 2):
                    raise RuntimeError(
                        "Non-manifold topology (some edges are part of more than two triangles)"
                    )

                # we can ignore border edges as these do not connect any faces
                manifold_edges = uc[ui] == 2

                # sort edges so that when applying to faces_enum, faces that
                # shared a particular edge will be next to each other.
                s = ui[manifold_edges].argsort()

                faces_enum = np.broadcast_to(
                    np.arange(self.n_faces)[:, None], self.faces.shape
                ).ravel()

                face_edges = faces_enum[manifold_edges][s].reshape(-1, 2)
                row_ind = face_edges.ravel()
                col_ind = face_edges[:, ::-1].ravel()
                shape = (self.n_faces, self.n_faces)
                data = np.ones(face_edges.size)
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

    def face_adjacency_matrix(self, include_self=False):
        return self.adjacency_matrix("face", include_self)

    def vertex_adjacency_matrix(self, include_self=False):
        return self.adjacency_matrix("vertex", include_self)

    def mass_matrix(self, mode="voronoi area", inverse: bool = False):
        match mode:
            case "degree":
                mass = self.degree()
            case "voronoi area":
                mass = self.voronoi_area()
            case _:
                raise ValueError("Invalid mass matrix ")
        return scipy.sparse.diags_array(1.0 / mass if inverse else mass)

    def stiffness_matrix(self, mode: str = "cotangent"):
        """Form a stiffness matrix from the geometry of the surface.

        Parameters
        ----------
        mode : str


        Returns
        -------
        S : scipy.sparse.csr_array
            The stiffness matrix.

        """
        edges = self.edges()

        match mode:
            case "uniform":
                # 1 if neighbor
                # data =
                data = 0.5 * np.ones_like(edges[:, 0]).ravel()
            case "distance-inverse":
                # distance to neighbor
                data = 0.5 * 1 / self.edges_norm().ravel()

            case "cotangent":
                # Stiffness matrix (cotangent of angles)
                #
                #   L[i,j] = L[j,i] = -0.5 * (cot[a] + cot[b])
                #
                # where a and b are the angles of the edge between i and j.
                cot = self.cotangents().ravel()
                data = 0.5 * cot

        data = np.broadcast_to(data[:, None], edges.shape).T.ravel()
        ix_row = edges.T.ravel()
        ix_col = edges[:, ::-1].T.ravel()
        S = scipy.sparse.csr_array(
            (data, (ix_row, ix_col)), shape=(self.n_vertices, self.n_vertices)
        )
        # S.setdiag(-S.sum(1))

        return S

    def laplacian_matrix(
        self,
        mass_matrix: str = "voronoi area",
        stiffness_matrix: str = "cotangent",
    ):
        """Compute the Laplacian matrix, L, of the discrete Laplace-Beltrami
        operator.

        Computes an estimate of the mean curvature over a function at each
        vertex using discrete Laplace-Beltrami operator. If `f` is None, we
        compute the curvature of the surface itself (i.e., using the vertex
        positions).

        The Laplace-Beltrami operator (also known as the mean curvature
        normal operator)

            K(i) = 2 * H(i) * n(i)
            K(i) = 0.5 * 1.0 / area * sum_{j in N(i)} [ cot(a_ij) + cot(b_ij) ] * (f_i - f_j)

        where N(i) is the neighborhood of i, n(i) is the normal at i, and H(i)
        is the mean curvature. The latter is therefore given by

            H(i) = 0.5 * n(i).T * K(i)  # signed
                = 0.5 * |K(i)|         # unsigned

        Parameters
        ----------
        mass_matrix : str
            The type of mass matrix to use.
        stiffness_matrix : str
            The type of stiffness matrix to use.

        Returns
        -------
        _type_
            _description_

        References
        ----------
        Meyer et al. (2003). Discrete Differential-Geometry Operators for
            Triangulated 2-Manifolds.
        https://computergraphics.stackexchange.com/questions/1718/what-is-the-simplest-way-to-compute-principal-curvature-for-a-mesh-triangle
        """
        M_inv = self.mass_matrix(mass_matrix, inverse=True)
        S = self.stiffness_matrix(stiffness_matrix)

        # Mass matrix (vertex areas)

        # edges = self.compute_edges()
        # diff_vector = np.diff(f[edges], axis=1).squeeze()
        # cot_vec_sum = torch.zeros_like(f)
        # cot_vec_sum = torch.index_add(cot_vec_sum, 1, edges[:, 0], cot * diff_vector)
        # cot_vec_sum = torch.index_add(cot_vec_sum, 1, edges[:, 1], -cot * diff_vector)
        # return 0.5 * 1.0 / atleast_nd_append(vertex_area, f.ndim) * cot_vec_sum

        return M_inv @ S


Surface = ManifoldSurface


class FixedTopologySurface(ManifoldSurface):
    def __init__(self, vertices, faces, **kwargs):
        # when the topology is fixed, we do not need to check it
        kwargs["check_topology"] = False
        super().__init__(vertices, faces, **kwargs)

    def subdivide(self, *args, **kwargs):
        return self.new(
            self.subdivide_vertices(*args, **kwargs),
            self.subdivide_faces(*args, **kwargs),
        )

    def subdivide_faces(self):
        raise NotImplementedError(
            "`FixedTopologySurface` is a base class and does not implement a subdivision scheme."
        )

    def subdivide_vertices(self):
        raise NotImplementedError(
            "`FixedTopologySurface` is a base class and does not implement a subdivision scheme."
        )

    def subsample(self):
        return self.new(self.subsample_vertices(), self.subsample_faces())

    def subsample_vertices(self):
        raise NotImplementedError(
            "`FixedTopologySurface` is a base class and does not implement a subsampling scheme."
        )

    def subsample_faces(self):
        raise NotImplementedError(
            "`FixedTopologySurface` is a base class and does not implement a subsampling scheme."
        )

    @classmethod
    def recursive_subdivision(cls, n: int, **kwargs):
        assert n >= 0
        surface = cls(**kwargs)
        surfaces = []
        if "faces" in kwargs:
            del kwargs["faces"]
        for _ in range(0, n):
            surface = cls(surface.subdivide_faces(), **kwargs)
            surfaces.append(surface)
        return surfaces


class DeepSurferSurface(FixedTopologySurface):
    def __init__(self, vertices, faces, **kwargs):
        assert len(vertices) == 245762
        assert len(faces) == 491520
        # this cannot be specified
        kwargs["edge_pairs"] = np.array([[0, 1], [1, 2], [2, 0]])
        super().__init__(vertices, faces, **kwargs)

        self.n_faces_prev = self.n_faces // 4
        self.n_vertices_prev = (self.n_faces_prev + 4) // 2
        self.n_faces_next = self.n_faces * 4
        self.n_vertices_next = (self.n_faces_next + 4) // 2

    def subsample_faces(self):
        original_vertices = self.faces < self.n_vertices_prev
        # these are faces consisting strictly of vertices from this level
        newest_faces = np.sum(original_vertices, 1) == 0
        A = self.adjacency_matrix("face")
        # neighbors of the newest faces
        nn = A[newest_faces].indices.reshape(-1, 3)
        # prev_faces_soup = self.faces[~newest_faces][nn]
        prev_faces_soup = self.faces[nn]
        return prev_faces_soup[original_vertices[nn]].reshape(-1, 3)

    def subsample_vertices(self):
        return self.vertices[: self.n_vertices_prev]

    def subdivide_faces(self, retain_edge_order: bool = False):
        """Subdivide all faces, increasing the face count by a factor of four.

        References
        ----------
        This is based on pytorch3d.ops.subdivide_meshes.
        """
        # The subdivision scheme of the face (V0,V1,V2)
        #
        #               V0
        #              /  \
        #             /    \
        #            /      \
        #           /        \
        #          /    f0    \
        #         /            \
        #        v0 ---------- v2
        #       /  \          /  \
        #      /    \   f3   /    \
        #     /      \      /      \
        #    /   f1   \    /   f2   \
        #   /          \  /          \
        # V1 ---------- v1 ---------- V2
        #
        # where V* = original vertices and v* = new vertices

        # These are the faces made up entirely of new vertices
        f3 = self.faces_to_edges(retain_edge_order) + self.n_vertices

        # Concatenate each "original" vertex with the vertices placed on it's
        # adjacent edges such that the orientation (e.g., counter-clockwise) is
        # preserved

        # original vertices
        V0 = self.faces[:, 0]
        V1 = self.faces[:, 1]
        V2 = self.faces[:, 2]
        # new vertices
        v0 = f3[:, 0]
        v1 = f3[:, 1]
        v2 = f3[:, 2]

        f0 = np.stack([V0, v0, v2], -1)
        f1 = np.stack([V1, v1, v0], -1)
        f2 = np.stack([V2, v2, v1], -1)

        return np.concat((f3, f0, f1, f2))

    def subdivide_vertices(self, retain_edge_order: bool = False):
        """From (V0,V1,V2), insert (v0, v1, v2), at the midpoint of the edges
        (V0, V1), (V0, V2), (V1, V2).
        """
        e = self.edges(unique=True, retain_edge_order=retain_edge_order)
        new_vertices = self.vertices[e].mean(-2)
        return np.concat((self.vertices, new_vertices))


class FsAverageSurface(FixedTopologySurface):
    def __init__(self, vertices):
        super().__init__(vertices, edge_pairs=np.array([[2, 0], [1, 2], [0, 1]]))

    @staticmethod
    def reorder_subdivided_faces(faces):
        return np.concat((faces[::4], faces.reshape(-1, 4, 3)[:, 1:].reshape(-1, 3)))

    def subdivide_faces(self):
        r"""Subdivide all faces, increasing the face count by a factor of four.

        FreeSurfer subdivision scheme

                      V0------------ v15---------- V5
                     /  \           / \           /
                    /    \   f4    /   \   f5    /
                   /      \       /     \       /
                  /        \     /       \     /
                 /    f0    \   /    f6   \   /
                /            \ /           \ /
               v14---------- v12 --------- v16
              /  \          /  \          /
             /    \   f3   /    \   f7   /
            /      \      /      \      /
           /   f3   \    /   f1   \    /
          /          \  /          \  /
        V3 ---------- v13---------- V4
                Originally,
          F0 = (V0, V3, V4).
          F1 = (V0, V4, V5).
        After subdivision, F0 and F1 are replaced by
          (f0, f1, f2, f3)
          (f4, f5, f6, f7)
        i.e., vertices are placed on edges in this order: [2,0], [1,2], [0,1]

        Notes
        -----
        In order to reproduce the faces from

            FREESURFER_HOME/average/surf/lh.sphere.ico{i}.reg

        we would need the following subdivision. However, when indexing into the
        vertices of fsaverage7 in order to obtain the lower resolution icos (e.g.,
        v[:N_FACES_ICO_0], v[:N_FACES_ICO_1], etc.), we simply have to reorder the
        faces each time!

        def fsaverage_reorder_faces_once(faces):
            return torch.cat((faces[::4], faces.reshape(-1, 4, 3)[:, 1:].reshape(-1, 3)))
        def get_fsaverage_topology(ico_order: int = 0):
            assert ico_order >= 0
            topologies = [topology := FsAverageTopology()]
            topologies_reordered = [topology_reordered := FsAverageTopology()]
            for i in range(0, ico_order):
                if i <= 3:
                    topology = FsAverageTopology(topology.subdivide_faces())
                    topologies.append(topology)
                if ico_order >= 5:
                    topology_reordered = FsAverageTopology(fsaverage_reorder_faces_once(topology_reordered.subdivide_faces()))
                    topologies_reordered.append(topology_reordered)
            return topologies + topologies_reordered[5:] if ico_order >= 5 else topologies

        References
        ----------
        This is based on pytorch3d.ops.subdivide_meshes.
        """

        new_vertices = np.arange(
            self.n_vertices,
            4 * self.n_faces,
        )

        # the vertex to be inserted at each edge
        ve = new_vertices[self.faces_to_edges]

        faces = torch.stack(
            (
                np.stack((self.faces[:, 0], ve[:, 2], ve[:, 0]), axis=1),
                np.stack((ve[:, 0], ve[:, 1], self.faces[:, 2]), axis=1),
                np.stack((ve[:, 2], ve[:, 1], ve[:, 0]), axis=1),
                np.stack((ve[:, 2], self.faces[:, 1], ve[:, 1]), axis=1),
            ),
            dim=1,
        ).reshape(4 * self.n_faces, 3)

        return self.reorder_subdivided_faces(faces)


class MultiSurface(ManifoldSurface):
    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, value):
        value = cortech.utils.atleast_nd_prepend(value, 3)
        assert value.ndim == 3
        self._vertices = value
        self.n_surfaces, self.n_vertices, self.n_dim = value.shape
        # self._ix_surfaces = np.arange(self.n_surfaces)[:, None]

    def mesh(self, subset=None):
        f = self.faces if subset is None else self.faces[subset]
        return self.vertices[np.arange(self.n_surfaces), f]

    def bounding_box(self):
        return np.stack((self.vertices.min((0, 1)), self.vertices.max((0, 1))))


class Sphere(ManifoldSurface):
    def __init__(
        self,
        vertices: npt.NDArray,
        faces: npt.NDArray,
        normalize: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(vertices, faces, space="scanner", geometry=None, **kwargs)
        # Ensure on unit sphere
        if normalize:
            self.vertices = cortech.utils.normalize(self.vertices, axis=-1)
        self._proj_matrix = None

    def to_spherical_coordinates(self):
        return cortech.sphere_utils.cart_to_sph(self.vertices)

    def set_projection(
        self, points: npt.NDArray, method: str = "linear", n_closest_vertices: int = 5
    ):
        """Project `points` to `self`, i.e., compute a mapping that can be used
        to map vertex data from `self` to `points`.

        The projection matrix is a sparse matrix with dimensions
        (len(points), self.n_vertices) where each row has exactly one
        (nearest) or three (linear) entries that sum to one.

        For example, to map data from fsaverage to subject space,

            data_on_fsavg = ...
            fsavg = SphericalRegistration( ... )
            subject = SphericalRegistration( ... )
            fsavg.project(subject.vertices)
            data_on_subject = fsavg.resample( data_on_fsavg )

        PARAMETERS
        ----------
        target :
            The target mesh (i.e., the mesh to interpolate *to*).
        n_nearest_vertices: int
            When using linear interpolation, we need to identify the triangle
            to which each vertex in `target` projects. Testing all target
            vertices against all triangles in `self` is expensive and
            inefficient, thus we compute an approximation by finding, for each
            vertex in `target`, the `n_nearest_vertices` closest vertices in
            `self`. We find the triangles to which these points belong and then
            test only against these triangles.
        """
        n_points, n_dim = points.shape

        match method:
            case "nearest":
                kdtree = scipy.spatial.cKDTree(self.vertices)
                cols = kdtree.query(points)[1]
                rows = np.arange(n_points)
                weights = np.ones(n_points, dtype=int)
            case "linear":
                tris, weights, _, _ = self.project_points(points, n_closest_vertices)
                rows = np.repeat(np.arange(n_points), n_dim)
                cols = self.faces[tris].ravel()
                weights = weights.ravel()
            case _:
                raise ValueError(
                    f"Invalid projection method, please select `nearest` or `linear` (got {method})."
                )

        self._proj_matrix = scipy.sparse.csr_array(
            (weights, (rows, cols)), shape=(n_points, self.n_vertices)
        )
        self._proj_matrix.sum_duplicates()

    def resample(self, values: npt.NDArray):
        """Pull values defined on `self` to the points used as input to
        `set_projection`.

        Parameters
        ----------
        values : npt.NDArray
            Data to map to the target points. The shape must be
            (self.n_vertices, ...)

        Returns
        -------
        mapped values: npt.NDArray
            Data mapped onto the target surface. The shape is
            (len(points), ...)
        """
        if self._proj_matrix is None:
            raise RuntimeError(
                "No projection matrix found. Please run `set_projection`."
            )
        return self._proj_matrix @ values

    def set_projection_and_resample(
        self, target_points: npt.NDArray, values: npt.NDArray, *args, **kwargs
    ):
        self.set_projection(target_points, *args, **kwargs)
        return self.resample(values)

    @classmethod
    def from_freesurfer(cls, filename: Path | str, **kwargs):
        """Read default and .srf files from FreeSurfer.

        We do not need metadata for spheres.


        Parameters
        ----------
        filename : Path | str
            File to read.

        Returns
        -------
        Surface :
            Instance of self.

        """
        v, f = cortech.freesurfer.read_geometry(filename)
        return cls(v, f, **kwargs)

    @classmethod
    def from_gifti(cls, filename: nib.GiftiImage | Path | str, **kwargs):
        """Read surface from Gifti file.

        We do not need metadata for spheres.

        Parameters
        ----------
        filename : Path | str
            File to read.

        Returns
        -------
        Surface :
            Instance of self.
        """
        if isinstance(filename, (Path, str)):
            gii = nib.load(filename)
        else:
            gii = filename
            assert isinstance(gii, nib.GiftiImage)
        v = gii.agg_data("NIFTI_INTENT_POINTSET").astype(float)
        f = gii.agg_data("NIFTI_INTENT_TRIANGLE")
        return cls(v, f, **kwargs)


def iterative_smooth_shape_by_curvature_threshold(
    s: ManifoldSurface,
    smooth_kwargs: list[dict] | None = None,
    remesh_kwargs: dict | None = None,
    decoupling_amount: float = 0.1,
    do_remesh: bool = True,
    do_decouple: bool = True,
):
    if smooth_kwargs is None:
        smooth_kwargs = [
            dict(time=1.0, n_iter=10, curv_threshold=0.2, ball_radius=0.5),
            dict(time=1.0, n_iter=10, curv_threshold=0.1, ball_radius=0.5),
            dict(time=1.0, n_iter=10, curv_threshold=0.05, ball_radius=0.5),
            dict(time=1.0, n_iter=5, curv_threshold=0.025, ball_radius=0.5),
            dict(time=1.0, n_iter=5, curv_threshold=0.0, ball_radius=0.5),
        ]

    if remesh_kwargs is None:
        # siso = self.adaptive_remeshing(0.5, 0.5, 2.0, n_iter=3, relative_edge_length=True)
        # siso = self.isotropic_remeshing(0.5, n_iter=3, relative_edge_length=True)
        remesh_kwargs = dict(target_edge_length=0.75, n_iter=5, relative=True)

    tri = None
    weight = None
    faces = s.faces.copy()
    for i, kw in enumerate(smooth_kwargs, 1):
        print(f"Iteration :: {i} of {len(smooth_kwargs)}")

        print(">> Smoothing")
        s.smooth_shape_by_curvature_threshold(**kw, inplace=True)
        if do_remesh:
            print(">> Remeshing")
            # s.isotropic_remeshing(**remesh_kwargs, inplace=True)
            s, tri, weight = s.remeshing_with_projection(
                "isotropic", tri, weight, remesh_kwargs=remesh_kwargs
            )
            # s.tangential_relaxation(n_iter=5, inplace=True)

        if do_decouple:
            if i < len(smooth_kwargs):
                curv = s.interpolated_corrected_curvatures()
                n = s.vertex_normals()
                mask = curv.H <= 0.0
                s.vertices[mask] = s.vertices[mask] - decoupling_amount * n[mask]

    # original vertices projected on final (remeshed) surface
    v = np.sum(s.mesh(tri) * weight[..., None], 1)
    # sulc_at_orig = np.sum(sulc_smooth[s.faces[tris1]] * weights1, 1)

    # assert not self.does_self_intersect()
    return s.new(v, faces)


# def inflate_surface(s: Surface, smooth_kwargs: list[dict] | None):
# inflate
# smooth_kwargs = [
#     dict(time=1.0, n_iter=10, curv_threshold=-0.2, ball_radius=0.5, apply_above_curv_threshold=False),
#     dict(time=1.0, n_iter=10, curv_threshold=-0.1, ball_radius=0.5, apply_above_curv_threshold=False),
#     dict(time=1.0, n_iter=10, curv_threshold=-0.05, ball_radius=0.5, apply_above_curv_threshold=False),
#     dict(time=1.0, n_iter=10, curv_threshold=-0.025, ball_radius=0.5, apply_above_curv_threshold=False),
#     dict(time=1.0, n_iter=10, curv_threshold=0.0, ball_radius=0.5, apply_above_curv_threshold=False),
# ]
