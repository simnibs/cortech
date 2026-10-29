import warnings

import numpy as np
import numpy.typing as npt

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

try:
    import tetgen
    HAS_TETGEN = True
except ImportError:
    HAS_TETGEN = False

from cortech.surface import ManifoldSurface, NonManifoldSurface
from  cortech.cgal import mesh_3
import cortech.cgal.tetrahedral_remeshing as tet_remesh
from cortech.utils import gradate_sizing_field

msg_pmap_err = "Data must equal the number of vertices ({nv}), faces ({nf}), or cells ({nc}). Got {n}."


class MeshComplex:
    def __init__(
        self,
        vertices: npt.ArrayLike,
        faces: npt.ArrayLike,
        cells: npt.ArrayLike,
        face_id: npt.ArrayLike | None = None,
        cell_id: npt.ArrayLike | None = None,
        vertex_data: dict | None = None,
        face_data: dict | None = None,
        cell_data: dict | None = None,
    ):
        """Mesh containing a mix of faces (triangles) and cells (tetrahedra)
        and associated properties (data).

        Parameters
        ----------
        property_maps : dict | None
            Dict of properties/data associates with each vertex, face, or cell.

        """
        self.vertices = vertices
        self.cells = cells
        self.faces = faces
        self.face_id = face_id
        self.cell_id = cell_id

        self._vertex_data = {}
        self._face_data = {}
        self._cell_data = {}

        if vertex_data is not None:
            for k, v in vertex_data.items():
                self.vertex_data[k] = v
        if face_data is not None:
            for k, v in face_data.items():
                self.face_data[k] = v
        if cell_data is not None:
            for k, v in cell_data.items():
                self.cell_data[k] = v

    # def cell_data(self):
    #     return {k: v for k, v in self._property_maps.items() if len(v) == self.n_cells}

    # def face_data(self):
    #     return {k: v for k, v in self._property_maps.items() if len(v) == self.n_faces}

    # def vertex_data(self):
    #     return {
    #         k: v for k, v in self._property_maps.items() if len(v) == self.n_vertices
    #     }

    # def __getitem__(self, key):
    #     return self._property_maps[key]

    # def __setitem__(self, key, value):
    #     assert len(value) in {self.n_vertices, self.n_faces, self.n_cells}, (
    #         msg_pmap_err.format(
    #             dict(nv=self.n_vertices, nf=self.n_faces, nc=self.n_cells, n=len(value))
    #         )
    #     )
    #     self._property_maps[key] = value

    @property
    def cells(self):
        return self._cells

    @cells.setter
    def cells(self, value):
        value = np.empty((0, 4), int) if value is None else value
        value = np.atleast_2d(value)
        assert value.ndim == 2
        self._cells = value
        self.n_cells, self.vertices_per_cell = value.shape

    @property
    def faces(self):
        return self._faces

    @faces.setter
    def faces(self, value):
        value = np.empty((0, 3), int) if value is None else value
        value = np.atleast_2d(value)
        assert value.ndim == 2
        self._faces = value
        self.n_faces, self.vertices_per_face = value.shape

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, value):
        value = np.atleast_2d(value)
        assert value.ndim == 2
        self._vertices = value
        self.n_vertices, self.n_dim = value.shape

    @property
    def face_id(self):
        return self._face_id

    @face_id.setter
    def face_id(self, value):
        value = np.asarray(value) if isinstance(value, (list, tuple)) else value
        if value is None:
            self._face_id = np.full(self.n_faces, 1)
        elif isinstance(value, int):
            self._face_id = np.full(self.n_faces, value)
        elif isinstance(value, np.ndarray) and np.isdtype(value.dtype, "integral"):
            assert len(value) == self.n_faces and value.ndim == 1
            self._face_id = np.asarray(value)
        else:
            raise ValueError(f"Invalid face_id ({value})")

    @property
    def cell_id(self):
        return self._cell_id

    @cell_id.setter
    def cell_id(self, value):
        value = np.asarray(value) if isinstance(value, (list, tuple)) else value
        if value is None:
            self._cell_id = np.full(self.n_cells, 1)
        elif isinstance(value, int):
            self._cell_id = np.full(self.n_cells, value)
        elif isinstance(value, np.ndarray) and np.isdtype(value.dtype, "integral"):
            assert len(value) == self.n_cells and value.ndim == 1
            self._cell_id = np.asarray(value)
        else:
            raise ValueError(f"Invalid cell_id ({value})")

    @property
    def cell_data(self):
        return self._cell_data

    @cell_data.setter
    def cell_data(self, k, v):
        assert len(v) == self.n_cells
        self._cell_data[k] = v

    @property
    def face_data(self):
        return self._face_data

    @face_data.setter
    def face_data(self, k, v):
        assert len(v) == self.n_faces
        self._face_data[k] = v

    @property
    def vertex_data(self):
        return self._vertex_data

    @vertex_data.setter
    def vertex_data(self, k, v):
        assert len(v) == self.n_vertices
        self._vertex_data[k] = v

    def gradate_sizing_field(self, h, g):
        """Enforce h(u) <= h(v) + g * |u-v|, i.e., ensure that the gradient of h
        (and thereby element size) does not change by more that `g`, which encodes
        how much edge lengths can change per unit distance.

        """
        return gradate_sizing_field(self.vertices, self.cells, h, g)

    def remesh(
        self,
        sizing: float | npt.ArrayLike | None,
        n_iterations: int = 1,
        remesh_boundaries: bool = True,
        check_triangulation: bool = True
    ):
        assert self.n_cells > 0, "Remeshing requires cells in the complex!"
        if np.any(self.cell_id < 1):
            warnings.warn("`cell_id` contains values smaller than 1. Cells marked as such will be interpreted as being outside of the domain by CGAL!")
        if np.any(self.face_id < 1):
            warnings.warn("`face_id` contains values smaller than 1. Faces marked as such will be interpreted as being outside of the domain by CGAL!")

        v, f, c, fid, cid = tet_remesh.remesh(
            self.vertices,
            self.faces if self.n_faces > 0 else None,
            self.cells,
            self.face_id if self.n_faces > 0 else None,
            self.cell_id,
            "uniform",
            sizing,
            remesh_boundaries,
            n_iterations,
            check_triangulation,
        )
        return MeshComplex(v,f,c,fid,cid)

    def remesh_protect_boundary(
        self,
        sizing: npt.ArrayLike,
        n_iterations: int = 1,
        edge_is_constrained: npt.ArrayLike | None = None,
        nb_neighbors_dim3: int = 30,
        nb_neighbors_dim2: int = 6,
        check_triangulation: bool = True
    ):
        assert self.n_cells > 0, "Remeshing requires cells in the complex!"
        if np.any(self.cell_id < 1):
            warnings.warn("`cell_id` contains values smaller than 1. Cells marked as such will be interpreted as being outside of the domain by CGAL!")
        if np.any(self.face_id < 1):
            warnings.warn("`face_id` contains values smaller than 1. Faces marked as such will be interpreted as being outside of the domain by CGAL!")

        v, f, c, fid, cid = tet_remesh.remesh_protect_boundary(
            sizing,
            self.vertices,
            self.cells,
            self.cell_id,
            self.faces if self.n_faces > 0 else None,
            self.face_id if self.n_faces > 0 else None,
            edge_is_constrained,
            n_iterations,
            nb_neighbors_dim3,
            nb_neighbors_dim2,
            check_triangulation
        )
        return MeshComplex(v,f,c,fid,cid)


    def to_unstructured_grid(
        self,
        include_faces=True,
        include_cells=True,
        remove_unused_points: bool = True
    ):
        assert HAS_PYVISTA, "Conversion to VTK UnstructuredGrid requires PyVista"
        cells_dict = {}

        if include_faces and self.n_faces > 0:
            cells_dict[pv.CellType.TRIANGLE] = self.faces
            face_id = self.face_id
        else:
            face_id = np.array([])
        if include_cells and self.n_cells > 0:
            cells_dict[pv.CellType.TETRA] = self.cells
            cell_id = self.cell_id
        else:
            cell_id = np.array([])
        grid = pv.UnstructuredGrid(cells_dict, self.vertices)
        grid["id"] = np.concat([face_id, cell_id])

        # for k, v in self.face_data.items():
        #     grid[k] = np.concat([v, self.cell_data[k]])

        if remove_unused_points:
            return grid.remove_unused_points()
        else:
            return grid

    def to_multiblock(self):
        assert HAS_PYVISTA, "Conversion to VTK MultiBlock requires PyVista"

        mb = pv.MultiBlock()
        if self.n_faces > 0:
            mb["faces"] = pv.make_tri_mesh(self.vertices, self.faces)
            mb["faces"]["id"] = self.face_id
            for k, v in self.face_data.items():
                mb["faces"][k] = v

        if self.n_cells > 0:
            mb["cells"] = self.to_unstructured_grid(include_faces=False)
            mb["cells"]["id"] = self.cell_id
            for k, v in self.cell_data.items():
                mb["cells"][k] = v
        return mb

    @classmethod
    def from_surface(cls, surface, **kwargs):
        v, f, c, fmap, cmap = mesh_3(
            surface.vertices, surface.faces, **kwargs
        )
        return cls(
            v, f, c, face_data=dict(subdomain=fmap), cell_data=dict(subdomain=cmap)
        )

    # @classmethod
    # def from_indicent_domains(cls, surfaces, incident_domains, **kwargs):
    #     v, f, c, fmap, cmap = make_mesh_complex(
    #         [s.vertices for s in surfaces],
    #         [s.faces for s in surfaces],
    #         incident_domains,
    #         **kwargs,
    #     )
    #     return cls(v, f, c), fmap, cmap

    @classmethod
    def from_triangle_soup(
        cls,
        triangle_soup: NonManifoldSurface,
        **kwargs
    ):
        """_summary_

        Parameters
        ----------
        triangle_soup : cortech.TriangleSoup
            _description_
        remesh_boundary : bool, optional
            If False, no Steiner points are added on the triangle faces.
        tetgen_kwargs : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        assert HAS_TETGEN, "TetGen is required but not found!"

        # if len(kwargs) == 0:
        #     kwargs = dict(minratio=1.1, mindihedral=10.0)
        #     # metric_scale=1.0,  # sizing
        #     nobisect=remesh_boundary,  # preserve boundary of input surface

        cdt = tetgen.TetGen(
            triangle_soup.vertices, triangle_soup.faces, triangle_soup.face_id
        )
        _ = cdt.tetrahedralize(**kwargs)

        # f_pmap = np.ones(
        #     len(f), int
        # )
        f_pmap = np.concat(
            (
                # np.ones(roi.n_faces, int),
                roi["tag"],
                np.full(box.n_faces, 21, int),
                np.full(layer4.n_faces, 22, int),
                # np.full(plane.n_faces, 22, int),
            )
        )

        # use pyvista
        roi_vol_only = roi_vol.extract_cells_by_type(pv.CellType.TETRA)
        cell_idx = roi_vol_only.find_containing_cell(v[c].mean(1))
        c_pmap = roi_vol_only["tag"][cell_idx]

        is_in_box = np.all(np.abs(v[c].mean(1) - box.center) < 0.5 * box_size, 1)
        c_pmap[is_in_box] = 21

        return cls(
            cdt.node,
            cdt.trifaces,
            cdt.elem,
            face_data=dict(subdomain=f_pmap),
            cell_data=dict(subdomain=c_pmap),
        )



    @classmethod
    def from_indicent_subdomains(
        cls,
        surfaces: list[ManifoldSurface] | tuple,
        indicent_subdomains: list[tuple[int, int]] | tuple,
        cell_size: float | None = None,
        cell_radius_edge_ratio: float = 2.0,
        edge_size: float | None = None,
        facet_angle: float = 25.0,
        facet_distance: float = 0.01,
        facet_size: float | None = None,
        factor: float = 1.0,
    ):
        """Generate a tetrahedral mesh from a list of surfaces defining a set
        of incident subdomains each with its own unique subdomain index
        (label).

        Parameters
        ----------
        surfaces : list[cortech.Surface] | tuple
            _description_
        indicent_subdomains : list[tuple[int,int]] | tuple
            _description_
        cell_size : float | None, optional
            (default = 2 * mean edge length).
        cell_radius_edge_ratio : float, optional
            _description_, by default 2.0
        edge_size : float | None, optional
            (default = mean edge length).
        facet_angle : float, optional
            _description_, by default 25.0
        facet_distance : float | None, optional
            (default = 0.01).
        facet_size : float | None, optional
            (default = 2.0 * mean edge length)
        factor : float, optional
            _description_, by default 1.0

        Returns
        -------
        _type_
            _description_
        """
        mean_edge_length = np.mean([s.edges_norm().mean() for s in surfaces])
        mean_edge_length = mean_edge_length * factor
        mesh_kw = dict(
            cell_radius_edge_ratio=cell_radius_edge_ratio,
            cell_size=cell_size or 2 * mean_edge_length,
            edge_size=edge_size or mean_edge_length,
            facet_angle=facet_angle,
            facet_distance=facet_distance,
            facet_size=facet_size or 2 * mean_edge_length,
        )
        v, f, c, fmap, cmap = mesh_3.make_mesh_complex(
            [s.vertices for s in surfaces],
            [s.faces for s in surfaces],
            indicent_subdomains,
            **mesh_kw,
        )
        return cls(v, f, c), fmap, cmap
