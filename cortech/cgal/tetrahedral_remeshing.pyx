from libcpp cimport bool as cppbool
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.vector cimport vector
import numpy as np
import numpy.typing as npt
cimport numpy as np

cdef extern from "tetrahedral_remeshing_src.cpp" namespace "cortech":
    cdef cppclass VolumeMesh:
        vector[vector[float]] vertices
        vector[vector[int]] faces
        vector[vector[int]] cells

    cdef cppclass VolumeMeshWithPMaps:
        vector[vector[float]] vertices
        vector[vector[int]] faces
        vector[vector[int]] cells
        vector[int] faces_pmap
        vector[int] cells_pmap

cdef extern from "tetrahedral_remeshing_src.cpp" nogil:
    VolumeMeshWithPMaps tetrahedral_remeshing_remesh(
        vector[vector[float]] vertices,
        vector[vector[int]] faces,
        vector[vector[int]] cells,
        vector[int] faces_pmap,
        vector[int] cells_pmap,
        vector[vector[int]] constrained_edges,
        # vector[vector[int]] constrained_faces,
        string sizing_field_type,
        float target_edge_length,
        cppbool remesh_boundaries,
        # vector[int] cell_is_selected,
        # bool smooth_constrained_edges
        int n_iterations,
        cppbool check_triangulation,
    )


def remesh(
    vertices: npt.ArrayLike,
    faces: npt.ArrayLike,
    cells: npt.ArrayLike,
    faces_pmap: npt.ArrayLike,
    cells_pmap: npt.ArrayLike,
    constrained_edges: npt.ArrayLike | None = None,
    # constrained_faces: npt.ArrayLike | None,
    sizing_field_type: str = "uniform",
    target_edge_length: float = 1.0,
    remesh_boundaries: bool = True,
    cell_is_selected: npt.ArrayLike | None = None,
    smooth_constrained_edges: bool = False,
    n_iterations: int = 1,
    check_triangulation: bool = True,
):
    constrained_edges = [] if constrained_edges is None else constrained_edges

    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef np.ndarray[int, ndim=2] cpp_c = np.ascontiguousarray(cells, dtype=np.int32)
    cdef np.ndarray[int] cpp_f_pmap = np.ascontiguousarray(faces_pmap, dtype=np.int32)
    cdef np.ndarray[int] cpp_c_pmap = np.ascontiguousarray(cells_pmap, dtype=np.int32)
    cdef np.ndarray[int] cpp_constr_edges = np.ascontiguousarray(constrained_edges, dtype=np.int32)
    # cdef np.ndarray[int] cpp_constr_faces = np.ascontiguousarray(constrained_faces or [], dtype=np.int32)

    # cdef np.ndarray[bool] cpp_c_sel = np.ascontiguousarray(cell_is_selected, dtype=np.bool)
    cdef VolumeMeshWithPMaps out
    cdef string cpp_sizing_field_type = sizing_field_type.encode() # to bytes

    out = tetrahedral_remeshing_remesh(
        cpp_v,
        cpp_f,
        cpp_c,
        cpp_f_pmap,
        cpp_c_pmap,
        cpp_constr_edges,
        # cpp_constr_faces,
        cpp_sizing_field_type,
        target_edge_length,
        remesh_boundaries,
        # cpp_c_sel,
        # smooth_constrained_edges,
        n_iterations,
        check_triangulation,
    )
    v = np.array(out.vertices, dtype=float)
    f = np.array(out.faces, dtype=int)
    t = np.array(out.cells, dtype=int)
    f_pmap = np.array(out.faces_pmap, dtype=int)
    t_pmap = np.array(out.cells_pmap, dtype=int)
    return v, f, t, f_pmap, t_pmap
