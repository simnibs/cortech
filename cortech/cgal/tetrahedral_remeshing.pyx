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
        vector[vector[double]] vertices,
        vector[vector[int]] faces,
        vector[vector[int]] cells,
        vector[int] faces_pmap,
        vector[int] cells_pmap,
        string sizing_field_type,
        float target_edge_length,
        cppbool remesh_boundaries,
        int n_iterations,
        cppbool check_triangulation,
    ) except +

    VolumeMeshWithPMaps tetrahedral_remeshing_remesh_protect_boundary(
        vector[vector[double]] vertices,
        vector[vector[int]] cells,
        vector[int] cells_pmap,
        vector[vector[int]] faces,
        vector[int] faces_pmap,
        vector[double] sizing,
        int n_iterations,
        int nb_neighbors_dim3,
        int nb_neighbors_dim2,
        cppbool check_triangulation,
    ) except +


def remesh(
    vertices: npt.ArrayLike,
    faces: npt.ArrayLike,
    cells: npt.ArrayLike,
    faces_pmap: npt.ArrayLike,
    cells_pmap: npt.ArrayLike,
    sizing_field_type: str = "uniform",
    target_edge_length: float = 1.0,
    remesh_boundaries: bool = True,
    n_iterations: int = 1,
    check_triangulation: bool = True,
):
    # float64 is required; otherwise numerical problems arise and boundary
    # remeshing tends to fail!
    cdef np.ndarray[double, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float64)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef np.ndarray[int, ndim=2] cpp_c = np.ascontiguousarray(cells, dtype=np.int32)
    cdef np.ndarray[int] cpp_f_pmap = np.ascontiguousarray(faces_pmap, dtype=np.int32)
    cdef np.ndarray[int] cpp_c_pmap = np.ascontiguousarray(cells_pmap, dtype=np.int32)

    cdef VolumeMeshWithPMaps out
    cdef string cpp_sizing_field_type = sizing_field_type.encode() # to bytes

    out = tetrahedral_remeshing_remesh(
        cpp_v,
        cpp_f,
        cpp_c,
        cpp_f_pmap,
        cpp_c_pmap,
        cpp_sizing_field_type,
        target_edge_length,
        remesh_boundaries,
        n_iterations,
        check_triangulation,
    )
    v = np.array(out.vertices, dtype=float)
    f = np.array(out.faces, dtype=int)
    t = np.array(out.cells, dtype=int)
    f_pmap = np.array(out.faces_pmap, dtype=int)
    t_pmap = np.array(out.cells_pmap, dtype=int)
    return v, f, t, f_pmap, t_pmap

def remesh_protect_boundary(
    sizing: npt.ArrayLike,
    vertices: npt.ArrayLike,
    cells: npt.ArrayLike,
    cells_pmap: npt.ArrayLike | None = None,
    faces: npt.ArrayLike | None = None,
    faces_pmap: npt.ArrayLike | None = None,
    n_iterations: int = 1,
    int nb_neighbors_dim3 = 30,
    int nb_neighbors_dim2 = 6,
    check_triangulation: bool = True,
):
    """Tetrahedral remeshing of interior cells and faces but not the outer
    boundary using a sizing field.

    Outer boundary vertices, edges, and faces are constrained meaning that

        1. edges and faces cannot be flipped
        2. vertices cannot removed by collapse or relocated by smoothing

    Furthermore, the sizing field class will prevent vertices and midpoints of
    edges on the outer boundary from being split, thus effectively preventing
    the outer boundary from being modified.

    Parameters
    ----------
    sizing
        Array specifying a target edge length for each vertex.
    vertices
    cells
    cells_pmap
        Property map (subdomain index) of each cell (default is a map of all
        ones).
    nb_neighbors_dim3
        Number of nearest neighbors to search for and interpolate sizing field
        from for points in cells (and on faces which are not specified in
        `faces`).
    nb_neighbors_dim2
        Number of nearest neighbors to search for and interpolate sizing field
        from for points on explicitly specified faces.

    References
    ----------
    https://doc.cgal.org/latest/Tetrahedral_remeshing/group__PkgTetrahedralRemeshingRef.html#ga263775c52eeb483a86a16aeb9eb31af0


    """
    assert len(vertices) == len(sizing)
    cells_pmap = np.ones(len(cells), int) if cells_pmap is None else cells_pmap
    assert len(cells) == len(cells_pmap)
    assert n_iterations > 0
    assert nb_neighbors_dim3 > 0
    assert nb_neighbors_dim2 > 0
    faces = np.array([[]], int) if faces is None else faces
    faces_pmap = np.array([], int) if faces_pmap is None else faces_pmap
    assert len((faces) == len(faces_pmap)) or (faces.size == faces_pmap.size)

    # if (cells_pmap < 1).any():
    #     warnings.warn("`cells_pmap` contains values smaller than 1. Cells marked as such will be interpreted as being outside of the domain by CGAL!")
    # if (faces_pmap < 1).any():
    #     warnings.warn("`faces_pmap` contains values smaller than 1. Cells marked as such will be interpreted as being outside of the domain by CGAL!")


    # float64 is required; otherwise numerical problems arise and boundary
    # remeshing tends to fail!
    cdef np.ndarray[double, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float64)
    cdef np.ndarray[int, ndim=2] cpp_c = np.ascontiguousarray(cells, dtype=np.int32)
    cdef np.ndarray[int] cpp_c_pmap = np.ascontiguousarray(cells_pmap, dtype=np.int32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef np.ndarray[int] cpp_f_pmap = np.ascontiguousarray(faces_pmap, dtype=np.int32)
    cdef np.ndarray[double] cpp_sizing = np.ascontiguousarray(sizing, dtype=np.float64)
    cdef VolumeMeshWithPMaps out

    out = tetrahedral_remeshing_remesh_protect_boundary(
        cpp_v,
        cpp_c,
        cpp_c_pmap,
        cpp_f,
        cpp_f_pmap,
        cpp_sizing,
        n_iterations,
        nb_neighbors_dim3,
        nb_neighbors_dim2,
        check_triangulation,
    )
    v = np.array(out.vertices, dtype=float)
    f = np.array(out.faces, dtype=int)
    t = np.array(out.cells, dtype=int)
    f_pmap = np.array(out.faces_pmap, dtype=int)
    t_pmap = np.array(out.cells_pmap, dtype=int)
    return v, f, t, f_pmap, t_pmap
