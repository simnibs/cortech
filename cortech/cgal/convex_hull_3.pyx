from libcpp.vector cimport vector
from libcpp.pair cimport pair
import numpy as np
import numpy.typing as npt
cimport numpy as np

cdef extern from "convex_hull_3_src.cpp" namespace "cortech":
    cdef cppclass SurfaceMesh:
        vector[vector[float]] vertices
        vector[vector[int]] faces

cdef extern from "convex_hull_3_src.cpp" nogil:
    SurfaceMesh convex_hull_3(
        vector[vector[float]] vertices,
    )

def convex_hull(vertices: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute the intersecting pairs of triangles in a surface mesh.

    Parameters
    ----------
    vertices : npt.ArrayLike

    faces : npt.ArrayLike

    Returns
    -------
    intersecting_pairs : npt

    """
    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef SurfaceMesh out

    out = convex_hull_3(cpp_v)

    return np.array(out.vertices, dtype=float), np.array(out.faces, dtype=int)
