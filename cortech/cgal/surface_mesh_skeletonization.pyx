from libcpp.vector cimport vector
import numpy as np
import numpy.typing as npt
cimport numpy as np


cdef extern from "surface_mesh_skeletonization_src.cpp" namespace "cortech":
    cdef cppclass SurfaceMesh:
        vector[vector[float]] vertices
        vector[vector[int]] faces

cdef extern from "surface_mesh_skeletonization_src.cpp" nogil:
    SurfaceMesh smskel_skeletonize(
        vector[vector[float]] vertices,
        vector[vector[int]] faces,
    )

def skeletonize(vertices: npt.ArrayLike, faces: npt.ArrayLike):
    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef SurfaceMesh out

    out = smskel_skeletonize(cpp_v, cpp_f)
    return np.array(out.vertices, float), np.array(out.faces, int)