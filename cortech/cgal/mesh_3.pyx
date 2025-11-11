from libcpp cimport bool as cppbool
from libcpp.pair cimport pair
from libcpp.vector cimport vector
import numpy as np
import numpy.typing as npt
cimport numpy as np

cdef extern from "mesh_3_src.cpp":
    cdef cppclass V2FII:
        vector[vector[float]] v0
        vector[vector[int]] v1
        vector[vector[int]] v2


cdef extern from "mesh_3_src.cpp" nogil:
    V2FII mesh3_make_mesh(
        vector[vector[float]] vertices,
        vector[vector[int]] faces,
    )

def make_mesh(vertices: npt.ArrayLike, faces: npt.ArrayLike):
    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef V2FII out

    out = mesh3_make_mesh(cpp_v, cpp_f)
    v = np.array(out.v0, dtype=float)
    f = np.array(out.v1, dtype=int)
    t = np.array(out.v2, dtype=int)
    return v, f, t
