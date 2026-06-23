from libcpp.vector cimport vector
import numpy as np
import numpy.typing as npt
cimport numpy as np


cdef extern from "constrained_triangulations_src.cpp":
    cdef cppclass MeshGeometry:
        vector[vector[float]] vertices
        # vector[vector[int]] faces
        vector[vector[int]] cells

cdef extern from "constrained_triangulations_src.cpp" nogil:
    MeshGeometry constrained_triangulation_make_ccdt(
        vector[vector[float]] vertices,
        vector[vector[int]] faces
    )

def make_ccdt(vertices: npt.ArrayLike, faces: npt.ArrayLike):
    cdef vector[vector[float]] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef vector[vector[int]] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef MeshGeometry out

    out = constrained_triangulation_make_ccdt(cpp_v, cpp_f)
    v = np.array(out.vertices, dtype=float)
    # f = np.array(out.faces, dtype=int)
    t = np.array(out.cells, dtype=int)
    return v, t
    # return v, f, t
