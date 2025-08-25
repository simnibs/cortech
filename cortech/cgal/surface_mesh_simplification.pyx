from libcpp cimport bool as cppbool
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from typing import Union
import numpy as np
import numpy.typing as npt
cimport numpy as np


cdef extern from "surface_mesh_simplification_src.cpp" nogil:
    pair[vector[vector[float]], vector[vector[int]]] sms_simplify(
        vector[vector[float]] vertices,
        vector[vector[int]] faces,
        int stop_face_count,
    )

def simplify(vertices, faces, stop_face_count):
    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    # we add one because the algorithm terminates when the number of faces
    # drop *below* `stop_face_count`
    cdef int cpp_stop = stop_face_count + 1

    cdef pair[vector[vector[float]], vector[vector[int]]] out

    out = sms_simplify(cpp_v, cpp_f, cpp_stop)

    v = np.array(out.first, dtype=float)
    f = np.array(out.second, dtype=int)

    return v,f