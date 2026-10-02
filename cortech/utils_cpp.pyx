from libcpp.vector cimport vector
import numpy as np
import numpy.typing as npt
cimport numpy as np

cdef extern from "utils_cpp_src.cpp":
    vector[int] map_values_from_source_to_target(vector[int],vector[int],vector[int]) except +
    vector[float] sizing_field_gradation(
        vector[vector[float]], vector[vector[unsigned int]], vector[float], float
    ) except +
def map_values(source: npt.ArrayLike, target: npt.ArrayLike, values: npt.ArrayLike):

    cdef np.ndarray[int] source_ = np.ascontiguousarray(source, dtype=np.int32)
    cdef np.ndarray[int] target_ = np.ascontiguousarray(target, dtype=np.int32)
    cdef np.ndarray[int] values_ = np.ascontiguousarray(values, dtype=np.int32)
    out = map_values_from_source_to_target(source_, target_, values_)
    return np.array(out, dtype=int)

def gradate_sizing_field(vertices, cells, h, g):
    assert vertices.shape[1] == 3
    assert cells.shape[1] == 4
    assert len(vertices) == len(h)
    assert g > 0.0

    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[unsigned int, ndim=2] cpp_f = np.ascontiguousarray(cells, dtype=np.uint32)
    cdef np.ndarray[float] cpp_h = np.ascontiguousarray(h, dtype=np.float32)
    cdef vector[float] out

    out = sizing_field_gradation(cpp_v, cpp_f, cpp_h, g)

    return np.array(out, dtype=float)