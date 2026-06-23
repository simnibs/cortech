from libcpp cimport bool as cppbool
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from typing import Union
import numpy as np
import numpy.typing as npt
cimport numpy as np


cdef extern from "alpha_wrap_3_src.cpp" nogil:
    pair[vector[vector[float]], vector[vector[int]]] aw3_alpha_wrap_3_points(
        vector[vector[float]] vertices,
        double alpha,
        double offset
    )
    pair[vector[vector[float]], vector[vector[int]]] aw3_alpha_wrap_3_surface(
        vector[vector[float]] vertices,
        vector[vector[int]] faces,
        double alpha,
        double offset
    )

def alpha_wrap_3_points(vertices: npt.NDArray, alpha: float, offset: float) -> tuple[npt.NDArray, npt.NDArray]:
    """Wraps a point cloud

    Parameters
    ----------
    vertices : npt.ArrayLike

    Returns
    -------

    """
    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef pair[vector[vector[float]], vector[vector[int]]] out

    out = aw3_alpha_wrap_3_points(cpp_v, alpha, offset)


    v = np.array(out.first, dtype=float)
    f = np.array(out.second, dtype=int)

    return v, f


def alpha_wrap_3_surface(
    vertices: npt.NDArray,
    faces: npt.NDArray,
    alpha: float,
    offset: float,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Wraps a point cloud

    Parameters
    ----------
    vertices : npt.ArrayLike

    Returns
    -------

    """
    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef pair[vector[vector[float]], vector[vector[int]]] out

    out = aw3_alpha_wrap_3_surface(cpp_v, cpp_f, alpha, offset)

    v = np.array(out.first, dtype=float)
    f = np.array(out.second, dtype=int)

    return v, f
