from libcpp.vector cimport vector
import numpy as np
import numpy.typing as npt
cimport numpy as np


cdef extern from "alpha_wrap_3_src.cpp" namespace "cortech":
    cdef cppclass SurfaceMesh:
        vector[vector[float]] vertices
        vector[vector[int]] faces

cdef extern from "alpha_wrap_3_src.cpp" nogil:
    SurfaceMesh aw3_alpha_wrap_3_points(
        vector[vector[float]] vertices,
        double alpha,
        double offset
    )
    SurfaceMesh aw3_alpha_wrap_3_surface(
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
    cdef SurfaceMesh out

    out = aw3_alpha_wrap_3_points(cpp_v, alpha, offset)

    return np.array(out.vertices, dtype=float), np.array(out.faces, dtype=int)


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
    cdef SurfaceMesh out

    out = aw3_alpha_wrap_3_surface(cpp_v, cpp_f, alpha, offset)

    return np.array(out.vertices, dtype=float), np.array(out.faces, dtype=int)
