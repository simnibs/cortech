from libcpp.vector cimport vector
import numpy as np
import numpy.typing as npt
cimport numpy as np

ctypedef vector[vector[float]] PointVector
ctypedef vector[vector[int]] IndexVector

cdef extern from "surface_mesh_segmentation_src.cpp" nogil:
    vector[float] ss_signed_distance_function(
        PointVector vertices,
        IndexVector faces,
        double cone_angle,
        unsigned int number_of_rays
    ) except +


def signed_distance_function(
    vertices,
    faces,
    cone_angle: float = 2.0943951023931953,
    number_of_rays: int = 25
):
    """

    cone_angle
        default = 2.0/3.0*pi
    """
    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef vector[float] out
    out = ss_signed_distance_function(cpp_v, cpp_f, cone_angle, number_of_rays)
    return np.array(out, dtype=float)