from libcpp cimport bool as cppbool
from libcpp.vector cimport vector
# from libcpp.pair cimport pair
import numpy as np
import numpy.typing as npt
cimport numpy as np


cdef extern from "aabb_tree_src.cpp" nogil:
    vector[double] aabb_distance(
        vector[vector[double]] vertices,
        vector[vector[int]] faces,
        vector[vector[double]] query_points,
        vector[vector[double]] query_hints,
        cppbool accelerate_distance_queries,
    )
    # pair[vector[vector[double]], vector[int]] aabb_closest_point_and_primitive(
    #     vector[vector[double]] vertices,
    #     vector[vector[int]] faces,
    #     vector[vector[double]] query_points,
    #     vector[vector[double]] query_hints,
    #     cppbool accelerate_distance_queries,
    # )


def distance(
        vertices: npt.NDArray,
        faces: npt.NDArray,
        query_points: npt.NDArray,
        query_hints: npt.NDArray | None = None,
        accelerate_distance_queries: bool = True,
    ) -> npt.NDArray:
    """

    """
    cdef np.ndarray[double, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float64)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef np.ndarray[double, ndim=2] cpp_qp = np.ascontiguousarray(query_points, dtype=np.float64)
    query_hints = np.array([[]], dtype=np.float64) if query_hints is None else query_hints
    cdef np.ndarray[double, ndim=2] cpp_qhp = np.ascontiguousarray(query_hints, dtype=np.float64)
    cdef cppbool cpp_accelerate_distance_queries = accelerate_distance_queries
    cdef vector[double] out

    out = aabb_distance(cpp_v, cpp_f, cpp_qp, cpp_qhp, cpp_accelerate_distance_queries)

    return np.array(out, dtype=query_points.dtype)


# def closest_point_and_primitive(
#         vertices: npt.NDArray,
#         faces: npt.NDArray,
#         query_points: npt.NDArray,
#         query_hints: npt.NDArray | None = None,
#         accelerate_distance_queries: bool = True,
#     ) -> npt.NDArray:
#     """

#     """
#     cdef np.ndarray[double, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=float)
#     cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
#     cdef np.ndarray[double, ndim=2] cpp_qp = np.ascontiguousarray(query_points, dtype=float)
#     query_hints = np.array([[]], dtype=float) if query_hints is None else query_hints
#     cdef np.ndarray[double, ndim=2] cpp_qh = np.ascontiguousarray(query_hints, dtype=float)
#     cdef cppbool cpp_accelerate_distance_queries = accelerate_distance_queries
#     cdef pair[vector[vector[double]],vector[int]] out

#     out = aabb_closest_point_and_primitive(cpp_v, cpp_f, cpp_qp, cpp_qh, cpp_accelerate_distance_queries)

#     closest_point = np.array(out.first, dtype=float)
#     closest_primitive = np.array(out.second, dtype=int)

#     return closest_point, closest_primitive
