from libcpp cimport bool as cppbool
# from libcpp.pair cimport pair
from libcpp.vector cimport vector
import numpy as np
import numpy.typing as npt
cimport numpy as np

ctypedef vector[vector[float]] PointVector
ctypedef vector[vector[int]] IndexVector

cdef extern from "boost_graph_src.cpp" namespace "cortech":
    cdef cppclass SurfaceMesh:
        vector[vector[float]] vertices
        vector[vector[int]] faces

cdef extern from "boost_graph_src.cpp":
    # vector[int] graph_expand_face_selection(
    #     PointVector vertices,
    #     IndexVector faces,
    #     vector[int] selection,
    #     unsigned int k
    # ) except +

    vector[cppbool] graph_expand_face_selection_for_removal(
        PointVector vertices,
        IndexVector faces,
        vector[int] selection,
    ) except +

    # vector[int] graph_expand_vertex_selection(
    #     PointVector vertices,
    #     IndexVector faces,
    #     vector[int] selection,
    #     unsigned int k
    # ) except +
    IndexVector graph_find_border_edges(
        PointVector vertices, IndexVector faces
    ) except +

# def expand_face_selection(vertices, faces, selection, k):
#     assert k > 0
#     cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
#     cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
#     cdef np.ndarray[int, ndim=1] cpp_sel = np.ascontiguousarray(selection, dtype=np.int32)
#     cdef unsigned int cpp_k = k
#     cdef vector[int] out
#     out = graph_expand_face_selection(cpp_v, cpp_f, cpp_sel, cpp_k)
#     return np.array(out, dtype=int)

def expand_face_selection_for_removal(vertices, faces, selection):
    assert np.issubdtype(selection.dtype, np.integer)

    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] cpp_sel = np.ascontiguousarray(selection, dtype=np.int32)
    cdef vector[cppbool] out
    out = graph_expand_face_selection_for_removal(cpp_v, cpp_f, cpp_sel)
    return np.array(out, dtype=bool)

# def expand_vertex_selection(vertices, faces, selection, k):
#     assert k > 0
#     cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
#     cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
#     cdef np.ndarray[int, ndim=1] cpp_sel = np.ascontiguousarray(selection, dtype=np.int32)
#     cdef unsigned int cpp_k = k
#     cdef vector[int] out
#     out = graph_expand_vertex_selection(cpp_v, cpp_f, cpp_sel, cpp_k)
#     return np.array(out, dtype=int)

def find_border_edges(vertices: npt.NDArray, faces: npt.NDArray) -> npt.NDArray:
    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef IndexVector out
    out = graph_find_border_edges(cpp_v, cpp_f)
    return np.array(out, dtype=int)