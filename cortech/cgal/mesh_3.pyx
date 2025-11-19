from libcpp cimport bool as cppbool
from libcpp.pair cimport pair
from libcpp.vector cimport vector
import numpy as np
import numpy.typing as npt
cimport numpy as np

cdef extern from "mesh_3_src.cpp":
    cdef cppclass V2FIIII:
        vector[vector[float]] vertices
        vector[vector[int]] faces
        vector[vector[int]] cells
        vector[int] faces_pmap
        vector[int] cells_pmap


cdef extern from "mesh_3_src.cpp" nogil:
    # V2FIIII mesh3_make_mesh(
    #     vector[vector[float]] vertices,
    #     vector[vector[int]] faces,
    #     float edge_size,
    #     float cell_radius_edge_ratio,
    #     float cell_size,
    #     float face_angle,
    #     float facet_distance,
    #     float facet_size
    # )

    # V2FIIII mesh3_make_mesh_bounding(
    #     vector[vector[float]] v_inside,
    #     vector[vector[int]] f_inside,
    #     vector[vector[float]] v_bounding,
    #     vector[vector[int]] f_bounding,
    #     float edge_size,
    #     float cell_radius_edge_ratio,
    #     float cell_size,
    #     float face_angle,
    #     float facet_distance,
    #     float facet_size
    # )

    V2FIIII mesh3_make_mesh_complex(
        vector[vector[vector[float]]] vertices,
        vector[vector[vector[int]]] faces,
        vector[pair[int,int]] incident_subdomains,
        float edge_size,
        float cell_radius_edge_ratio,
        float cell_size,
        float face_angle,
        float facet_distance,
        float facet_size
    )

# def make_mesh(
#     vertices: npt.ArrayLike,
#     faces: npt.ArrayLike,
#     edge_size: float,
#     cell_radius_edge_ratio: float,
#     cell_size: float,
#     facet_angle: float,
#     facet_distance: float,
#     facet_size: float
# ):
#     cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
#     cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
#     cdef V2FIIII out

#     out = mesh3_make_mesh(
#         cpp_v,
#         cpp_f,
#         edge_size,
#         cell_radius_edge_ratio,
#         cell_size,
#         facet_angle,
#         facet_distance,
#         facet_size
#     )
#     v = np.array(out.vertices, dtype=float)
#     f = np.array(out.faces, dtype=int)
#     t = np.array(out.cells, dtype=int)
#     f_pmap = np.array(out.faces_pmap, dtype=int)
#     t_pmap = np.array(out.cells_pmap, dtype=int)
#     return v, f, t, f_pmap, t_pmap

# def make_mesh_bounding(
#     v_inside: npt.ArrayLike,
#     f_inside: npt.ArrayLike,
#     v_bounding: npt.ArrayLike,
#     f_bounding: npt.ArrayLike,
#     edge_size: float,
#     cell_radius_edge_ratio: float,
#     cell_size: float,
#     facet_angle: float,
#     facet_distance: float,
#     facet_size: float
# ):
#     cdef np.ndarray[float, ndim=2] cpp_vi = np.ascontiguousarray(v_inside, dtype=np.float32)
#     cdef np.ndarray[int, ndim=2] cpp_fi = np.ascontiguousarray(f_inside, dtype=np.int32)
#     cdef np.ndarray[float, ndim=2] cpp_vb = np.ascontiguousarray(v_bounding, dtype=np.float32)
#     cdef np.ndarray[int, ndim=2] cpp_fb = np.ascontiguousarray(f_bounding, dtype=np.int32)
#     cdef V2FIIII out

#     out = mesh3_make_mesh_bounding(
#         cpp_vi,
#         cpp_fi,
#         cpp_vb,
#         cpp_fb,
#         edge_size,
#         cell_radius_edge_ratio,
#         cell_size,
#         facet_angle,
#         facet_distance,
#         facet_size
#     )
#     v = np.array(out.vertices, dtype=float)
#     f = np.array(out.faces, dtype=int)
#     t = np.array(out.cells, dtype=int)
#     f_pmap = np.array(out.faces_pmap, dtype=int)
#     t_pmap = np.array(out.cells_pmap, dtype=int)
#     return v, f, t, f_pmap, t_pmap


def make_mesh_complex(
    vertices: npt.ArrayLike,
    faces: npt.ArrayLike,
    incident_subdomains: list[tuple[int,int]],
    edge_size: float,
    cell_radius_edge_ratio: float,
    cell_size: float,
    facet_angle: float,
    facet_distance: float,
    facet_size: float
):
    cdef vector[vector[vector[float]]] cpp_v = [np.ascontiguousarray(v, dtype=np.float32) for v in vertices]
    cdef vector[vector[vector[int]]] cpp_f = [np.ascontiguousarray(f, dtype=np.int32) for f in faces]
    cdef vector[pair[int,int]] cpp_incident_subdomains = incident_subdomains
    cdef V2FIIII out

    out = mesh3_make_mesh_complex(
        cpp_v,
        cpp_f,
        cpp_incident_subdomains,
        edge_size,
        cell_radius_edge_ratio,
        cell_size,
        facet_angle,
        facet_distance,
        facet_size
    )
    v = np.array(out.vertices, dtype=float)
    f = np.array(out.faces, dtype=int)
    t = np.array(out.cells, dtype=int)
    f_pmap = np.array(out.faces_pmap, dtype=int)
    t_pmap = np.array(out.cells_pmap, dtype=int)
    return v, f, t, f_pmap, t_pmap

