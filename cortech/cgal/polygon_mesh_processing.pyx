from libcpp cimport bool as cppbool
from libcpp.pair cimport pair
from libcpp.vector cimport vector
import numpy as np
import numpy.typing as npt
cimport numpy as np

ctypedef vector[vector[float]] PointVector
ctypedef vector[vector[int]] IndexVector

cdef extern from "polygon_mesh_processing_src.cpp" namespace "cortech":
    cdef cppclass SurfaceMesh:
        vector[vector[float]] vertices
        vector[vector[int]] faces

    cdef cppclass SurfaceMeshWithFaceid:
        PointVector vertices
        IndexVector faces
        vector[int] face_id

    cdef cppclass SurfaceMeshWithPMaps:
        PointVector vertices
        IndexVector faces
        vector[int] vertices_pmap
        vector[int] faces_pmap

    cdef cppclass SurfaceMeshWithFaceidAndPMaps:
        PointVector vertices
        IndexVector faces
        vector[int] face_id
        vector[int] vertices_pmap
        vector[int] faces_pmap

cdef extern from "polygon_mesh_processing_src.cpp" nogil:
    SurfaceMesh pmp_autorefine_triangle_soup(
        PointVector vertices,
        IndexVector faces,
        cppbool apply_iterative_snap_rounding,
        unsigned int n_iter
    ) except +

    SurfaceMesh pmp_clip(
        PointVector vertices,
        IndexVector faces,
        vector[float] plane_origin,
        vector[float] plane_direction,
        # cppbool clip_volume
    ) except +
    pair[vector[int], vector[int]] pmp_connected_components(
        PointVector vertices,
        IndexVector faces,
    ) except +
    pair[vector[int], vector[int]] pmp_connected_components(
        PointVector vertices,
        IndexVector faces,
        IndexVector constrained_edges,
    ) except +
    SurfaceMesh pmp_duplicate_non_manifold_edges_in_polygon_soup(
        PointVector vertices, IndexVector faces
    ) except +

    SurfaceMesh pmp_duplicate_non_manifold_vertices(
        PointVector vertices, IndexVector faces
    ) except +
    pair[pair[SurfaceMeshWithPMaps,SurfaceMeshWithPMaps], pair[vector[vector[int]], vector[vector[int]]]] pmp_corefine(
        PointVector v0, IndexVector f0, PointVector v1, IndexVector f1, cppbool return_intersection_edges
    ) except +
    cppbool pmp_does_triangle_soup_self_intersect(PointVector vertices, IndexVector faces) except +
    IndexVector pmp_extract_boundary_cycles(
        PointVector vertices, IndexVector faces
    ) except +
    PointVector pmp_fair(
        PointVector vertices, IndexVector faces,vector[int] indices, int continuity
    ) except +

    SurfaceMeshWithFaceidAndPMaps pmp_flip_edges(
        PointVector vertices,
        IndexVector faces,
        IndexVector edges,
        vector[int] face_id,
    ) except +


    SurfaceMesh pmp_hole_fill_refine_fair(
        PointVector vertices, IndexVector faces
    ) except +
    IndexVector pmp_intersecting_meshes(
        PointVector vertices0,
        IndexVector faces0,
        PointVector vertices1,
        IndexVector faces1,
    ) except +
    cppbool pmp_is_polygon_soup_a_polygon_mesh(IndexVector faces) except +

    SurfaceMeshWithFaceidAndPMaps pmp_isotropic_remeshing(
        PointVector vertices,
        IndexVector faces,
        double target_edge_length,
        int n_iter,
        cppbool protect_constraints,
        cppbool collapse_constraints,
        cppbool do_split,
        cppbool do_collapse,
        cppbool do_flip,
        int number_of_relaxation_steps,
        cppbool disallow_surface_crossing,
        vector[int] face_id,
        vector[int] face_is_selected,
        vector[int] vertex_is_constrained,
    ) except +

    SurfaceMeshWithFaceidAndPMaps pmp_isotropic_remeshing(
        PointVector vertices,
        IndexVector faces,
        double target_edge_length,
        int n_iter,
        cppbool protect_constraints,
        cppbool collapse_constraints,
        cppbool do_split,
        cppbool do_collapse,
        cppbool do_flip,
        int number_of_relaxation_steps,
        cppbool disallow_surface_crossing,
        vector[int] face_id,
        vector[int] face_is_selected,
        vector[int] vertex_is_constrained,
        vector[vector[int]] edge_is_constrained,
    ) except +

    SurfaceMeshWithFaceidAndPMaps pmp_collapse_halfedges(
        PointVector vertices,
        IndexVector faces,
        IndexVector edges,
        vector[int] face_id,
    ) except +

    SurfaceMeshWithFaceidAndPMaps pmp_collapse_short_edges(
        PointVector vertices,
        IndexVector faces,
        double target_edge_length,
        vector[int] face_id,
        vector[int] face_is_selected,
        vector[int] vertex_is_constrained,
    ) except +

    SurfaceMeshWithFaceidAndPMaps pmp_collapse_short_edges(
        PointVector vertices,
        IndexVector faces,
        double target_edge_length,
        vector[int] face_id,
        vector[int] face_is_selected,
        vector[int] vertex_is_constrained,
        vector[vector[int]] edge_is_constrained,
    ) except +
    SurfaceMeshWithFaceidAndPMaps pmp_adaptive_remeshing(
        PointVector vertices,
        IndexVector faces,
        double error_tol,
        double edge_length_min,
        double edge_length_max,
        int n_iter,
        cppbool protect_constraints,
        cppbool collapse_constraints,
        cppbool do_split,
        cppbool do_collapse,
        cppbool do_flip,
        int number_of_relaxation_steps,
        cppbool disallow_surface_crossing,
        vector[int] face_id,
        vector[int] face_is_selected,
        vector[int] vertex_is_constrained,
    ) except +
    SurfaceMeshWithFaceidAndPMaps pmp_adaptive_remeshing(
        PointVector vertices,
        IndexVector faces,
        double error_tol,
        double edge_length_min,
        double edge_length_max,
        int n_iter,
        cppbool protect_constraints,
        cppbool collapse_constraints,
        cppbool do_split,
        cppbool do_collapse,
        cppbool do_flip,
        int number_of_relaxation_steps,
        cppbool disallow_surface_crossing,
        vector[int] face_id,
        vector[int] face_is_selected,
        vector[int] vertex_is_constrained,
        vector[vector[int]] edge_is_constrained,
    ) except +
    SurfaceMeshWithFaceidAndPMaps pmp_custom_remeshing(
        PointVector vertices,
        IndexVector faces,
        vector[float] sizing,
        int n_iter,
        cppbool protect_constraints,
        cppbool collapse_constraints,
        cppbool do_split,
        cppbool do_collapse,
        cppbool do_flip,
        int number_of_relaxation_steps,
        cppbool disallow_surface_crossing,
        vector[int] face_id,
        vector[int] face_is_selected,
        vector[int] vertex_is_constrained,
    ) except +
    SurfaceMeshWithFaceidAndPMaps pmp_custom_remeshing(
        PointVector vertices,
        IndexVector faces,
        vector[float] sizing,
        int n_iter,
        cppbool protect_constraints,
        cppbool collapse_constraints,
        cppbool do_split,
        cppbool do_collapse,
        cppbool do_flip,
        int number_of_relaxation_steps,
        cppbool disallow_surface_crossing,
        vector[int] face_id,
        vector[int] face_is_selected,
        vector[int] vertex_is_constrained,
        vector[vector[int]] edge_is_constrained,
    ) except +

    SurfaceMesh pmp_merge_duplicate_points_in_polygon_soup(
        PointVector vertices, IndexVector faces
    ) except +
    SurfaceMesh pmp_merge_duplicate_polygons_in_polygon_soup(
        PointVector vertices, IndexVector faces
    ) except +
    SurfaceMesh pmp_orient(
        PointVector vertices, IndexVector faces, cppbool outward_orientation
    ) except +
    pair[cppbool, SurfaceMesh] pmp_orient_polygon_soup(
        PointVector vertices, IndexVector faces
    ) except +

    vector[int] pmp_non_manifold_vertices(
        PointVector vertices, IndexVector faces
    ) except +

    SurfaceMeshWithPMaps pmp_stitch_borders(PointVector vertices, IndexVector faces) except +
    # SurfaceMesh pmp_snap_borders(
    #     PointVector vertices, IndexVector faces,
    # )
    vector[cppbool] pmp_points_inside(
        PointVector vertices,
        IndexVector faces,
        PointVector points,
        cppbool on_boundary_is_inside,
    ) except +

    # SurfaceMesh pmp_repair_mesh(
    #     PointVector vertices,
    #     IndexVector faces,
    # ) except +

    SurfaceMesh pmp_refine(
        PointVector vertices,
        IndexVector faces,
        vector[int] faces_to_refine,
        float density
    ) except +
    SurfaceMeshWithPMaps pmp_remove_almost_degenerate_faces(
        PointVector vertices,
        IndexVector faces,
        vector[int] face_is_selected,
        double cap_threshold,
        double needle_threshold,
        vector[int] vertex_is_constrained,
    ) except +
    SurfaceMeshWithPMaps pmp_remove_almost_degenerate_faces(
        PointVector vertices,
        IndexVector faces,
        vector[int] face_is_selected,
        double cap_threshold,
        double needle_threshold,
        vector[int] vertex_is_constrained,
        vector[vector[int]] edge_is_constrained,
    ) except +
    SurfaceMeshWithPMaps pmp_remove_self_intersections(
        PointVector vertices,
        IndexVector faces,
        vector[int] face_is_selected
    ) except +
    IndexVector pmp_self_intersections(PointVector vertices, IndexVector faces) except +

    # pair[vector[int], vector[int]] pmp_volume_connected_components(
    #     IndexVector faces,
    #     cppbool do_orientation_tests,
    #     cppbool do_self_intersection_tests,
    # ) except +

    PointVector pmp_smooth_shape(
        PointVector vertices,
        IndexVector faces,
        vector[int] constrained_vertices,
        double time,
        int n_iter
    ) except +

    PointVector pmp_smooth_shape_by_curvature_threshold(
        PointVector vertices,
        IndexVector faces,
        double time,
        int n_iter,
        double curv_threshold,
        cppbool apply_above_curv_threshold,
        double ball_radius,
    ) except +

    PointVector pmp_tangential_relaxation(
        PointVector vertices,
        IndexVector faces,
        vector[int] constrained_vertices,
        int n_iter
    ) except +

    vector[vector[float]] pmp_interpolated_corrected_curvatures(
        PointVector vertices,
        IndexVector faces,
    ) except +

    PointVector pmp_smooth_angle_and_area(
        PointVector vertices,
        IndexVector faces,
        vector[int] constrained_vertices,
        int niter,
        cppbool use_angle_smoothing,
        cppbool use_area_smoothing,
        cppbool use_delaunay_flips,
        cppbool use_safety_constraints
    ) except +

    SurfaceMeshWithFaceidAndPMaps pmp_split_edges(
        PointVector vertices,
        IndexVector faces,
        IndexVector edges,
        vector[int] face_id,
    ) except +

    SurfaceMeshWithFaceidAndPMaps pmp_split_long_edges(
        PointVector vertices,
        IndexVector faces,
        double sizing,
        vector[int] face_id,
    ) except +
    SurfaceMeshWithFaceidAndPMaps pmp_split_long_edges(
        PointVector vertices,
        IndexVector faces,
        double sizing,
        vector[int] face_id,
        IndexVector edges,
    ) except +

    SurfaceMesh pmp_split_with_plane(
        PointVector vertices,
        IndexVector faces,
        vector[float] plane_origin,
        vector[float] plane_direction,
    ) except +
    pair[SurfaceMesh,SurfaceMesh] pmp_split_with_surface(
        PointVector mesh_v,
        IndexVector mesh_f,
        PointVector splitter_v,
        IndexVector splitter_f
    ) except +

cdef _from_SurfaceMesh(SurfaceMesh out):
    return np.array(out.vertices, float), np.array(out.faces, int)

cdef _from_SurfaceMeshWithFaceid(SurfaceMeshWithFaceid out):
    v = np.array(out.vertices, float)
    f = np.array(out.faces, int)
    fid = np.array(out.face_id, int)
    return v, f, fid


cdef _from_SurfaceMeshWithPMaps(SurfaceMeshWithPMaps out):
    v = np.array(out.vertices, float)
    f = np.array(out.faces, int)
    v_pmap = np.array(out.vertices_pmap, int)
    f_pmap = np.array(out.faces_pmap, int)
    return v, f, v_pmap, f_pmap


cdef _from_SurfaceMeshWithFaceidAndPMaps(SurfaceMeshWithFaceidAndPMaps out):
    v = np.array(out.vertices, float)
    f = np.array(out.faces, int)
    fid = np.array(out.face_id, int)
    v_pmap = np.array(out.vertices_pmap, int)
    f_pmap = np.array(out.faces_pmap, int)
    return v, f, fid, v_pmap, f_pmap


# def snap_borders(vertices: npt.NDArray, faces: npt.NDArray):
#     cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
#     cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
#     cdef SurfaceMesh out

#     out = pmp_snap_borders(cpp_v, cpp_f)
#     v = np.array(out.first, dtype=float)
#     f = np.array(out.second, dtype=int)
#     return v, f


# def autorefine_triangle_soup(
#     vertices: npt.NDArray,
#     faces: npt.NDArray,
#     apply_iterative_snap_rounding: bool = True,
#     n_iter: int = 5
# ):
#     cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
#     cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
#     cdef cppbool cpp_snap = apply_iterative_snap_rounding
#     cdef unsigned int cpp_n_iter = <unsigned int>n_iter
#     cdef SurfaceMesh out
#     out = pmp_autorefine_triangle_soup(cpp_v, cpp_f, cpp_snap, cpp_n_iter)
#     return _from_SurfaceMesh(out)

def clip(
        vertices: npt.NDArray,
        faces: npt.NDArray,
        plane_origin: npt.NDArray,
        plane_direction: npt.NDArray,
        # clip_volume: bool = True,
    ) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute the intersecting pairs of triangles in a surface mesh.

    Parameters
    ----------
    vertices : npt.ArrayLike
    faces : npt.ArrayLike

    Returns
    -------
    intersecting_pairs : npt

    """
    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef np.ndarray[float] cpp_orig = np.ascontiguousarray(plane_origin, dtype=np.float32)
    cdef np.ndarray[float] cpp_dir = np.ascontiguousarray(plane_direction, dtype=np.float32)
    # cdef cppbool cpp_clip_volume = clip_volume
    cdef SurfaceMesh out

    out = pmp_clip(cpp_v, cpp_f, cpp_orig, cpp_dir)
    return _from_SurfaceMesh(out)

def connected_components(
    vertices: npt.ArrayLike,
    faces: npt.ArrayLike,
    constrained_edges: npt.ArrayLike | None = None
) -> tuple[npt.NDArray, npt.NDArray]:
    """Label connected components on a surface (graph).

    Parameters
    ----------
    vertices: npt.ArrayLike
        Vertices of the surfaces, shape = (N, 3).
    faces: npt.ArrayLike
        Faces of the surface, shape = (M, 3).
    constrained_edges: Union[npt.ArrayLike, None]


        The relevant function in CGAL takes a set of *edges* to constrain,
        however, this interface allows specifying *faces* instead such that
        only the *outer* edges of these faces are used as constraints. By
        default, no faces (edges) are constrained.

    Returns
    -------
    component_label : npt.NDArray
        The label associated with each face.
    component_size : npt.NDArray
        The size associated with each label.
    """
    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef np.ndarray[int, ndim=2] cpp_ecm

    cdef pair[vector[int], vector[int]] out

    if constrained_edges is None:
        out = pmp_connected_components(cpp_v, cpp_f)
    else:
        cpp_ecm = np.ascontiguousarray(constrained_edges, dtype=np.int32)
        out = pmp_connected_components(cpp_v, cpp_f, cpp_ecm)

    component_label = np.array(out.first, dtype=int)
    component_size = np.array(out.second, dtype=int)

    return component_label, component_size

def corefine(vertices0, faces0, vertices1, faces1, return_intersection_edges: bool):
    cdef np.ndarray[float, ndim=2] cpp_v0 = np.ascontiguousarray(vertices0, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f0 = np.ascontiguousarray(faces0, dtype=np.int32)
    cdef np.ndarray[float, ndim=2] cpp_v1 = np.ascontiguousarray(vertices1, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f1 = np.ascontiguousarray(faces1, dtype=np.int32)
    cdef pair[pair[SurfaceMeshWithPMaps,SurfaceMeshWithPMaps], pair[vector[vector[int]], vector[vector[int]]]] out
    out = pmp_corefine(cpp_v0, cpp_f0, cpp_v1, cpp_f1, return_intersection_edges)
    #return _from_SurfaceMesh(out.first), _from_SurfaceMesh(out.second)
    vf0 = _from_SurfaceMeshWithPMaps(out.first.first)
    vf1 = _from_SurfaceMeshWithPMaps(out.first.second)
    e0 = np.array(out.second.first, dtype=int)
    e1 = np.array(out.second.second, dtype=int)
    return vf0, vf1, e0, e1

def does_triangle_soup_self_intersect(vertices: npt.NDArray, faces: npt.NDArray) -> bool:
    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    return pmp_does_triangle_soup_self_intersect(cpp_v, cpp_f)

def duplicate_non_manifold_edges_in_polygon_soup(vertices,faces):
    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef SurfaceMesh out
    out = pmp_duplicate_non_manifold_edges_in_polygon_soup(cpp_v, cpp_f)
    return _from_SurfaceMesh(out)

def duplicate_non_manifold_vertices(vertices, faces):
    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef SurfaceMesh out
    out = pmp_duplicate_non_manifold_vertices(cpp_v, cpp_f)
    return _from_SurfaceMesh(out)

def extract_boundary_cycles(vertices: npt.NDArray, faces: npt.NDArray):
    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef IndexVector out
    out = pmp_extract_boundary_cycles(cpp_v, cpp_f)
    return out

def fair(vertices, faces, vertex_indices, continuity: int = 1):
    """Mesh fairing."""
    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef np.ndarray[int] cpp_vi = np.ascontiguousarray(vertex_indices, dtype=np.int32)
    cdef PointVector v
    assert 2 >= continuity >= 0, "continuity should be in {0,1,2}"

    v = pmp_fair(cpp_v, cpp_f, cpp_vi, continuity)

    return np.array(v, dtype=float)

def flip_edges(
    vertices: npt.ArrayLike,
    faces: npt.ArrayLike,
    edges: npt.ArrayLike,
    face_id: npt.ArrayLike | None = None,
):
    """Flip edges.

    Parameters
    ----------
    vertices: npt.ArrayLike
    faces: npt.ArrayLike

    Returns
    -------
    v : npt.NDArray
        The new vertices.
    f : npt.NDArray
        The new faces.

    References
    ----------

    https://doc.cgal.org/latest/Polygon_mesh_processing/group__PMP__meshing__grp.html#gaa5cc92275df27f0baab2472ecbc4ea3f

    """
    face_id = [] if face_id is None else face_id

    cdef np.ndarray[float, ndim=2] cv = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cf = np.ascontiguousarray(faces, dtype=np.int32)
    cdef np.ndarray[int, ndim=2] ce = np.ascontiguousarray(edges, dtype=np.int32)
    cdef np.ndarray[int] cpp_fid = np.ascontiguousarray(face_id, dtype=np.int32)
    cdef SurfaceMeshWithFaceidAndPMaps out

    out = pmp_flip_edges(cv, cf, ce, cpp_fid)

    return _from_SurfaceMeshWithFaceidAndPMaps(out)

def hole_fill_refine_fair(vertices: npt.NDArray, faces: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute the intersecting pairs of triangles in a surface mesh.

    Parameters
    ----------
    vertices : npt.ArrayLike
    faces : npt.ArrayLike

    Returns
    -------

    """
    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef SurfaceMesh out
    out = pmp_hole_fill_refine_fair(cpp_v, cpp_f)
    return _from_SurfaceMesh(out)

def interpolated_corrected_curvatures(vertices: npt.ArrayLike, faces: npt.ArrayLike):
    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef vector[vector[float]] curv

    curv = pmp_interpolated_corrected_curvatures(cpp_v, cpp_f)

    curv_arr = np.array(curv, dtype=float)
    k1 = np.ascontiguousarray(curv_arr[:,0])
    k2 = np.ascontiguousarray(curv_arr[:,1])
    H = np.ascontiguousarray(curv_arr[:,2])
    K = np.ascontiguousarray(curv_arr[:,3])
    k1_vec = np.ascontiguousarray(curv_arr[:,4:7])
    k2_vec = np.ascontiguousarray(curv_arr[:, 7:])

    return k1, k2, H, K, k1_vec, k2_vec

def intersecting_meshes(
    vertices0: npt.ArrayLike,
    faces0: npt.ArrayLike,
    vertices1: npt.ArrayLike,
    faces1: npt.ArrayLike
) -> npt.NDArray:
    """Compute the intersecting pairs of triangles in a surface mesh.

    Parameters
    ----------
    vertices0 : npt.ArrayLike
    faces0 : npt.ArrayLike
    vertices1 : npt.ArrayLike
    faces1 : npt.ArrayLike

    Returns
    -------
    intersecting_pairs : npt
        intersecting_pairs[:,0] contains the face indices for the first surface.
        intersecting_pairs[:,1] contains the face indices for the second surface.

    """
    cdef np.ndarray[float, ndim=2] cpp_v0 = np.ascontiguousarray(vertices0, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f0 = np.ascontiguousarray(faces0, dtype=np.int32)
    cdef np.ndarray[float, ndim=2] cpp_v1 = np.ascontiguousarray(vertices1, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f1 = np.ascontiguousarray(faces1, dtype=np.int32)

    cdef IndexVector intersecting_pairs # list of lists

    intersecting_pairs = pmp_intersecting_meshes(cpp_v0, cpp_f0, cpp_v1, cpp_f1)

    return np.array(intersecting_pairs, dtype=int)

def is_polygon_soup_a_polygon_mesh(faces: npt.ArrayLike) -> bool:
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    return pmp_is_polygon_soup_a_polygon_mesh(cpp_f)

def adaptive_remeshing(
    vertices: npt.ArrayLike,
    faces: npt.ArrayLike,
    error_tol: float,
    edge_length_min: float,
    edge_length_max: float,
    face_id: npt.ArrayLike | None = None,
    face_is_selected: npt.ArrayLike | None = None,
    vertex_is_constrained: npt.ArrayLike | None = None,
    edge_is_constrained: npt.ArrayLike | None = None,
    n_iter: int = 1,
    protect_constraints: bool = False,
    collapse_constraints: bool = True,
    do_split: bool = True,
    do_collapse: bool = True,
    do_flip: bool = True,
    number_of_relaxation_steps: int = 1,
    disallow_surface_crossing=False,
):
    """Isotropic surface remeshing. Remeshing is achieved by a combination of
    edge splits/flips/collapses, tangential relaxation, and projection back
    onto the original surface.

    Parameters
    ----------
    vertices: npt.ArrayLike
    faces: npt.ArrayLike
    target_edge_length: float
        The target edge length for the isotropic remesher. This defines the
        resolution of the resulting surface.
    n_iter: int
        Number of iterations of the above-mentioned atomic operations.

    Returns
    -------
    v : npt.NDArray
        The new vertices.
    f : npt.NDArray
        The new faces.

    References
    ----------

    https://doc.cgal.org/latest/Polygon_mesh_processing/group__PMP__meshing__grp.html#gaa5cc92275df27f0baab2472ecbc4ea3f

    """
    face_id = [] if face_id is None else face_id
    face_is_selected = [] if face_is_selected is None else face_is_selected
    vertex_is_constrained = [] if vertex_is_constrained is None else vertex_is_constrained

    cdef np.ndarray[float, ndim=2] cv = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cf = np.ascontiguousarray(faces, dtype=np.int32)
    cdef np.ndarray[int] cpp_faces_sel = np.ascontiguousarray(face_is_selected, dtype=np.int32)
    cdef np.ndarray[int] cpp_fid = np.ascontiguousarray(face_id, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] cpp_vcm = np.ascontiguousarray(vertex_is_constrained, dtype=np.int32)
    cdef np.ndarray[int, ndim=2] cpp_ecm
    cdef SurfaceMeshWithFaceidAndPMaps out

    if edge_is_constrained is None:
        out = pmp_adaptive_remeshing(
            cv,
            cf,
            error_tol,
            edge_length_min,
            edge_length_max,
            n_iter,
            protect_constraints,
            collapse_constraints,
            do_split,
            do_collapse,
            do_flip,
            number_of_relaxation_steps,
            disallow_surface_crossing,
            cpp_fid,
            cpp_faces_sel,
            cpp_vcm
        )
    else:
        cpp_ecm = np.ascontiguousarray(edge_is_constrained, dtype=np.int32)
        out = pmp_adaptive_remeshing(
            cv,
            cf,
            error_tol,
            edge_length_min,
            edge_length_max,
            n_iter,
            protect_constraints,
            collapse_constraints,
            do_split,
            do_collapse,
            do_flip,
            number_of_relaxation_steps,
            disallow_surface_crossing,
            cpp_fid,
            cpp_faces_sel,
            cpp_vcm,
            cpp_ecm
        )
    return _from_SurfaceMeshWithFaceidAndPMaps(out)


def custom_remeshing(
    vertices: npt.ArrayLike,
    faces: npt.ArrayLike,
    sizing: npt.ArrayLike,
    face_id: npt.ArrayLike | None = None,
    face_is_selected: npt.ArrayLike | None = None,
    vertex_is_constrained: npt.ArrayLike | None = None,
    edge_is_constrained: npt.ArrayLike | None = None,
    n_iter: int = 1,
    protect_constraints: bool = False,
    collapse_constraints: bool = True,
    do_split: bool = True,
    do_collapse: bool = True,
    do_flip: bool = True,
    number_of_relaxation_steps: int = 1,
    disallow_surface_crossing=False,
):
    """Isotropic surface remeshing. Remeshing is achieved by a combination of
    edge splits/flips/collapses, tangential relaxation, and projection back
    onto the original surface.

    Parameters
    ----------
    vertices: npt.ArrayLike
    faces: npt.ArrayLike
    target_edge_length: float
        The target edge length for the isotropic remesher. This defines the
        resolution of the resulting surface.
    n_iter: int
        Number of iterations of the above-mentioned atomic operations.

    Returns
    -------
    v : npt.NDArray
        The new vertices.
    f : npt.NDArray
        The new faces.

    References
    ----------

    https://doc.cgal.org/latest/Polygon_mesh_processing/group__PMP__meshing__grp.html#gaa5cc92275df27f0baab2472ecbc4ea3f

    """
    face_id = [] if face_id is None else face_id
    face_is_selected = [] if face_is_selected is None else face_is_selected
    vertex_is_constrained = [] if vertex_is_constrained is None else vertex_is_constrained

    cdef np.ndarray[float, ndim=2] cv = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cf = np.ascontiguousarray(faces, dtype=np.int32)
    cdef np.ndarray[float, ndim=1] csizing = np.ascontiguousarray(sizing, dtype=np.float32)
    cdef np.ndarray[int] cpp_faces_sel = np.ascontiguousarray(face_is_selected, dtype=np.int32)
    cdef np.ndarray[int] cpp_fid = np.ascontiguousarray(face_id, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] cpp_vcm = np.ascontiguousarray(vertex_is_constrained, dtype=np.int32)
    cdef np.ndarray[int, ndim=2] cpp_ecm
    cdef SurfaceMeshWithFaceidAndPMaps out

    if edge_is_constrained is None:
        out = pmp_custom_remeshing(
            cv,
            cf,
            csizing,
            n_iter,
            protect_constraints,
            collapse_constraints,
            do_split,
            do_collapse,
            do_flip,
            number_of_relaxation_steps,
            disallow_surface_crossing,
            cpp_fid,
            cpp_faces_sel,
            cpp_vcm
        )
    else:
        cpp_ecm = np.ascontiguousarray(edge_is_constrained, dtype=np.int32)
        out = pmp_custom_remeshing(
            cv,
            cf,
            csizing,
            n_iter,
            protect_constraints,
            collapse_constraints,
            do_split,
            do_collapse,
            do_flip,
            number_of_relaxation_steps,
            disallow_surface_crossing,
            cpp_fid,
            cpp_faces_sel,
            cpp_vcm,
            cpp_ecm
        )
    return _from_SurfaceMeshWithFaceidAndPMaps(out)


def isotropic_remeshing(
    vertices: npt.ArrayLike,
    faces: npt.ArrayLike,
    target_edge_length: float,
    face_id: npt.ArrayLike | None = None,
    face_is_selected: npt.ArrayLike | None = None,
    vertex_is_constrained: npt.ArrayLike | None = None,
    edge_is_constrained: npt.ArrayLike | None = None,
    n_iter: int = 1,
    protect_constraints: bool = False,
    collapse_constraints: bool = True,
    do_split: bool = True,
    do_collapse: bool = True,
    do_flip: bool = True,
    number_of_relaxation_steps: int = 1,
    disallow_surface_crossing=False,
):
    """Isotropic surface remeshing. Remeshing is achieved by a combination of
    edge splits/flips/collapses, tangential relaxation, and projection back
    onto the original surface.

    Parameters
    ----------
    vertices: npt.ArrayLike
    faces: npt.ArrayLike
    target_edge_length: float
        The target edge length for the isotropic remesher. This defines the
        resolution of the resulting surface.
    n_iter: int
        Number of iterations of the above-mentioned atomic operations.

    Returns
    -------
    v : npt.NDArray
        The new vertices.
    f : npt.NDArray
        The new faces.

    References
    ----------

    https://doc.cgal.org/latest/Polygon_mesh_processing/group__PMP__meshing__grp.html#gaa5cc92275df27f0baab2472ecbc4ea3f

    """
    face_id = [] if face_id is None else face_id
    face_is_selected = [] if face_is_selected is None else face_is_selected
    vertex_is_constrained = [] if vertex_is_constrained is None else vertex_is_constrained

    cdef np.ndarray[float, ndim=2] cv = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cf = np.ascontiguousarray(faces, dtype=np.int32)
    cdef np.ndarray[int] cpp_faces_sel = np.ascontiguousarray(face_is_selected, dtype=np.int32)
    cdef np.ndarray[int] cpp_fid = np.ascontiguousarray(face_id, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] cpp_vcm = np.ascontiguousarray(vertex_is_constrained, dtype=np.int32)
    cdef np.ndarray[int, ndim=2] cpp_ecm
    cdef SurfaceMeshWithFaceidAndPMaps out

    if edge_is_constrained is None:
        out = pmp_isotropic_remeshing(
            cv,
            cf,
            target_edge_length,
            n_iter,
            protect_constraints,
            collapse_constraints,
            do_split,
            do_collapse,
            do_flip,
            number_of_relaxation_steps,
            disallow_surface_crossing,
            cpp_fid,
            cpp_faces_sel,
            cpp_vcm
        )
    else:
        cpp_ecm = np.ascontiguousarray(edge_is_constrained, dtype=np.int32)
        out = pmp_isotropic_remeshing(
            cv,
            cf,
            target_edge_length,
            n_iter,
            protect_constraints,
            collapse_constraints,
            do_split,
            do_collapse,
            do_flip,
            number_of_relaxation_steps,
            disallow_surface_crossing,
            cpp_fid,
            cpp_faces_sel,
            cpp_vcm,
            cpp_ecm
        )
    return _from_SurfaceMeshWithFaceidAndPMaps(out)


def collapse_halfedges(
    vertices: npt.ArrayLike,
    faces: npt.ArrayLike,
    edges: npt.ArrayLike,
    face_id: npt.ArrayLike | None = None,
):
    """Collapse halfedges (i.e., directed edges).

    Parameters
    ----------
    vertices: npt.ArrayLike
    faces: npt.ArrayLike

    Returns
    -------
    v : npt.NDArray
        The new vertices.
    f : npt.NDArray
        The new faces.

    References
    ----------

    https://doc.cgal.org/latest/Polygon_mesh_processing/group__PMP__meshing__grp.html#gaa5cc92275df27f0baab2472ecbc4ea3f

    """
    face_id = [] if face_id is None else face_id

    cdef np.ndarray[float, ndim=2] cv = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cf = np.ascontiguousarray(faces, dtype=np.int32)
    cdef np.ndarray[int, ndim=2] ce = np.ascontiguousarray(edges, dtype=np.int32)
    cdef np.ndarray[int] cpp_fid = np.ascontiguousarray(face_id, dtype=np.int32)
    cdef SurfaceMeshWithFaceidAndPMaps out

    out = pmp_collapse_halfedges(cv, cf, ce, cpp_fid)

    return _from_SurfaceMeshWithFaceidAndPMaps(out)


def collapse_short_edges(
    vertices: npt.ArrayLike,
    faces: npt.ArrayLike,
    target_edge_length: float,
    face_id: npt.ArrayLike | None = None,
    face_is_selected: npt.ArrayLike | None = None,
    vertex_is_constrained: npt.ArrayLike | None = None,
    edge_is_constrained: npt.ArrayLike | None = None,
):
    """Isotropic surface remeshing. Remeshing is achieved by a combination of
    edge splits/flips/collapses, tangential relaxation, and projection back
    onto the original surface.

    Parameters
    ----------
    vertices: npt.ArrayLike
    faces: npt.ArrayLike
    target_edge_length: float
        The target edge length for the isotropic remesher. This defines the
        resolution of the resulting surface.
    n_iter: int
        Number of iterations of the above-mentioned atomic operations.

    Returns
    -------
    v : npt.NDArray
        The new vertices.
    f : npt.NDArray
        The new faces.

    References
    ----------

    https://doc.cgal.org/latest/Polygon_mesh_processing/group__PMP__meshing__grp.html#gaa5cc92275df27f0baab2472ecbc4ea3f

    """
    face_id = [] if face_id is None else face_id
    face_is_selected = [] if face_is_selected is None else face_is_selected
    vertex_is_constrained = [] if vertex_is_constrained is None else vertex_is_constrained

    cdef np.ndarray[float, ndim=2] cv = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cf = np.ascontiguousarray(faces, dtype=np.int32)
    cdef np.ndarray[int] cpp_faces_sel = np.ascontiguousarray(face_is_selected, dtype=np.int32)
    cdef np.ndarray[int] cpp_fid = np.ascontiguousarray(face_id, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] cpp_vcm = np.ascontiguousarray(vertex_is_constrained, dtype=np.int32)
    cdef np.ndarray[int, ndim=2] cpp_ecm
    cdef SurfaceMeshWithFaceidAndPMaps out

    if edge_is_constrained is None:
        out = pmp_collapse_short_edges(
            cv,
            cf,
            target_edge_length,
            cpp_fid,
            cpp_faces_sel,
            cpp_vcm
        )
    else:
        cpp_ecm = np.ascontiguousarray(edge_is_constrained, dtype=np.int32)
        out = pmp_collapse_short_edges(
            cv,
            cf,
            target_edge_length,
            cpp_fid,
            cpp_faces_sel,
            cpp_vcm,
            cpp_ecm
        )
    return _from_SurfaceMeshWithFaceidAndPMaps(out)




def merge_duplicate_points_in_polygon_soup(vertices: npt.ArrayLike, faces: npt.ArrayLike):
    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef SurfaceMesh out
    out = pmp_merge_duplicate_points_in_polygon_soup(cpp_v, cpp_f)
    return _from_SurfaceMesh(out)

def merge_duplicate_polygons_in_polygon_soup(vertices: npt.ArrayLike, faces: npt.ArrayLike):
    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef SurfaceMesh out
    out = pmp_merge_duplicate_polygons_in_polygon_soup(cpp_v, cpp_f)
    return _from_SurfaceMesh(out)

def non_manifold_vertices(vertices, faces):
    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef vector[int] out
    out = pmp_non_manifold_vertices(cpp_v, cpp_f)
    return np.array(out, dtype=int)

def orient(
    vertices: npt.ArrayLike,
    faces: npt.ArrayLike,
    outward_orientation: bool = True
):
    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef SurfaceMesh out
    out = pmp_orient(cpp_v, cpp_f, outward_orientation)
    return _from_SurfaceMesh(out)

def orient_polygon_soup(vertices: npt.ArrayLike, faces: npt.ArrayLike):
    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef pair[cppbool, SurfaceMesh] out
    out = pmp_orient_polygon_soup(cpp_v, cpp_f)
    status = out.first
    # status == true
    #   operation succeeded
    # status == false
    #   some points were duplicated thus producing a combinatorically manifold
    #   but self-intersecting mesh
    return status, _from_SurfaceMesh(out.second)

def points_inside_surface(
    vertices: npt.NDArray,
    faces: npt.NDArray,
    points: npt.NDArray,
    on_boundary_is_inside: bool = True,
) -> npt.NDArray:
    """

    Parameters
    ----------
    vertices : npt.ArrayLike

    faces : npt.ArrayLike

    on_boundary_is_inside: bool
        If true, label points on the boundary as being inside. Otherwise, label
        as being outside.

    Returns
    -------
    intersecting_pairs : npt

    """
    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef np.ndarray[float, ndim=2] cpp_p = np.ascontiguousarray(points, dtype=np.float32)
    cdef cppbool cpp_on_boundary_is_inside = on_boundary_is_inside
    cdef vector[cppbool] out

    out = pmp_points_inside(cpp_v, cpp_f, cpp_p, cpp_on_boundary_is_inside)

    return np.array(out, dtype=bool)


def refine(
    vertices: npt.NDArray,
    faces: npt.NDArray,
    density: float = 2.0,
    faces_to_refine: npt.NDArray | None = None,
):
    faces_to_refine = [] if faces_to_refine is None else faces_to_refine

    cdef np.ndarray[float, ndim=2] cv = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cf = np.ascontiguousarray(faces, dtype=np.int32)
    cdef np.ndarray[int] cpp_faces_to_refine = np.ascontiguousarray(faces_to_refine, dtype=np.int32)
    cdef SurfaceMesh out

    out =  pmp_refine(cv, cf, cpp_faces_to_refine, density)

    return _from_SurfaceMesh(out)

def remove_almost_degenerate_faces(
    vertices: npt.ArrayLike,
    faces: npt.ArrayLike,
    face_is_selected: npt.ArrayLike | None = None,
    cap_threshold: float = -0.9396926207859083,
    needle_threshold: float = 4.0,
    vertex_is_constrained: npt.ArrayLike | None = None,
    edge_is_constrained: npt.ArrayLike | None = None,
):
    face_is_selected = [] if face_is_selected is None else face_is_selected
    vertex_is_constrained = [] if vertex_is_constrained is None else vertex_is_constrained

    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] cpp_select = np.ascontiguousarray(face_is_selected, dtype=np.int32)

    # cdef double cpp_cap_threshold = cap_threshold
    # cdef double cpp_needle_threshold = needle_threshold

    cdef np.ndarray[int, ndim=1] cpp_vcm = np.ascontiguousarray(vertex_is_constrained, dtype=np.int32)
    cdef np.ndarray[int, ndim=2] cpp_ecm
    cdef SurfaceMeshWithPMaps out

    if edge_is_constrained is None:
        out = pmp_remove_almost_degenerate_faces(
            cpp_v, cpp_f, cpp_select, cap_threshold, needle_threshold, cpp_vcm
        )
    else:
        cpp_ecm = np.ascontiguousarray(edge_is_constrained, dtype=np.int32)
        out = pmp_remove_almost_degenerate_faces(
            cpp_v, cpp_f, cpp_select, cap_threshold, needle_threshold, cpp_vcm, cpp_ecm
        )
    return _from_SurfaceMeshWithPMaps(out)

def remove_self_intersections(
    vertices: npt.NDArray,
    faces: npt.NDArray,
    face_is_selected: npt.ArrayLike | None = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute the intersecting pairs of triangles in a surface mesh.

    Parameters
    ----------
    vertices : npt.ArrayLike

    faces : npt.ArrayLike

    Returns
    -------
    intersecting_pairs : npt

    """
    face_is_selected = [] if face_is_selected is None else face_is_selected

    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] cpp_select = np.ascontiguousarray(face_is_selected, dtype=np.int32)
    cdef SurfaceMeshWithPMaps out

    out = pmp_remove_self_intersections(cpp_v, cpp_f, cpp_select)
    return _from_SurfaceMeshWithPMaps(out)

# def repair_mesh(
#         vertices: npt.NDArray, faces: npt.NDArray,
#     ) -> tuple[npt.NDArray, npt.NDArray]:
#     cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
#     cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)

#     cdef SurfaceMesh out

#     out = pmp_repair_mesh(cpp_v, cpp_f)
#      return _from_SurfaceMesh(out)

def self_intersections(vertices: npt.ArrayLike, faces: npt.ArrayLike) -> npt.NDArray:
    """Compute the intersecting pairs of triangles in a surface mesh.

    Parameters
    ----------
    vertices : npt.ArrayLike

    faces : npt.ArrayLike

    Returns
    -------
    intersecting_pairs : npt

    """
    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef IndexVector intersecting_pairs

    intersecting_pairs = pmp_self_intersections(cpp_v, cpp_f)

    return np.array(intersecting_pairs, dtype=int)

def smooth_angle_and_area(
        vertices: npt.ArrayLike,
        faces: npt.ArrayLike,
        constrained_vertices: npt.ArrayLike | None = None,
        n_iter: int = 1,
        use_angle_smoothing: bool = True,
        use_area_smoothing: bool = False,
        use_delaunay_flips: bool = True,
        use_safety_constraints: bool = False
    ):
    """Vertex smoothing preserving shape.

    Parameters
    ----------
    vertices: npt.ArrayLike
    faces: npt.ArrayLike
    constrained_vertices : npt.ArrayLike | None
    niter: int
    use_angle_smoothing: bool = True
    use_area_smoothing: bool = True
        This needs the Ceres solver library which we do not use be default.
    use_delaunay_flips: bool = True
    use_safety_constraints: bool = False

    Returns
    -------


    """
    constrained_vertices = [] if constrained_vertices is None else constrained_vertices

    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef np.ndarray[int] cpp_constrained_vertices = np.ascontiguousarray(constrained_vertices, dtype=np.int32)
    cdef PointVector v

    v = pmp_smooth_angle_and_area(
        cpp_v,
        cpp_f,
        cpp_constrained_vertices,
        n_iter,
        use_angle_smoothing,
        use_area_smoothing,
        use_delaunay_flips,
        use_safety_constraints
    )
    return np.array(v, dtype=float)

def smooth_shape(
    vertices: npt.ArrayLike,
    faces: npt.ArrayLike,
    constrained_vertices: npt.ArrayLike | None = None,
    time: float = 0.1,
    n_iter: int = 1
) -> npt.NDArray:
    """Shape smoothing using mean curvature flow.

    Parameters
    ----------
    time : float
        Determines the step size in the smoothing procedure (higher values
        means more aggressive smoothing).


    Returns
    -------
    The smoothed vertices.

    References
    ----------
    https://doc.cgal.org/latest/Polygon_mesh_processing/
    https://doc.cgal.org/latest/Polygon_mesh_processing/group__PMP__meshing__grp.html#ga57fa999abe8dc557003482444df2a189
    """
    constrained_vertices = [] if constrained_vertices is None else constrained_vertices

    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef np.ndarray[int] cpp_constrained_vertices = np.ascontiguousarray(constrained_vertices, dtype=np.int32)
    cdef PointVector v

    v = pmp_smooth_shape(cpp_v, cpp_f, cpp_constrained_vertices, time, n_iter)

    return np.array(v, dtype=float)

def smooth_shape_by_curvature_threshold(
        vertices: npt.ArrayLike,
        faces: npt.ArrayLike,
        time: float = 0.1,
        n_iter: int = 1,
        curv_threshold: float = 0.0,
        apply_above_curv_threshold: bool = True,
        ball_radius: float = -1.0,
    ) -> npt.NDArray:
    """Mean curvature flow constraining vertices whose curvature is either
    above or below a certain threshold. This allows strict shrinking or
    inflation of the surface whereas the standard mean curvature flow of
    vertices (as performed by `smooth_shape`) will shrink convex areas and
    inflate concave areas.
    The default settings (`curv_threshold = 0.0` and
    `apply_above_curv_threshold = True`) results in strict shrinkage.

    Parameters
    ----------
    vertices
        Vertices of the surface.
    faces
        Faces of the surface.
    time : float
        Amount of smoothing to apply at each iteration (higher values means
        more aggressive smoothing).
    n_iter : int
        Number of curvature estimation and smoothing steps to apply.
    curv_threshold : float
        Apply smoothing to vertices above/below `curv_threshold` (default = 0.0).
    apply_above_curv_threshold : bool
        If true, apply smoothing to vertices whose curvature is *above*
        `curv_threshold` (and vice verse if false) (default = True). If
        true (and curv_threshold = 0.0), the surface will strictly shrink.
    ball_radius : float
        Smooth curvature estimates within a ball with `ball_radius`. Must
        be > 0.0 except -1.0 which disables smoothing (default = -1.0).

    Returns
    -------
    The smoothed vertices.

    References
    ----------
    https://doc.cgal.org/latest/Polygon_mesh_processing/
    https://doc.cgal.org/latest/Polygon_mesh_processing/group__PMP__meshing__grp.html#ga57fa999abe8dc557003482444df2a189
    """
    assert ball_radius == -1.0 or ball_radius > 0.0
    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef cppbool cpp_apply_above_curv_threshold = apply_above_curv_threshold
    cdef PointVector v

    v = pmp_smooth_shape_by_curvature_threshold(cpp_v, cpp_f, time, n_iter, curv_threshold, cpp_apply_above_curv_threshold, ball_radius)

    return np.array(v, dtype=float)

def split_edges(
    vertices: npt.ArrayLike,
    faces: npt.ArrayLike,
    edges: npt.ArrayLike,
    face_id: npt.ArrayLike | None = None,
):
    """Collapse halfedges (i.e., directed edges).

    Parameters
    ----------
    vertices: npt.ArrayLike
    faces: npt.ArrayLike

    Returns
    -------
    v : npt.NDArray
        The new vertices.
    f : npt.NDArray
        The new faces.

    References
    ----------

    https://doc.cgal.org/latest/Polygon_mesh_processing/group__PMP__meshing__grp.html#gaa5cc92275df27f0baab2472ecbc4ea3f

    """
    face_id = [] if face_id is None else face_id

    cdef np.ndarray[float, ndim=2] cv = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cf = np.ascontiguousarray(faces, dtype=np.int32)
    cdef np.ndarray[int, ndim=2] ce = np.ascontiguousarray(edges, dtype=np.int32)
    cdef np.ndarray[int] cpp_fid = np.ascontiguousarray(face_id, dtype=np.int32)
    cdef SurfaceMeshWithFaceidAndPMaps out

    out = pmp_split_edges(cv, cf, ce, cpp_fid)

    return _from_SurfaceMeshWithFaceidAndPMaps(out)


def split_long_edges(
    vertices: npt.NDArray,
    faces: npt.NDArray,
    sizing: float,
    face_id: npt.NDArray | None = None,
    edges: npt.NDArray | None = None
):
    no_edges = edges is None
    face_id = [] if face_id is None else face_id

    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] cpp_fid = np.ascontiguousarray(face_id, dtype=np.int32)
    cdef SurfaceMeshWithFaceidAndPMaps out

    if no_edges:
        out = pmp_split_long_edges(cpp_v, cpp_f, sizing, cpp_fid)
    else:
        cpp_edges = np.ascontiguousarray(edges, dtype=np.int32)
        out = pmp_split_long_edges(cpp_v, cpp_f, sizing, cpp_fid, cpp_edges)
    return _from_SurfaceMeshWithFaceidAndPMaps(out)

def split_with_plane(
    vertices: npt.NDArray,
    faces: npt.NDArray,
    plane_origin: npt.NDArray,
    plane_direction: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    """Split surface with a plane.

    Parameters
    ----------
    vertices : npt.ArrayLike
    faces : npt.ArrayLike

    Returns
    -------
    intersecting_pairs : npt

    """
    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef np.ndarray[float] cpp_orig = np.ascontiguousarray(plane_origin, dtype=np.float32)
    cdef np.ndarray[float] cpp_dir = np.ascontiguousarray(plane_direction, dtype=np.float32)
    cdef SurfaceMesh out
    out = pmp_split_with_plane(cpp_v, cpp_f, cpp_orig, cpp_dir)
    return _from_SurfaceMesh(out)

def split_with_surface(
    surface_v: npt.NDArray,
    surface_f: npt.NDArray,
    splitter_v: npt.NDArray,
    splitter_f: npt.NDArray,
) -> tuple[tuple[npt.NDArray, npt.NDArray],tuple[npt.NDArray, npt.NDArray]]:
    """Split surface with another surface.

    Parameters
    ----------
    surface_v : npt.ArrayLike
    surface_f : npt.ArrayLike
    splitter_v : npt.ArrayLike
    splitter_f : npt.ArrayLike

    Returns
    -------

    """
    cdef np.ndarray[float, ndim=2] cpp_s_v = np.ascontiguousarray(surface_v, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_s_f = np.ascontiguousarray(surface_f, dtype=np.int32)
    cdef np.ndarray[float, ndim=2] cpp_spl_v = np.ascontiguousarray(splitter_v, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_spl_f = np.ascontiguousarray(splitter_f, dtype=np.int32)
    cdef pair[SurfaceMesh,SurfaceMesh] out
    out = pmp_split_with_surface(cpp_s_v, cpp_s_f, cpp_spl_v, cpp_spl_f)
    return _from_SurfaceMesh(out.first),_from_SurfaceMesh(out.second)

def stitch_borders(vertices: npt.NDArray, faces: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef SurfaceMeshWithPMaps out
    out = pmp_stitch_borders(cpp_v, cpp_f)
    return _from_SurfaceMeshWithPMaps(out)

def tangential_relaxation(
        vertices: npt.ArrayLike,
        faces: npt.ArrayLike,
        constrained_vertices: npt.ArrayLike | None = None,
        n_iter: int = 1,
    ) -> npt.NDArray:
    """Tangential relaxation of vertices.

    Parameters
    ----------

    Returns
    -------
    The smoothed vertices.

    References
    ----------
    https://doc.cgal.org/latest/Polygon_mesh_processing/
    https://doc.cgal.org/latest/Polygon_mesh_processing/group__PMP__meshing__grp.html#ga57fa999abe8dc557003482444df2a189
    """
    constrained_vertices = [] if constrained_vertices is None else constrained_vertices

    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef np.ndarray[int] cpp_constrained_vertices = np.ascontiguousarray(constrained_vertices, dtype=np.int32)
    cdef PointVector v

    v = pmp_tangential_relaxation(cpp_v, cpp_f, cpp_constrained_vertices, n_iter)

    return np.array(v, dtype=float)

# def volume_connected_components(faces, do_orientation_tests = False, do_self_intersection_tests = False):
#     cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
#     cdef cppbool cpp_do_orientation_tests = do_orientation_tests
#     cdef cppbool cpp_do_self_intersection_tests = do_self_intersection_tests

#     cdef pair[vector[int], vector[int]] out

#     out = pmp_volume_connected_components(
#         cpp_f,
#         cpp_do_orientation_tests,
#         cpp_do_self_intersection_tests
#     )
#     component_label = np.array(out.first, dtype=int)
#     component_size = np.array(out.second, dtype=int)

#     return component_label, component_size
