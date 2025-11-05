from libcpp cimport bool as cppbool
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from typing import Union
import numpy as np
import numpy.typing as npt
cimport numpy as np


cdef extern from "polygon_mesh_processing_src.cpp" nogil:
    vector[cppbool] pmp_points_inside_surface(
        vector[vector[float]] vertices,
        vector[vector[int]] faces,
        vector[vector[float]] points,
        cppbool on_boundary_is_inside,
    )

    pair[vector[vector[float]], vector[vector[int]]] pmp_split(
        vector[vector[float]] vertices,
        vector[vector[int]] faces,
        vector[float] plane_origin,
        vector[float] plane_direction,
    )

    pair[vector[vector[float]], vector[vector[int]]] pmp_hole_fill_refine_fair(
        vector[vector[float]] vertices,
        vector[vector[int]] faces,
    )

    pair[vector[vector[float]], vector[vector[int]]] pmp_clip(
        vector[vector[float]] vertices,
        vector[vector[int]] faces,
        vector[float] plane_origin,
        vector[float] plane_direction,
        # cppbool clip_volume
    )

    # pair[vector[vector[float]], vector[vector[int]]] pmp_repair_mesh(
    #     vector[vector[float]] vertices,
    #     vector[vector[int]] faces,
    # )

    pair[vector[vector[float]], vector[vector[int]]] pmp_remove_self_intersections(
        vector[vector[float]] vertices,
        vector[vector[int]] faces,
    )

    vector[vector[int]] pmp_self_intersections(
        vector[vector[float]] vertices,
        vector[vector[int]] faces,
    )

    vector[vector[int]] pmp_intersecting_meshes(
        vector[vector[float]] vertices0,
        vector[vector[int]] faces0,
        vector[vector[float]] vertices1,
        vector[vector[int]] faces1,
    )

    pair[vector[int], vector[int]] pmp_connected_components(
        vector[vector[float]] vertices,
        vector[vector[int]] faces,
        vector[int] contrained_faces,
    )

    # pair[vector[int], vector[int]] pmp_volume_connected_components(
    #     vector[vector[int]] faces,
    #     cppbool do_orientation_tests,
    #     cppbool do_self_intersection_tests,
    # )

    vector[vector[float]] pmp_smooth_shape(
        vector[vector[float]] vertices,
        vector[vector[int]] faces,
        vector[int] constrained_vertices,
        double time,
        int n_iter
    )

    vector[vector[float]] pmp_smooth_shape_by_curvature_threshold(
        vector[vector[float]] vertices,
        vector[vector[int]] faces,
        double time,
        int n_iter,
        double curv_threshold,
        cppbool apply_above_curv_threshold,
        double ball_radius,
    )

    vector[vector[float]] pmp_tangential_relaxation(
        vector[vector[float]] vertices,
        vector[vector[int]] faces,
        vector[int] constrained_vertices,
        int n_iter
    )

    vector[vector[float]] pmp_interpolated_corrected_curvatures(
        vector[vector[float]] vertices,
        vector[vector[int]] faces,
    )

    vector[vector[float]] pmp_smooth_angle_and_area(
        vector[vector[float]] vertices,
        vector[vector[int]] faces,
        vector[int] constrained_vertices,
        int niter,
        cppbool use_angle_smoothing,
        cppbool use_area_smoothing,
        cppbool use_delaunay_flips,
        cppbool use_safety_constraints
    )

    vector[vector[float]] pmp_fair(
        vector[vector[float]] vertices,
        vector[vector[int]] faces,
        vector[int] indices,
    )

    pair[vector[vector[float]], vector[vector[int]]] pmp_isotropic_remeshing(
        vector[vector[float]] vertices,
        vector[vector[int]] faces,
        double target_edge_length,
        int n_iter,
    )

    # pair[vector[vector[float]], vector[vector[int]]] pmp_corefine_and_union(
    #     vector[vector[float]] vertices1,
    #     vector[vector[int]] faces1,
    #     vector[vector[float]] vertices2,
    #     vector[vector[int]] faces2,
    # )


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

    out = pmp_points_inside_surface(cpp_v, cpp_f, cpp_p, cpp_on_boundary_is_inside)

    return np.array(out, dtype=bool)

def hole_fill_refine_fair(
        vertices: npt.NDArray,
        faces: npt.NDArray,
    ) -> tuple[npt.NDArray, npt.NDArray]:
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

    cdef pair[vector[vector[float]], vector[vector[int]]] out

    out = pmp_hole_fill_refine_fair(cpp_v, cpp_f)

    v = np.array(out.first, dtype=float)
    f = np.array(out.second, dtype=int)

    return v, f


def split(
        vertices: npt.NDArray,
        faces: npt.NDArray,
        plane_origin: npt.NDArray,
        plane_direction: npt.NDArray
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

    cdef pair[vector[vector[float]], vector[vector[int]]] out

    out = pmp_split(cpp_v, cpp_f, cpp_orig, cpp_dir)

    v = np.array(out.first, dtype=float)
    f = np.array(out.second, dtype=int)

    return v, f

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

    cdef pair[vector[vector[float]], vector[vector[int]]] out

    out = pmp_clip(cpp_v, cpp_f, cpp_orig, cpp_dir)

    v = np.array(out.first, dtype=float)
    f = np.array(out.second, dtype=int)

    return v, f

# def repair_mesh(
#         vertices: npt.NDArray, faces: npt.NDArray,
#     ) -> tuple[npt.NDArray, npt.NDArray]:
#     cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
#     cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)

#     cdef pair[vector[vector[float]], vector[vector[int]]] out

#     out = pmp_repair_mesh(cpp_v, cpp_f)

#     v = np.array(out.first, dtype=float)
#     f = np.array(out.second, dtype=int)

#     return v, f


def remove_self_intersections(
        vertices: npt.NDArray, faces: npt.NDArray,
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

    cdef pair[vector[vector[float]], vector[vector[int]]] out

    out = pmp_remove_self_intersections(cpp_v, cpp_f)

    v = np.array(out.first, dtype=float)
    f = np.array(out.second, dtype=int)

    return v, f


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

    cdef vector[vector[int]] intersecting_pairs # list of lists

    intersecting_pairs = pmp_self_intersections(cpp_v, cpp_f)

    return np.array(intersecting_pairs, dtype=int)


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

    cdef vector[vector[int]] intersecting_pairs # list of lists

    intersecting_pairs = pmp_intersecting_meshes(cpp_v0, cpp_f0, cpp_v1, cpp_f1)

    return np.array(intersecting_pairs, dtype=int)


def connected_components(
        vertices: npt.ArrayLike,
        faces: npt.ArrayLike,
        constrained_faces: npt.ArrayLike | None = None
    ) -> tuple[npt.NDArray, npt.NDArray]:
    """Label connected components on a surface (graph).

    Parameters
    ----------
    vertices: npt.ArrayLike
        Vertices of the surfaces, shape = (N, 3).
    faces: npt.ArrayLike
        Faces of the surface, shape = (M, 3).
    constrained_faces: Union[npt.ArrayLike, None]
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
    constrained_faces = [] if constrained_faces is None else constrained_faces
    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef np.ndarray[int] cpp_constrained_faces = np.ascontiguousarray(constrained_faces, dtype=np.int32)
    cdef pair[vector[int], vector[int]] out

    out = pmp_connected_components(cpp_v, cpp_f, cpp_constrained_faces)
    component_label = np.array(out.first, dtype=int)
    component_size = np.array(out.second, dtype=int)

    return component_label, component_size


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


def smooth_shape(
        vertices: npt.ArrayLike,
        faces: npt.ArrayLike,
        constrained_vertices: Union[npt.ArrayLike, None] = None,
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
    cdef vector[vector[float]] v

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
    cdef vector[vector[float]] v

    v = pmp_smooth_shape_by_curvature_threshold(cpp_v, cpp_f, time, n_iter, curv_threshold, cpp_apply_above_curv_threshold, ball_radius)

    return np.array(v, dtype=float)

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

def tangential_relaxation(
        vertices: npt.ArrayLike,
        faces: npt.ArrayLike,
        constrained_vertices: Union[npt.ArrayLike, None] = None,
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
    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef np.ndarray[int] cpp_constrained_vertices = np.ascontiguousarray(constrained_vertices or [], dtype=np.int32)
    cdef vector[vector[float]] v

    v = pmp_tangential_relaxation(cpp_v, cpp_f, cpp_constrained_vertices, n_iter)

    return np.array(v, dtype=float)


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
    constrained_vertices : Union[npt.ArrayLike, None]
    niter: int
    use_angle_smoothing: bool = True
    use_area_smoothing: bool = True
        This needs the Ceres solver library which we do not use be default.
    use_delaunay_flips: bool = True
    use_safety_constraints: bool = False

    Returns
    -------


    """
    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef np.ndarray[int] cpp_constrained_vertices = np.ascontiguousarray(constrained_vertices or [], dtype=np.int32)
    cdef vector[vector[float]] v

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


def fair(vertices, faces, vertex_indices):
    """Mesh fairing."""
    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef np.ndarray[int] cpp_vi = np.ascontiguousarray(vertex_indices, dtype=np.int32)
    cdef vector[vector[float]] v

    v = pmp_fair(cpp_v, cpp_f, cpp_vi)

    return np.array(v, dtype=float)


def isotropic_remeshing(
        vertices: npt.ArrayLike,
        faces: npt.ArrayLike,
        target_edge_length: float,
        n_iter: int = 1,
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
    cdef np.ndarray[float, ndim=2] cpp_v = np.ascontiguousarray(vertices, dtype=np.float32)
    cdef np.ndarray[int, ndim=2] cpp_f = np.ascontiguousarray(faces, dtype=np.int32)
    cdef pair[vector[vector[float]], vector[vector[int]]] out

    out = pmp_isotropic_remeshing(
        cpp_v, cpp_f, target_edge_length, n_iter
    )
    v = np.array(out.first, dtype=float)
    f = np.array(out.second, dtype=int)

    return v, f


# def corefine_and_union(vertices1, faces1, vertices2, faces2):
#     cdef np.ndarray[float, ndim=2] cpp_v1 = np.ascontiguousarray(vertices1, dtype=np.float32)
#     cdef np.ndarray[int, ndim=2] cpp_f1 = np.ascontiguousarray(faces1, dtype=np.int32)
#     cdef np.ndarray[float, ndim=2] cpp_v2 = np.ascontiguousarray(vertices2, dtype=np.float32)
#     cdef np.ndarray[int, ndim=2] cpp_f2 = np.ascontiguousarray(faces2, dtype=np.int32)
#     cdef pair[vector[vector[float]], vector[vector[int]]] out

#     out = pmp_corefine_and_union(cpp_v1, cpp_f1, cpp_v2, cpp_f2)
#     v = np.array(out.first, dtype=float)
#     f = np.array(out.second, dtype=int)

#     return v, f
