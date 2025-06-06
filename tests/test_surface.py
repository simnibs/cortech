import copy

import numpy as np
import pytest

from cortech.surface import Surface
import cortech.utils


@pytest.fixture
def diamond(diamond_vertices, diamond_faces):
    return Surface(diamond_vertices, diamond_faces)


@pytest.fixture
def diamond_intersect(diamond, diamond_barycenters):
    # Create face intersections by moving vertex 0 through the plane of face 4
    diamond_intersect = copy.deepcopy(diamond)
    diamond_intersect.vertices[0] = diamond_barycenters[4] * 1.1
    return diamond_intersect


@pytest.fixture
def sphere(sphere_tuple):
    return Surface(*sphere_tuple)


def sph_to_cart(theta, phi):
    """
    points : r, theta, phi in columns
    """
    theta = np.atleast_2d(theta)
    phi = np.atleast_2d(phi)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.squeeze(np.stack([x, y, z], axis=1))


class TestSurface:
    def test_create_surface(self, sphere_tuple):
        s = Surface(*sphere_tuple)
        np.testing.assert_allclose(s.vertices, sphere_tuple[0])
        np.testing.assert_allclose(s.faces, sphere_tuple[1])

    @pytest.mark.parametrize("include_self", [False, True])
    def test_compute_vertex_adjacency(
        self, include_self, diamond, diamond_adjacency_matrix
    ):
        a = diamond.compute_vertex_adjacency(include_self)
        a_true = diamond_adjacency_matrix
        if include_self:
            a_true = a_true + np.eye(diamond.n_vertices)
        np.testing.assert_array_equal(a.todense(), a_true)

    def test_compute_face_barycenters(self, diamond, diamond_barycenters):
        b = diamond.compute_face_barycenters()
        np.testing.assert_allclose(b, diamond_barycenters)

    def test_compute_face_normals(self, diamond):
        n = diamond.compute_face_normals()
        n_true = cortech.utils.normalize(diamond.compute_face_barycenters(), axis=1)
        np.testing.assert_allclose(n, n_true)

    def test_compute_vertex_normals(self, diamond):
        n = diamond.compute_vertex_normals()
        n_true = cortech.utils.normalize(diamond.vertices, axis=1)
        np.testing.assert_allclose(n, n_true)

    def test_compute_principal_curvatures(self):
        pass

    # @pytest.mask.parametrize("radius", [0.5, 1.0, 5.0])
    # def test_compute_curvature(self, radius):
    #     sphere = Surface(*fibonacci_sphere(10000, radius))
    #     curv = sphere.compute_curvature()
    #     curvs = sphere.compute_curvature(smooth_iter=10)

    #     k1_true = -1.0/radius
    #     k2_true = k1_true
    #     H_true = k1_true
    #     K_true = 2 * H_true

    #     theta_resolution = 200
    #     phi_resolution = 100
    #     theta = np.linspace(0, 2 * np.pi, theta_resolution)
    #     phi = np.linspace(0, np.pi, phi_resolution)

    #     p = sph_to_cart(np.repeat(theta, len(phi)), np.tile(phi, len(theta)))

    #     p = 0.5 * p
    #     v,f = convex_hull(p)

    #     pd = pv.make_tri_mesh(v, f)
    #     pd.save("test3.vtk")

    #     sphere = pv.Sphere(0.5, theta_resolution=200, phi_resolution=100)
    #     surf = Surface(sphere.points, sphere.faces.reshape(-1, 4)[:,1:])
    #     curv = surf.compute_curvature()

    #     np.testing.assert_allclose(curv.k1, k1_true)
    #     np.testing.assert_allclose(curv.k2, k2_true, atol=0.4)
    #     np.testing.assert_allclose(curv.H, H_true, atol=0.4)
    #     np.testing.assert_allclose(curv.K, K_true, atol=0.8)

    #     np.testing.assert_allclose(curvs.k1, k1_true)
    #     np.testing.assert_allclose(curvs.k2, k2_true)
    #     np.testing.assert_allclose(curvs.H, H_true)
    #     np.testing.assert_allclose(curvs.K, K_true)

    # def test_convex_hull(self, diamond):
    #     p = np.concatenate(
    #         (diamond.vertices, np.array([[0.0,0.0,0.0]]), np.array([[1.0,1.0,1.0]])), axis=0)
    #     diamond_copy = diamond.

    #     hull = Surface.convex_hull(p)

    # def test_k_ring_neighbors():

    #     s = Surface(vertices, faces)
    #     knn,kr = s.k_ring_neighbors(1, 0)
    #     knn[0] == 0
    #     knn[1:5] ==
    #     n = s.k_ring_neighbors(2, 0)

    def test_remove_self_intersections(self, diamond_intersect):
        """Basic check that *something* sensible is being done."""
        assert len(diamond_intersect.self_intersections()) > 0
        diamond_clean = diamond_intersect.remove_self_intersections()
        assert len(diamond_clean.self_intersections()) == 0

    def test_self_intersections(self, diamond_intersect):
        sif = diamond_intersect.self_intersections()
        np.testing.assert_allclose(sif, [[3, 4], [1, 4], [2, 4]])

    def test_connected_components(self, diamond):
        cc_label, cc_size = diamond.connected_components()
        np.testing.assert_allclose(cc_label, 0)
        np.testing.assert_allclose(cc_size, diamond.n_faces)

    def test_connected_components_constrained(self, diamond):
        cc_label, cc_size = diamond.connected_components([0, 1, 2, 3])
        n = diamond.n_faces // 2
        np.testing.assert_allclose(
            cc_label, np.concatenate((np.full(n, 0), np.full(n, 1)))
        )
        np.testing.assert_allclose(cc_size, [n, n])

    def test_points_inside_surface(self, diamond, diamond_barycenters, eps=1e-6):
        # Move points inwards
        is_inside = diamond.points_inside_surface(diamond_barycenters * (1 - eps))
        np.testing.assert_allclose(is_inside, True)

    def test_points_outside_surface(self, diamond, diamond_barycenters, eps=1e-6):
        # Move points outwards
        is_inside = diamond.points_inside_surface(diamond_barycenters * (1 + eps))
        np.testing.assert_allclose(is_inside, False)

    def test_shape_smooth(self):
        pass

    def test_taubin_smooth(self):
        pass

    def test_gaussian_smooth(self):
        pass

    def test_get_triangle_neighbors(self):
        """Triangulate an array of points and test neighbors like"""

        # Points
        # (coords)       (indices)
        #  1    .  .  .   2  5  9
        #  0    .  .  .   1  4  7
        # -1    .  .  .   0  3  6
        #      -1  0  1

        # Triangle indices
        # | 3 /  \ 7 |
        # |  /    \  |
        # | / 2  6 \ |
        # | \ 1  5 / |
        # |  \    /  |
        # | 0 \  / 4 |
        tris = np.array(
            [
                [0, 3, 1],
                [1, 3, 4],
                [3, 7, 4],
                [3, 6, 7],
                [1, 5, 2],
                [1, 4, 5],
                [4, 7, 5],
                [5, 7, 8],
            ]
        )
        pttris = transformations._get_triangle_neighbors(tris, nr=9)
        pttris_expected = [
            [0],
            [0, 1, 4, 5],
            [4],
            [0, 1, 2, 3],
            [1, 2, 5, 6],
            [4, 5, 6, 7],
            [3],
            [2, 3, 6, 7],
            [7],
        ]
        for i, j in zip(pttris, pttris_expected):
            assert all(i == j)

    def test_get_nearest_triangles_on_surface(self):
        """Same triangulation as before."""
        tris = np.array(
            [
                [0, 3, 1],
                [1, 3, 4],
                [3, 7, 4],
                [3, 6, 7],
                [1, 5, 2],
                [1, 4, 5],
                [4, 7, 5],
                [5, 7, 8],
            ]
        )
        x, y, z = (-1, 0, 1), (-1, 0, 1), (0,)
        points = np.dstack(np.meshgrid(x, y, z)).reshape(-1, 3)
        surf = dict(points=points, tris=tris)

        test_points = np.array(
            [
                [-0.9, -0.9, 1.0],
                [-0.9, 0.9, 1.0],
                [0.9, -0.9, 1.0],
                [0.9, 0.9, 1.0],
            ]
        )

        # all
        pttris = transformations._get_nearest_triangles_on_surface(test_points, surf, 1)
        pttris_exp = [[0], [3], [4], [7]]

        for pt, pte in zip(pttris, pttris_exp):
            assert all(pt == pte)

        # subset
        subset = np.array([0, 1, 3, 4])
        pttris = transformations._get_nearest_triangles_on_surface(
            test_points, surf, 1, subset
        )
        pttris_exp = [[0], [0, 1, 2, 3], [0, 1, 4, 5], [1, 2, 5, 6]]

        for pt, pte in zip(pttris, pttris_exp):
            assert all(pt == pte)

    def test_project_points_to_surface(self):
        """Make a surface consisting of one triangle and project a point from each
        'region' to it.
        """
        surf = dict(
            points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            tris=np.array([[0, 1, 2]]),
        )

        # Test a point in each region
        points = np.array(
            [
                [0.25, 0.25, 0.67],  # Region 0
                [1, 1, 0.27],  # Region 1
                [-0.25, 2, 1.1],  # Region 2
                [-1, 0.5, -1.4],  # Region 3
                [-1, -1, 0.53],  # Region 4
                [0.5, -1, -0.77],  # Region 5
                [2, -0.25, -0.16],
            ]
        )  # Region 6
        pttris = np.atleast_2d(np.zeros(len(points), dtype=int)).T

        # Expected output
        tris = np.zeros(len(points), dtype=int)
        weights = np.array(
            [
                [0.5, 0.25, 0.25],
                [0, 0.5, 0.5],
                [0, 0, 1],
                [0.5, 0, 0.5],
                [1, 0, 0],
                [0.5, 0.5, 0],
                [0, 1, 0],
            ]
        )
        projs = np.array(
            [
                [0.25, 0.25, 0],
                [0.5, 0.5, 0],
                [0, 1, 0],
                [0, 0.5, 0],
                [0, 0, 0],
                [0.5, 0, 0],
                [1, 0, 0],
            ]
        )
        dists = np.linalg.norm(points - projs, axis=1)

        # Actual output
        t, w, p, d = transformations._project_points_to_surface(points, surf, pttris)

        np.testing.assert_array_equal(t, tris)
        np.testing.assert_allclose(w, weights)
        np.testing.assert_allclose(p, projs)
        np.testing.assert_allclose(d, dists)

    def test_prune(self, diamond):
        # *Prepend* fake vertices, adjust faces accordingly, and prune to
        # recover the original mesh.
        d = copy.deepcopy(diamond)
        d.vertices = np.concatenate((np.ones_like(d.vertices), d.vertices), axis=0)
        d.faces += diamond.n_vertices
        d.prune()

        np.testing.assert_allclose(d.vertices, diamond.vertices)
        np.testing.assert_allclose(d.faces, diamond.faces)

    def test_from_freesurfer_subject_dir(self):
        pass
