import copy
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np
import pytest
from scipy.spatial import cKDTree

from cortech.surface import Sphere, Surface
from cortech.freesurfer import VolumeGeometry
import cortech.utils


@pytest.fixture
def triangulation():
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
    x, y, z = (-1, 0, 1), (-1, 0, 1), (0,)
    return Surface(
        np.dstack(np.meshgrid(x, y, z)).reshape(-1, 3),
        np.array(
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
        ),
    )


@pytest.fixture
def one_triangle():
    return Surface(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), np.array([[0, 1, 2]]))


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


@pytest.fixture
def sphere_reg(sphere_tuple):
    return Sphere(*sphere_tuple)


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

    def test_self_intersections(self, diamond_intersect):
        sif = diamond_intersect.self_intersections()
        np.testing.assert_allclose(sif, [[3, 4], [1, 4], [2, 4]])

    def test_remove_self_intersections(self, diamond_intersect):
        """Basic check that *something* sensible is being done."""
        assert len(diamond_intersect.self_intersections()) > 0
        diamond_clean = diamond_intersect.remove_self_intersections()
        assert len(diamond_clean.self_intersections()) == 0

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

    def test_get_triangle_neighbors(self, triangulation):
        pttris = triangulation.get_triangle_neighbors()
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
        assert all(all(i == j) for i, j in zip(pttris, pttris_expected))

    @pytest.mark.parametrize("subset", [None, np.array([0, 1, 3, 4])])
    def test_get_closest_triangles(self, triangulation, subset):
        test_points = np.array(
            [
                [-0.9, -0.9, 1.0],
                [-0.9, 0.9, 1.0],
                [0.9, -0.9, 1.0],
                [0.9, 0.9, 1.0],
            ]
        )
        pttris = triangulation.get_closest_triangles(test_points, 1, subset)
        if subset is None:
            pttris_exp = [[0], [3], [4], [7]]
        else:
            pttris_exp = [[0], [0, 1, 2, 3], [0, 1, 4, 5], [1, 2, 5, 6]]

        assert all(all(pt == pte) for pt, pte in zip(pttris, pttris_exp))

    def test_project_points(self, one_triangle):
        """Make a surface consisting of one triangle and project a point from each
        'region' to it.
        """
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
        t, w, p, d = one_triangle.project_points(points, pttris)

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
        d = d.prune()

        np.testing.assert_allclose(d.vertices, diamond.vertices)
        np.testing.assert_allclose(d.faces, diamond.faces)

    @pytest.mark.parametrize("space", ["scanner", "surface"])
    def test_to_surface_ras(self, diamond_vertices, diamond_faces, space):
        """Test conversion of coordinates to scanner/surface ras."""
        geom = VolumeGeometry(
            valid=True,
            cosines=[[-1, 0, 0], [0, 0, 1], [0, -1, 0]],
            cras=[10.0, 5.0, 0.0],
        )
        ras2tkr = geom.get_affine("tkr", fr="scanner")
        diamond_vertices_tkr = nib.affines.apply_affine(ras2tkr, diamond_vertices)

        s_ras = Surface(diamond_vertices, diamond_faces, "scanner", geom)
        s_tkr = Surface(diamond_vertices_tkr, diamond_faces, "surface", geom)

        assert not np.allclose(s_ras.vertices, s_tkr.vertices)

        match space:
            case "scanner":
                s_tkr.to_scanner_ras()
            case "surface":
                s_ras.to_surface_ras()
        np.testing.assert_allclose(s_ras.vertices, s_tkr.vertices)


class TestSurfaceLoad:
    def test_from_freesurfer(self, BERT_DIR):
        s = Surface.from_freesurfer(BERT_DIR / "surf" / "lh.white")
        assert s.n_vertices == 2562
        assert s.n_faces == 5120

    def test_from_gifti(self, BERT_DIR):
        s = Surface.from_gifti(BERT_DIR / "surf" / "lh.white.gii")
        assert s.n_vertices == 2562
        assert s.n_faces == 5120

    @pytest.mark.skip
    def test_from_vtk(self, BERT_DIR):
        s = Surface.from_vtk(BERT_DIR / "surf" / "lh.white.vtk")
        assert s.n_vertices == 2562
        assert s.n_faces == 5120

    def test_from_freesurfer_subject_dir(self, BERT_DIR):
        s = Surface.from_freesurfer_subject_dir(BERT_DIR, "lh.white")
        assert s.n_vertices == 2562
        assert s.n_faces == 5120

    @pytest.mark.parametrize("filename", ["lh.white", "lh.pial"])
    def test_from_file(self, BERT_DIR, filename):
        s0 = Surface.from_file(BERT_DIR / "surf" / filename)
        s1 = Surface.from_freesurfer(BERT_DIR / "surf" / filename)
        np.testing.assert_allclose(s0.vertices, s1.vertices)
        np.testing.assert_allclose(s0.faces, s1.faces)

    def test_read_dataspace(self, BERT_DIR):
        s_tkr = Surface.from_file(BERT_DIR / "surf" / "lh.white")
        s_ras = Surface.from_file(BERT_DIR / "surf" / "lh.white.scanner")

        assert s_tkr.is_surface_ras()
        assert s_ras.is_scanner_ras()
        assert not np.allclose(s_tkr.vertices, s_ras.vertices)

    def test_read_vol_geom(self, BERT_DIR, VOL_GEOM):
        s = Surface.from_file(BERT_DIR / "surf" / "lh.white")

        assert s.geometry.valid
        assert s.geometry.filename == VOL_GEOM["filename"]
        np.testing.assert_allclose(s.geometry.volume, VOL_GEOM["volume"])
        np.testing.assert_allclose(s.geometry.voxelsize, VOL_GEOM["voxelsize"])
        np.testing.assert_allclose(
            s.geometry.cras, VOL_GEOM["cras"], rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(s.geometry.cosines, VOL_GEOM["cosines"])

    def test_read_vol_geom_gifti(self, BERT_DIR):
        sfs = Surface.from_file(BERT_DIR / "surf" / "lh.white")
        sgii = Surface.from_file(BERT_DIR / "surf" / "lh.white.gii")

        assert sfs.geometry.valid == sgii.geometry.valid
        assert sfs.geometry.filename == sgii.geometry.filename
        np.testing.assert_allclose(sfs.geometry.volume, sgii.geometry.volume)
        np.testing.assert_allclose(sfs.geometry.voxelsize, sgii.geometry.voxelsize)
        np.testing.assert_allclose(
            sfs.geometry.cras, sgii.geometry.cras, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(sfs.geometry.cosines, sgii.geometry.cosines)

    def test_read_vol_geom_invalid(self, BERT_DIR):
        s = Surface.from_file(BERT_DIR / "surf" / "lh.white.no_volgeom")
        assert not s.geometry.valid


class TestSurfaceSave:
    @pytest.mark.parametrize(
        "filename",
        [
            "lh.white",
            "lh.white.gii",
            "lh.white.no_volgeom",  # has vol geom but valid = False
            "lh.white.no_volgeom.gii",  # has no vol geom
            "lh.white.stripped",  # has no vol geom
        ],
    )
    @pytest.mark.parametrize("ext", ["", ".gii"])
    def test_save(self, BERT_DIR, filename, ext):
        """Test that volume geometry information (whether present or not) is
        correctly preserved upon saving and loading a file.
        """
        s = Surface.from_file(BERT_DIR / "surf" / filename)

        with tempfile.TemporaryDirectory() as tmpdir:
            p = (Path(tmpdir) / "tmpfile").with_suffix(ext)
            s.save(p)
            t = Surface.from_file(p)

        np.testing.assert_allclose(s.vertices, t.vertices)
        np.testing.assert_allclose(s.faces, t.faces)

        assert s.geometry.valid == t.geometry.valid
        if filename == "lh.white.no_volgeom" and ext == ".gii":
            # This file actually has vol geom info but `valid` is set to False.
            # When the vol geom is invalid, we don't write it into the gifti
            # file as otherwise it would be considered valid!
            return
        assert s.geometry.filename == t.geometry.filename
        np.testing.assert_allclose(s.geometry.volume, t.geometry.volume)
        np.testing.assert_allclose(s.geometry.voxelsize, t.geometry.voxelsize)
        np.testing.assert_allclose(s.geometry.cras, t.geometry.cras)
        np.testing.assert_allclose(s.geometry.cosines, t.geometry.cosines)


@pytest.mark.parametrize("method", ["nearest", "linear"])
class TestSphere:
    def test_project_and_resample(self, sphere_reg, method):
        # Project 'source_field' defined on 'fibonacci_sphere' to dest_points.
        rng = np.random.default_rng(0)

        # Generate a smooth field to resample
        source_field = sphere_reg.vertices[:, 0]

        # Generate a point for each face on sphere_reg (to resample the field
        # to)
        weights = rng.uniform(size=(sphere_reg.n_faces, 3))
        weights /= weights.sum(1, keepdims=True)

        # We cheat and use Surface instead of SphericalRegistration to avoid
        # the vertices being projected back on the sphere, hence rendering the
        # comparison to the `weights` array invalid
        n = sphere_reg.n_faces
        # the triangles of the surface being morphed *to* are unused so just
        # generate some random ones
        dest_points = np.sum(sphere_reg.as_mesh() * weights[..., None], 1)
        dest_sphere = Surface(dest_points, rng.integers(1, n, (2 * n - 4, 3)))

        dest_field = sphere_reg.project_and_resample(
            dest_sphere, source_field, method=method
        )

        match method:
            case "nearest":
                _, closest = cKDTree(sphere_reg.vertices).query(dest_sphere.vertices)
                # Test projection
                np.testing.assert_allclose(sphere_reg._mapping_matrix.data, 1)
                # Test sampling
                np.testing.assert_allclose(source_field[closest], dest_field)
            case "linear":
                # The weights in _mapping_matrix are sorted by face index,
                # hence we need to sort the `weights` array before checking
                weights_sorted = weights[
                    np.repeat(np.arange(sphere_reg.n_faces), 3).reshape(-1, 3),
                    sphere_reg.faces.argsort(1),
                ]
                # Test projection
                np.testing.assert_allclose(
                    sphere_reg._mapping_matrix.data, weights_sorted.ravel()
                )
                # Test sampling
                np.testing.assert_allclose(
                    np.sum(source_field[sphere_reg.faces] * weights, 1), dest_field
                )
