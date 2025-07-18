import numpy as np

from cortech.freesurfer import VolumeGeometry


class TestVolumeGeometry:
    def test_init_from_xyz_and_cos(self, VOL_GEOM):
        keys = {"valid", "filename", "volume", "voxelsize", "cras"}
        keys_cos = keys | {"cosines"}
        keys_xyz = keys | {"xras", "yras", "zras"}

        geom_cos = VolumeGeometry(**{k: VOL_GEOM[k] for k in keys_cos})
        geom_xyz = VolumeGeometry(**{k: VOL_GEOM[k] for k in keys_xyz})

        np.testing.assert_allclose(geom_cos.cosines, geom_xyz.cosines)

    def test_get_affine(self, VOL_GEOM):
        keys = {"valid", "filename", "volume", "voxelsize", "cras", "cosines"}

        geometry = VolumeGeometry(**{k: VOL_GEOM[k] for k in keys})

        # vox2ras
        affine = geometry.get_affine("scanner")
        np.testing.assert_allclose(affine, VOL_GEOM["vox2ras"], atol=1e-5)

        # vox2tkr
        affine = geometry.get_affine("surface")
        np.testing.assert_allclose(affine, VOL_GEOM["vox2ras_tkr"], atol=1e-5)

        # tkr2ras
        affine = geometry.get_affine("scanner", fr="surface")
        np.testing.assert_allclose(
            affine,
            VOL_GEOM["vox2ras"] @ np.linalg.inv(VOL_GEOM["vox2ras_tkr"]),
            rtol=1e-5,
            atol=1e-5,
        )

        # ras2tkr
        affine = geometry.get_affine("surface", fr="scanner")
        np.testing.assert_allclose(
            affine,
            VOL_GEOM["vox2ras_tkr"] @ np.linalg.inv(VOL_GEOM["vox2ras"]),
            rtol=1e-5,
            atol=1e-5,
        )

        # ras2ras (= I)
        affine = geometry.get_affine("scanner", fr="scanner")
        np.testing.assert_allclose(affine, np.eye(*affine.shape), rtol=1e-5, atol=1e-5)
