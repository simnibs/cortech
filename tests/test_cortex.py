import numpy as np
import pytest

from cortech import Cortex, Hemisphere, Surface


class TestHemisphere:
    def test_init(self, diamond_vertices, diamond_faces):
        name = "lh"
        white = Surface(diamond_vertices, diamond_faces)
        pial = Surface(diamond_vertices, diamond_faces)
        hemi = Hemisphere(name, white, pial)

        np.testing.assert_allclose(hemi.white.vertices, white.vertices)
        np.testing.assert_allclose(hemi.white.vertices, pial.vertices)
        assert hemi.name == name
        assert not hemi.has_registration()

    @pytest.mark.parametrize("hemi", ["lh", "rh"])
    def test_from_freesurfer_subject_dir(self, BERT_DIR, hemi):
        hemi = Hemisphere.from_freesurfer_subject_dir(BERT_DIR, hemi)

        assert hemi.white.n_vertices == 2562
        assert hemi.white.n_faces == 5120
        assert hemi.pial.n_vertices == 2562
        assert hemi.pial.n_faces == 5120


class TestCortex:
    def test_from_freesurfer_subject_dir(self, BERT_DIR):
        cortex = Cortex.from_freesurfer_subject_dir(BERT_DIR)

        for hemi in cortex:
            assert hemi.white.n_vertices == 2562
            assert hemi.white.n_faces == 5120
            assert hemi.pial.n_vertices == 2562
            assert hemi.pial.n_faces == 5120
