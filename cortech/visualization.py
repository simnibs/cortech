import nibabel as nib
import numpy as np
import pyvista as pv

import cortech.freesurfer

class SurfaceVisualizer:
    def __init__(self, subject: str = "fsaverage"):
        """A class that provides access to the FsAverage surface files included
        with SimNIBS.

        fsavg40 = SurfaceVisualizer("fsaverage")
        fsavg40.get_morph_data("thickness")
        fsavg40.get_annotation("Yeo2011_17Networks_N1000")

        PARAMETERS
        ----------
        resolution : int
            The FsAverage resolution factor. Available resolutions are
            10, 40, 160 denoting the approximate number of thousands of
            vertices per hemisphere, e.g., 160 has ~160,000 vertices per
            hemisphere and correspondings to the default `fsaverage` template
            (default = 160).
        """
        assert cortech.freesurfer.HOME is not None, "Could not find FREESURFER_HOME"

        self.subject_dir = cortech.freesurfer.HOME / "subjects" / subject
        self.subpaths = {
            "surface": self.subject_dir / "surf",
            "morph_data": self.subject_dir / "surf",
            "annot": self.subject_dir / "label",
        }

    def _get_files(self, what, path):
        return {h: self.subpaths[path] / ".".join((h, what)) for h in cortech.freesurfer.HEMISPHERES}

    def get_surface(self, surface):
        assert surface in cortech.freesurfer.GEOMETRY, f"{surface} is not a valid surface."
        files = self._get_files(surface, "surface")
        return {
            k: dict(zip(("points", "tris"), nib.freesurfer.read_geometry(v)))
            for k, v in files.items()
        }

    def get_morph_data(self, data):
        assert data in cortech.freesurfer.MORPH_DATA, f"{data} is not a valid morph data type."
        files = self._get_files(data, "morph_data")
        return {k: nib.freesurfer.read_morph_data(v) for k, v in files.items()}

    def get_annotation(self, annot):
        # only aparc and aparc.a2009s are available for fsaverage5 and fsaverage6
        assert annot in cortech.freesurfer.ANNOT, f"{annot} is not a valid annotation."

        files = self._get_files(f"{annot}.annot", "annot")
        keys = ("labels", "ctab", "names")
        return {
            k: dict(zip(keys, nib.freesurfer.read_annot(v))) for k, v in files.items()
        }


class FsAveragePlotter:
    def __init__(self, subject: str = "fsaverage", surface: str = "inflated"):
        self.fsavg = SurfaceVisualizer(subject)
        self.brain = self.surface_as_multiblock(surface)
        if surface == "inflated":
            self.brain["lh"].points[:, 0] -= np.abs(self.brain["lh"].points[:, 0].max())
            self.brain["rh"].points[:, 0] += np.abs(self.brain["rh"].points[:, 0].min())
        self.overlays = self.brain.copy()

    def surface_as_multiblock(self, surface):
        """Return the specified surface as a PyVista MultiBlock object."""
        surf = self.fsavg.get_surface(surface)
        mb = pv.MultiBlock()
        for hemi in surf:
            mb[hemi] = pv.make_tri_mesh(surf[hemi]["points"], surf[hemi]["tris"])
        return mb

    def add_curvature(self):
        curv = self.fsavg.get_morph_data("curv")
        for h in curv:
            self.brain[h]["curv"] = curv[h].astype(float)
            self.brain[h]["curv_bin"] = np.where(curv[h] > 0, 1 / 3, 2 / 3)

    def add_overlay(self, data, name: str):
        for h in cortech.freesurfer.HEMISPHERES:
            self.overlays[h][name] = data[h].astype(float)

    def remove_overlay(self, name: str):
        for h in cortech.freesurfer.HEMISPHERES:
            self.overlays[h].point_data.remove(name)

    def remove_all_overlays(self):
        for h in cortech.freesurfer.HEMISPHERES:
            self.overlays[h].clear_data()

    def set_active_overlay(self, name: str):
        assert name is not None
        for h in cortech.freesurfer.HEMISPHERES:
            self.overlays[h].set_active_scalars(name)

    def apply_threshold(self, threshold, use_abs: bool = False):
        overlay_thres = pv.MultiBlock()
        for h in cortech.freesurfer.HEMISPHERES:
            point_data = self.overlays[h].active_scalars
            cell_data = point_data[self.brain[h].faces.reshape(-1, 4)[:, 1:]].mean(-1)
            cell_data = np.abs(cell_data) if use_abs else cell_data
            overlay_thres[h] = self.overlays[h].remove_cells(cell_data < threshold)
        return overlay_thres

    def plot(
        self,
        overlay=None,
        threshold=None,
        use_abs=False,
        name=None,
        brain_kwargs=None,
        overlay_kwargs=None,
        plotter_kwargs=None,
        plotter=None,
    ):
        name = name or "temporary scalars"
        kw_brain = {}  # dict(
        #            scalars="curv_bin", cmap="gray", clim=(0, 1), show_scalar_bar=False
        #        )
        if brain_kwargs:
            kw_brain.update(brain_kwargs)
        kw_overlay = dict(
            annotations={threshold: "Threshold"} if threshold else None,
            # cmap="jet" if use_abs else "hot",
            # clim=None if use_abs else (0, None),
        )
        if overlay_kwargs:
            kw_overlay.update(overlay_kwargs)
        kw_plotter = plotter_kwargs or {}

        p = plotter or pv.Plotter(**kw_plotter)
        # Only plot background mesh if there is a "texture" or the overlay is
        # thresholded or there is no overlay
        if (
            all(h.active_scalars is not None for h in self.brain)
            or threshold
            or overlay is None
        ):
            p.add_mesh(self.brain, **kw_brain)

        if overlay is not None:
            if isinstance(overlay, str):
                self.set_active_overlay(overlay)
                remove_overlay = False
            else:
                # Temporarily add as overlay
                self.add_overlay(overlay, name)
                self.set_active_overlay(name)
                remove_overlay = True

            over = (
                self.apply_threshold(threshold, use_abs)
                if threshold
                else self.overlays.copy()
            )
            p.add_mesh(over, **kw_overlay)

            if remove_overlay:
                self.remove_overlay(name)

        p.view_xy(True)
        # p.camera.zoom(np.sqrt(2))
        p.camera.zoom(1.3)

        return p

def plot_surface(
        surface, scalars=None, plotter=None, mesh_kwargs=None, plotter_kwargs=None
    ):
    mesh_kwargs = mesh_kwargs or {}
    plotter_kwargs = plotter_kwargs or {}

    # if not isinstance(pv.PolyData):
    mesh = pv.make_tri_mesh(surface.vertices, surface.faces)

    p = plotter or pv.Plotter(**plotter_kwargs)
    p.add_mesh(mesh, scalars=scalars, **mesh_kwargs)
    # p.view_xy(True)
    # p.camera.zoom(1.3)
    p.show()
