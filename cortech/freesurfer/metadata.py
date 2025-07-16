from collections import OrderedDict
import warnings

import nibabel as nib
import numpy as np
import numpy.typing as npt


class VolumeGeometry:
    def __init__(
        self,
        valid: bool,
        filename: str,
        volume: npt.ArrayLike | None = None,
        voxelsize: npt.ArrayLike | None = None,
        xras: npt.ArrayLike | None = None,
        yras: npt.ArrayLike | None = None,
        zras: npt.ArrayLike | None = None,
        cras: npt.ArrayLike | None = None,
        cosines: npt.ArrayLike | None = None,
    ):
        """FreeSurfer volume geometry information."""
        assert valid in {False, True}
        self.valid = valid
        self.filename = filename
        self.volume = volume
        self.voxelsize = voxelsize

        if cosines is None:
            if any([xras is None, yras is None, zras is None]):
                if self.valid:
                    raise ValueError(
                        "VolumeGeometry was set to as `valid` but x/y/zras was not specified."
                    )
                xras = np.array([-1.0, 0.0, 0.0])
                yras = np.array([0.0, 0.0, -1.0])
                zras = np.array([0.0, 1.0, 0.0])
            cosines = self._cosines_from_xyz(xras, yras, zras)
        self.cosines = cosines

        if cras is None:
            if self.valid:
                raise ValueError(
                    "VolumeGeometry was set to as `valid` but cras was not specified."
                )
            cras = np.zeros(3)
        self.cras = np.asarray(cras)

        self.tkrcosines = np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]])

    @staticmethod
    def _cosines_from_xyz(xras, yras, zras) -> npt.NDArray:
        return np.column_stack([xras, yras, zras])

    @staticmethod
    def _xyz_from_cosines(cosines) -> dict[str, npt.NDArray]:
        return dict(xras=cosines[:, 0], yras=cosines[:, 1], zras=cosines[:, 2])

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, value):
        if value is not None:
            assert len(value) == 3
            self._volume = np.asarray(value, dtype=int)
        else:
            self._volume = np.array([256, 256, 256])

    @property
    def voxelsize(self):
        return self._voxelsize

    @voxelsize.setter
    def voxelsize(self, value):
        if value is not None:
            assert len(value) == 3
            self._voxelsize = np.asarray(value, dtype=int)
        else:
            self._voxelsize = np.ones(3)

    def get_affine_vox2space(self, space):
        match space:
            case "tkr" | "tkreg" | "tkregister" | "surface":
                mat = self.tkrcosines * self.voxelsize
                trans = nib.affines.from_matvec(mat, -mat @ self.volume / 2)
            case "scanner" | "ras":
                mat = self.cosines * self.voxelsize
                trans = nib.affines.from_matvec(mat, self.cras - mat @ self.volume / 2)
            case "voxel":
                trans = np.eye(4)
            case _:
                raise ValueError(f"Invalid space: {space}")
        return trans

    def get_affine(self, to: str, *, fr: str = "voxel"):
        """Get affine transformations for between voxel and world spaces.
        Valid word spaces are

            Native scanner space    (scanner, ras)
            Native FreeSurfer space (tkr, tkreg, tkregister, surface)

        Parameters
        ----------
        to : str
            Space to transform to.
        fr : str, optional
            Space to transform from (default = voxel).

        Returns
        -------
        trans
            Transformation from `fr` to `to`.

        Raises
        ------
        ValueError
            When to/fr is invalid.
        """
        vox2to = self.get_affine_vox2space(to)
        fr2vox = np.linalg.inv(self.get_affine_vox2space(fr))
        return vox2to @ fr2vox

    def as_gifti_dict(self):
        d = {}
        if self.volume is not None:
            d["VolGeomWidth"] = self.volume[0]
            d["VolGeomHeight"] = self.volume[1]
            d["VolGeomDepth"] = self.volume[2]
        if self.voxelsize is not None:
            d["VolGeomXsize"] = self.voxelsize[0]
            d["VolGeomYsize"] = self.voxelsize[1]
            d["VolGeomZsize"] = self.voxelsize[2]
        if self.cosines is not None:
            ras = self._xyz_from_cosines(self.cosines)
            for ax0, v in ras.items():
                for i, ax1 in enumerate("RAS"):
                    d[f"VolGeom{ax0.upper()}_{ax1}"] = v[i]
        if self.cras is not None:
            for i, ax1 in enumerate("RAS"):
                d[f"VolGeomC_{ax1}"] = v[i]
            # SurfaceCenterX = ,
            # SurfaceCenterY = ,
            # SurfaceCenterZ = ,
        return d

    def as_freesurfer_dict(self):
        d = OrderedDict(
            valid=str(int(self.valid)),
            filename=str(self.filename),
        )
        if self.volume is not None:
            d["volume"] = self.volume
        if self.voxelsize is not None:
            d["voxelsize"] = self.voxelsize
        if self.cosines is not None:
            d |= self._xyz_from_cosines(self.cosines)
        if self.cras is not None:
            d["cras"] = self.cras
        return d

    @classmethod
    def from_freesurfer_metadata_dict(cls, meta):
        inputs = {
            k.lstrip("VolGeom"): v for k, v in meta.items() if k.startswith("VolGeom")
        }
        return cls(**inputs)


class MetaData:
    def __init__(
        self,
        real_ras: bool = True,
        geometry: dict | VolumeGeometry | None = None,
    ):
        """FreeSurfer metadata."""
        self.real_ras = real_ras
        self.geometry = geometry

    @property
    def geometry(self):
        return self._geometry

    @geometry.setter
    def geometry(self, value):
        if isinstance(value, dict):
            self._geometry = VolumeGeometry(**value)
        elif isinstance(value, VolumeGeometry):
            self._geometry = value
        elif value is None:
            self._geometry = VolumeGeometry(False, "")
        else:
            raise ValueError("Invalid geometry")

    def is_scanner_ras(self):
        return self.real_ras

    def is_surface_ras(self):
        return not self.real_ras

    # def as_freesurfer_dict(self):
    #     meta = dict(
    #         head=np.array(
    #             [
    #                 cortech.freesurfer.Tag.OLD_USEREALRAS,
    #                 self.real_ras,
    #                 cortech.freesurfer.Tag.OLD_SURF_GEOM,
    #             ],
    #             dtype=np.int32,
    #         )
    #     )
    #     return meta | self.geometry.as_freesurfer_dict()
