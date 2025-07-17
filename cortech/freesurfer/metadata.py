from collections import OrderedDict

import nibabel as nib
import numpy as np
import numpy.typing as npt


class VolumeGeometry:
    def __init__(
        self,
        valid: bool,
        filename: str | None = None,
        volume: npt.ArrayLike | None = None,
        voxelsize: npt.ArrayLike | None = None,
        xras: npt.ArrayLike | None = None,
        yras: npt.ArrayLike | None = None,
        zras: npt.ArrayLike | None = None,
        cras: npt.ArrayLike | None = None,
        cosines: npt.ArrayLike | None = None,
    ):
        """FreeSurfer volume geometry information."""
        assert valid in {False, True}, f"`valid` must be True or False (got {valid})"
        self.valid = valid
        self.filename = filename
        if self.filename is not None:
            assert isinstance(self.filename, str)
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
            self.cosines = self._cosines_from_xyz(xras, yras, zras)
        else:
            self.cosines = cosines

        if cras is None:
            if self.valid:
                raise ValueError(
                    "VolumeGeometry was set to as `valid` but cras was not specified."
                )
            cras = np.zeros(3)
        self.cras = np.asarray(cras)

        self.tkrcosines = np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]])

        self._niistring_to_fsstring = dict(
            NIFTI_XFORM_UNKNOWN="tkr", NIFTI_XFORM_SCANNER_ANAT="scanner"
        )
        self._fsstring_to_niistring = {
            v: k for k, v in self._niistring_to_fsstring.items()
        }

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
        if self.filename is not None:
            d["VolGeomFname"] = self.filename
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
                    d[f"VolGeom{ax0[0].upper()}_{ax1}"] = v[i]
        if self.cras is not None:
            for i, ax1 in enumerate("RAS"):
                d[f"VolGeomC_{ax1}"] = v[i]
            # SurfaceCenterX = ,
            # SurfaceCenterY = ,
            # SurfaceCenterZ = ,
        return d

    def as_freesurfer_dict(self):
        d = OrderedDict(valid=self.valid)
        if self.filename is not None:
            d["filename"] = self.filename
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


def read_metadata_gifti(gii: nib.GiftiImage):
    """Read metadata associated with a gifti image.

    vertex_data, face_data, and extra_data should be specified as a mapping of
    (key to read, parser) where parser is used to interpret the value found
    in the metadata of the gifti image.

    Parameters
    ----------
    gii : nib.GiftiImage
        _description_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """

    vertices = gii.darrays[0]
    # faces = gii.darrays[1]

    # Read real_ras
    coordsys = vertices.coordsys
    # We ignore these. They can be derived from the volume geometry
    # xform = coordsys.xform
    # xformspace = coordsys.xformspace
    dataspace = coordsys.dataspace
    # nib.nifti1.xform_codes.label[coordsys]
    nii_code = nib.nifti1.xform_codes.niistring[dataspace]
    match nii_code:
        case "NIFTI_XFORM_UNKNOWN":
            # assert xformspace == "NIFTI_XFORM_SCANNER_ANAT"
            space = "surface"
        case "NIFTI_XFORM_SCANNER_ANAT":
            # assert xformspace == "NIFTI_XFORM_UNKNOWN"
            space = "scanner"
        case _:
            raise ValueError(f"Unknown dataspace {nii_code}")

    # Read the volume geometry
    m = vertices.meta
    n_fields = 0
    vol_geom = {}

    if "VolGeomFname" in m:
        vol_geom["filename"] = m["VolGeomFname"]
        n_fields += 1

    try:
        vol_geom["volume"] = np.array(
            [int(m[f"VolGeom{k}"]) for k in ("Width", "Height", "Depth")]
        )
        n_fields += 3
    except KeyError:
        pass

    try:
        vol_geom["voxelsize"] = np.array(
            [float(m[f"VolGeom{k}size"]) for k in ("X", "Y", "Z")]
        )
        n_fields += 3
    except KeyError:
        pass

    try:
        for i in "XYZC":
            vol_geom[f"{i.lower()}ras"] = np.array(
                [float(m[f"VolGeom{i}_{k}"]) for k in "RAS"]
            )
        n_fields += 12
    except KeyError:
        pass

    # This is how validity of the volume geometry is determined in FreeSurfer
    # when reading a gifti file
    # https://github.com/freesurfer/freesurfer/blob/920f33cade45b901f702192ace64b37ef2c4b3e1/utils/gifti.cpp#L645
    vol_geom["valid"] = n_fields == 19

    return space, VolumeGeometry(**vol_geom)
