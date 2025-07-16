# THE FOLLOWING CODE IS TAKEN FROM nibabel.freesurfer.io AND MODIFIED TO ALLOW
# READING FREESURFER SURFACE FILES IN SCANNER RAS (IN ADDITION TO OR SURFACE
# RAS)

from collections import namedtuple, OrderedDict
from enum import IntEnum
import getpass
import time
import warnings

import numpy as np


global METADATA
METADATA = namedtuple("Metadata", ["real_ras", "vol_geom"])


# For all tags, see https://github.com/freesurfer/freesurfer/blob/dev/include/tags.h
class Tag(IntEnum):
    OLD_USEREALRAS = 2
    # TAG_CMDLINE = 3
    OLD_SURF_GEOM = 20


KnownTags = set(i.value for i in Tag)


def _read_tag(fobj):
    """Read a tag."""
    tag = np.fromfile(fobj, ">i4", 1)
    return None if len(tag) == 0 else tag[0]


def _read_userealras_old(fobj):
    return dict(real_ras=bool(np.fromfile(fobj, ">i4", 1)[0]))


def _read_surf_geom_old(fobj):
    """Read volume geometry (in old format) associated with the surface."""
    surf_geom = OrderedDict()
    for key in (
        "valid",
        "filename",
        "volume",
        "voxelsize",
        "xras",
        "yras",
        "zras",
        "cras",
    ):
        pair = fobj.readline().decode("utf-8").split("=")
        if pair[0].strip() != key or len(pair) != 2:
            raise OSError("Error parsing volume info.")
        if key in ("valid", "filename"):
            surf_geom[key] = pair[1].strip()
        elif key == "volume":
            surf_geom[key] = np.array(pair[1].split(), int)
        else:
            surf_geom[key] = np.array(pair[1].split(), float)
    return OrderedDict(vol_geom=surf_geom)


def _read_tag_value(fobj, tag):
    """Read the value associated with a tag."""
    if tag == Tag.OLD_USEREALRAS:
        return _read_userealras_old(fobj)
    elif tag == Tag.OLD_SURF_GEOM:
        return _read_surf_geom_old(fobj)
    else:
        raise ValueError(f"Unable to read tag: {tag}.")


def _fread3(fobj):
    """Read a 3-byte int from an open binary file object

    Parameters
    ----------
    fobj : file
        File descriptor

    Returns
    -------
    n : int
        A 3 byte int
    """
    b1, b2, b3 = np.fromfile(fobj, ">u1", 3).astype(np.int64)
    return (b1 << 16) + (b2 << 8) + b3


def _fread3_many(fobj, n):
    """Read 3-byte ints from an open binary file object.

    Parameters
    ----------
    fobj : file
        File descriptor

    Returns
    -------
    out : 1D array
        An array of 3 byte int
    """
    b1, b2, b3 = np.fromfile(fobj, ">u1", 3 * n).reshape(-1, 3).astype(int).T
    return (b1 << 16) + (b2 << 8) + b3


def _write_bytes(fobj, value, dtype):
    fobj.write(np.asarray(value, dtype=dtype).tobytes())


def _write_real_ras(fobj, value):
    assert value in {False, True}
    _write_bytes(fobj, [Tag.OLD_USEREALRAS, value], ">i4")


def _write_vol_geom_old(fobj, volume_info):
    """Helper for serializing the volume info."""
    _write_bytes(fobj, Tag.OLD_SURF_GEOM, ">i4")

    keys = [
        "valid",
        "filename",
        "volume",
        "voxelsize",
        "xras",
        "yras",
        "zras",
        "cras",
    ]
    diff = set(volume_info.keys()).difference(keys)
    if len(diff) > 0:
        raise ValueError(f"Invalid volume info: {diff.pop()}.")

    for key in keys:
        if key in ("valid", "filename"):
            val = volume_info[key]
            fobj.write(f"{key} = {val}\n".encode())
        elif key == "volume":
            val = volume_info[key]
            fobj.write(f"{key} = {val[0]} {val[1]} {val[2]}\n".encode())
        else:
            val = volume_info[key]
            fobj.write(
                f"{key:6s} = {val[0]:.10g} {val[1]:.10g} {val[2]:.10g}\n".encode()
            )


def _read_metadata(fobj):
    metadata = {}
    while (tag := _read_tag(fobj)) is not None:
        if tag in KnownTags:
            metadata |= _read_tag_value(fobj, tag)
        else:
            # We encountered a tag that we don't know how to read so we have to
            # terminate
            break
    return METADATA(**metadata)


def _write_metadata(fobj, real_ras=None, vol_geom=None):
    if real_ras is not None:
        _write_real_ras(fobj, real_ras)
    if vol_geom is not None:
        _write_vol_geom_old(fobj, vol_geom)
    # CMDLINE


def read_geometry(filepath, read_metadata=False, read_stamp=False):
    """Read a triangular format Freesurfer surface mesh.

    Parameters
    ----------
    filepath : str
        Path to surface file.
    read_metadata : bool, optional
        If True, read and return metadata as key-value pairs.

        Valid keys:

        * 'head' : array of int
        * 'valid' : str
        * 'filename' : str
        * 'volume' : array of int, shape (3,)
        * 'voxelsize' : array of float, shape (3,)
        * 'xras' : array of float, shape (3,)
        * 'yras' : array of float, shape (3,)
        * 'zras' : array of float, shape (3,)
        * 'cras' : array of float, shape (3,)

    read_stamp : bool, optional
        Return the comment from the file

    Returns
    -------
    coords : numpy array
        nvtx x 3 array of vertex (x, y, z) coordinates.
    faces : numpy array
        nfaces x 3 array of defining mesh triangles.
    volume_info : OrderedDict
        Returned only if `read_metadata` is True.  Key-value pairs found in the
        geometry file.
    create_stamp : str
        Returned only if `read_stamp` is True.  The comment added by the
        program that saved the file.
    """
    volume_info = OrderedDict()

    TRIANGLE_MAGIC = 16777214
    QUAD_MAGIC = 16777215
    NEW_QUAD_MAGIC = 16777213
    with open(filepath, "rb") as fobj:
        magic = _fread3(fobj)
        if magic in (QUAD_MAGIC, NEW_QUAD_MAGIC):  # Quad file
            nvert = _fread3(fobj)
            nquad = _fread3(fobj)
            (fmt, div) = (">i2", 100.0) if magic == QUAD_MAGIC else (">f4", 1.0)
            coords = np.fromfile(fobj, fmt, nvert * 3).astype(np.float64) / div
            coords = coords.reshape(-1, 3)
            quads = _fread3_many(fobj, nquad * 4)
            quads = quads.reshape(nquad, 4)
            #
            #   Face splitting follows
            #
            faces = np.zeros((2 * nquad, 3), dtype=int)
            nface = 0
            for quad in quads:
                if (quad[0] % 2) == 0:
                    faces[nface] = quad[0], quad[1], quad[3]
                    nface += 1
                    faces[nface] = quad[2], quad[3], quad[1]
                    nface += 1
                else:
                    faces[nface] = quad[0], quad[1], quad[2]
                    nface += 1
                    faces[nface] = quad[0], quad[2], quad[3]
                    nface += 1

        elif magic == TRIANGLE_MAGIC:  # Triangle file
            create_stamp = fobj.readline().rstrip(b"\n").decode("utf-8")
            fobj.readline()
            vnum = np.fromfile(fobj, ">i4", 1)[0]
            fnum = np.fromfile(fobj, ">i4", 1)[0]
            coords = np.fromfile(fobj, ">f4", vnum * 3).reshape(vnum, 3)
            faces = np.fromfile(fobj, ">i4", fnum * 3).reshape(fnum, 3)

            if read_metadata:
                volume_info = _read_metadata(fobj)
        else:
            raise ValueError("File does not appear to be a Freesurfer surface")

    coords = coords.astype(np.float64)  # XXX: due to mayavi bug on mac 32bits

    ret = (coords, faces)
    if read_metadata:
        if len(volume_info) == 0:
            warnings.warn("No volume information contained in the file")
        ret += (volume_info,)
    if read_stamp:
        ret += (create_stamp,)

    return ret


def write_geometry(
    filepath, coords, faces, create_stamp=None, *, real_ras=None, vol_geom=None
):
    """Write a triangular format Freesurfer surface mesh.

    Parameters
    ----------
    filepath : str
        Path to surface file.
    coords : numpy array
        nvtx x 3 array of vertex (x, y, z) coordinates.
    faces : numpy array
        nfaces x 3 array of defining mesh triangles.
    create_stamp : str, optional
        User/time stamp (default: "created by <user> on <ctime>")
    volume_info : dict-like or None, optional
        Key-value pairs to encode at the end of the file.

        Valid keys:

        * 'head' : array of int
        * 'valid' : str
        * 'filename' : str
        * 'volume' : array of int, shape (3,)
        * 'voxelsize' : array of float, shape (3,)
        * 'xras' : array of float, shape (3,)
        * 'yras' : array of float, shape (3,)
        * 'zras' : array of float, shape (3,)
        * 'cras' : array of float, shape (3,)

    """
    magic_bytes = np.array([255, 255, 254], dtype=np.uint8)

    if create_stamp is None:
        create_stamp = f"created by {getpass.getuser()} on {time.ctime()}"

    with open(filepath, "wb") as fobj:
        magic_bytes.tofile(fobj)
        fobj.write((f"{create_stamp}\n\n").encode())

        np.array([coords.shape[0], faces.shape[0]], dtype=">i4").tofile(fobj)

        # Coerce types, just to be safe
        coords.astype(">f4").reshape(-1).tofile(fobj)
        faces.astype(">i4").reshape(-1).tofile(fobj)

        _write_metadata(fobj, real_ras, vol_geom)
