from libcpp.vector cimport vector
import numpy as np
import numpy.typing as npt
cimport numpy as np

cdef extern from "helpers.h":
    cdef cppclass V2FIIII:
        vector[vector[float]] vertices
        vector[vector[int]] faces
        vector[vector[int]] cells
        vector[int] faces_pmap
        vector[int] cells_pmap

    cdef struct SurfaceMesh:
        vector[vector[float]] vertices
        vector[vector[int]] faces

    cdef struct VolumeMesh:
        vector[vector[float]] vertices
        vector[vector[int]] faces
        vector[vector[int]] cells

    cdef struct SurfaceMeshWithPMaps:
        vector[vector[float]] vertices
        vector[vector[int]] faces
        vector[int] faces_pmap

    cdef struct VolumeMeshWithPMaps:
        vector[vector[float]] vertices
        vector[vector[int]] faces
        vector[vector[int]] cells
        vector[int] faces_pmap
        vector[int] cells_pmap



