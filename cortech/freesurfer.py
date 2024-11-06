import os
from pathlib import Path

if "FREESURFER_HOME" in os.environ:
    HAS_FREESURFER = True
    HOME = Path(os.environ["FREESURFER_HOME"])
else:
    HAS_FREESURFER = False
    HOME = Path()

HEMISPHERES = {"lh", "rh"}
MORPH_DATA = {"area", "curv", "sulc", "thickness"}
GEOMETRY = {"inflated", "pial", "sphere", "white"}
ANNOT = {
    "aparc",
    "aparc.a2005s",
    "aparc.a2009s",
    "oasis.chubs",
    "PALS_B12_Brodmann",
    "PALS_B12_Lobes",
    "PALS_B12_OrbitoFrontal",
    "PALS_B12_Visuotopic",
    "Yeo2011_7Networks_N1000",
    "Yeo2011_17Networks_N1000",
}
