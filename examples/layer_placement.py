import nibabel as nib

from cortech import Hemisphere, Surface
import cortech.freesurfer

if not cortech.freesurfer.HAS_FREESURFER:
    raise ValueError("Could not find `FREESURFER_HOME`.")

subject_dir = cortech.freesurfer.HOME / "subjects" / "bert"

print(f"Using subject in {subject_dir}")

bert = Hemisphere.from_freesurfer_subject_dir(subject_dir, "lh")

# 0. Prerequisites for layer estimation

thickness = bert.compute_thickness()
# Compute the average curvature over white and pial surfaces
curv = bert.compute_average_curvature(curv_kwargs=dict(smooth_iter=10))

# 1. Estimate layers with both methods
# We use the mean curvature (H)

equivol = bert.estimate_layers(thickness, curv.H, 0.5, method="equivolume")
equidist = bert.estimate_layers(thickness, curv.H, 0.5, method="equidistance")

ev = Surface(equivol, bert.white.faces, bert.white.metadata)
ed = Surface(equidist, bert.white.faces, bert.white.metadata)

# nib.freesurfer.write_geometry("lh.equivol", ev.vertices, ev.faces)
# nib.freesurfer.write_geometry("lh.equidist", ed.vertices, ed.faces)

# 2. Estimate layers at several fractions

# equivols has shape (n_frac, n_vertices, coordinates)
# equivols = bert.estimate_layers(thickness, curv.H, [0.25, 0.5, 0.75], method="equivolume")
# for i,v in enumerate(equivols):
#     ev = Surface(v, bert.white.faces, bert.white.metadata)
#     nib.freesurfer.write_geometry(f"lh.equivol.{i}", ev.vertices, ev.faces)


