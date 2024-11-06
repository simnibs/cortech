import pyvista as pv

from cortech.surface import Surface
from cortech.cortex import Hemisphere

# setup lh of a subject
sub04_lh = Hemisphere.from_freesurfer_subject_dir(
    "/home/jesperdn/INN_JESPER/nobackup/projects/anateeg/freesurfer/sub-04",
    "lh",
)

bert = Hemisphere.from_freesurfer_subject_dir(
    "/mnt/depot64/freesurfer/freesurfer.7.4.0/subjects/bert",
    "lh",
)

knn, kr = bert.white.k_ring_neighbors(5)


v,f = nib.freesurfer.read_geometry("/mnt/scratch/personal/jesperdn/topofit-ours/ABIDE/sub-0014/lh.white.pred")
s = Surface(v,f)
curv = s.compute_curvature(smooth_iter=10)

v,f = nib.freesurfer.read_geometry("/mnt/projects/CORTECH/nobackup/training_data/full/ABIDE/sub-0014/lh.white")
st = Surface(v,f)
curvt = s.compute_curvature(smooth_iter=10)


v,f = nib.freesurfer.read_geometry("/mrhome/jesperdn/repositories/brainsynth/brainsynth/resources/sphere-reg.srf")
s = Surface(v,f)
s.save("test.vtk", scalars=dict(true=curvt.H, pred=curv.H))


import cortech.cgal.aabb_tree

b = bert.white.compute_face_barycenters()
n = bert.white.compute_face_normals()

points = b[:1000] + 0.1 * n[:1000]
v = bert.white.vertices
f = bert.white.faces

x = cortech.cgal.aabb_tree.distance(v,f,points)

curv = bert.compute_average_curvature(curv_kwargs=dict(smooth_iter=5))
thickness = bert.compute_thickness()

equivol = bert.place_layers(thickness, curv.H, 0.5, method="equi-volume")
equidist = bert.place_layers(thickness, curv.H, 0.5, method="equi-distance")

ev = Surface(equivol, bert.white.faces, bert.white.metadata)
ed = Surface(equidist, bert.white.faces, bert.white.metadata)

nib.freesurfer.write_geometry("/mrhome/jesperdn/nobackup/bert_equivol", ev.vertices, ev.faces)
nib.freesurfer.write_geometry("/mrhome/jesperdn/nobackup/bert_equidist", ed.vertices, ed.faces)


import cortech.cgal.polygon_mesh_processing as pmp


x = pmp.points_inside_surface(bert.white.vertices, bert.white.faces, bert.pial.vertices)
# y = pmp.points_inside_surface(bert.white.vertices, bert.white.faces, bert.pial.vertices, False)



hemi = "rh"

v,f = nib.freesurfer.read_geometry("/mrhome/jesperdn/repositories/brainsynth/brainsynth/resources/sphere-reg.srf")
topo = SphericalRegistration(v,f)

v,f = nib.freesurfer.read_geometry(f"/mnt/depot64/freesurfer/freesurfer.7.4.1/subjects/fsaverage/surf/{hemi}.sphere")
fsavg = SphericalRegistration(v,f)
label, color, name = nib.freesurfer.read_annot(f"/mnt/depot64/freesurfer/freesurfer.7.4.1/subjects/fsaverage/label/{hemi}.aparc.annot")

is_medial_wall = label == -1

fsavg.compute_projection(topo, method="nearest")
out = fsavg.resample(is_medial_wall)
out = out.astype(is_medial_wall.dtype)

np.save("/home/jesperdn/nobackup/medial_wall.npy", out)

outi = np.where(out)[0]

v,f = nib.freesurfer.read_geometry("/mrhome/jesperdn/repositories/brainsynth/brainsynth/resources/cortex-int-lh.srf")
v[:, 0] *= -1
f = f[:, (0,2,1)]
s = Surface(v,f)
s.plot(out, plotter_kwargs=dict(notebook=False))



v,f = nib.freesurfer.read_geometry("/mrhome/jesperdn/repositories/brainsynth/brainsynth/resources/cortex-int-lh.srf")
s = Surface(v,f)
s.plot(out, plotter_kwargs=dict(notebook=False))

# Visualization

# visualize with kwargs
curv = bert.white.compute_curvature()
curv1 = bert.white.compute_curvature(smooth_iter=10)
bert.white.plot(
    curv.H,
    mesh_kwargs=dict(show_edges=True),
    plotter_kwargs=dict(notebook=False)
)


# Smoothing

v = bert.white.gaussian_smooth(n_iter=25)
gs = Surface(v, bert.white.faces)
gs.plot()

v = bert.white.taubin_smooth(n_iter=25)
ts = Surface(v, bert.white.faces)
ts.plot()

# ---

sub04_lh.white.remove_self_intersections()
sub04_lh.pial.remove_self_intersections()

# sub04_lh.decouple_brain_surfaces()

wm_curv = sub04_lh.white.compute_curvature()
wm_curv = sub04_lh.white.compute_curvature(smooth_iter=10)

pial_curv = sub04_lh.pial.compute_curvature()
pial_curv = sub04_lh.pial.compute_curvature(smooth_iter=10)

# compute the average curvature (white/2 + pial/2)
curv = sub04_lh.compute_average_curvature(curv_kwargs=dict(smooth_iter=10))

# setup lh of a fsaverage
# 'fsaverage' is special; it just grabs from $FS_HOME/subjects/fsaverage
fsavg_lh = Hemisphere.from_freesurfer_subject_dir("fsaverage", "lh")

# compute projection from subject to fsaverage. This is stored internally
sub04_lh.spherical_registration.compute_projection(fsavg_lh.spherical_registration)
# apply projection
resampled_white_H = sub04_lh.spherical_registration.resample(white_H)
resampled_pial_H = sub04_lh.spherical_registration.resample(pial_H)

# Visualize

# show on subject

m = pv.make_tri_mesh(sub04_lh.white.vertices, sub04_lh.white.faces)
m["H"] = white_H
m.plot(show_edges=False)

m = pv.make_tri_mesh(sub04_lh.pial.vertices, sub04_lh.pial.faces)
m["H"] = pial_H
m.plot(show_edges=False)

# show on fsaverage

q = pv.make_tri_mesh(fsavg_lh.white.vertices, fsavg_lh.white.faces)
q["H"] = resampled_white_H
q.plot()

q = pv.make_tri_mesh(fsavg_lh.pial.vertices, fsavg_lh.pial.faces)
q["H"] = resampled_pial_H
q.plot()

# show on sphere.reg

q = pv.make_tri_mesh(
    fsavg_lh.spherical_registration.vertices,
    fsavg_lh.spherical_registration.faces
)
q["H"] = resampled_white_H
q.plot()

q = pv.make_tri_mesh(
    fsavg_lh.spherical_registration.vertices,
    fsavg_lh.spherical_registration.faces
)
q["H"] = resampled_pial_H
q.plot()
