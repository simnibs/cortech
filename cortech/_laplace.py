import subprocess
import time

import numpy as np
from scipy.spatial import cKDTree, ConvexHull

import simnibs
from simnibs.mesh_tools import cython_msh

import simnibs.mesh_tools.cgal.polygon_mesh_processing as pmp


# hemisphere object (white, pial surfaces)
# -> correct self-intersections in each surface
# -> decouple white and pial surfaces
# -> produce volume mesh
# ->


geo = """Mesh.Algorithm3D=4;
Mesh.Optimize=0;
Mesh.OptimizeNetgen=0;

Merge "/home/jesperdn/nobackup/bigbrain/lh.white.decouple.smooth.stl";
Merge "/home/jesperdn/nobackup/bigbrain/lh.pial.decouple.smooth.stl";

Surface Loop(1) = {1}; // white
Surface Loop(2) = {2}; // pial

Volume(1) = {1, 2};    // skull (outside brain, inside skull)

// LHS: target surface region number, RHS: surface number (i.e. from merge ...)
Physical Surface(1001) = {1};
Physical Surface(1002) = {2};

// LHS: target volume region number, RHS: volume number
Physical Volume(2) = {1};
"""

filename = "/home/jesperdn/nobackup/bigbrain/brain.geo"
with open(filename, "w") as f:
    f.write(geo)

gmsh = "/home/jesperdn/repositories/simnibs/simnibs/external/bin/linux/gmsh"
cmd = f"{gmsh} -bin -3 -format msh22 -o /home/jesperdn/nobackup/bb_lh_vol.msh {filename}"
subprocess.run(cmd.split())

# run mmg to remesh
filename_in = "/home/jesperdn/nobackup/bb_lh_vol.msh"
filename_out = "/home/jesperdn/nobackup/bb_lh_vol_remesh.msh"
cmd = f"mmg3d_O3 -v 6 -nosurf -rmc -in {filename_in} -out {filename_out}"
subprocess.run(cmd.split())

mesh = simnibs.mesh_tools.mesh_io.read_msh(filename_out)
mesh_tets = mesh.crop_mesh(elm_type=4)

wm_pos = mesh.nodes.node_coord[np.unique(mesh.elm.node_number_list[mesh.elm.tag1==1001, :3]) - 1]
gm_pos = mesh.nodes.node_coord[np.unique(mesh.elm.node_number_list[mesh.elm.tag1==1002, :3]) - 1]

wmi = np.unique(mesh.elm.node_number_list[mesh.elm.tag1==1001, :3])
gmi = np.unique(mesh.elm.node_number_list[mesh.elm.tag1==1002, :3])


tree = cKDTree(mesh_tets.nodes.node_coord)
wmd,wmi = tree.query(wm_pos)
gmd,gmi = tree.query(gm_pos)

wm_vertices = wmi
gm_vertices = gmi

mesh_vol = prepare_for_field_line_tracing(mesh)

from scipy.spatial import ConvexHull
hull = ConvexHull(bert.pial.vertices)
s = Surface(hull.points, hull.simplices)
s.prune()

hull = bert.pial.convex_hull()
bert.pial.n_vertices

v,f = hull.isotropic_remeshing(target_edge_length=1.0)

hull_iso = Surface(v,f)
n = hull_iso.compute_vertex_normals()

hull_iso.plot(mesh_kwargs=dict(show_edges=True), plotter_kwargs=dict(notebook=False))
hull_iso.vertices += 5.0 * n

pv.make_tri_mesh(hull_iso.vertices, hull_iso.faces).save("/home/jesperdn/nobackup/hulliso.vtk")
pv.make_tri_mesh(bert.pial.vertices, bert.pial.faces).save("/home/jesperdn/nobackup/bert.vtk")


bert.pial.plot(x, plotter_kwargs=dict(notebook=False))

m = pv.make_tri_mesh(hull.points, hull.simplices)
m.plot(notebook=False)




from simnibs.mesh_tools.meshing import _mesh_surfaces
from simnibs.mesh_tools.mesh_io import make_surface_mesh

v = np.concatenate((bert.white.vertices, bert.pial.vertices))
f = np.concatenate((bert.white.faces, bert.pial.faces + bert.white.n_vertices))
s = Surface(v,f)
ii = s.self_intersections()

white = pv.make_tri_mesh(bert.white.vertices, bert.white.faces)
white.save("/home/jesperdn/nobackup/inner.stl")

bbox = np.stack((bert.pial.vertices.min(0) - 5.0, bert.pial.vertices.max(0) + 5.0))
BOX = pv.Box(np.array(bbox.T.ravel()), level=25).triangulate().smooth(100)
BOX.save("/home/jesperdn/nobackup/outer.stl")


# bbox = np.stack((bert.pial.vertices.min(0) + 10.0, bert.pial.vertices.max(0) - 10.0))
# BOX = pv.Box(np.array(bbox.T.ravel()), level=25).triangulate().smooth(100)
# BOX.save("/home/jesperdn/nobackup/inner.stl")

surfaces = [
    make_surface_mesh(bert.white.vertices, bert.pial.faces + 1),
    # make_surface_mesh(bert.pial.vertices, bert.pial.faces + 1),
]
mesh = _mesh_surfaces(
        surfaces,
        subdomains = [(1,0)],#[(1,2), (2,0)],
        facet_angle = 30,
        facet_size = 10,
        facet_distance = 0.1,
        cell_radius_edge_ratio = 2,
        cell_size = 10,
        optimize = False,
    )
mesh.write()



def volume_mesh(self):

    optimize = False

    surfaces = [
        make_surface_mesh(self.white.vertices, self.white.faces + 1),
        make_surface_mesh(self.pial.vertices, self.pial.faces + 1),
    ]

    return _mesh_surfaces(
        surfaces,
        subdomains = [(1,2), (2,0)],
        facet_angle = 30,
        facet_size = 10,
        facet_distance = 0.1,
        cell_radius_edge_ratio = 2,
        cell_size = 10,
        optimize = optimize,
    )


abs_potential_diff = 1000
start_vertices =  np.unique(mesh.elm.node_number_list[mesh.elm.tag1==1002, :3])-1

faces = mesh.elm.node_number_list[mesh.elm.tag1==1002, :3]-1

vertices_used = np.unique(faces)
reindexer = np.zeros(len(mesh.nodes.node_coord), dtype=faces.dtype)
reindexer[vertices_used] = np.arange(vertices_used.size, dtype=faces.dtype)

vertices = mesh.nodes.node_coord[vertices_used]
faces = reindexer[faces]

nib.freesurfer.write_geometry(
    "/home/jesperdn/nobackup/laplace_gm",
    vertices,
    faces,
)


mesh = simnibs.mesh_tools.mesh_io.read_msh("/home/jesperdn/nobackup/bigbrain/brain.remesh.msh")
mesh_vol = prepare_for_field_line_tracing(mesh)
mesh_vol.write("/home/jesperdn/nobackup/bigbrain/brain.remesh.sol.msh")

elements = np.concatenate(
    (np.full(mesh_vol.elm.nr, 4)[:,None], mesh_vol.elm.node_number_list-1),
    axis=1
)
cell_types = np.full(mesh_vol.elm.nr, fill_value=pv.CellType.TETRA, dtype=np.uint8)
# p = pv.PolyData(mesh_vol.nodes.node_coord, elements.ravel())
p = pv.UnstructuredGrid(
    elements, cell_types, mesh_vol.nodes.node_coord
)
p["potential [nodes]"] = mesh_vol.nodedata[1].value
p["potential [cells]"] = mesh_vol.elmdata[0].value
p["E [nodes]"] = mesh_vol.nodedata[2].value
p["inner"] = np.zeros(p.n_points)
p["inner"][np.unique(mesh.elm.node_number_list[mesh.elm.tag1==1001, :3])-1] = 1
p["outer"] = np.zeros(p.n_points)
p["outer"][np.unique(mesh.elm.node_number_list[mesh.elm.tag1==1002, :3])-1] = 1
# p["E [cells]"] = E_elm
p.save("/home/jesperdn/nobackup/bigbrain/brain.remesh.sol.vtk")

abs_potential_diff = 1000
start_vertices = np.unique(mesh.elm.node_number_list[mesh.elm.tag1==1002, :3])-1

all_pos, all_pot, valid_iter = euler_forward(
    mesh_vol, start_vertices, abs_potential_diff, h_max=1.0, thickness=None
)

v2use = np.abs(all_pot - 500).argmin(0)

m = pv.make_tri_mesh(all_pos[v2use, np.arange(len(v2use))], faces)
m["V"] = all_pot[v2use, np.arange(len(v2use))]
m["valid iter"] = valid_iter
m.save("/home/jesperdn/nobackup/bigbrain/laplace_0.5.vtk")

nib.freesurfer.write_geometry(
    "/home/jesperdn/nobackup/bigbrain/laplace_0.5",
    all_pos[v2use, np.arange(len(v2use))],
    faces,
)

def prepare_for_field_line_tracing(mesh, potential_limits=None):
    """Solve a PDE where a certain potential is set on the white and pial
    surfaces.
    """

    v_inner = np.unique(mesh.elm.node_number_list[mesh.elm.tag1==1001, :3])-1
    v_outer = np.unique(mesh.elm.node_number_list[mesh.elm.tag1==1002, :3])-1

    mesh_vol = mesh.crop_mesh(elm_type=4)
    assert np.allclose(mesh.nodes.node_coord, mesh_vol.nodes.node_coord)

    cond = simnibs.simulation.sim_struct.SimuList(mesh_vol).cond2elmdata()

    potential_limits = potential_limits or dict(inner=0, outer=1000)

    # Define boundary conditions
    dirichlet = simnibs.simulation.fem.DirichletBC(
        np.concatenate((v_inner+1, v_outer+1)),
        np.concatenate((
            np.full(v_inner.size, potential_limits["inner"]),
            np.full(v_outer.size, potential_limits["outer"])))
    )

    # Solve
    laplace_eq = simnibs.simulation.fem.FEMSystem(mesh_vol, cond, dirichlet, store_G=True)
    potential = laplace_eq.solve()
    potential = np.clip(potential, potential_limits["inner"], potential_limits["outer"])

    potential_elm = potential[mesh_vol.elm.node_number_list-1]

    # Compute E field
    E_elm = - np.sum(laplace_eq._G * potential_elm[..., None], 1)

    # E_mag_elm = np.linalg.norm(E_elm, axis=1)

    # Interpolate E field to nodes

    # SPR interpolation matrix
    M = mesh_vol.interp_matrix(
        mesh_vol.nodes.node_coord, out_fill='nearest', th_indices=None, element_wise=True
    )
    E = M @ E_elm
    E_mag = np.linalg.norm(E, axis=1)

    # Normalized field vector
    N = np.divide(E, E_mag[:, None], where=E_mag[:, None]>0)

    # N_elm = np.divide(E_elm, E_mag_elm[:, None], where=E_mag_elm[:, None]>0)
    N_elm = N[mesh_vol.elm.node_number_list-1]


    is_valid = E_mag.squeeze() > 1e-8
    # E = E[is_valid]
    # E_mag = E_mag[is_valid]

    print("E magnitude (minimum)", E_mag.min())


    mesh_vol.nodedata.append(simnibs.mesh_tools.mesh_io.NodeData(potential, "potential (node)", mesh_vol))
    mesh_vol.nodedata.append(simnibs.mesh_tools.mesh_io.NodeData(E, "E (node)", mesh_vol))
    mesh_vol.nodedata.append(simnibs.mesh_tools.mesh_io.NodeData(E_mag, "|E| (node)", mesh_vol))
    mesh_vol.nodedata.append(simnibs.mesh_tools.mesh_io.NodeData(N, "N (node)", mesh_vol))
    mesh_vol.nodedata.append(simnibs.mesh_tools.mesh_io.NodeData(is_valid, "valid (node)", mesh_vol))

    # mesh_vol.elmdata.append(simnibs.mesh_tools.mesh_io.ElementData(potential_elm, "potential (elm)", mesh_vol))
    mesh_vol.elmdata.append(simnibs.mesh_tools.mesh_io.ElementData(E_elm, "E (elm)", mesh_vol))
    # mesh_vol.elmdata.append(simnibs.mesh_tools.mesh_io.ElementData(E_mag_elm, "|E| (elm)", mesh_vol))
    # mesh_vol.elmdata.append(simnibs.mesh_tools.mesh_io.ElementData(N_elm, "N (elm)", mesh_vol))

    # elements = np.concatenate(
    #     (np.full(mesh_vol.elm.nr, 4)[:,None], mesh_vol.elm.node_number_list-1),
    #     axis=1
    # )
    # cell_types = np.full(mesh_vol.elm.nr, fill_value=pv.CellType.TETRA, dtype=np.uint8)
    # # p = pv.PolyData(mesh_vol.nodes.node_coord, elements.ravel())
    # p = pv.UnstructuredGrid(
    #     elements, cell_types, mesh_vol.nodes.node_coord
    # )
    # p["potential [nodes]"] = potential
    # p["potential [cells]"] = potential_elm
    # p["E [nodes]"] = E
    # p["E [cells]"] = E_elm
    # p.save("/home/jesperdn/nobackup/mesh.vtk")

    # p = pv.make_tri_mesh(
    #     mesh.nodes.node_coord[start_vertices], mesh.elm.node_number_list[mesh.elm.tag1==1001, :3]-1
    # )
    # p.save("/home/jesperdn/nobackup/mesh_0.vtk")

    # i = 2
    # every_pos = mesh.nodes.node_coord[start_vertices].copy()
    # every_pos[mesh_vol.field["valid (node)"].value[start_vertices]] = all_pos[i]
    # p = pv.make_tri_mesh(
    #     every_pos, mesh.elm.node_number_list[mesh.elm.tag1==1001, :3]-1
    # )
    # p.save(f"/home/jesperdn/nobackup/mesh_{i}.vtk")


    return mesh_vol


def prepare_for_tetrahedron_with_points(mesh):
    indices_tetra = mesh.elm.tetrahedra
    nodes_tetra = np.array(mesh.nodes[mesh.elm[indices_tetra]], float)
    th_baricenters = nodes_tetra.mean(1)

    # Calculate a few things we will use later
    _, faces_tetra, adjacency_list = mesh.elm.get_faces(indices_tetra)
    faces_tetra = np.array(faces_tetra, dtype=int)
    adjacency_list = np.array(adjacency_list, dtype=int)

    kdtree = cKDTree(th_baricenters)

    return faces_tetra, nodes_tetra, adjacency_list, kdtree, indices_tetra


def tetrahedron_with_points(points, faces_tetra, nodes_tetra, adjacency_list, indices_tetra, init_tetra):


    tetra_index = cython_msh.find_tetrahedron_with_points(
        np.array(points, float), nodes_tetra, init_tetra, faces_tetra, adjacency_list
    )


    # calculate baricentric coordinates
    inside = tetra_index != -1

    M = np.transpose(
        nodes_tetra[tetra_index[inside], :3] - nodes_tetra[tetra_index[inside], 3, None],
        (0, 2, 1)
    )
    baricentric = np.zeros((len(points), 4), dtype=float)
    baricentric[inside, :3] = np.linalg.solve(
        M, points[inside] - nodes_tetra[tetra_index[inside], 3]
    )
    baricentric[inside, 3] = 1 - np.sum(baricentric[inside], axis=1)

    # Return indices
    tetra_index[inside] = indices_tetra[tetra_index[inside]]

    return tetra_index-1, baricentric

def euler_forward(
    mesh_vol, start_vertices, abs_potential_diff, h_max=0.1, thickness=None,
    ):


    h_min = 0.01
    h_max = 0.1 # maximum stepsize (in mm)

    is_valid = mesh_vol.field["valid (node)"].value

    # collect necessary quantities
    # nodes
    pot = mesh_vol.field["potential (node)"].value
    N = mesh_vol.field["N (node)"].value
    E_mag = mesh_vol.field["|E| (node)"].value
    # elements
    # N_elm = mesh_vol.field["N (elm)"].value
    # E_mag_elm = mesh_vol.field["|E| (elm)"].value
    pot_elm = pot[mesh_vol.elm.node_number_list-1]
    N_elm = N[mesh_vol.elm.node_number_list-1]
    E_mag_elm = E_mag[mesh_vol.elm.node_number_list-1]

    t0 = time.perf_counter()

    # intialize the random walk to tetrahedron with closest baricenter
    # subsequent iterations use the previously found tetrahedron at starting
    # point
    faces_tetra, nodes_tetra, adjacency_list, kdtree, indices_tetra = prepare_for_tetrahedron_with_points(
        mesh_vol
    )
    t1 = time.perf_counter()
    print("time elapsed (initialize tetrahedron search)", t1-t0, "seconds")

    for i in start_vertices:

        pos = mesh_vol.nodes.node_coord[i]
        _, tetra_index = np.array(kdtree.query(pos), int)

        positions = []
        potential = []
        while True:

            tetra_index, coo_bari = tetrahedron_with_points(
                np.atleast_2d(pos), faces_tetra, nodes_tetra, adjacency_list, indices_tetra, np.atleast_1d(tetra_index),
            )
            if tetra_index<=0:
                # find the intersection and put that as the final pos with
                # potential=0?

                # ADJUSTED POS
                # positions.append(pos)

                break
            else:
                positions.append(pos)

                # sampled direction
                dydt = np.sum(N_elm[tetra_index] * coo_bari[...,None], axis=1)

                # determine step size
                # step size is based on magnitude of the E field
                a = np.sum(E_mag_elm[tetra_index] * coo_bari, axis=1)
                h = np.clip(a, h_min, h_max)

                pos = pos + h * dydt
                pos = pos.squeeze()

                # sampled_potential
                potential.append(np.sum(pot_elm[tetra_index] * coo_bari, axis=1))

        positions = np.array(positions)
        potential = np.array(potential)

        #p = pv.MultipleLines(points=positions)
        #p["potential"] = potential
        #mb[f"{ii}"] = p

    # mb.save(f"/home/jesperdn/nobackup/line_trace1.vtm")



def euler_forward(
        mesh_vol, start_vertices, abs_potential_diff, h_max=0.1, thickness=None
    ):

    h_min = 0.01
    h_max = 0.1 # maximum stepsize (in mm)

    is_valid = mesh_vol.field["valid (node)"].value

    # collect necessary quantities
    # nodes
    pot = mesh_vol.field["potential (node)"].value
    N = mesh_vol.field["N (node)"].value
    E_mag = mesh_vol.field["|E| (node)"].value
    # elements
    # N_elm = mesh_vol.field["N (elm)"].value
    # E_mag_elm = mesh_vol.field["|E| (elm)"].value
    pot_elm = pot[mesh_vol.elm.node_number_list-1]
    N_elm = N[mesh_vol.elm.node_number_list-1]
    E_mag_elm = E_mag[mesh_vol.elm.node_number_list-1]

    t0 = time.perf_counter()

    # intialize the random walk to tetrahedron with closest baricenter
    # subsequent iterations use the previously found tetrahedron at starting
    # point
    faces_tetra, nodes_tetra, adjacency_list, kdtree, indices_tetra = prepare_for_tetrahedron_with_points(
        mesh_vol
    )
    t1 = time.perf_counter()
    print("time elapsed (initialize tetrahedron search)", t1-t0, "seconds")

    # cc = np.array([-37.21862022426466, -10.461344922585802, 28])
    # cc = np.array([-47.794715881347656, -1.8441352844238281, 38.11970901489258])
    # i = np.linalg.norm(mesh_vol.nodes.node_coord[start_vertices] - cc, axis=1).argsort()[:1000]
    # i = start_vertices[i]
    # #print(i)

    # mb = pv.MultiBlock()
    # niters = []
    # for ii in i:

    #     pos = mesh_vol.nodes.node_coord[ii]
    #     _, tetra_index = np.array(kdtree.query(pos), int)



    #     positions = []
    #     potential = []
    #     # for _ in range(400):
    #     niter = 0
    #     while True:

    #         tetra_index, coo_bari = tetrahedron_with_points(
    #             np.atleast_2d(pos), faces_tetra, nodes_tetra, adjacency_list, indices_tetra, np.atleast_1d(tetra_index),
    #         )
    #         if tetra_index<=0:
    #             # find the intersection and put that as the final pos with potential=0?
    #             break
    #         else:
    #             positions.append(pos)
    #             dydt = np.sum(N_elm[tetra_index] * coo_bari[...,None], axis=1)

    #             a = np.sum(E_mag_elm[tetra_index] * coo_bari, axis=1)
    #             h = np.minimum(h_max, np.maximum(h_min, a))
    #             pos = pos + h * dydt

    #             pos = pos.squeeze()

    #             potential.append(np.sum(pot_elm[tetra_index] * coo_bari, axis=1))

    #         # print(h, tetra_index, coo_bari)
    #         niter += 1
    #     niters.append(niter)

    #     positions = np.array(positions)
    #     p = pv.MultipleLines(points=positions)
    #     p["potential"] = potential
    #     mb[f"{ii}"] = p

    # mb.save(f"/home/jesperdn/nobackup/line_trace1.vtm")

    valid_gm = start_vertices[is_valid[start_vertices]]

    # bbox = np.array([[-66, -27, 14], [-22, -7, 54]])

    # x = mesh_tets.nodes.node_coord[valid_gm]
    # in_bbox = np.all((x >= bbox[0]) & (x <= bbox[1]), 1)

    # valid_gm = valid_gm[in_bbox]


    thickness = np.zeros(valid_gm.size)

    # iteration 0
    y = pot[valid_gm]
    pos = mesh_vol.nodes.node_coord[valid_gm]

    # target gradient in V: we aim for stepping 1 % of the way at each iteration
    V_stepsize = 0.01 * abs_potential_diff

    # Starting position for walking algorithm: the closest baricenter
    _, tetra_index = np.array(kdtree.query(pos), int)


    # Initialize
    still_valid = tetra_index>0
    tmp = tetra_index>0

    dydt = N[valid_gm[still_valid]]
    h = np.minimum(h_max, V_stepsize / E_mag[valid_gm[still_valid]])

    valid_iterations = np.zeros(valid_gm.size, int)

    # thickness_increment = [np.zeros(valid_gm.size)]
    sampled_y = [y.copy()]

    max_iter = 200


    i = 0
    sampled_positions = [pos.copy()]
    while True:
        i += 1

        if i > max_iter:
            break

        # Forward Euler step
        pos_next = pos[still_valid] + h[:, None] * dydt

        # idx = thickness[still_valid]>target_frac
        # y_prev[still_valid][idx] + y[still_valid][idx]

        # Find tetrahedron in which each point is located
        # index is zero-based!
        tetra_index, coo_bari = tetrahedron_with_points(
            pos_next, faces_tetra, nodes_tetra, adjacency_list, indices_tetra, tetra_index[tmp]
        )
        # assert np.all(tetra_index >= 0)
        tmp = tetra_index>=0

        if not tmp.any():
            print(f"no more valid vertices. exiting at {i}")
            break

        # Accept move and update thickness for points which are still inside
        # the domain
        still_valid[still_valid] = tmp
        pos[still_valid] = pos_next[tmp]

        x = np.zeros(valid_gm.size)
        x[still_valid] = h[tmp]
        #thickness_increment.append(x)

        thickness[still_valid] += h[tmp]

        valid_iterations[still_valid] += 1


        # we could calculate the exact point where the field line crosses the
        # mesh but perhaps that is not really worth it given that the lines
        # seem to terminate at >97% thickness

        # y_tmp = np.sum(potential_tetrahedra[tetra_index[tmp]] * coo_bari[tmp], 1)
        # idx = y_tmp > target_frac
        # y[still_valid][tmp]

        # linear interpolation

        # REMOVE; only for diagnostics...
        y[still_valid] = np.sum(pot_elm[tetra_index[tmp]] * coo_bari[tmp], 1)
        print(f"{i:3d} : {y[still_valid].min():10.3f} {y[still_valid].mean():10.3f} {y[still_valid].max():10.3f} {still_valid.sum()/len(still_valid):10.3f}")

        dydt = np.sum(N_elm[tetra_index[tmp]] * coo_bari[tmp, :, None], 1)
        dydt_norm = np.linalg.norm(dydt, axis=1, keepdims=True)
        dydt = np.divide(dydt, dydt_norm, where=dydt_norm>0) # check zeros...

        # update h for next iteration: sample |E| at current position to
        # determine step size
        E_mag_sample = np.sum(E_mag_elm[tetra_index[tmp]] * coo_bari[tmp], 1)
        h = np.minimum(h_max, V_stepsize / E_mag_sample)

        sampled_positions.append(pos.copy())
        sampled_y.append(y.copy())


    # thickness_increment = np.array(thickness_increment)
    all_pos = np.array(sampled_positions)
    sampled_y = np.array(sampled_y)

    print("time elapsed (trace field lines)", time.perf_counter()-t1, "seconds")

    return all_pos, sampled_y, valid_iterations


# fit parametric for (polynomial of degree n); n = 4 seems to work okay

def parameterize_field_lines(all_pos, h_history, valid_iterations, order=5):

    # parameteric curve is valid for [0, ..., thickness] at each point

    n_vertices = len(valid_iterations)
    min_number_of_points = order ** 3

    cumulative_thickness = h_history.cumsum(0)

    parameters = np.zeros((n_vertices, order, 3))
    residuals = np.zeros((n_vertices, 3))

    for i in range(n_vertices):
        if valid_iterations[i] < min_number_of_points:
            continue

        c = all_pos[0, i] # intercept
        Y = all_pos[1:valid_iterations[i]+1, i]
        t = cumulative_thickness[1:valid_iterations[i]+1, i] # coordinates

        A = np.stack([t**i for i in range(1,order+1)], axis=1)
        b = Y-c
        X, res, _, _ = np.linalg.lstsq(A, b)
        parameters[i] = X
        residuals[i] = res

    return parameters, residuals