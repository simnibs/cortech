import numpy as np
import pyvista as pv

import simnibs

f = "/mnt/projects/skull_reco/head_and_shoulders/simnibs_simulation2/head_and_shoulders_TDCS_1_scalar.msh"

m = simnibs.mesh_tools.mesh_io.read(f)


mtris = m.crop_mesh(elm_type=2)
mtets = m.crop_mesh(elm_type=4)

# SPR interpolation matrix
M = mtets.interp_matrix(
    mtets.nodes.node_coord, out_fill='nearest', th_indices=None, element_wise=True
)
for data in mtets.elmdata:
    mtets.nodedata.append(simnibs.mesh_tools.mesh_io.NodeData(M @ data.value, data.field_name, mtets))
mtets.elmdata = [] # we don't need the element data anymore

mb = pv.MultiBlock()

for t in np.unique(mtris.elm.tag1):
    mc = mtris.crop_mesh(tags=t)

    grid = pv.make_tri_mesh(mc.nodes.node_coord, mc.elm.node_number_list[:, :3]-1)

    # for data in mc.nodedata:
    #     grid[data.field_name] = data.value
    # for data in mc.elmdata:
    #     grid[data.field_name] = data.value

    mb[str(t)] = grid

for t in np.unique(mtets.elm.tag1):
    mc = mtets.crop_mesh(tags=t)

    cells = np.concatenate((np.full((mc.elm.nr, 1), 4), mc.elm.node_number_list-1), axis=1).ravel()
    cell_type = np.full(mc.elm.nr, pv.CellType.TETRA, dtype=np.uint8)
    points = mc.nodes.node_coord

    grid = pv.UnstructuredGrid(cells, cell_type, points)

    for data in mc.nodedata:
        grid[data.field_name] = data.value
    for data in mc.elmdata:
        grid[data.field_name] = data.value

    mb[str(t)] = grid

mb.save("/home/jesperdn/nobackup/simulation.vtm")

# m = pv.read_meshio(
#     "/mrhome/jesperdn/INN_JESPER/projects/facerecognition/simnibs_template/sub-mni152/m2m_sub-mni152/sub-mni152.msh"
# )
m = pv.read_meshio(f)



tris = m.celltypes == 5
tets = m.celltypes == 10

mtris = m.remove_cells(tets, inplace=False)
m.remove_cells(tris)
mtets = m

mb = pv.MultiBlock()
for t in np.unique(mtris["gmsh:physical"]):
    mb[str(t)] = mtris.remove_cells(mtris["gmsh:physical"] != t, inplace=False)
    mb[str(t)].clear_data()
for t in np.unique(mtets["gmsh:physical"]):
    mb[str(t)] = mtets.remove_cells(mtets["gmsh:physical"] != t, inplace=False)
    mb[str(t)].clear_data()

mb.save("/mrhome/jesperdn/nobackup/test.vtm")

