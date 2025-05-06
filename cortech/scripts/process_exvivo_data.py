from cortech.cortex import Hemisphere
import nibabel as nib
import numpy as np
import numpy.typing as npt
import shutil
import os
from cortech.constants import Curvature
from scipy.spatial import KDTree
from pathlib import Path
from typing import Union
import pandas as pd


def _compute_fs_style_thickness(coord1: npt.NDArray[float], coord2: npt.NDArray[float]) -> npt.NDArray[float]:
    """This function calculates a FreeSurfer "style" thickness by finding the minimum distance between a white matter
    node and every pial surface node, doing the same the other way around, and averaging those two minimum distances
    for every node. It will also ensure the thickness estimate is between 0 and 5 mm.

    Parameters
    ----------
    coord1 : np.array(float)
                Node coordinates of the first surface
    coord2 : np.array(float)
                Node coordinates of the second surface

    Returns
    -------
    thickness : np.array(float)
                    The average, clipped cortical thickness estimate
    """


    tree1 = KDTree(coord1)
    tree2 = KDTree(coord2)

    # Compute the closest distance one way
    dist1, inds1 = tree1.query(coord2)

    # And the other way
    dist2, inds2 = tree2.query(coord1)

    av_min_dist = (dist1 + dist2) / 2

    return np.clip(av_min_dist, 0, 5)

def _save_overlay(data: npt.NDArray[float], filepath: os.PathLike) -> None:
    """A utility function to save overlays.

    Parameters
    ----------
    data : np.array(float)
                Overlay data to be save.
    filepath : os.PathLike
                A path object giving the filename where the overlay should be saved.

    Returns
    -------
    """

    overlay = nib.freesurfer.mghformat.MGHImage(
        data.astype("float32"), np.eye(4)
    )
    nib.save(overlay, filepath)

def _map_list_to_fsaverage(
    surf: Hemisphere,
    outpath_sub: os.PathLike,
    surf_directory: os.PathLike,
    quantities_to_map: list[str],
    hemi: str,
    unique_labels: npt.NDArray[int],
    label_aparc: npt.NDArray[int],
    quant_dict_name: str = "cortical_measures",
) -> None:
    """This function maps the different quantities listed in the "quantities_to_map" list to fsaverage and saves them as overlays.
    It also writes out a dictionary that saves the average and standard deviation at the different cortical regions as defined in
    the aseg+aparc parcellation.

    Parameters
    ----------
    surf : Hemisphere
                A Hemisphere object that stores the surfaces along with the spherical registration
    outpath_sub : os.PathLike
                A path object pointing to a folder where the results should be saved
    surf_path : os.PathLike
                A path object pointing to the surface folder inside the standard FS output folder
    quantities_to_map : list[str]
                A list of quantities to save and map to the fsaverge template. These
                strings in this list should correspond to the filenames that FS uses
                in its outputs, e.g., thickness.
    hemi : str
                A string denoting the hemisphere ("lh"/"rh")
    unique_labels : np.array(int)
                A numpy array storing the list of unique labels in the aseg+aparc parcellation
    label_aparc : np.array(int)
                A numpy array storing the parcellation as read from the aseg+aparc file
    thickness_dict_name : str
                Name for the dictionary file that saves the results

    Returns
    -------
    """

    quant_dict = {}
    quant_dict["aparc_labels"] = (
        unique_labels + 1000 if hemi in "lh" else unique_labels + 2000
    )
    # Map thickness data to fsaverage
    for stuff in quantities_to_map:
        # Read in the quantity
        data = nib.freesurfer.io.read_morph_data(
            surf_directory / f"{hemi}.{stuff}"
        )
        quant_av = []
        quant_std = []
        for l in unique_labels:
            quant_av.append(data[label_aparc == l].mean())
            quant_std.append(data[label_aparc == l].std())

        quant_dict[stuff + "_avg"] = np.array(quant_av)
        quant_dict[stuff + "_std"] = np.array(quant_std)
        data_on_fsaverage = surf.spherical_registration.resample(data)
        _save_overlay(data_on_fsaverage, outpath_sub / f"{hemi}.{stuff}.fsaverage.mgh")

    # Save the thickess dict
    df = pd.DataFrame.from_dict(quant_dict, orient="index")
    df.to_csv(outpath_sub / Path(f"{quant_dict_name}.csv"))


def _map_curvatures_to_fsaverage(
    curv_dict: dict[str, Curvature],
    surf: Hemisphere,
    outpath_sub: os.PathLike,
    hemi: str,
) -> None:
    """This function maps the different curvatures (from the different surfaces such as the white matter) to fsaverage

    Parameters
    ----------
    curv_dict : dict[str, Curvature]
                A dictionary of curvature objects storing the curvatures of the different surfaces.
                The keys refer to the surfaces.
    surf : Hemisphere
                A Hemisphere object that stores the surfaces along with the spherical registration
    outpath_sub : os.PathLike
                A path object pointing to a folder where the results should be saved
    hemi : str
                A string denoting the hemisphere ("lh"/"rh")

    Returns
    -------
    """

    for surf_name in curv_dict.keys():
        for k, n in zip(curv_dict[surf_name], curv_dict[surf_name]._fields):
            curv_on_fsaverage = surf.spherical_registration.resample(k)
            _save_overlay(curv_on_fsaverage, outpath_sub / f"{hemi}.{n}.{surf_name}.fsaverage.mgh")


def _compute_and_save_thickness(
    surf: Hemisphere, outpath_sub: os.PathLike, hemi: str
) -> None:
    """This function maps node-to-node distance between the wm and infra-supra, infra-supra and pial, and wm and pial to the fsaverage.

    Parameters
    ----------
    surf : Hemisphere
                A Hemisphere object that stores the surfaces along with the spherical registration
    outpath_sub : os.PathLike
                A path object pointing to a folder where the results should be saved
    hemi : str
                A string denoting the hemisphere ("lh"/"rh")

    Returns
    -------
    """
    v_white = surf.white.vertices
    v_pial = surf.pial.vertices
    v_inf = surf.inf.vertices
    thickness_cortex_node_to_node = _compute_fs_style_thickness(v_white, v_pial)
    thickness_inf_node_to_node = _compute_fs_style_thickness(v_white, v_inf)
    thickness_sup_node_to_node = _compute_fs_style_thickness(v_inf, v_pial)
    thk_cortech = {
        "thickness_cortex": thickness_cortex_node_to_node,
        "thicnkess_inf": thickness_inf_node_to_node,
        "thickness_sup": thickness_sup_node_to_node,
    }
    for t in thk_cortech.keys():
        t_on_fsav = surf.spherical_registration.resample(thk_cortech[t])
        _save_overlay(t_on_fsav, outpath_sub / f"{hemi}.{t}.fsaverage.mgh")


def _sweep_isovolume_and_isodistance_fractions(surf: Hemisphere, outpath_sub: os.PathLike, curv: Curvature, hemi: str, unique_labels: npt.NDArray[int],label_aparc: npt.NDArray[int], n_steps: int = 61, min_frac: float = 0.2, max_frac: float = 0.8) -> None:
    """This function maps node-to-node distance between the wm anGd infra-supra, infra-supra and pial, and wm and pial to the fsaverage.

    Parameters
    ----------
    surf : Hemisphere
                A Hemisphere object that stores the surfaces along with the spherical registration
    outpath_sub : os.PathLike
                A path object pointing to a folder where the results should be saved
    curv : Curvature
                A curvature object storing the average curvature of the white and pial surfaces
    unique_labels : np.array(int)
                A numpy array storing the list of unique labels in the aseg+aparc parcellation
    label_aparc : np.array(int)
                A numpy array storing the parcellation as read from the aseg+aparc file
    n_steps : np.int
                The number of steps with which to sweep the grid search from min_frac to max_frac
    min_frac : float
                The minimum distance/volume fraction from which to start the grid search
    max_frac : float
                The maximum distance/volume fraction from where to end the grid search
    hemi : str
                A string denoting the hemisphere ("lh"/"rh")

    Returns
    -------
    """

    vol_fracs = np.linspace(min_frac, max_frac, n_steps)
    orig_inf_vertices = surf.inf.vertices
    thickness = surf.compute_thickness()
    for surf_placement in ["equi-volume", "equi-distance"]:
        outpath_sub_method = outpath_sub / surf_placement

        if outpath_sub_method.exists():
            shutil.rmtree(outpath_sub_method)

        outpath_sub_method.mkdir()

        frac_array_avg = np.zeros((len(unique_labels), n_steps))
        frac_array_std = np.zeros((len(unique_labels), n_steps))
        intensity_differences = np.zeros((n_steps))
        average_intensities = np.zeros((n_steps))

        for i, vol_frac in enumerate(vol_fracs):
            v = surf.place_layers(
                thickness, curv.H, vol_frac, method=surf_placement
            )
            surf.inf.vertices = v

            nib.freesurfer.write_geometry(
                outpath_sub_method / f"{hemi}.inf.{surf_placement}.{vol_frac}",
                v,
                surf.inf.faces,
            )


            # Calculate the distance error
            inf_dist = np.linalg.norm(
                orig_inf_vertices - surf.inf.vertices, axis=1
            )
            _save_overlay(inf_dist, outpath_sub_method / f"{hemi}.distance.error.infra.supra.{vol_frac}.{surf_placement}.mgh")

            # Also map to fsaverage
            data_on_fsaverage = surf.spherical_registration.resample(inf_dist)
            _save_overlay(data_on_fsaverage, outpath_sub_method / f"{hemi}.distance.error.infra.supra.{vol_frac}.{surf_placement}.fsaverage.mgh")

            # Calculate the inf thickness at the prediction
            inf_thick = _compute_fs_style_thickness(surf.white.vertices, surf.inf.vertices)
            _save_overlay(inf_thick, outpath_sub_method / f"{hemi}.inf.thickness.{vol_frac}.{surf_placement}.mgh")

            # Also map to fsaverage
            data_on_fsaverage = surf.spherical_registration.resample(inf_thick)
            _save_overlay(data_on_fsaverage, outpath_sub_method / f"{hemi}.inf.thickness.{vol_frac}.{surf_placement}.fsaverage.mgh")


            # Calculate the sup thickness at the prediction
            sup_thick = _compute_fs_style_thickness(surf.inf.vertices, surf.pial.vertices)
            _save_overlay(sup_thick, outpath_sub_method / f"{hemi}.sup.thickness.{vol_frac}.{surf_placement}.mgh")

            # Also map to fsaverage
            data_on_fsaverage = surf.spherical_registration.resample(sup_thick)
            _save_overlay(data_on_fsaverage, outpath_sub_method / f"{hemi}.sup.thickness.{vol_frac}.{surf_placement}.fsaverage.mgh")

            for j, l in enumerate(unique_labels):
                frac_array_avg[j, i] = inf_dist[label_aparc == l].mean()
                frac_array_std[j, i] = inf_dist[label_aparc == l].std()

        # Find the best fractions and save them

        global_min = np.argmin(np.sum(frac_array_avg, axis=0))
        area_min = np.argmin(frac_array_avg, axis=1)
        min_fracs = vol_fracs[area_min]

        errors_dict = {}
        errors_dict["aparc_labels"] = (
            unique_labels + 1000 if hemi in "lh" else unique_labels + 2000
        )
        errors_dict["vol_fracs"] = min_fracs
        errors_dict["dist_error_avg"] = frac_array_avg[
            np.array(range(len(unique_labels))), area_min
        ]
        errors_dict["dist_error_std"] = frac_array_std[
            np.array(range(len(unique_labels))), area_min
        ]

        df = pd.DataFrame.from_dict(errors_dict, orient="index")
        df.to_csv(outpath_sub_method / Path(surf_placement + "_errors.csv"))


def _compute_surface_gradients(surf: Hemisphere, data: npt.NDArray[float]) -> None:

    """Fit a plane to each node (and its neighbors) to approximate the first order gradients.
    Basically for each node we are solving:

    |x1 - x0, y1 - y0, z1 - z0, 1|   |grad_x(f)_(x0,y0,z0)|    |f(x1, y1, z1)|
    |x2 - x0. y1 - y0, z1 - z0, 1|   |grad_y(f)_(x0,y0,z0)|    |f(x2, y2, z1)|
    |   .        .        .     .| * |grad_z(f)_(x0,y0,z0)|  = |     .       |
    |   .        .        .     .|   |f(x0, y0, z0)       |    |     .       |
    |xn - x0, yn - y0, zn - z0, 1|                             |f(xn, yn, zn)|

    Obviously using the MRI world coordinate system is maybe not the best for this
    because we presumably would like to have these gradients along the surface.
    A more natural coordinate system for the cortex is perhaps the one spanned
    by the principal curvature directions. Once we have those we just project
    the coordinates on the left hand side to that coordinate system and solve.

    Parameters
    ----------
       surf: Hemisphere
                A hemisphere object storing the surfaces
       data: np.array(float)
                Some data that lives on the nodes of the
                surface of which gradient we are interested in.
    Returns:
       coeffs: np.array(float)
                A nodes x 3 array that stores the first order
                gradient of the data projected to the principal
                curvature directions along with the bias or
                constant term.
    """

    # Get the principal curvature directions
    _, E = surf.inf.compute_principal_curvatures()

    # The unknowns (3x1), two gradients and the bias
    num_params = 2
    coeffs = np.zeros((surf.inf.n_vertices, num_params))

    # number of neighbors
    adjacency_matrix = surf.inf.compute_vertex_adjacency()
    m = np.array(adjacency_matrix.sum(1)).squeeze().astype(int)
    muq = np.unique(m)


    # Loop over the nodes with the same amount of neighbors
    # This way we can solve all the linear systems with the
    # same size at the same time.
    for mm in muq:
        i = np.where(m == mm)[0]
        nid = adjacency_matrix[i].indices.reshape(-1, mm)

        # Get the coordinates of the neighbors in MRI space
        coords_neighbors = surf.inf.vertices[nid, ...]
        coords_neighbors = np.moveaxis(coords_neighbors, 0, 1)

        # Get the coordinates of the center nodes
        coords_centers = surf.inf.vertices[i, ...]

        # Compute the difference
        coord_difference = coords_neighbors - coords_centers

        # Get the principal curvature directions, at the center
        E_tmp = E[i, ...].swapaxes(0,1)

        # Project the coordinates to curvature directions
        projected = np.einsum('ijk,ljk->ilj',E_tmp,coord_difference)

        # Massage this to the correct form (nodes x ngbrs x coeffs)
        projected = projected.swapaxes(0,-1)

        # Do the linear fit node-wise
        U, S, Vt = np.linalg.svd(projected, full_matrices=False)

        # Get the target values at the nodes we are dealing with
        neighbor_data = data[nid, ...].swapaxes(0,1)
        center_node_data = data[i, ...]
        data_difference = neighbor_data - center_node_data
        data_difference = data_difference.swapaxes(0,1)

        betas_tmp = np.squeeze(
            Vt.swapaxes(1, 2)
            @ (U.swapaxes(1, 2) @ data_difference[..., None] / S[..., None])
        )

        coeffs[i, :] = betas_tmp

    return coeffs

def _compute_gradients_on_surface(surf: Hemisphere, outpath_sub: os.PathLike, surf_directory: os.PathLike, curv_dict: dict[str, Curvature], quantities_to_map: list[str], hemi: str) -> None:
    """This function computes gradients of the various metrics living on the surface nodes and maps those to fsaverage.

    Parameters
    ----------
    surf : Hemisphere
                A Hemisphere object that stores the surfaces along with the spherical registration
    outpath_sub : os.PathLike
                A path object pointing to a folder where the results should be saved
    surf_path : os.PathLike
                A path object pointing to the surface folder inside the standard FS output folder
    curv_dict : dict[str, Curvature]
                A dictionary of curvature objects storing the curvatures of the different surfaces.
                The keys refer to the surfaces.
    quantities_to_map : list[str]
                A list of quantities to save and map to the fsaverge template. These
                strings in this list should correspond to the filenames that FS uses
                in its outputs, e.g., thickness.
    hemi : str
                A string denoting the hemisphere ("lh"/"rh")

    Returns
    -------
    """

    for surf_name in curv_dict.keys():
        for k, n in zip(curv_dict[surf_name], curv_dict[surf_name]._fields):
            coeffs_tmp = _compute_surface_gradients(surf, k)
            mag_coeff = np.sqrt(np.sum(coeffs_tmp**2,1))
            _save_overlay(mag_coeff, outpath_sub / f"{hemi}.{n}.gradient.magnitude.{surf_name}.mgh")
            mag_coeff_on_fsaverage = surf.spherical_registration.resample(mag_coeff)
            _save_overlay(mag_coeff_on_fsaverage, outpath_sub / f"{hemi}.{n}.gradient.magnitude.{surf_name}.fsaverage.mgh")
            for c in range(coeffs_tmp.shape[-1]):
                coeff = coeffs_tmp[:,c]
                _save_overlay(coeff, outpath_sub / f"{hemi}.{n}.gradient.{c}.{surf_name}.mgh")
                coeff_on_fsaverage = surf.spherical_registration.resample(coeff)
                _save_overlay(coeff_on_fsaverage, outpath_sub / f"{hemi}.{n}.gradient.{c}.{surf_name}.fsaverage.mgh")


    for stuff in quantities_to_map:
        # Read in the quantity
        data = nib.freesurfer.io.read_morph_data(
            surf_directory / f"{hemi}.{stuff}"
        )

        coeffs_tmp = _compute_surface_gradients(surf, data)

        mag_coeff = np.sqrt(np.sum(coeffs_tmp**2,1))
        _save_overlay(mag_coeff, outpath_sub / f"{hemi}.{stuff}.gradient.magnitude.mgh")
        mag_coeff_on_fsaverage = surf.spherical_registration.resample(mag_coeff)
        _save_overlay(mag_coeff_on_fsaverage, outpath_sub / f"{hemi}.{stuff}.gradient.magnitude.fsaverage.mgh")
        for c in range(coeffs_tmp.shape[-1]):
            coeff = coeffs_tmp[:,c]
            _save_overlay(coeff, outpath_sub / f"{hemi}.{stuff}.gradient.{c}.mgh")
            coeff_on_fsaverage = surf.spherical_registration.resample(coeff)
            _save_overlay(coeff_on_fsaverage, outpath_sub / f"{hemi}.{stuff}.gradient.{c}.fsaverage.mgh")


def _compute_fsaverage_stats(outpath: os.PathLike) -> None:
    """This function maps node-to-node distance between the wm and infra-supra, infra-supra and pial, and wm and pial to the fsaverage.

    Parameters
    ----------
    outpath_sub : os.PathLike
                A path object pointing to a folder where the results should be saved
    Returns
    -------
    """

    # Get the subject names
    sub_names = [sub.stem for sub in outpath.glob('*/')]

    # Next grab all the files that have been mapped to fsaverage
    # while splitting the hemi part off
    file_names = [fname.name.split('h.')[-1] for fname in (outpath / sub_names[0]).glob('*fsaverage*')]

    # Peek into one of the files to see how many nodes we are dealing with
    tmp_fname = outpath / sub_names[0] / f"lh.{file_names[0]}"
    if not tmp_fname.exists():
        tmp_fname = outpath / sub_names[0] / f"rh.{file_names[0]}"

    num_nodes = nib.load(tmp_fname).shape[0]
    # Okay we need to collect stats for all the files in the list for every subject, but split between hemis
    # Let's make a big list of lists because I don't know how many hemis we have a priori
    hemi_stats = {'lh': [[] for i in range(len(file_names))], 'rh': [[] for i in range(len(file_names))]}

    for sub in sub_names:
        # Check hemi
        subpath = outpath / sub
        for fname in subpath.glob('*fsaverage*'):
            # Which hemi are we dealing with
            hemi = 'lh' if 'lh' in fname.stem else 'rh'
            # Grab the index of this file in the file name list
            index = file_names.index(fname.name.split('h.')[-1])
            # Now we can stick the data into the correct_location in the dict
            # or rather append it to the list
            data = nib.load(fname).get_fdata().squeeze()
            hemi_stats[hemi][index].append(data)


    # Compute the stats
    for i, name in enumerate(file_names):
        name_stub = name.split('.mgh')[0]
        for hemi in ['lh', 'rh']:
            vals = np.array(hemi_stats[hemi][i]).T
            mean = vals.mean(axis=1)
            std = vals.std(axis=1)
            _save_overlay(mean, outpath / f"{hemi}.{name_stub}.mean.mgh")
            _save_overlay(std, outpath / f"{hemi}.{name_stub}.std.mgh")


def main(
    data_path: os.PathLike,
    output_path: os.PathLike,
    quantities_to_map: list[str],
    smoothing_steps_surface: int = 0,
    smoothing_steps_curvature: int = 0,
    sphere_reg_name = 'sphere.reg'
) -> None:
    """This function takes in a path pointing to processed Freesurfer runs, an output path to
    store the results, and a list of quantities to store and map to the fsaverage template,
    e.g., thickness curvature etc.

    Parameters
    ----------
    data_path : os.PathLike
                A path object pointing to a folder with FS outputs
    output_path : os.PathLike
                A path object pointing to a folder where the results should be saved
    quantities_to_map : list[str]
                A list of quantities to save and map to the fsaverge template. These
                strings in this list should correspond to the filenames that FS uses
                in its outputs, e.g., thickness.
    smoothing_steps_surface : int
                How many Taubin smoothing steps to run on the surfaces
    smoothing_steps_curvature : int
                How many smoothing steps to run on the curvature
    sphere_reg_name : str
                Name of the spherical registration file

    Returns
    -------
    """

    # data_path = Path("/autofs/space/rauma_001/users/op035/data/exvivo/hires_surf/")
    # out_path = Path(
    #     "/autofs/space/rauma_001/users/op035/data/exvivo/hires_surf/analysis/cortech_thicknesses"
    # )

    # What do we want to map to fsaverage?
    # stuff_to_map = ["thickness", "thickness.inf.pial", "thickness.wm.inf"]

    # Start processing subjects in the FS path
    for sub_path in data_path.iterdir():
        sub_name = sub_path.stem
        print(f"Processing sub {sub_name}")

        # Create an output folder, if one exits overwrite it
        outpath_sub = out_path / Path(sub_name)

        if outpath_sub.exists():
            shutil.rmtree(outpath_sub)

        outpath_sub.mkdir()

        # Get the surf-directory
        surf_dir = sub_path / Path("surf")

        # Check if we have lh or rh or both
        hemi = [x.stem for x in surf_dir.glob("*.white")]

        for h in hemi:
            # Load all the data we need for this hemi
            # Surfaces: wm, gm, inf-sup
            # Set up the surface
            surf = Hemisphere.from_freesurfer_subject_dir(sub_path, h, inf="inf", spherical_registration=sphere_reg_name)
            surf.white.taubin_smooth(n_iter=smoothing_steps_surface, inplace=True)
            surf.pial.taubin_smooth(n_iter=smoothing_steps_surface, inplace=True)
            surf.inf.taubin_smooth(n_iter=smoothing_steps_surface, inplace=True)

            # Technically we wouldn't have to read this in every time but it's not that bad
            fsavg = surf.from_freesurfer_subject_dir("fsaverage", h)
            surf.spherical_registration.compute_projection(fsavg.spherical_registration)

            # Load the aparc+aseg labeling
            label_aparc, _, _ = nib.freesurfer.io.read_annot(
                surf_dir.parent / Path("label") / Path(h + ".aparc.annot")
            )

            unique_labels = np.unique(label_aparc)
            unique_labels = unique_labels[unique_labels > 0]

            # Calculate curvature
            curv_wm = surf.white.compute_curvature(
                smooth_iter=smoothing_steps_curvature
            )
            curv_gm = surf.pial.compute_curvature(smooth_iter=smoothing_steps_curvature)
            curv_inf = surf.inf.compute_curvature(smooth_iter=smoothing_steps_curvature)
            curv = surf.compute_average_curvature(
                curv_kwargs={"smooth_iter": smoothing_steps_curvature}
            )

            # Map the data in the provided list to fsaverage and also save a dictionary including the
            # average value as well as the standard deviation in the different cortical regions as
            # defined in the aparc+aseg
            _map_list_to_fsaverage(
                surf,
                outpath_sub,
                surf_dir,
                quantities_to_map,
                h,
                unique_labels,
                label_aparc,
            )

            # Map curvatures to fsaverage as well. This function could probably be combined with the one above.
            curv_dict = {"wm": curv_wm, "gm": curv_gm, "inf": curv_inf, "avg": curv}
            _map_curvatures_to_fsaverage(
                curv_dict,
                surf,
                outpath_sub,
                h,
            )

            # Compute the node-to-node thickness (distance), so we can compare the models fairly.
            # This is mainly done so that the models can be easily fitted because the error will
            # be computed as the distance between corresponding nodes (see below). Also the
            # linear model will be later fitted with the node-to-node distance, although
            # it could be also fitted with the FS way of computing distance. We just have not
            # implemented that yet.
            _compute_and_save_thickness(surf, outpath_sub, h)

            # Alright, grid search the iso-distance and -volume parameters so we can
            # optimize the distance or volume fraction locally and globally.
            _sweep_isovolume_and_isodistance_fractions(surf, outpath_sub, curv, h, unique_labels, label_aparc)

            # Compute gradients of the metrics on the surface
            _compute_gradients_on_surface(surf, outpath_sub, surf_dir, curv_dict, quantities_to_map, h)


if __name__ == '__main__':
    smooth_steps_surf = 5
    smooth_steps_curv = 0
    stuff_to_map = ["thickness", "thickness.inf.pial", "thickness.wm.inf"]
    data_path = Path("/autofs/space/rauma_001/users/op035/data/exvivo/hires_surf/runs/")
    out_path = Path(f"/autofs/space/rauma_001/users/op035/data/exvivo/hires_surf/analysis/cortech_thicknesses/smooth_step_surf_{smooth_steps_surf}_smooth_steps_curv_{smooth_steps_curv}_josa")

    if out_path.exists():
        shutil.rmtree(out_path)

    out_path.mkdir()

    main(data_path, out_path, stuff_to_map, smoothing_steps_surface = smooth_steps_surf, smoothing_steps_curvature = smooth_steps_curv, sphere_reg_name='josa.sphere.reg')
    _compute_fsaverage_stats(out_path)
