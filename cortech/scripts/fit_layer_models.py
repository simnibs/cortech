import os
import shutil
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.spatial import KDTree

from cortech.cortex import Hemisphere


def _compute_fs_style_thickness(
    coord1: np.array(float), coord2: np.array(float)
) -> np.array(float):
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


def _equivolume_fit_smooth(distance_error, knn, number_of_nodes):
    """Find the equivolume fraction with the lowest error for each node (plus neighbors) for a given node

    Args:
       distance_error: np.array (number_of_nodes x num_isovol_fractions x num_subjects)
       knn: list of np.array(int)
            Lists the node indices of node i (the row index), including the node itself
       number_of_nodes: int

    Returns:
        min_index: np.array (number_of_nodes x 1)

    """
    #

    m = np.array([x.size for x in knn])
    muq = np.unique(m)
    min_index = np.zeros((number_of_nodes, 1))

    # Take the average error over subjects
    av_error = np.mean(distance_error, -1)

    for mm in muq:
        i = np.where(m == mm)[0]
        nid = np.array([knn[j] for j in i])
        # Grab the neighbor errors
        predictors_tmp = av_error[nid, ...]
        min_indexes = np.argmin(predictors_tmp, -1)
        counts = [np.bincount(min_indexes[i]) for i in range(min_indexes.shape[0])]
        min_ind = np.array([np.argmax(x) for x in counts])
        min_index[i] = min_ind[:, None]

    return min_index


def _equivolume_fit_node(distance_error, number_of_nodes):
    """Find the equivolume fraction with the lowest error for each node (plus neighbors) for a given node

    Args:
       distance_error: np.array (number_of_nodes x num_isovol_fractions x num_subjects)
       knn: list of np.array(int)
            Lists the node indices of node i (the row index), including the node itself
       number_of_nodes: int

    Returns:
        min_index: np.array (number_of_nodes x 1)

    """
    #

    # Take the average error over subjects
    av_error = np.mean(distance_error, -1)
    min_indexes = np.argmin(av_error, -1)

    return min_indexes


def _equivolume_fit_per_region(
    distance_error, label_aparc, unique_labels, number_of_nodes
):
    """Find the equivolume fraction with the lowest error for each node (plus neighbors) for a given node

    Args:
       distance_error: np.array (number_of_nodes x num_isovol_fractions x num_subjects)
       label_aparc: np.array (number_of_nodes x 1)
       unique_labels: np.array
       number_of_nodes: int

    Returns:
        min_index: np.array (number_of_nodes x 1)

    """
    #

    # Take the average error over subjects
    av_error = np.mean(distance_error, -1)

    min_index = -1 * np.ones((number_of_nodes, 1))

    for l in unique_labels:
        inds = np.where(label_aparc == l)[0]
        predictors_tmp = av_error[inds, ...]
        min_ind = np.argmin(predictors_tmp, -1)
        min_ind = np.argmax(np.bincount(min_ind))
        min_index[inds] = min_ind

    return min_index


def _equivolume_fit_global(distance_error):
    """Find the equivolume fraction with the lowest error for each node (plus neighbors) for a given node

    Args:
       distance_error: np.array (number_of_nodes x num_isovol_fractions x num_subjects)

    Returns:
        min_index: np.array (number_of_nodes x 1)

    """
    #

    # Take the average error over subjects
    av_error = np.mean(distance_error, -1)
    min_index = np.argmin(av_error, -1)

    return min_index


def _linear_fit_neighborhood(
    training_predictors: np.array(float),
    training_values: np.array(float),
    knn: list[np.array(float)],
    number_of_nodes: int,
) -> np.array(float):
    """Fit a multiple linear regression model node-wise to predictors and target values

    Args:
       training_predictors: np.array (subjects x nodes x predictors)
       training_values: np.array (subjects x nodes)
       knn: list[np.array(float)] (nodes x neighbors)
       number_of_nodes: int (num_nodes)

    Returns:
        betas: np.array (nodes x predictors)

    """

    num_predictors = training_predictors.shape[-1]
    num_subjects = training_predictors.shape[0]
    # m = np.array(adjacency_matrix.sum(1)).squeeze().astype(int)
    # number of neighbors
    m = np.array([x.size for x in knn])
    muq = np.unique(m)

    # To do a smooth fit we take each node and its connected n-neighborhood
    # and fit the betas to those values. This means that the subject x nodes
    # predictor matrix will now become (subjects + subjects*node_neighbors) x nodes
    # NOTE: The nodes in fsaverage have different amounts of neighbors
    # although most (almost all) have 6 neighbors. In any case the fits have to be done
    # separately for those two cases as the stacking and fitting can't be different.
    # It's pretty fast anyway.

    betas = np.zeros((number_of_nodes, num_predictors))
    predicted_values = np.zeros((number_of_nodes, 1))

    # Move the nodes to first dim
    training_predictors = training_predictors.swapaxes(0, 1)
    training_values = training_values.swapaxes(0, 1)

    for mm in muq:
        i = np.where(m == mm)[0]
        nid = np.array([knn[j] for j in i])
        # Grab the values from the neighbors
        predictors_tmp = training_predictors[nid, ...]

        # Reshape so that it's nodes x ngbrs x predictors
        predictors_tmp = predictors_tmp.reshape((nid.shape[0], -1, num_predictors))

        # Do the linear fit node-wise
        U, S, Vt = np.linalg.svd(predictors_tmp, full_matrices=False)

        # Get the target values at the nodes we are dealing with
        target_values_tmp = training_values[nid, ...].reshape((nid.shape[0], -1))

        betas_tmp = np.squeeze(
            Vt.swapaxes(1, 2)
            @ (U.swapaxes(1, 2) @ target_values_tmp[..., None] / S[..., None])
        )

        betas[i, :] = betas_tmp

    return betas


def _linear_fit(training_predictors, training_values):
    """Fit a multiple linear regression model node-wise to predictors and target values

    Args:
       training_predictors: np.array (subjects x nodes x predictors)
       training_values: np.array (subjects x nodes)

    Returns:
        betas: np.array (nodes x predictors)

    """

    # Do the linear fit node-wise
    U, S, Vt = np.linalg.svd(training_predictors, full_matrices=False)

    # Get the target values at the nodes we are dealing with

    betas = np.squeeze(
        Vt.swapaxes(1, 2)
        @ (U.swapaxes(1, 2) @ training_values[..., None] / S[..., None])
    )

    return betas


def _prepare_data_for_linear_fit(
    surface_data_path: os.PathLike,
    target_name: str,
    predictor_names: list[str],
    clip_range: tuple[float] = (0.1, 99.9),
) -> np.array(float) | np.array(float) | np.array(float) | list[str]:
    """This function prepares the predictors and
    target variables for the linear fitting. The
    predictors are centered to zero.

    Parameters
    ----------
        surface_data_path : os.PathLike
                          A path object pointing to the processed surface data as output by the exvivo processing script
        target_name : str
                    A string giving the file name of the target value to be predicted (typically the thickness of the infra-granular layer)
        predictor_names : list[str]
                    A list of strings with the filenames of the predictor variables.
        clip_range : tuple(float)
                   A tuple listing the lower and upper percentile for clipping extreme values
    Returns
    -------
        predictors : np.array(float)
                   The centered predictor variables for fitting
        target_values : np.array(float)
                      The target values (thickness fractions)
        orig_thicknesses : np.array(float)
                         The original thickness values

    """

    target_values = []
    predictor_values = []
    measurement_dict = {key: [] for key in predictor_names}
    sub_names = []
    hemis = []
    num_rh = 0

    for p in surface_data_path.glob("*/"):
        if not p.is_dir() or "model" in str(p) or "plot" in str(p):
            continue

        sub = p.stem
        sub_names.append(sub)
        print(sub)
        out_path = surface_data_path / Path(sub)
        hemi = [
            x.stem.split(".")[0] for x in out_path.glob(f"*.{predictor_names[0]}.mgh")
        ][0]
        hemis.append(hemi)
        print(hemi)

        # Grab the target values
        prepend = "to.lh.rh" if ("rh" in hemi or "to" in hemi) else "lh"
        target_name_tmp = f"{prepend}.{target_name}.mgh"
        tmp = nib.load(out_path / target_name_tmp).get_fdata().squeeze().T
        target_values.append(tmp)

        for pname in measurement_dict:
            tmp = nib.load(out_path / f"{prepend}.{pname}.mgh").get_fdata().squeeze().T
            measurement_dict[pname].append(tmp)

    # Make the data numpy friendly
    targets = np.array(target_values)

    for key in measurement_dict:
        measurement_dict[key] = np.array(measurement_dict[key])

    # Grab the thickness data
    thickness_dict = dict(
        filter(lambda item: "thickness" in item[0], measurement_dict.items())
    )
    thicknesses = list(thickness_dict.values())[0]
    # Keep a copy around
    orig_thicknesses = thicknesses.copy().T

    target_fraction = targets / (thicknesses + np.finfo(float).eps)
    target_fraction = np.clip(target_fraction, 0, 1).T

    gaussian_curv = np.ones_like(target_fraction)

    # Clip the precictors and compute gaussian curvature
    for key in measurement_dict:
        measure_tmp = measurement_dict[key]
        for s in range(measure_tmp.shape[0]):
            measure_tmp_sub = measure_tmp[s, :]
            perc = np.percentile(measure_tmp_sub, clip_range)
            measure_tmp[s, :] = np.clip(measure_tmp[s, :], perc[0], perc[1])

        measurement_dict[key] = measure_tmp.T
        # if "k1" in key or "k2" in key:
        #     gaussian_curv = gaussian_curv * measurement_dict[key]

    # Center the predictors
    for key in measurement_dict:
        measure_tmp = measurement_dict[key]
        measure_tmp = measure_tmp - measure_tmp.mean(axis=0)[None, :]
        measurement_dict[key] = measure_tmp

    # Add gaussian curvature into the dict
    measurement_dict["k1k2.avg.fsaverage"] = (
        measurement_dict["k1.avg.fsaverage"] * measurement_dict["k2.avg.fsaverage"]
    )
    # thicknesses = thicknesses.T
    target_values = target_fraction.T

    # Create an array of the predictors with a dummy for the fraction
    predictors = [np.ones_like(target_fraction)]
    for key in measurement_dict:
        print(key)
        # Skip the thickness values because we are predicting a fraction
        if "thickness" in key:
            continue
        predictors.append(measurement_dict[key])

    predictors = np.array(predictors).swapaxes(0, -1)
    inf_thickness = targets
    thickness = orig_thicknesses.T
    names = list(measurement_dict.keys())[1:]

    return predictors, target_values, inf_thickness, thickness, names, sub_names, hemis


def _cv_linear_fit(
    predictors: np.array(float),
    target_thicknesses: np.array(float),
    inf_thickness: np.array(float),
    thickness: np.array(float),
    outfiles: list[str],
    outpath: os.PathLike,
    fsav_path: str,
    surf_path: str,
    number_of_nodes: int,
    knn: list[np.array(int)],
    sub_names: list[str],
    hemis: list[str],
    sphere_reg_name="sphere.reg",
    smooth_steps_surf=5,
    smooth_steps_curv=0,
):
    """Cross-validate and save the linear model

    Args:
       predictors: np.array (subjects x nodes x predictors)
       target_thicknesses: np.array (subjects x nodes)
       inf_thicknesses: np.array (subjects x nodes)
       thicknesses: np.array (subjects x nodes)
       outfiles: list
       outpath: Path-object
       number_of_nodes: int
       knn: list of np.array(int)
            Lists the node indices of node i (the row index), including the node itself

    Returns:

    """

    num_subjects = predictors.shape[0]
    number_of_nodes = predictors.shape[1]
    betas = []
    abs_error = []
    r2s = []
    num_rh = 0
    for h in hemis:
        if "rh" in h:
            num_rh += 1

    pred_error = np.zeros((number_of_nodes, num_subjects))
    pred_error_only_fsav = np.zeros((number_of_nodes, num_subjects))
    if outpath.exists():
        shutil.rmtree(outpath)

    outpath.mkdir()

    for s, sub in enumerate(sub_names):
        selector = [x for x in range(num_subjects) if x != s]
        if 1:
            beta = _linear_fit_neighborhood(
                predictors[selector, ...],
                target_thicknesses[selector, ...],
                knn,
                number_of_nodes,
            )

        else:
            beta = _linear_fit(
                predictors[selector, :, :].swapaxes(0, -1),
                target_thicknesses[selector, :].swapaxes(0, -1),
            )

        error_on_subject, error_on_fsav = _predict_on_sub(
            surf_path,
            sub,
            beta,
            "linear_model",
            hemis[s],
            fsav_path,
            sphere_reg_name,
            smooth_steps_surf=smooth_steps_surf,
            smooth_steps_curv=smooth_steps_curv,
        )

        thickness_frac_on_fsav = np.einsum("ij, ij -> i", predictors[s, :, :], beta)
        thickness_frac_on_fsav = np.clip(thickness_frac_on_fsav, 0, 1)
        pred_error_only_fsav[:, s] = (
            inf_thickness[s, :] - thickness_frac_on_fsav * thickness[s, :]
        ).squeeze()
        overlay = nib.freesurfer.mghformat.MGHImage(
            error_on_subject.astype("float32"), np.eye(4)
        )
        nib.save(overlay, outpath / Path(f"{hemis[s]}.error.{sub}.linear_model.mgh"))
        overlay = nib.freesurfer.mghformat.MGHImage(
            error_on_fsav.astype("float32"), np.eye(4)
        )
        nib.save(
            overlay,
            outpath / Path(f"lh.error.{sub}.linear_model.fsaverage.mgh"),
        )

        pred_error[:, s] = error_on_fsav.squeeze()

        # Save coeffs per subject
        for i, n in enumerate(outfiles):
            b = beta[:, i]
            overlay = nib.freesurfer.mghformat.MGHImage(b.astype("float32"), np.eye(4))
            nib.save(overlay, outpath / Path(f"lh.{n}.sub.{s}.mgh"))

    # tot = target_thicknesses - np.mean(predicted, axis=1)[:, None]
    # r2 = 1 - np.sum(res**2, axis=1) / (np.sum(tot**2, axis=1) + np.finfo(float).eps)
    # r2 = r2_score(
    #     target_thicknesses.T, predicted.T, multioutput="raw_values", force_finite=True
    # )
    abs_error = np.mean(np.abs(pred_error), axis=1)
    abs_error_fsav = np.mean(np.abs(pred_error_only_fsav), axis=1)

    overlay = nib.freesurfer.mghformat.MGHImage(
        abs_error_fsav.astype("float32"), np.eye(4)
    )
    nib.save(overlay, outpath / Path("lh.abs.average.error.ONLY.FSAV.linear_model.mgh"))
    # overlay = nib.freesurfer.mghformat.MGHImage(r2.astype("float32"), np.eye(4))
    # nib.save(overlay, outpath / Path(f"lh.r2.mgh"))

    overlay = nib.freesurfer.mghformat.MGHImage(abs_error.astype("float32"), np.eye(4))
    nib.save(overlay, outpath / Path("lh.abs.average.error.linear_model.mgh"))

    # Also fit with all subjects to get the betas
    beta = _linear_fit_neighborhood(
        predictors,
        target_thicknesses,
        knn,
        number_of_nodes,
    )

    for i, n in enumerate(outfiles):
        b = beta[:, i]
        overlay = nib.freesurfer.mghformat.MGHImage(b.astype("float32"), np.eye(4))
        nib.save(overlay, outpath / Path(f"lh.{n}.linear_model.mgh"))
        call = f"mris_apply_reg --src {outpath}/lh.{n}.linear_model.mgh --trg {outpath}/rh.{n}.linear_model.mgh --streg {fsav_path}/xhemi/surf/rh.fsaverage_sym.sphere.reg {fsav_path}/surf/rh.fsaverage_sym.sphere.reg"
        os.system(call)


def _cv_equivol_fit(
    data_path,
    outpath,
    surf_path,
    number_of_nodes,
    number_of_subjects,
    knn: list[np.array(int)],
    fsav_path: str,
    sphere_reg_name="sphere.reg",
    min_frac=0.2,
    max_frac=0.8,
    num_fracs=61,
    smooth_steps_surf=5,
    smooth_steps_curv=0,
):
    """Cross-validate and save the best local and global equivolume parameters.

    Args:
       data_path: Path-object: path to the processed surface models
       outpath: Path-object: path for saveing
       number_of_nodes: int: number of nodes on the surface (fsaverage)
       number_of_subjects: int
       knn: list of np.array(int)
            Lists the node indices of node i (the row index), including the node itself
       fsav: str
            Path to fsaverage
       min_frac: float: minimum fraction tested for the isovolume model.
            Note: these models are assumed to be placed already using the ex-vivo processing script,
            so if you want to change these, they need to be changed in that script too.
       max_frac: float: maximum fraction tested for the isovolume model. Note above applies.
       num_fracs: int: number of steps by which the fractions are stepped from min to max
       sphere_reg_name: str: name of the spherical registration file
    Returns:

    """

    if outpath.exists():
        shutil.rmtree(outpath)

    outpath.mkdir()
    # Read the isovolume surface errors computed by the ex-vivo processing script.
    isovol_errors = np.zeros((number_of_nodes, num_fracs, number_of_subjects))
    fracs = np.linspace(min_frac, max_frac, num_fracs)
    sub_id = 0
    sub_names = []
    hemis = []
    num_rh = 0
    for p in data_path.glob("*/"):
        if not p.is_dir() or "model" in str(p) or "plot" in str(p):
            continue

        sub = p.stem
        print(sub)
        sub_names.append(sub)

        model_path = data_path / sub / "equivolume"
        print(model_path)
        hemi = [x.stem.split(".")[0] for x in model_path.glob("*.inf.equivolume.0.8")][
            0
        ]
        hemis.append(hemi)
        glob_pattern = ".distance.error.infra.supra.*.equivolume.fsaverage.mgh"
        if "rh" in hemi:
            glob_pattern = "to.lh.rh" + glob_pattern
            num_rh += 1
        else:
            glob_pattern = "lh" + glob_pattern

        for i, fname in enumerate(sorted(model_path.glob(glob_pattern))):
            # print(fname)
            tmp = nib.load(model_path / fname).get_fdata()
            isovol_errors[:, i, sub_id] = np.squeeze(tmp)
        sub_id += 1

    # Now compute the prediction error for a node and its neighborhood.
    # Pick the isovolume fraction, which minimizes the error locally.
    pred_error = np.zeros((number_of_nodes, number_of_subjects))
    for s, sub in enumerate(sub_names):
        selector = [x for x in range(number_of_subjects) if x != s]
        fractions = _equivolume_fit_smooth(
            isovol_errors[:, :, selector], knn, number_of_nodes
        )

        fracs_tmp = fracs[fractions.astype("int")].squeeze()
        error_on_subject, error_on_fsav = _predict_on_sub(
            surf_path,
            sub,
            fracs_tmp,
            "equivolume_local",
            hemis[s],
            fsav_path,
            sphere_reg_name,
            smooth_steps_surf=smooth_steps_surf,
            smooth_steps_curv=smooth_steps_curv,
        )
        overlay = nib.freesurfer.mghformat.MGHImage(
            error_on_subject.astype("float32"), np.eye(4)
        )
        nib.save(
            overlay, outpath / Path(f"{hemis[s]}.error.{sub}.equivolume.neighbors.mgh")
        )
        overlay = nib.freesurfer.mghformat.MGHImage(
            error_on_fsav.astype("float32"), np.eye(4)
        )
        nib.save(
            overlay,
            outpath
            / Path(f"{hemis[s]}.error.{sub}.equivolume.neighbors.fsaverage.mgh"),
        )

        pred_error[:, s] = error_on_fsav.squeeze()

    abs_error = np.mean(np.abs(pred_error), axis=1)

    overlay = nib.freesurfer.mghformat.MGHImage(abs_error.astype("float32"), np.eye(4))
    nib.save(overlay, outpath / Path("lh.abs.average.error.equivolume.neighbors.mgh"))

    pred_error = np.zeros((number_of_nodes, number_of_subjects))
    for s, sub in enumerate(sub_names):
        selector = [x for x in range(number_of_subjects) if x != s]
        fractions = _equivolume_fit_node(isovol_errors[:, :, selector], number_of_nodes)

        fracs_tmp = fracs[fractions.astype("int")].squeeze()
        error_on_subject, error_on_fsav = _predict_on_sub(
            surf_path,
            sub,
            fracs_tmp,
            "equivolume_local",
            hemis[s],
            fsav_path,
            sphere_reg_name,
            smooth_steps_surf=smooth_steps_surf,
            smooth_steps_curv=smooth_steps_curv,
        )
        overlay = nib.freesurfer.mghformat.MGHImage(
            error_on_subject.astype("float32"), np.eye(4)
        )
        nib.save(
            overlay, outpath / Path(f"{hemis[s]}.error.{sub}.equivolume.per.node.mgh")
        )
        overlay = nib.freesurfer.mghformat.MGHImage(
            error_on_fsav.astype("float32"), np.eye(4)
        )
        nib.save(
            overlay,
            outpath
            / Path(
                f"{hemis[s]}.error.{sub}.equivolume.neighbors.per.node.fsaverage.mgh"
            ),
        )

        pred_error[:, s] = error_on_fsav.squeeze()

    abs_error = np.mean(np.abs(pred_error), axis=1)

    overlay = nib.freesurfer.mghformat.MGHImage(abs_error.astype("float32"), np.eye(4))
    nib.save(overlay, outpath / Path("lh.abs.average.error.equivolume.per.node.mgh"))
    # Fit regionally
    HOME = Path(os.environ["FREESURFER_HOME"])
    label_aparc, _, _ = nib.freesurfer.io.read_annot(
        HOME / "subjects" / "fsaverage" / "label" / "lh.aparc.annot"
    )

    unique_labels = np.unique(label_aparc)
    unique_labels = unique_labels[unique_labels > 0]

    pred_error = np.zeros((number_of_nodes, number_of_subjects))
    for s, sub in enumerate(sub_names):
        selector = [x for x in range(number_of_subjects) if x != s]
        fractions = _equivolume_fit_per_region(
            isovol_errors[:, :, selector], label_aparc, unique_labels, number_of_nodes
        )

        fractions_not_specified = fractions == -1
        fractions[fractions_not_specified] = 0
        fracs_tmp = fracs[fractions.astype("int")].squeeze()

        error_on_subject, error_on_fsav = _predict_on_sub(
            surf_path,
            sub,
            fracs_tmp,
            "equivolume_local",
            hemis[s],
            fsav_path,
            sphere_reg_name,
            smooth_steps_surf=smooth_steps_surf,
            smooth_steps_curv=smooth_steps_curv,
        )

        overlay = nib.freesurfer.mghformat.MGHImage(
            error_on_subject.astype("float32"), np.eye(4)
        )
        nib.save(
            overlay, outpath / Path(f"{hemis[s]}.error.{sub}.equivolume.regions.mgh")
        )
        overlay = nib.freesurfer.mghformat.MGHImage(
            error_on_fsav.astype("float32"), np.eye(4)
        )
        nib.save(
            overlay,
            outpath / Path(f"lh.error.{sub}.equivolume.regions.fsaverage.mgh"),
        )

        pred_error[:, s] = error_on_fsav.squeeze()

    fractions_not_specified = pred_error == np.nan
    pred_error[fractions_not_specified] = 0
    abs_error = np.mean(np.abs(pred_error), axis=1)
    abs_error[np.any(fractions_not_specified, axis=1)] = -1

    overlay = nib.freesurfer.mghformat.MGHImage(abs_error.astype("float32"), np.eye(4))
    nib.save(overlay, outpath / Path("lh.abs.average.error.equivolume.regions.mgh"))

    # Also fit on all subs
    fractions = _equivolume_fit_smooth(isovol_errors, knn, number_of_nodes)

    overlay = nib.freesurfer.mghformat.MGHImage(
        fracs[fractions.astype("int")].astype("float32"), np.eye(4)
    )
    nib.save(overlay, outpath / Path("lh.equivolume.neighbor.frac.mgh"))
    call = f"mris_apply_reg --src {outpath}/lh.equivolume.neighbor.frac.mgh --trg {outpath}/rh.equivolume.neighbor.frac.mgh --streg {fsav_path}/xhemi/surf/rh.fsaverage_sym.sphere.reg {fsav_path}/surf/rh.fsaverage_sym.sphere.reg"
    os.system(call)

    # Also fit the global equivol fraction
    pred_error_global = np.zeros((number_of_nodes, number_of_subjects))
    minimum_fraction_indices = np.zeros(number_of_subjects)
    for s, sub in enumerate(sub_names):
        selector = [x for x in range(number_of_subjects) if x != s]
        isovol_training = isovol_errors[:, :, selector]
        average_error = np.mean(np.mean(isovol_training, axis=0), axis=-1)
        minimum_fraction_index = np.argmin(average_error)
        min_average_error = average_error[minimum_fraction_index]

        print(
            f"Minimum average error: {min_average_error} minimum fraction index: {minimum_fraction_index}"
        )
        fracs_tmp = fracs[minimum_fraction_index].squeeze()
        error_on_subject, error_on_fsav = _predict_on_sub(
            surf_path,
            sub,
            fracs_tmp,
            "equivolume_global",
            hemis[s],
            fsav_path,
            sphere_reg_name,
            smooth_steps_surf=smooth_steps_surf,
            smooth_steps_curv=smooth_steps_curv,
        )

        overlay = nib.freesurfer.mghformat.MGHImage(
            error_on_subject.astype("float32"), np.eye(4)
        )
        nib.save(
            overlay, outpath / Path(f"{hemis[s]}.error.{sub}.equivolume.global.mgh")
        )
        overlay = nib.freesurfer.mghformat.MGHImage(
            error_on_fsav.astype("float32"), np.eye(4)
        )
        nib.save(
            overlay,
            outpath / Path(f"lh.error.{sub}.equivolume.global.fsaverage.mgh"),
        )

        pred_error_global[:, s] = error_on_fsav.squeeze()

        minimum_fraction_indices[s] = fracs[minimum_fraction_index]

    abs_error = np.mean(np.abs(pred_error_global), axis=1)

    overlay = nib.freesurfer.mghformat.MGHImage(abs_error.astype("float32"), np.eye(4))
    nib.save(overlay, outpath / Path("lh.abs.average.error.global.mgh"))
    np.savetxt(
        outpath / Path("best_fraction_per_subject.csv"),
        minimum_fraction_indices,
        delimiter=",",
    )

    # And also fit the best overall
    isovol_training = isovol_errors
    average_error = np.mean(np.mean(isovol_training, axis=0), axis=-1)
    minimum_fraction_index = np.argmin(average_error)
    min_average_error = average_error[minimum_fraction_index]

    print(
        f"Minimum average error: {min_average_error} minimum fraction index: {minimum_fraction_index}"
    )
    np.savetxt(
        outpath / Path("best_fraction_overall.csv"),
        np.atleast_1d(np.array(fracs[minimum_fraction_index])),
        delimiter=",",
    )


def _predict_on_sub(
    surf_path,
    sub,
    fracs_tmp,
    method,
    hemi,
    fsavpath,
    sphere_reg_name="sphere.reg",
    smooth_steps_surf=5,
    smooth_steps_curv=0,
):
    """Predict and compute infra-supra surface error in subject space and map back to fsaverage

    Args:
        surf_path: Path-object
                   Path to a folder with fsruns
        sub: str
             Fsrun folder name
        fracs_tmp: np.array(float) | float
                   Fitted fractions used for prediction
        method: str
                Method for predicting the surface
        hemi: str
              lh or rh
        fsavpath: str
              Path to fsaverage
        sphere_reg_name: str
              Name of the spherical registration file name, using josa by default
    """

    path_tmp = surf_path / sub
    surf_path_tmp = path_tmp / "surf"
    hemi = [x.stem for x in surf_path_tmp.glob("*.white")]
    for h in hemi:
        print(f"Predicting on {sub} using {method}")
        surf_tmp = Hemisphere.from_freesurfer_subject_dir(
            path_tmp, h, inf="inf", spherical_registration=sphere_reg_name
        )
        surf_tmp.white.smooth_taubin(n_iter=smooth_steps_surf, inplace=True)
        surf_tmp.pial.smooth_taubin(n_iter=smooth_steps_surf, inplace=True)
        fsavg = Hemisphere.from_freesurfer_subject_dir("fsaverage", h)
        surf_tmp.spherical_registration.compute_projection(fsavg.spherical_registration)
        fsavg.spherical_registration.compute_projection(surf_tmp.spherical_registration)

        if "rh" in hemi and fracs_tmp.size > 1:
            with tempfile.TemporaryDirectory() as tmpdirname:
                tmpdir = Path(tmpdirname)
                print(f"Mapping fracs to the right hemisphere for subject {sub}")
                # Write a temporary fracs file on disk
                # If we have multiple channels, loop over them
                if fracs_tmp.ndim > 1 and fracs_tmp.shape[1] > 1:
                    fracs_on_rh = np.zeros_like(fracs_tmp)
                    for c in range(fracs_tmp.shape[1]):
                        overlay = nib.freesurfer.mghformat.MGHImage(
                            fracs_tmp[:, c].astype("float32"), np.eye(4)
                        )
                        nib.save(overlay, tmpdir / "fracs_tmp.mgh")
                        call = f"mris_apply_reg --src {tmpdirname}/fracs_tmp.mgh --trg {tmpdirname}/lh-on-rh.fracs_tmp.mgh --streg {fsav_path}/xhemi/surf/rh.fsaverage_sym.sphere.reg {fsav_path}/surf/rh.fsaverage_sym.sphere.reg"
                        os.system(call)
                        # Read the mapped values
                        fracs_on_rh[:, c] = (
                            nib.load(tmpdir / "lh-on-rh.fracs_tmp.mgh")
                            .get_fdata()
                            .squeeze()
                        )
                else:
                    overlay = nib.freesurfer.mghformat.MGHImage(
                        fracs_tmp.astype("float32"), np.eye(4)
                    )
                    nib.save(overlay, tmpdir / "fracs_tmp.mgh")
                    call = f"mris_apply_reg --src {tmpdirname}/fracs_tmp.mgh --trg {tmpdirname}/lh-on-rh.fracs_tmp.mgh --streg {fsav_path}/xhemi/surf/rh.fsaverage_sym.sphere.reg {fsav_path}/surf/rh.fsaverage_sym.sphere.reg"
                    os.system(call)
                    # Read the mapped values
                    fracs_on_rh = (
                        nib.load(tmpdir / "lh-on-rh.fracs_tmp.mgh")
                        .get_fdata()
                        .squeeze()
                    )

                # Next map it to the subject
                fracs_on_rh_sub = fsavg.spherical_registration.resample(fracs_on_rh)
                # Set the prediction up
                tmp_dict = {}
                tmp_dict[method] = fracs_on_rh_sub
                surf_tmp.infra_supra_model = tmp_dict
                inf_prediction_tmp = surf_tmp.fit_infra_supra_border(
                    curv_args={"smooth_iter": smooth_steps_curv}
                )
                # Calculate error
                inf_thickness_true = _compute_fs_style_thickness(
                    surf_tmp.white.vertices, surf_tmp.inf.vertices
                )
                inf_thickness_estimated = _compute_fs_style_thickness(
                    surf_tmp.white.vertices, inf_prediction_tmp[method]
                )
                error_subject = inf_thickness_true - inf_thickness_estimated
                # Map back to fsaverage space and to the left hemi
                error_fsav_rh = surf_tmp.spherical_registration.resample(error_subject)
                overlay = nib.freesurfer.mghformat.MGHImage(
                    error_fsav_rh.astype("float32"), np.eye(4)
                )
                if 0:
                    debug_save_path = Path(
                        f"/mnt/projects/CORTECH/nobackup/exvivo/derivatives/exvivo_surface_analysis/smooth_step_surf_{smooth_steps_surf}_smooth_steps_curv_{smooth_steps_curv}_no_josa/"
                    )
                    metadata = surf_tmp.white.metadata
                    nib.freesurfer.write_geometry(
                        debug_save_path / sub / f"{h}.inf.predicted.{method}",
                        inf_prediction_tmp[method] + metadata["cras"],
                        surf_tmp.white.faces,
                    )

                nib.save(overlay, tmpdir / "error_fsav_rh_tmp.mgh")
                call = f"mris_apply_reg --src {tmpdirname}/error_fsav_rh_tmp.mgh --trg {tmpdirname}/rh-on-lh.error_fsav.mgh --streg {fsav_path}/xhemi/surf/lh.fsaverage_sym.sphere.reg {fsav_path}/surf/lh.fsaverage_sym.sphere.reg"
                os.system(call)
                error_fsav = (
                    nib.load(tmpdir / "rh-on-lh.error_fsav.mgh").get_fdata().squeeze()
                )

        elif "lh" in hemi and fracs_tmp.size > 1:
            # No need to map to rh here
            fracs_on_lh_sub = fsavg.spherical_registration.resample(fracs_tmp).squeeze()
            tmp_dict = {}
            tmp_dict[method] = fracs_on_lh_sub
            surf_tmp.infra_supra_model = tmp_dict
            inf_prediction_tmp = surf_tmp.fit_infra_supra_border(
                curv_args={"smooth_iter": smooth_steps_curv}
            )
            # Calculate error
            inf_thickness_true = _compute_fs_style_thickness(
                surf_tmp.white.vertices, surf_tmp.inf.vertices
            )
            inf_thickness_estimated = _compute_fs_style_thickness(
                surf_tmp.white.vertices, inf_prediction_tmp[method]
            )
            error_subject = inf_thickness_true - inf_thickness_estimated
            error_fsav = surf_tmp.spherical_registration.resample(error_subject)

            if 0:
                debug_save_path = Path(
                    f"/mnt/projects/CORTECH/nobackup/exvivo/derivatives/exvivo_surface_analysis/smooth_step_surf_{smooth_steps_surf}_smooth_steps_curv_{smooth_steps_curv}_no_josa/"
                )
                metadata = surf_tmp.white.metadata
                nib.freesurfer.write_geometry(
                    debug_save_path / sub / f"{h}.inf.predicted.{method}",
                    inf_prediction_tmp[method] + metadata["cras"],
                    surf_tmp.white.faces,
                )
        elif fracs_tmp.size == 1:
            # If we only have a single, global fraction no mapping from fsav is needed.
            tmp_dict = {}
            tmp_dict[method] = fracs_tmp
            surf_tmp.infra_supra_model = tmp_dict
            inf_prediction_tmp = surf_tmp.fit_infra_supra_border(
                curv_args={"smooth_iter": smooth_steps_curv}
            )
            # Calculate error
            inf_thickness_true = _compute_fs_style_thickness(
                surf_tmp.white.vertices, surf_tmp.inf.vertices
            )
            inf_thickness_estimated = _compute_fs_style_thickness(
                surf_tmp.white.vertices, inf_prediction_tmp[method]
            )
            if 0:
                debug_save_path = Path(
                    f"/mnt/projects/CORTECH/nobackup/exvivo/derivatives/exvivo_surface_analysis/smooth_step_surf_{smooth_steps_surf}_smooth_steps_curv_{smooth_steps_curv}_no_josa/"
                )
                metadata = surf_tmp.white.metadata
                nib.freesurfer.write_geometry(
                    debug_save_path / sub / f"{h}.inf.predicted.{method}",
                    inf_prediction_tmp[method] + metadata["cras"],
                    surf_tmp.white.faces,
                )
            error_subject = inf_thickness_true - inf_thickness_estimated
            if "lh" in hemi:
                error_fsav = surf_tmp.spherical_registration.resample(error_subject)
            else:
                with tempfile.TemporaryDirectory() as tmpdirname:
                    # Map back to fsaverage space and to the left hemi
                    tmpdir = Path(tmpdirname)
                    error_fsav_rh = surf_tmp.spherical_registration.resample(
                        error_subject
                    )
                    overlay = nib.freesurfer.mghformat.MGHImage(
                        error_fsav_rh.astype("float32"), np.eye(4)
                    )
                    nib.save(overlay, tmpdir / "error_fsav_rh_tmp.mgh")
                    call = f"mris_apply_reg --src {tmpdirname}/error_fsav_rh_tmp.mgh --trg {tmpdirname}/rh-on-lh.error_fsav.mgh --streg {fsav_path}/xhemi/surf/lh.fsaverage_sym.sphere.reg {fsav_path}/surf/lh.fsaverage_sym.sphere.reg"
                    os.system(call)
                    error_fsav = (
                        nib.load(tmpdir / "rh-on-lh.error_fsav.mgh")
                        .get_fdata()
                        .squeeze()
                    )

        else:
            raise Exception("Unknown hemisphere or fractions size!")

        return error_subject, error_fsav


def _cv_equidist_fit(
    data_path,
    outpath,
    surf_path,
    fsav_path,
    number_of_nodes,
    number_of_subjects,
    knn: list[np.array(int)],
    sphere_reg_name="sphere.reg",
    min_frac=0.2,
    max_frac=0.8,
    num_fracs=61,
    smooth_steps_surf=5,
    smooth_steps_curv=0,
):
    """Cross-validate and save the linear model

    Args:
       data_path: Path-object
       outpath: Path-object
       surf_path: Path-object
       fsav_path: Path-object
       number_of_nodes: int
       number_of_subjects: int
       knn: list of np.array(int)
            Lists the node indices of node i (the row index), including the node itself
       min_frac: float: minimum fraction tested for the isovolume model.
            Note: these models are assumed to be placed already using the ex-vivo processing script,
            so if you want to change these, they need to be changed in that script too.
       max_frac: float: maximum fraction tested for the isovolume model. Note above applies.
       num_fracs: int: number of steps by which the fractions are stepped from min to max
    Returns:

    """

    if outpath.exists():
        shutil.rmtree(outpath)

    outpath.mkdir()

    isodist_errors = np.zeros((number_of_nodes, num_fracs, number_of_subjects))
    fracs = np.linspace(0.2, 0.8, num_fracs)
    sub_id = 0
    sub_names = []
    hemis = []
    num_rh = 0
    for p in data_path.glob("*/"):
        if not p.is_dir() or "model" in str(p) or "plot" in str(p):
            continue

        sub = p.stem
        print(sub)
        sub_names.append(sub)

        out_path = data_path / sub / "equidistance"

        hemi = [x.stem.split(".")[0] for x in out_path.glob("*.inf.equidistance.0.8")][
            0
        ]
        hemis.append(hemi)
        glob_pattern = ".distance.error.infra.supra.*.equidistance.fsaverage.mgh"
        if "rh" in hemi:
            num_rh += 1
            glob_pattern = "to.lh.rh" + glob_pattern
        else:
            glob_pattern = "lh" + glob_pattern

        for i, fname in enumerate(sorted(out_path.glob(glob_pattern))):
            tmp = nib.load(out_path / fname).get_fdata()
            isodist_errors[:, i, sub_id] = np.squeeze(tmp)
        sub_id += 1

    pred_error = np.zeros((number_of_nodes, number_of_subjects))
    for s, sub in enumerate(sub_names):
        selector = [x for x in range(len(sub_names)) if x != s]
        fractions = _equivolume_fit_smooth(
            isodist_errors[:, :, selector], knn, number_of_nodes
        )

        fracs_tmp = fracs[fractions.astype("int")]
        error_on_subject, error_on_fsav = _predict_on_sub(
            surf_path,
            sub,
            fracs_tmp,
            "equidistance_local",
            hemis[s],
            fsav_path,
            sphere_reg_name,
            smooth_steps_surf=smooth_steps_surf,
            smooth_steps_curv=smooth_steps_curv,
        )
        overlay = nib.freesurfer.mghformat.MGHImage(
            error_on_subject.astype("float32"), np.eye(4)
        )
        nib.save(
            overlay,
            outpath / Path(f"{hemis[s]}.error.{sub}.equidistance.neighbors.mgh"),
        )
        overlay = nib.freesurfer.mghformat.MGHImage(
            error_on_fsav.astype("float32"), np.eye(4)
        )
        nib.save(
            overlay,
            outpath
            / Path(f"{hemis[s]}.error.{sub}.equidistance.neighbors.fsaverage.mgh"),
        )

        pred_error[:, s] = error_on_fsav.squeeze()

    abs_error = np.mean(np.abs(pred_error), axis=1)

    overlay = nib.freesurfer.mghformat.MGHImage(abs_error.astype("float32"), np.eye(4))
    nib.save(overlay, outpath / Path("lh.abs.average.error.equidistance.neighbors.mgh"))

    pred_error = np.zeros((number_of_nodes, number_of_subjects))
    for s, sub in enumerate(sub_names):
        selector = [x for x in range(len(sub_names)) if x != s]
        fractions = _equivolume_fit_node(
            isodist_errors[:, :, selector], number_of_nodes
        )

        fracs_tmp = fracs[fractions.astype("int")]
        error_on_subject, error_on_fsav = _predict_on_sub(
            surf_path,
            sub,
            fracs_tmp,
            "equidistance_local",
            hemis[s],
            fsav_path,
            sphere_reg_name,
            smooth_steps_surf=smooth_steps_surf,
            smooth_steps_curv=smooth_steps_curv,
        )
        overlay = nib.freesurfer.mghformat.MGHImage(
            error_on_subject.astype("float32"), np.eye(4)
        )
        nib.save(
            overlay,
            outpath / Path(f"{hemis[s]}.error.{sub}.equidistance.per.node.mgh"),
        )
        overlay = nib.freesurfer.mghformat.MGHImage(
            error_on_fsav.astype("float32"), np.eye(4)
        )
        nib.save(
            overlay,
            outpath
            / Path(
                f"{hemis[s]}.error.{sub}.equidistance.neighbors.per.node.fsaverage.mgh"
            ),
        )

        pred_error[:, s] = error_on_fsav.squeeze()

    abs_error = np.mean(np.abs(pred_error), axis=1)

    overlay = nib.freesurfer.mghformat.MGHImage(abs_error.astype("float32"), np.eye(4))
    nib.save(overlay, outpath / Path("lh.abs.average.error.equidistance.per.node.mgh"))
    # Fit regionally
    HOME = Path(os.environ["FREESURFER_HOME"])
    label_aparc, _, _ = nib.freesurfer.io.read_annot(
        HOME / "subjects" / "fsaverage" / "label" / "lh.aparc.annot"
    )

    unique_labels = np.unique(label_aparc)
    unique_labels = unique_labels[unique_labels > 0]

    pred_error = np.zeros((number_of_nodes, number_of_subjects))
    for s, sub in enumerate(sub_names):
        selector = [x for x in range(number_of_subjects) if x != s]
        fractions = _equivolume_fit_per_region(
            isodist_errors[:, :, selector], label_aparc, unique_labels, number_of_nodes
        )

        fractions_not_specified = fractions == -1
        fractions[fractions_not_specified] = 0
        fracs_tmp = fracs[fractions.astype("int")]

        error_on_subject, error_on_fsav = _predict_on_sub(
            surf_path,
            sub,
            fracs_tmp,
            "equidistance_local",
            hemis[s],
            fsav_path,
            sphere_reg_name,
            smooth_steps_surf=smooth_steps_surf,
            smooth_steps_curv=smooth_steps_curv,
        )

        overlay = nib.freesurfer.mghformat.MGHImage(
            error_on_subject.astype("float32"), np.eye(4)
        )
        nib.save(
            overlay, outpath / Path(f"{hemis[s]}.error.{sub}.equidistance.regions.mgh")
        )
        overlay = nib.freesurfer.mghformat.MGHImage(
            error_on_fsav.astype("float32"), np.eye(4)
        )
        nib.save(
            overlay,
            outpath
            / Path(f"{hemis[s]}.error.{sub}.equidistance.regions.fsaverage.mgh"),
        )

        pred_error[:, s] = error_on_fsav.squeeze()

    fractions_not_specified = pred_error == np.nan
    pred_error[fractions_not_specified] = 0
    abs_error = np.mean(np.abs(pred_error), axis=1)
    abs_error[np.any(fractions_not_specified, axis=1)] = -1

    overlay = nib.freesurfer.mghformat.MGHImage(abs_error.astype("float32"), np.eye(4))
    nib.save(overlay, outpath / Path("lh.abs.average.error.equidistance.regions.mgh"))

    # Get the best over all subjects
    fractions = _equivolume_fit_smooth(isodist_errors, knn, number_of_nodes)

    overlay = nib.freesurfer.mghformat.MGHImage(
        fracs[fractions.astype("int")].astype("float32"), np.eye(4)
    )
    nib.save(overlay, outpath / Path("lh.equidistance.neighbor.frac.mgh"))
    # Map it to rh
    call = f"mris_apply_reg --src {outpath}/lh.equidistance.neighbor.frac.mgh --trg {outpath}/rh.equidistance.neighbor.frac.mgh --streg {fsav_path}/xhemi/surf/rh.fsaverage_sym.sphere.reg {fsav_path}/surf/rh.fsaverage_sym.sphere.reg"
    os.system(call)

    # Also fit the global equidist fraction
    pred_error_global = np.zeros((number_of_nodes, number_of_subjects))
    minimum_fraction_indices = np.zeros(number_of_subjects)
    for s, sub in enumerate(sub_names):
        selector = [x for x in range(number_of_subjects) if x != s]
        isodist_training = isodist_errors[:, :, selector]
        average_error = np.mean(np.mean(isodist_training, axis=0), axis=-1)
        minimum_fraction_index = np.argmin(average_error)
        min_average_error = average_error[minimum_fraction_index]

        print(
            f"Minimum average error: {min_average_error} minimum fraction index: {minimum_fraction_index}"
        )
        fracs_tmp = fracs[minimum_fraction_index].squeeze()
        error_on_subject, error_on_fsav = _predict_on_sub(
            surf_path,
            sub,
            fracs_tmp,
            "equidistance_global",
            hemis[s],
            fsav_path,
            sphere_reg_name,
            smooth_steps_surf=smooth_steps_surf,
            smooth_steps_curv=smooth_steps_curv,
        )

        overlay = nib.freesurfer.mghformat.MGHImage(
            error_on_subject.astype("float32"), np.eye(4)
        )
        nib.save(
            overlay, outpath / Path(f"{hemis[s]}.error.{sub}.equidistance.global.mgh")
        )
        overlay = nib.freesurfer.mghformat.MGHImage(
            error_on_fsav.astype("float32"), np.eye(4)
        )
        nib.save(
            overlay,
            outpath / Path(f"{hemis[s]}.error.{sub}.equidistance.global.fsaverage.mgh"),
        )

        pred_error_global[:, s] = error_on_fsav.squeeze()

        minimum_fraction_indices[s] = fracs[minimum_fraction_index]

    abs_error = np.mean(np.abs(pred_error_global), axis=1)

    overlay = nib.freesurfer.mghformat.MGHImage(abs_error.astype("float32"), np.eye(4))
    nib.save(overlay, outpath / Path("lh.abs.average.error.global.mgh"))
    np.savetxt(
        outpath / Path("best_fraction_per_subject.csv"),
        minimum_fraction_indices,
        delimiter=",",
    )

    # Get the best over all
    isodist_training = isodist_errors
    average_error = np.mean(np.mean(isodist_training, axis=0), axis=-1)
    minimum_fraction_index = np.argmin(average_error)
    min_average_error = average_error[minimum_fraction_index]
    print(
        f"Minimum average error: {min_average_error} minimum fraction index: {minimum_fraction_index}"
    )

    np.savetxt(
        outpath / Path("best_fraction_overall.csv"),
        np.atleast_1d(np.array(fracs[minimum_fraction_index])),
        delimiter=",",
    )


def main(
    surf_data_path: os.PathLike,
    output_path: os.PathLike,
    fs_run_path: os.PathLike,
    fsav_path: os.PathLike,
    fsav: Hemisphere,
    predictor_names: list[str],
    target_name: str,
    data_path: os.PathLike,
    neighborhood_size: int = 1,
    sphere_reg_name="sphere.reg",
    smooth_steps_surf=5,
    smooth_steps_curv=0,
) -> None:
    """This function takes in a path pointing to the processed surface data, i.e., data mapped to fsaverage,
    and fits various models to predict the infra-supra-border. The models considered are: isodistance, isovolume,
    and a general linear model that uses the quantities listed in the predictor_names list as the predictors.

    Parameters
    ----------
    surf_data_path : os.PathLike
                   A path object pointing to a folder with processed surface data as output by the "process_exvivo_data.py" script.
    output_path : os.PathLike
                A path object pointing to a folder where the results should be saved
    fsrun_path : os.PathLike
                A path object pointing to a folder where the results should be saved
    fsav_path : os.PathLike
                A path object pointing to the fsaverage folder
    fsav : Hemisphere
         A hemisphere object storing the left hemi of the fsaverage subject
    predictor_names: list[str]
                A list of quantity names to treat as the predictors for the linear model. These should correspond to the filenames as output by the processing script.
    target_name: str
                A string (name) for the target variable of interest.
    data_path : os.PathLike
                A path object pointing to a folder where the data is stored.
    neighborhood_size: int
                An integer defining the size of the neighborhood of a node to include in the fit. 0 would be only the node itself, 1 would refer to the nearest connected neighbors, 2 to the second-order neighbors etc.
    sphere_reg_name: str
                Name of the spherical registration file.
    smooth_steps_surf: int
                Number of smoothing steps for the surface.
    smooth_steps_curv: int
                Number of smoothing steps for the curvature.
    Returns
    -------
    """

    # Before fitting the models, let's map all the single right hemi results to the left hemi
    _map_rh_to_lh(surf_data_folder, fsav_path)

    # Okay start out by fitting the linear model

    predictors, target_values, inf_thickness, thickness, names, sub_names, hemis = (
        _prepare_data_for_linear_fit(
            surf_data_path, target_name, predictor_names, clip_range=(0.1, 99.9)
        )
    )

    knn, kr = fsav.white.k_ring_neighbors(neighborhood_size)
    number_of_nodes = fsav.white.n_vertices

    # Run leave-one-out cross-validation for linear model

    out_files = ["intercept", "beta_k1", "beta_k2", "beta_k1k2"]
    outpath = data_path / "linear_model_test_prediction"

    _cv_linear_fit(
        predictors,
        target_values,
        inf_thickness,
        thickness,
        out_files,
        outpath,
        fsav_path,
        fs_run_path,
        number_of_nodes,
        knn,
        sub_names,
        hemis,
        sphere_reg_name,
        smooth_steps_surf=smooth_steps_surf,
        smooth_steps_curv=smooth_steps_curv,
    )

    # # Next cross-validate the isovolume values
    outpath = data_path / "equivolume_model_test_prediction"
    number_of_subjects = predictors.shape[0]

    _cv_equivol_fit(
        data_path,
        outpath,
        fs_run_path,
        number_of_nodes,
        number_of_subjects,
        knn,
        fsav_path,
        sphere_reg_name,
        smooth_steps_surf=smooth_steps_surf,
        smooth_steps_curv=smooth_steps_curv,
    )

    # # Next cross-validate the isodistance values
    outpath = data_path / "equidistance_model_test_prediction"
    number_of_subjects = predictors.shape[0]
    _cv_equidist_fit(
        data_path,
        outpath,
        fs_run_path,
        fsav_path,
        number_of_nodes,
        number_of_subjects,
        knn,
        sphere_reg_name,
        smooth_steps_surf=smooth_steps_surf,
        smooth_steps_curv=smooth_steps_curv,
    )


def _map_rh_to_lh(surf_data_path: os.PathLike, fsav_path: os.PathLike):
    """This function maps the right hemi results to the left hemi so we can fit the models
    using all the data.
    Parameters
    ----------
    surf_data_path : os.PathLike
                   A path object pointing to a folder with processed surface data as output by the "process_exvivo_data.py" script.
    fsav : os.PathLike
         A path object pointing to the fsaverage subject

    Returns
    -------
    """

    for f in surf_data_path.glob("**/*"):
        if "lh." in f.name or "fsaverage" not in f.name:
            continue

        print(f"Processing: {f}")
        call = (
            "mris_apply_reg --src "
            + str(f)
            + " --trg "
            + str(f.parent)
            + "/to.lh."
            + f.name
            + " --streg "
            + str(fsav_path)
            + "/xhemi/surf/lh.fsaverage_sym.sphere.reg "
            + str(fsav_path)
            + "/surf/lh.fsaverage_sym.sphere.reg"
        )
        os.system(call)


if __name__ == "__main__":
    # Grab the FS dir from the environ
    fs_home = Path(os.environ["FREESURFER_HOME"])
    if not fs_home.exists():
        raise Exception("FREESURFER_HOME not set")

    fsav_path = fs_home / "subjects" / "fsaverage"
    fsav = Hemisphere.from_freesurfer_subject_dir(fsav_path, "lh")

    smooth_steps_surf = 5
    smooth_steps_curv = 20
    surf_data_folder = Path(
        f"/mnt/projects/CORTECH/nobackup/exvivo/derivatives/exvivo_surface_analysis/smooth_step_surf_{smooth_steps_surf}_smooth_steps_curv_{smooth_steps_curv}_josa/"
    )

    if "josa" in str(surf_data_folder):
        sphere_reg_name = "josa.sphere.reg"
    else:
        sphere_reg_name = "sphere.reg"

    # surf_data_folder = Path(
    #     "/autofs/space/rauma_001/users/op035/data/exvivo/hires_surf/analysis/output_smooth20/"
    # )

    fs_run_folder = Path(
        "/mnt/projects/CORTECH/nobackup/exvivo/derivatives/final_surfaces/"
    )
    # Could use some the thickness gradient as well?
    # predictor_names = ['thickness_cortex.fsaverage', 'k1.avg.fsaverage', 'k2.avg.fsaverage', 'thickness.gradient.magnitude.fsaverage']
    predictor_names = [
        "thickness_cortex.fsaverage",
        "k1.avg.fsaverage",
        "k2.avg.fsaverage",
    ]

    # predictor_names = [
    #     "thickness.fsaverage",
    #     "k1.fsaverage",
    #     "k2.fsaverage",
    # ]

    target_name = "thickness.wm.inf.fsaverage"

    main(
        surf_data_folder,
        surf_data_folder,
        fs_run_folder,
        fsav_path,
        fsav,
        predictor_names,
        target_name,
        surf_data_folder,
        sphere_reg_name=sphere_reg_name,
        smooth_steps_surf=smooth_steps_surf,
        smooth_steps_curv=smooth_steps_curv,
    )

    # stuff_to_map = ["thickness", "thickness.inf.pial", "thickness.wm.inf"]
    # out_path = Path(f"/autofs/space/rauma_001/users/op035/data/exvivo/hires_surf/analysis/cortech_thicknesses/smooth_step_surf_{smooth_steps_surf}_smooth_steps_curv_{smooth_steps_curv}")
