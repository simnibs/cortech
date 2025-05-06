import numpy as np
import nibabel as nib
from pathlib import Path
import shutil

import os
from sklearn.metrics import r2_score
import multiprocessing as mp
from functools import partial
import sys
from cortech.cortex import Hemisphere


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


def _equivolume_fit_per_region(distance_error, label_aparc, unique_labels, number_of_nodes):
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
        predictors_tmp = av_error[inds,...]
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

    return min_index


def _linear_fit_neighborhood(
    training_predictors: np.array(float), training_values: np.array(float), knn: list[np.array(float)], number_of_nodes: int
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

    for mm in muq:
        i = np.where(m == mm)[0]
        nid = np.array([knn[j] for j in i])
        # Grab the values from the neighbors

        predictors_tmp = training_predictors[:, nid, :]

        # Reshape so that it's nodes x ngbrs x predictors
        predictors_tmp = predictors_tmp.reshape((nid.shape[0], -1, num_predictors))

        # Do the linear fit node-wise
        U, S, Vt = np.linalg.svd(predictors_tmp, full_matrices=False)

        # Get the target values at the nodes we are dealing with
        target_values_tmp = training_values[..., nid].reshape((nid.shape[0], -1))

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


def _prepare_data_for_linear_fit(surface_data_path: os.PathLike, target_name: str, predictor_names: list[str], clip_range: tuple[float]=(0.1, 99.9)) -> np.array(float) | np.array(float) | np.array(float) | list[str]:
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

    for p in surface_data_path.glob("*/"):
        if not p.is_dir() or "model" in str(p) or "plot" in str(p):
            continue

        sub = p.stem
        print(sub)
        out_path = surface_data_path / Path(sub)
        hemi = [x.stem.split(".")[0] for x in out_path.glob(f"*.{predictor_names[0]}.mgh")][0]
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
    thickness_dict = dict(filter(lambda item: 'thickness' in item[0], measurement_dict.items()))
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
            measure_tmp_sub = measure_tmp[s,:]
            perc = np.percentile(measure_tmp_sub, clip_range)
            measure_tmp[s,:] = np.clip(measure_tmp[s,:], perc[0], perc[1])


        measurement_dict[key] = measure_tmp.T
        if 'k1' in key or 'k2' in key:
            gaussian_curv = gaussian_curv * measurement_dict[key]

    # Add gaussian curvature into the dict
    measurement_dict['k1k2.fsaverage'] = gaussian_curv

    # Center the predictors
    for key in measurement_dict:
        measure_tmp = measurement_dict[key]
        measure_tmp = measure_tmp - measure_tmp.mean(axis=1)[:, None]
        measurement_dict[key] = measure_tmp

    # thicknesses = thicknesses.T
    target_values = target_fraction.T

    # Create an array of the predictors with a dummy for the fraction
    predictors = [np.ones_like(target_fraction)]
    for key in measurement_dict:
        # Skip the thickness values because we are predicting a fraction
        if 'thickness' in key:
            continue
        predictors.append(measurement_dict[key])

    predictors = np.array(predictors).swapaxes(0,-1)
    inf_thickness = targets
    thickness = orig_thicknesses.T
    names = list(measurement_dict.keys())[1:]

    return predictors, target_values, inf_thickness, thickness, names


def _cv_linear_fit(
        predictors: np.array(float), target_thicknesses: np.array(float), inf_thickness: np.array(float), thickness: np.array(float), outfiles: list[str], outpath: os.PathLike, number_of_nodes: int, knn: list[np.array(int)],
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
    betas = []
    predicted = np.zeros((num_subjects, number_of_nodes))
    abs_error = []
    r2s = []

    if outpath.exists():
        shutil.rmtree(outpath)

    outpath.mkdir()

    for s in range(num_subjects):
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
                predictors[:, :, selector],
                target_thicknesses[:, selector],
            )

        pred = np.einsum("ij, ij -> i", predictors[s,...].squeeze(), beta)
        # Clip the fraction to between 0 and 1
        pred = np.clip(pred, 0, 1)
        pred = np.nan_to_num(pred)
        predicted[s, :] = pred
        # Calculate the pred error in mm
        res = inf_thickness[s, :] - pred * thickness[s, :]
        abs_error = np.abs(res)

        # Save absolute error per subject
        overlay = nib.freesurfer.mghformat.MGHImage(
            abs_error.astype("float32"), np.eye(4)
        )
        nib.save(overlay, outpath / Path(f"lh.abs.error.sub.{s}.mgh"))

        # Save coeffs per subject
        for i, n in enumerate(outfiles):
            b = beta[:, i]
            overlay = nib.freesurfer.mghformat.MGHImage(b.astype("float32"), np.eye(4))
            nib.save(overlay, outpath / Path(f"lh.{n}.sub.{s}.mgh"))


    res = inf_thickness - predicted * thickness
    # tot = target_thicknesses - np.mean(predicted, axis=1)[:, None]
    # r2 = 1 - np.sum(res**2, axis=1) / (np.sum(tot**2, axis=1) + np.finfo(float).eps)
    r2 = r2_score(
        target_thicknesses.T, predicted.T, multioutput="raw_values", force_finite=True
    )
    abs_error = np.mean(np.abs(res), axis=1)

    overlay = nib.freesurfer.mghformat.MGHImage(r2.astype("float32"), np.eye(4))
    nib.save(overlay, outpath / Path(f"lh.r2.mgh"))

    overlay = nib.freesurfer.mghformat.MGHImage(abs_error.astype("float32"), np.eye(4))
    nib.save(overlay, outpath / Path(f"lh.abs.error.average.mgh"))

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
        nib.save(overlay, outpath / Path(f"lh.{n}.all.subjects.smoothed.mgh"))

def _cv_equivol_fit(data_path, outpath, number_of_nodes, number_of_subjects, knn: list[np.array(int)], min_frac=0.2, max_frac=0.8, num_fracs=61):
    """Cross-validate and save the best local and global equivolume parameters.

    Args:
       data_path: Path-object: path to the processed surface models
       outpath: Path-object: path for saveing
       number_of_nodes: int: number of nodes on the surface (fsaverage)
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
    # Read the isovolume surface errors computed by the ex-vivo processing script.
    isovol_errors = np.zeros((number_of_nodes, num_fracs, number_of_subjects))
    sub_id = 0
    for p in data_path.glob("*/"):
        if not p.is_dir() or "model" in str(p) or "plot" in str(p):
            continue

        sub = p.stem
        print(sub)

        model_path = data_path / sub / "equi-volume"
        hemi = [x.stem.split(".")[0] for x in model_path.glob("*.inf.equi-volume.0.8")][0]
        glob_pattern = ".distance.error.infra.supra.*.equi-volume.fsaverage.mgh"
        if "rh" in hemi:
            glob_pattern = "to.lh.rh" + glob_pattern
        else:
            glob_pattern = "lh" + glob_pattern

        for i, fname in enumerate(sorted(model_path.glob(glob_pattern))):
            print(fname)
            tmp = nib.load(model_path / fname).get_fdata()
            isovol_errors[:, i, sub_id] = np.squeeze(tmp)
        sub_id += 1


    # Now compute the prediction error for a node and its neighborhood.
    # Pick the isovolume fraction, which minimizes the error locally.
    pred_error = np.zeros((number_of_nodes, number_of_subjects))
    for s in range(number_of_subjects):
        selector = [x for x in range(number_of_subjects) if x != s]
        fractions = _equivolume_fit_smooth(
            isovol_errors[:, :, selector], knn, number_of_nodes
        )

        isovol_tmp = isovol_errors[:, :, s]
        pred_error[:, s] = isovol_tmp[
            np.arange(number_of_nodes), np.squeeze(fractions.astype(int))
        ]
        res = pred_error[:, s]
        abs_error = np.abs(res)

        # Save absolute error per subject
        overlay = nib.freesurfer.mghformat.MGHImage(
            abs_error.astype("float32"), np.eye(4)
        )
        nib.save(overlay, outpath / Path(f"lh.abs.error.sub.{s}.smoothing.mgh"))

    abs_error = np.mean(np.abs(pred_error), axis=1)

    overlay = nib.freesurfer.mghformat.MGHImage(abs_error.astype("float32"), np.eye(4))
    nib.save(overlay, outpath / Path(f"lh.abs.error.average.smoothing.mgh"))

    # Fit regionally
    HOME = Path(os.environ["FREESURFER_HOME"])
    label_aparc, _, _ = nib.freesurfer.io.read_annot(HOME / "subjects"/ "fsaverage" / "label" / "lh.aparc.annot")

    unique_labels = np.unique(label_aparc)
    unique_labels = unique_labels[unique_labels > 0]

    pred_error = np.zeros((number_of_nodes, number_of_subjects))
    for s in range(number_of_subjects):
        selector = [x for x in range(number_of_subjects) if x != s]
        fractions = _equivolume_fit_per_region(
            isovol_errors[:, :, selector], label_aparc, unique_labels, number_of_nodes
        )

        isovol_tmp = isovol_errors[:, :, s]
        fractions_not_specified = fractions == -1
        fractions[fractions_not_specified] = 0
        pred_error[:, s] = isovol_tmp[
            np.arange(number_of_nodes), np.squeeze(fractions.astype(int))
        ]
        res = pred_error[:, s]
        abs_error = np.abs(res)
        abs_error[fractions_not_specified.squeeze()] = -1
        pred_error[fractions_not_specified.squeeze(), s] = np.nan

        # Save absolute error per subject
        overlay = nib.freesurfer.mghformat.MGHImage(
            abs_error.astype("float32"), np.eye(4)
        )
        nib.save(overlay, outpath / Path(f"lh.abs.error.sub.{s}.per.region.mgh"))

    fractions_not_specified = pred_error == np.nan
    pred_error[fractions_not_specified] = 0
    abs_error = np.mean(np.abs(pred_error), axis=1)
    abs_error[np.any(fractions_not_specified,axis=1)] = -1

    overlay = nib.freesurfer.mghformat.MGHImage(abs_error.astype("float32"), np.eye(4))
    nib.save(overlay, outpath / Path(f"lh.abs.error.average.per.region.mgh"))

    # Also fit on all subs
    fractions = _equivolume_fit_smooth(
            isovol_errors, knn, number_of_nodes
        )

    overlay = nib.freesurfer.mghformat.MGHImage(fractions.astype("float32"), np.eye(4))
    nib.save(overlay, outpath / Path(f"lh.best.local.isovol.frac.mgh"))

    # Also fit the global equivol fraction
    pred_error_global = np.zeros((number_of_nodes, number_of_subjects))
    minimum_fraction_indices = np.zeros(number_of_subjects)
    fracs = np.linspace(min_frac, max_frac, num_fracs)
    for s in range(number_of_subjects):
        selector = [x for x in range(number_of_subjects) if x != s]
        isovol_training = isovol_errors[:, :, selector]
        average_error = np.mean(np.mean(isovol_training, axis=0), axis=-1)
        minimum_fraction_index = np.argmin(average_error)
        min_average_error = average_error[minimum_fraction_index]

        print(f"Minimum average error: {min_average_error} minimum fraction index: {minimum_fraction_index}")
        isovol_tmp = isovol_errors[:, minimum_fraction_index, s]
        pred_error_global[:, s] = isovol_tmp
        res = pred_error_global[:, s]
        abs_error = np.abs(res)
        minimum_fraction_indices[s] = fracs[minimum_fraction_index]

        # Save absolute error per subject
        overlay = nib.freesurfer.mghformat.MGHImage(
            abs_error.astype("float32"), np.eye(4)
        )
        nib.save(overlay, outpath / Path(f"lh.abs.error.global.sub.{s}.smoothing.mgh"))

    abs_error = np.mean(np.abs(pred_error_global), axis=1)

    overlay = nib.freesurfer.mghformat.MGHImage(abs_error.astype("float32"), np.eye(4))
    nib.save(overlay, outpath / Path(f"lh.abs.error.global.average.smoothing.mgh"))
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

    print(f"Minimum average error: {min_average_error} minimum fraction index: {minimum_fraction_index}")
    np.savetxt(
        outpath / Path("best_fraction_overall.csv"),
        np.atleast_1d(np.array(fracs[minimum_fraction_index])),
        delimiter=",",
    )

def _cv_equidist_fit(data_path, outpath, number_of_nodes, number_of_subjects, knn: list[np.array(int)], min_frac=0.2, max_frac=0.8, num_fracs=61):
    """Cross-validate and save the linear model

    Args:
       data_path: Path-object
       outpath: Path-object
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
    sub_id = 0
    for p in data_path.glob("*/"):
        if not p.is_dir() or "model" in str(p) or "plot" in str(p):
            continue

        sub = p.stem
        print(sub)

        out_path = data_path / sub / "equi-distance"

        hemi = [x.stem.split(".")[0] for x in out_path.glob("*.inf.equi-distance.0.8")][0]
        glob_pattern = ".distance.error.infra.supra.*.equi-distance.fsaverage.mgh"
        if "rh" in hemi:
            glob_pattern = "to.lh.rh" + glob_pattern
        else:
            glob_pattern = "lh" + glob_pattern

        for i, fname in enumerate(sorted(out_path.glob(glob_pattern))):
            print(fname)
            tmp = nib.load(out_path / fname).get_fdata()
            isodist_errors[:, i, sub_id] = np.squeeze(tmp)
        sub_id += 1


    pred_error = np.zeros((number_of_nodes, number_of_subjects))
    for s in range(number_of_subjects):
        selector = [x for x in range(number_of_subjects) if x != s]
        fractions = _equivolume_fit_smooth(
            isodist_errors[:, :, selector], knn, number_of_nodes
        )

        isodist_tmp = isodist_errors[:, :, s]
        pred_error[:, s] = isodist_tmp[
            np.arange(number_of_nodes), np.squeeze(fractions.astype(int))
        ]
        res = pred_error[:, s]
        abs_error = np.abs(res)

        # Save absolute error per subject
        overlay = nib.freesurfer.mghformat.MGHImage(
            abs_error.astype("float32"), np.eye(4)
        )
        nib.save(overlay, outpath / Path(f"lh.abs.error.sub.{s}.smoothing.mgh"))

    abs_error = np.mean(np.abs(pred_error), axis=1)

    # overlay = nib.freesurfer.mghformat.MGHImage(r2.astype("float32"), np.eye(4))
    # nib.save(overlay, outpath / Path(f"lh.r2.smoothing.mgh"))

    overlay = nib.freesurfer.mghformat.MGHImage(abs_error.astype("float32"), np.eye(4))
    nib.save(overlay, outpath / Path(f"lh.abs.error.average.smoothing.mgh"))

    # Fit regionally
    HOME = Path(os.environ["FREESURFER_HOME"])
    label_aparc, _, _ = nib.freesurfer.io.read_annot(HOME / "subjects"/ "fsaverage" / "label" / "lh.aparc.annot")

    unique_labels = np.unique(label_aparc)
    unique_labels = unique_labels[unique_labels > 0]

    pred_error = np.zeros((number_of_nodes, number_of_subjects))
    for s in range(number_of_subjects):
        selector = [x for x in range(number_of_subjects) if x != s]
        fractions = _equivolume_fit_per_region(
            isodist_errors[:, :, selector], label_aparc, unique_labels, number_of_nodes
        )

        isodist_tmp = isodist_errors[:, :, s]
        fractions_not_specified = fractions == -1
        fractions[fractions_not_specified] = 0
        pred_error[:, s] = isodist_tmp[
            np.arange(number_of_nodes), np.squeeze(fractions.astype(int))
        ]
        res = pred_error[:, s]
        abs_error = np.abs(res)
        abs_error[fractions_not_specified.squeeze()] = -1
        pred_error[fractions_not_specified.squeeze(), s] = np.nan

        # Save absolute error per subject
        overlay = nib.freesurfer.mghformat.MGHImage(
            abs_error.astype("float32"), np.eye(4)
        )
        nib.save(overlay, outpath / Path(f"lh.abs.error.sub.{s}.per.region.mgh"))

    fractions_not_specified = pred_error == np.nan
    pred_error[fractions_not_specified] = 0
    abs_error = np.mean(np.abs(pred_error), axis=1)
    abs_error[np.any(fractions_not_specified,axis=1)] = -1

    overlay = nib.freesurfer.mghformat.MGHImage(abs_error.astype("float32"), np.eye(4))
    nib.save(overlay, outpath / Path(f"lh.abs.error.average.per.region.mgh"))

    # Get the best over all subjects
    fractions = _equivolume_fit_smooth(
        isodist_errors, knn, number_of_nodes
    )
    overlay = nib.freesurfer.mghformat.MGHImage(fractions.astype("float32"), np.eye(4))
    nib.save(overlay, outpath / Path(f"lh.best.local.isodist.frac.mgh"))


    # Also fit the global equidist fraction
    pred_error_global = np.zeros((number_of_nodes, number_of_subjects))
    minimum_fraction_indices = np.zeros(number_of_subjects)
    fracs = np.linspace(0.2, 0.8, num_fracs)
    for s in range(number_of_subjects):
        selector = [x for x in range(number_of_subjects) if x != s]
        isodist_training = isodist_errors[:, :, selector]
        average_error = np.mean(np.mean(isodist_training, axis=0), axis=-1)
        minimum_fraction_index = np.argmin(average_error)
        min_average_error = average_error[minimum_fraction_index]

        print(f"Minimum average error: {min_average_error} minimum fraction index: {minimum_fraction_index}")
        isodist_tmp = isodist_errors[:, minimum_fraction_index, s]
        pred_error_global[:, s] = isodist_tmp
        res = pred_error_global[:, s]
        abs_error = np.abs(res)
        minimum_fraction_indices[s] = fracs[minimum_fraction_index]

        # Save absolute error per subject
        overlay = nib.freesurfer.mghformat.MGHImage(
            abs_error.astype("float32"), np.eye(4)
        )
        nib.save(overlay, outpath / Path(f"lh.abs.error.global.sub.{s}.smoothing.mgh"))

    abs_error = np.mean(np.abs(pred_error_global), axis=1)

    overlay = nib.freesurfer.mghformat.MGHImage(abs_error.astype("float32"), np.eye(4))
    nib.save(overlay, outpath / Path(f"lh.abs.error.global.average.smoothing.mgh"))
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
    print(f"Minimum average error: {min_average_error} minimum fraction index: {minimum_fraction_index}")

    np.savetxt(
        outpath / Path("best_fraction_overall.csv"),
        np.atleast_1d(np.array(fracs[minimum_fraction_index])),
        delimiter=",",
    )


def main(
    surf_data_path: os.PathLike,
    output_path: os.PathLike,
    fsav_path: os.PathLike,
    fsav: Hemisphere,
    predictor_names: list[str],
    target_name: str,
    data_path: os.PathLike,
    neighborhood_size: int=1,
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
    Returns
    -------
    """

    # Before fitting the models, let's map all the single right hemi results to the left hemi
    # _map_rh_to_lh(surf_data_folder, fsav_path)

    # Okay start out by fitting the linear model

    predictors, target_values, inf_thickness, thickness, names = _prepare_data_for_linear_fit(surf_data_path, target_name, predictor_names, clip_range=(0.1, 99.9))

    knn, kr = fsav.white.k_ring_neighbors(neighborhood_size)  
    number_of_nodes = fsav.white.n_vertices

    # Run leave-one-out cross-validation for linear model

    # out_files = ["intercept", "beta_k1", "beta_k2", "beta_k1k2"]
    # outpath = data_path / f"linear_model_neighborhood_size_{neighborhood_size}"

    # _cv_linear_fit(
    # predictors,
    # target_values,
    # inf_thickness,
    # thickness,
    # out_files,
    # outpath,
    # number_of_nodes,
    # knn,
    # )


    # Next cross-validate the isovolume values
    # outpath = data_path / "equivolume_model_smoothed"
    # number_of_subjects = predictors.shape[0]

    # _cv_equivol_fit(data_path, outpath, number_of_nodes, number_of_subjects, knn)

    # Next cross-validate the isodistance values
    outpath = data_path / "equidistance_model_smoothed"
    number_of_subjects = predictors.shape[0]
    _cv_equidist_fit(data_path, outpath, number_of_nodes, number_of_subjects, knn)

def _map_rh_to_lh(
    surf_data_path: os.PathLike,
    fsav_path: os.PathLike):

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

    for f in surf_data_path.glob('**/*'):
        if 'lh.' in f.name or not 'fsaverage' in f.name:
                continue

        print(f'Processing: {f}')
        call = 'mris_apply_reg --src ' + str(f) + ' --trg ' +  str(f.parent) + '/to.lh.' + f.name + ' --streg ' + str(fsav_path) + '/xhemi/surf/lh.fsaverage_sym.sphere.reg ' + str(fsav_path) + '/surf/lh.fsaverage_sym.sphere.reg'
        os.system(call)

if __name__ == '__main__':

    # Grab the FS dir from the environ
    fs_home = Path(os.environ["FREESURFER_HOME"])
    if not fs_home.exists():
        raise Exception("FREESURFER_HOME not set")

    fsav_path = fs_home / "subjects" / "fsaverage"
    fsav = Hemisphere.from_freesurfer_subject_dir(fsav_path, "lh")

    surf_data_folder = Path("/autofs/space/rauma_001/users/op035/data/exvivo/hires_surf/analysis/cortech_thicknesses/smooth_step_surf_5_smooth_steps_curv_0/")
    # Could use some the thickness gradient as well?
    # predictor_names = ['thickness_cortex.fsaverage', 'k1.avg.fsaverage', 'k2.avg.fsaverage', 'thickness.gradient.magnitude.fsaverage']
    predictor_names = ['thickness_cortex.fsaverage', 'k1.avg.fsaverage', 'k2.avg.fsaverage']
    target_name = 'thickness.wm.inf.fsaverage'

    main(surf_data_folder, surf_data_folder, fsav_path, fsav, predictor_names, target_name, surf_data_folder)

    # stuff_to_map = ["thickness", "thickness.inf.pial", "thickness.wm.inf"]
    # out_path = Path(f"/autofs/space/rauma_001/users/op035/data/exvivo/hires_surf/analysis/cortech_thicknesses/smooth_step_surf_{smooth_steps_surf}_smooth_steps_curv_{smooth_steps_curv}")
