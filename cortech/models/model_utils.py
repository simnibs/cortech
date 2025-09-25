import numpy as np
import nibabel as nib
from pathlib import Path
from cortech.models import MODEL_DIR


def load_model(model_name, local, hemi, registration="spherical"):
    assert model_name in {"equivolume", "equidistance", "linear_model"}
    assert hemi in {"lh", "rh"}
    assert local in {"local", "global"}
    assert registration in {"spherical", "josa"}

    linear_model_files = [
        "intercept.linear_model.mgh",
        "beta_k1.linear_model.mgh",
        "beta_k2.linear_model.mgh",
        "beta_k1k2.linear_model.mgh",
    ]

    model_dir = Path(MODEL_DIR)
    model_dict = {}
    if "equivolume" in model_name or "equidistance" in model_name:
        if "global" in local:
            global_frac = np.loadtxt(
                model_dir / model_name / registration / "global_fraction.csv"
            )
            model_dict[f"{model_name}_{local}"] = np.atleast_1d(global_frac.item())
        elif "local" in local:
            local_frac_im = nib.load(
                model_dir
                / model_name
                / registration
                / f"{hemi}.{model_name}.neighbor.frac.mgh"
            )
            local_frac_fsav = local_frac_im.get_fdata().squeeze()
            model_dict[f"{model_name}_{local}"] = local_frac_fsav
    elif "linear" in model_name:
        # NOTE: for the linear model the order matters! I'll keep it general now,
        # i.e., the order of the input files defines the parameter order.
        # For the models I have fitted it should be [intercept, k1, k2, k1k2]
        # I should find a better way to save this so that the function actually
        # predicting the surface now has fixed parameters.
        parameters_fsav = []
        for parameter_file in linear_model_files:
            param_im = nib.load(
                model_dir / model_name / registration / f"{hemi}.{parameter_file}"
            )
            param_tmp = param_im.get_fdata().squeeze()
            parameters_fsav.append(param_tmp)

        parameters_fsav = np.array(parameters_fsav).transpose()
        model_dict[f"{model_name}_{local}"] = parameters_fsav

    return model_dict
