import sys
from pathlib import Path

from cortech.cortex import Hemisphere
from cortech.cortex import Surface
import nibabel as nib
import numpy as np
import numpy.typing as npt
import shutil

from pathlib import Path
from typing import Union


# Paths to the various surface models, don't change these

curv_smoothing = {}
curv_smoothing['equidistance_global'] = 0
curv_smoothing['equidistance_local'] = 0
curv_smoothing['equivolume_global'] = 0
curv_smoothing['equivolume_local'] = 15
curv_smoothing['linear_model'] = 15

curv_smoothing_josa = {}
curv_smoothing_josa['equidistance_global'] = 0
curv_smoothing_josa['equidistance_local'] = 0
curv_smoothing_josa['equivolume_global'] = 0
curv_smoothing_josa['equivolume_local'] = 15
curv_smoothing_josa['linear_model'] = 0

model_dict_lh = {}
model_dict_lh['equidistance_global'] = './models/equidistance/spherical_reg/global_fraction.csv'
model_dict_lh['equidistance_local'] = './models/equidistance/spherical_reg/lh.equidistance.neighbor.frac.mgh'
model_dict_lh['equivolume_global'] = './models/equivolume/spherical_reg/global_fraction.csv'
model_dict_lh['equivolume_local'] = './models/equivolume/spherical_reg/lh.equivolume.neighbor.frac.mgh'
model_dict_lh['linear_model'] = ['./models//linear_model/spherical_reg/lh.intercept.linear_model.mgh',
                                 './models//linear_model/spherical_reg/lh.beta_k1.linear_model.mgh',
                                 './models//linear_model/spherical_reg/lh.beta_k2.linear_model.mgh',
                                 './models//linear_model/spherical_reg/lh.beta_k1k2.linear_model.mgh']

model_dict_rh = {}
model_dict_rh['equidistance_global'] = './models/equidistance/spherical_reg/global_fraction.csv'
model_dict_rh['equidistance_local'] = './models/equidistance/spherical_reg/rh.equidistance.neighbor.frac.mgh'
model_dict_rh['equivolume_global'] = './models/equivolume/spherical_reg/global_fraction.csv'
model_dict_rh['equivolume_local'] = './models/equivolume/spherical_reg/rh.equivolume.neighbor.frac.mgh'
model_dict_rh['linear_model'] = ['./models//linear_model/spherical_reg/rh.intercept.linear_model.mgh',
                                 './models//linear_model/spherical_reg/rh.beta_k1.linear_model.mgh',
                                 './models//linear_model/spherical_reg/rh.beta_k2.linear_model.mgh',
                                 './models//linear_model/spherical_reg/rh.beta_k1k2.linear_model.mgh']

model_dict = {'lh': model_dict_lh, 'rh': model_dict_rh}

model_dict_josa_lh = {}
model_dict_josa_lh['equidistance_global'] = './models/equidistance/josa/global_fraction.csv'
model_dict_josa_lh['equidistance_local'] = './models/equidistance/josa/lh.equidistance.neighbor.frac.mgh'
model_dict_josa_lh['equivolume_global'] = './models/equivolume/josa/global_fraction.csv'
model_dict_josa_lh['equivolume_local'] = './models/equivolume/josa/lh.equivolume.neighbor.frac.mgh'
model_dict_josa_lh['linear_model'] = ['./models//linear_model/josa/lh.intercept.linear_model.mgh',
                                      './models//linear_model/josa/lh.beta_k1.linear_model.mgh',
                                      './models//linear_model/josa/lh.beta_k2.linear_model.mgh',
                                      './models//linear_model/josa/lh.beta_k1k2.linear_model.mgh']

model_dict_josa_rh = {}
model_dict_josa_rh['equidistance_global'] = './models/equidistance/josa/global_fraction.csv'
model_dict_josa_rh['equidistance_local'] = './models/equidistance/josa/rh.equidistance.neighbor.frac.mgh'
model_dict_josa_rh['equivolume_global'] = './models/equivolume/josa/global_fraction.csv'
model_dict_josa_rh['equivolume_local'] = './models/equivolume/josa/rh.equivolume.neighbor.frac.mgh'
model_dict_josa_rh['linear_model'] = ['./models//linear_model/josa/rh.intercept.linear_model.mgh',
                                      './models//linear_model/josa/rh.beta_k1.linear_model.mgh',
                                      './models//linear_model/josa/rh.beta_k2.linear_model.mgh',
                                      './models//linear_model/josa/rh.beta_k1k2.linear_model.mgh']

model_dict_josa = {'lh': model_dict_josa_lh, 'rh': model_dict_josa_rh}

# Path to where the FS runs are. This one needs to be changed per data set.
data_path = Path("/mnt/projects/CORTECH/nobackup/hires_invivo_datasets/highres_invivo_amsterdam/fsruns_hires")
# data_path = Path("/autofs/space/rauma_001/users/op035/data/HCP_test/")
# data_path = Path("/autofs/space/rauma_001/users/op035/data/test_invivo_fit//")
# data_path = Path("/autofs/space/rauma_001/users/op035/data/hires_invivo_datasets/highres_invivo_amsterdam/fsruns_hires/")
# data_path = Path("/autofs/space/rauma_001/users/op035/data/exvivo/derivatives/surface_reconstructions_with_retrained_multiresolution_unet_model/")

# Get fsaverage

for p in data_path.glob("*/"):
    print(p)

    if not p.is_dir() or "fsaverage" in str(p):
        continue

    sub = p.stem
    print(f"Processing subject {sub}")

    for hemi in ['lh', 'rh']:

        if not (p / "surf" / f"{hemi}.white").exists() or not (p / "surf" / f"{hemi}.pial").exists():
            print(f"Cannot find surface for subject {sub}")
            continue

        for model in model_dict[hemi].keys():
            surf = Hemisphere.from_freesurfer_subject_dir(p ,hemi, infra_supra_model_type_and_path={model: model_dict[hemi][model]}, registration='sphere.reg')
            surf.white.smooth_taubin(n_iter=5, inplace=True)
            surf.pial.smooth_taubin(n_iter=5, inplace=True)
            surfs_tmp = surf.fit_infra_supra_border(curv_args={"smooth_iter": curv_smoothing[model]}, return_surface=True)
            surfs_tmp[model].save(f"{p}/surf/{hemi}.inf.{model}")

        for model in model_dict_josa[hemi].keys():
            surf = Hemisphere.from_freesurfer_subject_dir(p ,hemi, infra_supra_model_type_and_path={model: model_dict_josa[hemi][model]}, registration='josa.sphere.reg')
            surf.white.smooth_taubin(n_iter=5, inplace=True)
            surf.pial.smooth_taubin(n_iter=5, inplace=True)
            surfs_tmp = surf.fit_infra_supra_border(curv_args={"smooth_iter": curv_smoothing_josa[model]}, return_surface=True)
            surfs_tmp[model].save(f"{p}/surf/{hemi}.inf.josa.{model}")
