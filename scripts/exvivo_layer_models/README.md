# Ex vivo layer models

Scripts for fitting the infra-supragranular border models in `cortech/models`
from ex vivo FreeSurfer runs.

1. `process_exvivo_data.py` maps surface measures of each subject to fsaverage
   and sweeps equivolume/equidistance layer fractions.
2. `fit_layer_models.py` fits the linear, equivolume, and equidistance models
   on the processed data, cross-validates them (leave-one-out), and writes the
   final models.
3. `plot_fsaverage_results.py` plots group thickness, curvature, gradients,
   and the model fits on the inflated fsaverage (optional).
4. `compute_thickness_cov.py` computes and plots the coefficient of variation
   across hemispheres of the layer thicknesses, with and without curvature
   correction (optional).
5. `sweep_smoothing.py` runs 1. and 2. for several amounts of smoothing and
   selects the best setting of each model.
6. `border_aligned_profiles.py` compares border placement methods by
   sampling an intensity volume along WM-border-pial trajectories (optional).

## Requirements

- cortech installed (e.g., the `cortex` environment). The plotting scripts
  also need `pyvista` and `matplotlib`.
- FreeSurfer: `FREESURFER_HOME` set and `mris_apply_reg` on `PATH`. It is
  used to map between hemispheres via `fsaverage_sym`. All models are fitted
  on lh of fsaverage, with right hemispheres mapped to lh.

Each subject folder in the input folder needs

```
<subject>/surf/?h.{white,pial,inf}      # inf = infra-supragranular border
<subject>/surf/?h.sphere.reg            # or the file given by --sphere-reg
<subject>/surf/?h.{thickness,thickness.inf.pial,thickness.wm.inf}   # --quantities
<subject>/label/?h.aparc.annot
```

Subjects can have one or both hemispheres. Folders whose name contains
`fsaverage` are skipped.

## Usage

Run the scripts from this folder (or give their full path):

```bash
SUBJECTS=/mnt/projects/CORTECH/nobackup/exvivo/derivatives/final_surfaces
OUT=/mnt/projects/CORTECH/nobackup/exvivo/derivatives/exvivo_surface_analysis/surf5_curv20

python process_exvivo_data.py $SUBJECTS $OUT --surf-smooth 5 --curv-smooth 20
python fit_layer_models.py $OUT
python plot_fsaverage_results.py $OUT          # plots in $OUT/plots
python compute_thickness_cov.py $OUT           # results in $OUT/thickness_cov
```

Use `--help` on either script to see all options. The defaults reproduce the
settings the scripts were previously hard-coded with:

| option (process) | default | |
|---|---|---|
| `--surf-smooth` | 5 | Taubin smoothing iterations (a=0.6, b=-0.61) of white, pial, inf |
| `--curv-smooth` | 20 | smoothing iterations of the curvature estimates |
| `--sphere-reg` | `sphere.reg` | registration to fsaverage, e.g. `josa.sphere.reg` |
| `--frac-min/--frac-max/--n-fracs` | 0.2/0.8/61 | layer fractions in the sweep |
| `--quantities` | thickness thickness.inf.pial thickness.wm.inf | morphometry mapped to fsaverage |
| `--skip-gradients` | off | skip surface gradients (not used by the fits, slow) |

| option (fit) | default | |
|---|---|---|
| `--neighborhood-size` | 1 | k-ring neighborhood for the local fits (0 = vertex only) |
| `--models` | all | any of `linear_model equivolume equidistance` |
| `--subjects` | all | subset of the processed subjects |
| `--target` | `thickness.wm.inf` | linear model: thickness below the border |
| `--thickness` | `thickness_cortex` | linear model: total thickness |
| `--curvature-surface` | `avg` | linear model: curvatures used as predictors |

| option (plot) | default | |
|---|---|---|
| `--models-dir` | `<processed>/models_neighborhood_1` | fit to plot (skipped if missing) |
| `--out-dir` | `<processed>/plots` | |
| `--subjects` | all | subset of the processed subjects |

| option (CoV) | default | |
|---|---|---|
| `--curvature-surface` | `inf` | surface whose mean curvature is regressed out |
| `--curv-smooth` | 10 | smoothing iterations of that mean curvature |
| `--out-dir` | `<processed>/thickness_cov` | |
| `--subjects` | all | subset of the processed subjects |

The other scripts read the subjects folder, smoothing, registration, and
fraction grid from the processing output, so these only need to be set once.
`compute_thickness_cov.py` recomputes the thicknesses in subject space (the
curvature correction is done per hemisphere before mapping to fsaverage) with
the same surface smoothing and registration, so its uncorrected maps equal
`group/all.on.lh.thickness_<cortex|inf|sup>.mean.mgh`.

### Parameter sweeps

`sweep_smoothing.py` runs processing and fitting for several amounts of
smoothing (by default surface smoothing 5 and curvature smoothing 0, 5, 10,
15, 20, neighborhood size 1) and selects the setting with the lowest
cross-validation error for each model (linear model, local and global
equivolume, local and global equidistance):

```bash
python sweep_smoothing.py $SUBJECTS $ANALYSIS
python sweep_smoothing.py $SUBJECTS $ANALYSIS --surf-smooth 0 5 10 --curv-smooth 0 20 --neighborhood-sizes 1 2
```

Each processing run goes in `$ANALYSIS/surf<S>_curv<C>` and each fit in its
`models_neighborhood_<k>` subfolder. Finished runs are skipped, so an
interrupted sweep is resumed by rerunning the command (`--overwrite` redoes
everything). Other options of the two scripts can be passed with
`--process-args` and `--fit-args`, e.g., `--process-args "--n-fracs 31"`.
Outputs:

- `sweep_summary.csv`: cross-validation error of every model and setting.
- `best_models.csv`: the selected setting of each model. A model must be
  applied with the curvature smoothing it was fitted with (`curv_smooth`).
  The cross-validation error of the selected setting is slightly
  optimistic, as the same errors are used for the selection, so
  `selection_corrected_error` gives an approximately corrected error (the
  setting is selected for each left out hemisphere without its own error;
  see the script's docstring). `selection_agreement` is how often that
  per-hemisphere selection agrees with the overall one.
- `best_models/<method>/`: the final models of the selected settings, in the
  layout of `cortech/models/<method>/<spherical|josa>/`.

Gradients are skipped in the sweep unless `--gradients` is given. Single runs
of the other scripts do not overwrite existing output folders unless
`--overwrite` is given.

## Reproducibility

- `<out>/processing_config.json` stores all processing parameters, the
  subjects/hemispheres that were processed, and the cortech git commit
  (`-dirty` if there were uncommitted changes).
- `<out>/models_neighborhood_<k>/fit_config.json` stores the fitting
  parameters together with the processing config.

To reproduce a run, check out the recorded commit and rerun with the recorded
parameters.

## Outputs

### Processing (`<out>/`)

Per subject, `<out>/<subject>/`:

- `?h.<name>.fsaverage.mgh`: morphometry (`thickness`, ...), curvatures
  (`<k1|k2|H|K>.<white|pial|inf|avg>`), and FreeSurfer-style thicknesses
  (`thickness_cortex` white-pial, `thickness_inf` white-inf, `thickness_sup`
  inf-pial) on fsaverage.
- `?h.cortical_measures.csv`: mean/std of the morphometry per aparc region.
- `<equivolume|equidistance>/`: the sweep. For each fraction `f`:
  the layer surface `?h.inf.<method>.<f>`, and
  `?h.<infsup_distance_error|inf_thickness|sup_thickness>.<method>.<f>[.fsaverage].mgh`
  (distance from the predicted to the true border, and the thicknesses below
  and above the predicted border). `?h.<method>_errors.csv` gives the best
  fraction per aparc region.
- `?h.<measure>.gradient.<magnitude|0|1>[.<surface>][.fsaverage].mgh`:
  surface gradients (unless `--skip-gradients`).
- For right hemispheres, every fsaverage overlay also has an
  `rh-on-lh.<name>.fsaverage.mgh` version mapped to lh.

`<out>/group/`: mean and std over subjects of each fsaverage overlay, per
hemisphere (`lh.`, `rh.`) and for all subjects on lh (`all.on.lh.`).

### Fitting (`<out>/models_neighborhood_<k>/`)

- `cv_summary.csv`: mean absolute cross-validation error (mm) of the infra
  thickness for every model. `cv_errors.csv` gives the error of each left
  out hemisphere (mean over fsaverage vertices); its mean over hemispheres is
  the error in `cv_summary.csv`.
- `linear_model/`: final model `?h.<intercept|beta_k1|beta_k2|beta_k1k2>.linear_model.mgh`.
- `equivolume/`, `equidistance/`: final local model `?h.<method>.neighbor.frac.mgh`
  and global model `global_fraction.csv`. `global_fraction_loo.csv` gives the
  global fraction in each cross-validation fold.
- Cross-validation, per left-out subject: `?h.error.<subject>.<model>.mgh`
  (subject space) and `<lh|rh-on-lh>.error.<subject>.<model>.fsaverage.mgh`, and
  `lh.abs.average.error.<model>.mgh` (mean absolute error over subjects).
  `<model>` is `linear_model` or `<method>.<neighbors|per.vertex|regions|global>`.
  `lh.abs.average.error.linear_model.fsaverage_prediction.mgh` is the error
  when the linear model is evaluated directly on fsaverage. Leave-one-out
  coefficients are saved as `lh.<parameter>.loo.<subject>.<hemi>.mgh`.

### Plots (`<out>/plots/`)

PNGs of the inflated lh of fsaverage with a histogram, named
`<mean|std>_<quantity>.png` for group maps over hemispheres (thickness,
infra/supra fraction, curvatures `<H|K|k1|k2|shape_index>_<inf|wm_pial>`,
gradient magnitudes), `corr_<infra|supra>_thickness_<curvature>_inf.png`
(correlation over hemispheres of layer thickness and absolute curvature),
`<equidistance|equivolume>_<global|neighbors>_abs_error.png`,
`<method>_neighbor_fraction.png`, `linear_model_<abs_error|parameter>.png`,
and `fsaverage_avg_<sulc|curv>.png`. `plot_config.json` records the inputs.

### Thickness CoV (`<out>/thickness_cov/`)

`lh.<mean|cov>_<cortical|infra|supra>_thickness[_corrected].mgh` and PNGs of
the same name, `cov_histograms.png`, and `summary.csv` (mean over cortex
vertices of every map and median CoV). `cov_config.json` records the inputs.

### Border-aligned intensity profiles

`border_aligned_profiles.py` samples an intensity volume (qT1, R1, T2*, ...,
registered to the scanner space of the surfaces) of one subject along
WM-border-pial trajectories for each border method, and measures how well
each border keeps laminar intensity features aligned across gyri and sulci
(see the script's docstring for the method). Borders are either read from
`surf/?h.inf.<method>` or, with `--models-dir`, predicted with the fitted
models (as in the cross-validation, including the surface smoothing):

```bash
python border_aligned_profiles.py --subject <sub> --volume <sub>/mri/qT1.nii.gz \
    --models-dir $OUT/models_neighborhood_1 \
    --methods inf equidistance equivolume linear_model --out $OUT/profiles/<sub>
```

`inf` is the true border (`surf/?h.inf`). Subjects that were used to fit the
models are evaluated in-sample. Outputs in `--out`:

- `metrics.csv`: long format (hemi, method, axis, region, metric, value) for
  the normalised and distance axes and the whole cortex and each parcel of
  `--annot`: `gyral_sulcal_shift` (positive = features deeper in gyri),
  `gyral_sulcal_corr`, `gyral_sulcal_rms`, `within_sd`, `border_curv_slope`,
  `profile_curv_slope`, `n_vertices`.
- `border_diagnostics.csv`: median border fraction on the WM-pial line and
  median/95th percentile distance from the line.
- `?h.border_intensity.<method>.mgh`: intensity at the border (subject space).
- `?h.profiles.png`: mean gyral and sulcal profiles on both axes and the
  across-vertex SD for each method.
- `?h.profiles.npz` (with `--save-profiles`): z-scored profiles of the
  vertices in the mask, the axes, curvature, area, gyral/sulcal masks.
- `profiles_config.json`: the parameters of the run.

### Updating the models in cortech

The final model files use the layout of `cortech/models`, so they can be
copied in directly (`spherical` for `sphere.reg`, `josa` for
`josa.sphere.reg`):

```bash
M=$OUT/models_neighborhood_1
cp $M/linear_model/?h.*.linear_model.mgh                 cortech/models/linear_model/spherical/
cp $M/equivolume/?h.equivolume.neighbor.frac.mgh $M/equivolume/global_fraction.csv     cortech/models/equivolume/spherical/
cp $M/equidistance/?h.equidistance.neighbor.frac.mgh $M/equidistance/global_fraction.csv cortech/models/equidistance/spherical/
```
