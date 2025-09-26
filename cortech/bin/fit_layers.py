import argparse
import ast
from pathlib import Path
from cortech.cortex import Cortex, Hemisphere
from cortech.surface import Surface, Sphere


def fit_layers(
    model_type,
    layer_type=None,
    fractions=None,
    freesurfer_dir=None,
    output_path=None,
    global_flag=False,
    registration="spherical",
    input_surfaces=None,
):
    """
    Fit layers based on the provided data and model type.

    Parameters:
    model_type: The type of model to use for fitting. Options are 'equidistance', 'equivolume', and 'linear_model'.
    layer_type: The type of layer to fit. Options are 'layers' and 'infra-supra-border' (only applicable for equidistance and equivolume).
    fractions: A list of floats between 0 and 1 (only used when layer_type is 'layers').
    freesurfer_dir: Path to the FreeSurfer directory.
    output_path: Path to save the output surfaces.
    global_flag: A boolean flag to indicate global processing.
    registration: The type of registration to use, either 'spherical' or 'josa'.
    input_surfaces: A list of two surface file paths.

    Returns:
    None

    """
    # If input_surfaces is provided, create Hemisphere objects instead of Cortex
    if input_surfaces is not None:
        surfaces_in = [Surface.from_file(path) for path in input_surfaces[:2]]
        hemi = "lh" if "lh" in input_surfaces[0] else "rh"
        sphere = Sphere.from_file(input_surfaces[-1])
        hemispheres = [Hemisphere(hemi, *surfaces_in, registration=sphere)]
    else:
        # If freesurfer_dir exists, create a Cortex object
        cortex = (
            Cortex.from_freesurfer_subject_dir(
                freesurfer_dir,
                registration="sphere.reg"
                if "spherical" in registration
                else "josa.sphere.reg",
            )
            if freesurfer_dir is not None
            else None
        )
        hemispheres = [cortex.lh, cortex.rh]

    if input_surfaces is None and freesurfer_dir is None:
        raise ValueError("FreeSurfer directory of input surfaces need to be provided")

    # Set output_path to freesurfer_dir / "surf" if output_path is None and freesurfer_dir is not None
    if output_path is None and freesurfer_dir is not None:
        output_path = Path(freesurfer_dir) / "surf"

    # Placeholder for fitting logic
    if model_type == "equidistance":
        if layer_type not in ["layers", "infra-supra-border"]:
            raise ValueError(
                "For 'equidistance', layer_type must be 'layers' or 'infra-supra-border'."
            )
        if layer_type == "layers":
            if fractions is None:
                raise ValueError(
                    "Fractions must be provided when layer_type is 'layers'."
                )
            if any(x < 0 or x > 1 for x in fractions):
                raise ValueError(
                    "All elements in the fractions list must be between 0 and 1."
                )

            # Loop over hemispheres in the cortex object and save the surfaces
            for hemisphere in hemispheres:
                surfaces = hemisphere.estimate_layers(
                    method="equidistance", frac=fractions, return_surface=True
                )
                if type(surfaces) is not tuple:
                    surfaces = tuple([surfaces])

                for frac, surface in zip(fractions, surfaces):
                    surface.save(
                        Path(output_path) / f"{hemisphere.name}.{model_type}.{frac}"
                    )
        elif layer_type == "infra-supra-border":
            model_name = (
                "equidistance",
                "global" if global_flag else "local",
                registration,
            )
            for hemisphere in hemispheres:
                hemisphere.set_infra_supra_model(model_name)
                infra_supra_surface = hemisphere.fit_infra_supra_border(
                    curv_args={"smooth_iter": 0}
                    if global_flag
                    else {"smooth_iter": 15},
                    return_surface=True,
                )
                infra_supra_surface[f"{model_name[0]}_{model_name[1]}"].save(
                    Path(output_path)
                    / f"{hemisphere.name}.inf.{model_type}.{('global' if global_flag else 'local')}"
                )

    elif model_type == "equivolume":
        if layer_type not in ["layers", "infra-supra-border"]:
            raise ValueError(
                "For 'equivolume', layer_type must be 'layers' or 'infra-supra-border'."
            )
        if layer_type == "layers":
            if fractions is None:
                raise ValueError(
                    "Fractions must be provided when layer_type is 'layers'."
                )
            if any(x < 0 or x > 1 for x in fractions):
                raise ValueError(
                    "All elements in the fractions list must be between 0 and 1."
                )

            # Loop over hemispheres in the cortex object and save the surfaces
            for hemisphere in hemispheres:
                surfaces = hemisphere.estimate_layers(
                    method="equivolume", frac=fractions, return_surface=True
                )
                if type(surfaces) is not tuple:
                    surfaces = tuple([surfaces])
                for frac, surface in zip(fractions, surfaces):
                    surface.save(
                        Path(output_path) / f"{hemisphere.name}.{model_type}.{frac}"
                    )
        elif layer_type == "infra-supra-border":
            model_name = (
                "equivolume",
                "global" if global_flag else "local",
                registration,
            )
            for hemisphere in hemispheres:
                hemisphere.set_infra_supra_model(model_name)
                infra_supra_surface = hemisphere.fit_infra_supra_border(
                    curv_args={"smooth_iter": 0}
                    if global_flag
                    else {"smooth_iter": 15},
                    return_surface=True,
                )
                infra_supra_surface[f"{model_name[0]}_{model_name[1]}"].save(
                    Path(output_path)
                    / f"{hemisphere.name}.inf.{model_type}.{('global' if global_flag else 'local')}"
                )

    elif model_type == "linear_model":
        if layer_type != "infra-supra-border":
            raise ValueError(
                "For 'linear_model', layer_type must be 'infra-supra-border'."
            )
        model_name = ("linear_model", "local", registration)
        for hemisphere in hemispheres:
            hemisphere.set_infra_supra_model(model_name)
            infra_supra_surface = hemisphere.fit_infra_supra_border(
                curv_args={"smooth_iter": 0}
                if registration == "josa"
                else {"smooth_iter": 15},
                return_surface=True,
            )
            infra_supra_surface[f"{model_name[0]}_{model_name[1]}"].save(
                Path(output_path) / f"{hemisphere.name}.inf.{model_type}"
            )

    else:
        raise ValueError(
            "Invalid model type. Choose from 'equidistance', 'equivolume', or 'linear_model'."
        )

    return None


def main():
    parser = argparse.ArgumentParser(
        prog="fit_layers",
        description="Fit cortical layers based on the different models.",
        usage="""
        Usage examples:
        Fit three layers with volume fractions 0.1, 0.5, and 0.8 given the white and pial surface and save to an output path:
        fit_layers equivolume --layer_type layers --fractions "[0.1, 0.5, 0.8]" --input_surfaces lh.white lh.pial lh.sphere.reg --output_path /tmp/outpath

        Fit two layers with volume fractions 0.2 and 0.6 given a path two a recon-all output (processes both hemispheres):
        fit_layers equivolume --layer_type layers --fractions "[0.2, 0.6]" --freesurfer_dir $SUBJECTS_DIR/subid

        Fit the infra-supra border using a fitted local equdistance model (distance fractions fitted at each node of the fsaverage surface). Requires FreeSurfer.
        fit_layers equidistance --layer_type infra-supra-border --freesurfer_dir $SUBJECTS_DIR/subid

        The same as above but use the equivolume model but with a single fitted global parameter over the whole cortex (--isglobal).
        fit_layers equivolume --layer_type infra-supra-border --isglobal --freesurfer_dir $SUBJECTS_DIR/subid

        The same as above but use a linear model fitted to the local curvature and thickness. Requires FreeSurfer.
        fit_layers linear_model --layer_type infra-supra-border --freesurfer_dir $SUBJECTS_DIR/subid

        """,
    )
    parser.add_argument(
        "model_type",
        type=str,
        choices=["equidistance", "equivolume", "linear_model"],
        help="The type of model to use for fitting.",
    )
    parser.add_argument(
        "--layer_type",
        type=str,
        choices=["layers", "infra-supra-border"],
        help="The type of fit: layers or only the infra-supra-border. Option layers only applicable for equidistance and equivolume.",
    )
    parser.add_argument(
        "--fractions",
        type=str,
        help='A list of floats between 0 and 1 (only for layer_type "layers").',
    )
    parser.add_argument(
        "--freesurfer_dir",
        type=str,
        help="Path to a subject directory output by recon-all.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the output surfaces (if freesurfer_dir is provided, surfaces are saved into the surf/ folder).",
    )
    parser.add_argument(
        "--isglobal",
        action="store_true",
        help="For the infra-supra-border fitting you can use a locally (default) or globally fitted model",
    )
    parser.add_argument(
        "--registration",
        type=str,
        choices=["spherical", "josa"],
        default="spherical",
        help="Switching between spherical registration models. Only affects the infra-supra-border option. Default is spherical.",
    )
    parser.add_argument(
        "--input_surfaces",
        type=str,
        nargs=3,
        help="Path to white and pial surfaces, and the spherical registration (in that order) for a hemi. Can be used instead of the freesurfer_dir, but needs output_dir to be defined.",
    )

    args = parser.parse_args()
    fractions = ast.literal_eval(args.fractions) if args.fractions else None
    freesurfer_dir = Path(args.freesurfer_dir) if args.freesurfer_dir else None
    output_path = Path(args.output_path) if args.output_path else None
    fit_layers(
        args.model_type,
        args.layer_type,
        fractions,
        freesurfer_dir,
        output_path,
        args.isglobal,
        args.registration,
        args.input_surfaces,
    )


if __name__ == "__main__":
    main()
