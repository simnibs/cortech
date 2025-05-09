from collections import namedtuple

import numpy as np
import numpy.typing as npt
import nibabel as nib

import cortech.utils
from cortech.surface import Surface, SphericalRegistration
from cortech.constants import Curvature


class Hemisphere:
    """A class containing surfaces delineating the white-gray matter boundary
    and the gray matter-CSF (pial) boundary, and optionally the infra-supra-layer boundary.

    Additionally, it may contain information about layers.
    """

    def __init__(
        self,
        white: Surface,
        pial: Surface,
        inf=None,
        spherical_registration: None | SphericalRegistration = None,
        infra_supra_model=None,
    ) -> None:
        self.white = white
        self.pial = pial
        self.inf = inf
        self.spherical_registration = spherical_registration
        self.infra_supra_model = infra_supra_model

    def has_spherical_registration(self):
        return self.spherical_registration is not None

    def has_infra_supra_model(self):
        return self.infra_supra_model is not None

    def compute_node_to_node_difference(self) -> npt.NDArray[float]:
        """Calculate thickness at each vertex of node-matched surfaces."""

        # FIXME this should be a better estimate than simple node-to-node
        # distance
        vi = self.white.vertices
        vo = self.pial.vertices
        return np.linalg.norm(vo - vi, axis=1)

    def compute_thickness(self) -> npt.NDArray[float]:
        """This function calculates a FreeSurfer "style" thickness by finding the minimum distance between a white matter
        node and every pial surface node, doing the same the other way around, and averaging those two minimum distances
        for every node. It will also ensure the thickness estimate is between 0 and 5 mm.

        Parameters
        ----------

        Returns
        -------
        thickness : np.array(float)
                        The average, clipped cortical thickness estimate
        """

        vi = self.white.vertices
        vo = self.pial.vertices

        tree1 = KDTree(vi)
        tree2 = KDTree(vo)

        # Compute the closest distance one way
        dist1, inds1 = tree1.query(vo)

        # And the other way
        dist2, inds2 = tree2.query(vi)

        av_min_dist = (dist1 + dist2) / 2

        return np.clip(av_min_dist, 0, 5)

    def compute_average_curvature(
        self,
        white_curv: None | Curvature = None,
        pial_curv: None | Curvature = None,
        curv_kwargs: None | dict = None,
    ):
        """Average curvature estimates of white and pial surfaces."""
        curv_kwargs = curv_kwargs or {}
        white_curv = white_curv or self.white.compute_curvature(**curv_kwargs)
        pial_curv = pial_curv or self.pial.compute_curvature(**curv_kwargs)
        return Curvature(
            **{
                k: 0.5 * (getattr(white_curv, k) + getattr(pial_curv, k))
                for k in white_curv._fields
            }
        )

    def compute_equivolume_fraction(
        self,
        thickness: npt.NDArray,
        curv: npt.NDArray,
        vol_frac: npt.NDArray | float = 0.5,
    ):
        """Compute the distance fraction (between inner and outer surface whose
        distance is `thickness`) which yields the desired volume fraction at each
        position.

        Parameters
        ----------
        curv : npt.NDArray
            Curvature estimate at each position.
        thickness : npt.NDArray
            Thickness estimate at each position.
        vol_frac : float
            The desired volume fraction. Defaults to 0.5.

        Returns
        -------
        dist_frac : npt.NDArray
            Position-wise distance fraction which yields the desired volume
            fraction.

        Notes
        -----
        We expect positive curvature in sulci and negative curvature on gyral
        crowns.

        On gyral crowns, the curvature is negative because the surface bends away
        from the normals. Since a radius must be positive, we ignore the sign when
        computing the radius of the sphere which gives the desired volume fraction.

        In the sulci, the curvature is positive as the surface bends towards the
        normal. However, we are still placing the center of the sphere on the side
        of the white matter (there is nothing to distinguish concave and convex
        when we ignore the sign of the curvature). Assuming the sphere was placed
        on the pial side, we could find the desired volume fraction by first
        estimating the radius of the sphere at 1-frac (since 1-vol_frac from the
        outer side is vol_frac from the inner side) and then subtract this
        (relative to the inner radius) from the thickness to get a distance from
        the inner surface rather than the outer.

        References
        ----------
        Michiel Kleinnijenhuis et al. (2015). Diffusion tensor characteristics of
            gyrencephaly using high resolution diffusion MRI in vivo at 7T.
        """
        frac = np.atleast_1d(vol_frac)

        # Name variables in accordance with the reference
        R = 1 / np.abs(curv)
        R3 = R**3
        T = thickness
        r = np.zeros((len(frac), len(curv)))

        pos = curv > 0
        neg = curv < 0

        r[:, neg] = cortech.utils.compute_sphere_radius(frac, T[neg], R[neg], R3[neg])
        r[:, pos] = cortech.utils.compute_sphere_radius(
            1 - frac, T[pos], R[pos], R3[pos]
        )
        # r[curv == 0] should be `frac` but we take care of this in `dist_frac` below

        # r is a radius in between R and R+T so subtract R to get back to distances
        # that relate to the thickness
        r -= R

        # Correct positive curvature positions
        r[:, pos] *= -1
        r[:, pos] += T[pos]

        # Ensure r is valid
        _ = np.clip(r, 0, T, out=r)

        # Now compute the pointwise distance fraction
        dist_frac = np.zeros_like(r)
        _ = np.divide(r, T, out=dist_frac, where=T > 0)

        return dist_frac.squeeze()

    def fit_infra_supra_border(self):
        """Fit the infra supra border using models fitted from ex-vivo data.

        Parameters
        ----------

        Returns
        -------
        surfaces : dict{method, Surface}
                   A dictionary of infra-supra surfaces fitted with different methods.

        """
        if not self.has_infra_supra_model():
            print('No model for the infra supra border loaded!')
            return

        curv = self.compute_average_curvature()
        thickness = self.compute_thickness()

        surfaces = {}
        
        for model in self.infra_supra_model:
            if "equivolume" in model:
                surfaces[model] = self.estimate_layers(method="equivolume", frac=self.infra_supra_model[model], thickness=thickness, curv=curv.H)
            elif "equidistance" in model:
                surfaces[model] = self.estimate_layers(method="equidistance", frac=self.infra_supra_model[model])
            elif "linear" in model:
                surfaces[model] = self._predict_linear_model(self.infra_supra_model[model], curv)
            else:
                print("Unknown model!")

        return surfaces

    def _layer_from_distance_fraction(self, f: float | npt.NDArray = 0.5):
        """_summary_

        Parameters
        ----------
        vi : npt.NDArray
        vo : npt.NDArray
        frac : float | npt.NDArray
            Fraction measured from inner surface, e.g., 0.25 would be one quarter
            of the way from the inner to the outer surface. Defaults to 0.5.
            frac = float | (1, ) | (n_frac,) | (n_vertices, ) | (n_frac, n_vertices)

        Returns
        -------
        _type_: _description_
        """
        f = np.atleast_1d(f)
        if f.ndim == 1:
            if f.shape[0] == self.white.n_vertices:
                f = f[None]
        elif f.ndim == 2:
            fd2 = f.shape[1]
            assert fd2 == 1 or fd2 == self.white.n_vertices
        f = cortech.utils.atleast_nd(f, 3)
        return np.squeeze((1 - f) * self.white.vertices + f * self.pial.vertices)


    def _predict_linear_model(parameters: npt.NDArray, curv: Curvature, clip_range=(0.01, 99.9)):
        """Predict the infra supra border using a linear model fitted on the ex-vivo data.
        NOTE: This a little suboptimal at the moment as the model is fixed, i.e., it needs
        [1, k1, k2, k1k2] whereas the parameters are passed in and are thus more general.
        A better way would be to store the parameters and the needed predictors together.

        Parameters
        ----------
        parameters : np.array(float), number_of_nodes x number_of_parameters
                     The parameters of the linear model.
        curv : Curvature
               A curvature object storing the average curvature values.
        clip_range : tuple(float)
               Percentile ranges to clip the curvature values.

        Returns
        -------
        surface : Surface
                  The predicted infra-supra surface
        """

        pk1 = np.percentile(curv.k1, clip_range)
        pk2 = np.percentile(curv.k2, clip_range)

        k1 = np.clip(curv.k1, pk1[0], pk1[1]).T
        k2 = np.clip(curv.k2, pk2[0], pk2[1]).T

        k1 = k1 - k1.mean()
        k2 = k2 - k2.mean()

        k1k2 = k1*k2
        dummy = np.ones_like(k1)

        predictors = np.stack([dummy, k1, k2, k1k2])

        frac = np.einsum("ij, ij -> i", predictors, parameters)
        frac = np.clip(frac, 0, 1)
        surface = self._layer_from_distance_fraction(frac)

        return surface
        


    def estimate_layers(
        self,
        method: str = "equivolume",
        frac: float | npt.NDArray = 0.5,
        thickness: npt.NDArray | None = None,
        curv: str | npt.NDArray = "H",
        curv_kwargs: dict | None = None,
    ):
        """Estimate layers at `frac`. Given an estimate of the thickness at
        each vertex (and, for the equivolume model a curvature estimate),
        return the positions of one or more layers defined by the fraction in
        `frac`.

        Currently, this function relies on vertex-to-vertex correspondence
        between white and pial surfaces in that layers are placed

        Parameters
        ----------
        method : str
            The layer placement method to use. This determines how `frac` is to
            be interpreted.
        frac : float | NDArray
            The fraction(s) in between white and pial surfaces at which to
            estimate layer(s). When `method` is equidistance, `frac` is a
            distance fraction. When `method` is equivolume, `frac` is a volume
            fraction.
        thickness :
            The thickness at each (white, pial) vertex pairs. If None, it will
            be estimated.
        curv :
            Curvature estimate at each (white, pial) vertex pairs. If a string,
            it should be an attribute of cortech.constants.Curvature. By
            default, the mean curvature is estimated and used.

        Returns
        -------
        Position of vertices at the desired (distance or volume) fractions
        (n_vertices, n_frac).

        """
        if method in {"equidistance", "equivolume"}:
            match method:
                case "equivolume":
                    if thickness is None:
                        thickness = self.compute_thickness()
                    match curv:
                        case str():
                            curv = getattr(
                                self.compute_average_curvature(curv_kwargs=curv_kwargs),
                                curv,
                            )
                        case _:
                            curv = np.asarray(curv)
                    frac = self.compute_equivolume_fraction(thickness, curv, frac)
                case "equidistance":
                    pass
            return self._layer_from_distance_fraction(frac)
        elif method == "laplace":
            return

    @classmethod
    def from_freesurfer_subject_dir(
        cls,
        sub_dir,
        hemi,
        white="white",
        pial="pial",
        inf=None,
        sphere="sphere",
        spherical_registration="sphere.reg",
        infra_supra_model_type_and_path=None,
    ):
        assert hemi in {"lh", "rh"}

        white = Surface.from_freesurfer_subject_dir(sub_dir, f"{hemi}.{white}")
        pial = Surface.from_freesurfer_subject_dir(sub_dir, f"{hemi}.{pial}")
        if inf is not None:
            inf = Surface.from_freesurfer_subject_dir(sub_dir, f"{hemi}.{inf}")

        if spherical_registration is not None:
            spherical_registration = SphericalRegistration.from_freesurfer_subject_dir(
                sub_dir, f"{hemi}.{spherical_registration}"
            )
        if infra_supra_model_type_and_path is not None:
            infra_supra_model = {}
            number_of_nodes = white.n_vertices
            for model_type in infra_supra_model_type_and_path.keys():
                if "equivolume" in model_type or "equidistance" in model_type:
                    if "global" in model_type:
                        global_frac = np.loadtxt(infra_supra_model_type_and_path[model_type])
                        infra_supra_model[model_type] = global_frac.item()
                    elif "local" in model_type:
                        local_frac_im = nib.load(infra_supra_model_type_and_path[model_type])
                        local_frac_fsav = local_frac_im.get_fdata().squeeze()
                        local_frac = spherical_registration.resample(local_frac_fsav)
                        infra_supra_model[model_type] = local_frac
                elif "linear" in model_type:
                    # NOTE: for the linear model the order matters! I'll keep it general now,
                    # i.e., the order of the input files defines the parameter order.
                    # For the models I have fitted it should be [intercept, k1, k2, k1k2]
                    # I should find a better way to save this so that the function actually
                    # predicting the surface now has fixed parameters.
                    parameters_fsav = []
                    for parameter_file in infra_supra_model_type_and_path[model_type]:
                        param_im = nib.load(parameter_file)
                        param_tmp = param_im.get_fdata().squeeze()
                        parameters_fsav.append(param_tmp)

                    parameters_fsav = np.array(parameters_fsav).transpose()
                    parameters = spherical_registration.resample(parameters_fsav)
                    infra_supra_model[model_type] = parameters

            if not infra_supra_model:
                print('Model loading failed, using the default (equivolume with fraction 0.5)')
                infra_supra_model['equivolume_global'] = 0.5

        return cls(white, pial, inf=inf, spherical_registration=spherical_registration, infra_supra_model=infra_supra_model)


class Cortex:
    def __init__(self, lh: Hemisphere, rh: Hemisphere) -> None:
        """
        An iterator over hemisphere objects.

        Parameters
        ----------
        lh, rh : Hemisphere
            _description_
        """
        self.lh = lh
        self.rh = rh
        self.hemispheres = [self.lh, self.rh]

    def __len__(self):
        return len(self.hemispheres)

    def __getitem__(self, index):
        return self.hemispheres[index]

    @staticmethod
    def iterate_over_hemispheres(method):
        def wrapper(self, *args, **kwargs):
            t = namedtuple(f"{method.__name__}_result", ("lh", "rh"))
            return t(*[getattr(i, method.__name__)(*args, **kwargs) for i in self])

        return wrapper

    @iterate_over_hemispheres
    def compute_average_curvature(self):
        pass

    @iterate_over_hemispheres
    def compute_thickness(self):
        pass

    @iterate_over_hemispheres
    def compute_equivolume_fraction(self):
        pass

    @iterate_over_hemispheres
    def estimate_layers(self):
        pass

    @iterate_over_hemispheres
    def has_spherical_registration(self):
        pass

    @classmethod
    def from_freesurfer_subject_dir(cls, sub_dir, *args, **kwargs):
        return cls(
            Hemisphere.from_freesurfer_subject_dir(sub_dir, "lh", *args, **kwargs),
            Hemisphere.from_freesurfer_subject_dir(sub_dir, "rh", *args, **kwargs),
        )

    def __str__(self):
        s = "\n".join(f"{h} : {i}" for h, i in zip(("lh", "rh"), self))
        return s
