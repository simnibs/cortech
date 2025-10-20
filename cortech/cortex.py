from collections import namedtuple
import functools
from pathlib import Path

import numpy as np
import numpy.typing as npt
import nibabel as nib
from scipy.spatial import KDTree

import cortech.utils
from cortech.surface import Surface, Sphere
from cortech.constants import Curvature
from cortech.models.model_utils import load_model


class Hemisphere:
    """A class containing surfaces delineating the white-gray matter boundary
    and the gray matter-CSF (pial) boundary, and optionally the infra-supra-layer boundary.

    Additionally, it may contain information about layers.
    """

    def __init__(
        self,
        name: str,
        white: Surface,
        pial: Surface,
        sphere: Sphere | None = None,
        registration: Sphere | None = None,
        inf: Surface | None = None,
        infra_supra_model=None,
    ) -> None:
        assert name in {"lh", "rh"}
        self.name = name
        self.white = white
        self.pial = pial
        self.sphere = sphere
        self.registration = registration
        self.inf = inf
        self.infra_supra_model = infra_supra_model

        self._surfaces = [self.white, self.pial]
        if self.sphere is not None:
            self._surfaces.append(self.sphere)
        if self.registration is not None:
            self._surfaces.append(self.registration)
        if self.inf is not None:
            self._surfaces.append(self.inf)

    def has_registration(self):
        return self.registration is not None

    def has_infra_supra_model(self):
        return self.infra_supra_model is not None

    def compute_node_to_node_difference(self) -> npt.NDArray[float]:
        """Calculate thickness at each vertex of node-matched surfaces."""

        # FIXME this should be a better estimate than simple vertex-to-vertex
        # distance?
        return np.linalg.norm(self.pial.vertices - self.white.vertices, axis=1)

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
        vol_frac: float | npt.NDArray = 0.5,
    ):
        """Compute the distance fraction (between inner and outer surface whose
        distance is `thickness`) which yields the desired volume fraction at
        each position.

        Parameters
        ----------
        curv : npt.NDArray
            Curvature estimate at each position.
        thickness : npt.NDArray
            Thickness estimate at each position.
        vol_frac : float | npt.NDArray
            The desired volume fraction(s). If a float, estimate a single
            distance fraction per vertex. If an ndarray with one dimension,
            assume that `vol_frac` gives a number of fractions to estimate for
            each vertex. If an ndarray with two dimensions, assume that
            `vol_frac` is (n_frac, n_vertices), i.e., the fraction is provided
            explicitly for each vertex (default = 0.5).

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

        nv = len(thickness)

        # Name variables in accordance with the reference
        R = 1 / np.abs(curv)
        R3 = R**3
        T = thickness

        match frac.ndim:
            case 1:
                # broadcast frac against vertices
                r = np.zeros((len(frac), nv))
                frac = np.broadcast_to(frac[:, None], r.shape)
            case 2:
                # assume frac is (n_frac, n_vertices)
                assert frac.shape[-1] == nv
                r = np.zeros(frac.shape)

        pos = curv > 0
        neg = curv < 0

        r[:, neg] = cortech.utils.compute_sphere_radius(
            frac[:, neg], T[neg], R[neg], R3[neg]
        )
        r[:, pos] = cortech.utils.compute_sphere_radius(
            1 - frac[:, pos], T[pos], R[pos], R3[pos]
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

    def fit_infra_supra_border(self, curv_args=None, return_surface=False):
        """Fit the infra supra border using models fitted from ex-vivo data.

        Parameters
        ----------

        Returns
        -------
        surfaces : dict{method, Surface}
                   A dictionary of infra-supra surfaces fitted with different methods.

        """
        if not self.has_infra_supra_model():
            print("No model for the infra supra border loaded!")
            return

        thickness = self.compute_thickness()
        curv = self.compute_average_curvature(curv_kwargs=curv_args)
        surfaces = {}

        for model in self.infra_supra_model:
            print(f"Estimating surfaces using: {model}")
            if "equivolume" in model:
                if (
                    self.infra_supra_model[model].size > 1
                    and not self.infra_supra_model[model].shape[0] == 1
                ):
                    self.infra_supra_model[model] = np.expand_dims(
                        self.infra_supra_model[model], axis=0
                    )
                surfaces[model] = self.estimate_layers(
                    method="equivolume",
                    frac=self.infra_supra_model[model],
                    thickness=thickness,
                    curv=curv.H,
                    return_surface=return_surface,
                )
            elif "equidistance" in model:
                surfaces[model] = self.estimate_layers(
                    method="equidistance",
                    frac=self.infra_supra_model[model],
                    return_surface=return_surface,
                )
            elif "linear" in model:
                surf_tmp = self._predict_linear_model(
                    self.infra_supra_model[model], curv
                )
                if return_surface:
                    surfaces[model] = self.white.new_from(surf_tmp)
                else:
                    surfaces[model] = surf_tmp
            else:
                print("Unknown model!")

        return surfaces

    def _map_infra_supra_model_to_subject(
        self, parameters_fsav: npt.NDArray, registration=None
    ):
        """Map local infra-supra models from fsaverage to subject space.

        Parameters
        ----------
        parameters_fsav : A parameter array of size nodes x params
        registration : An optional spherical registration

        Returns
        -------

        """

        fsavg = Hemisphere.from_freesurfer_subject_dir(
            "fsaverage", self.name, registration="sphere.reg"
        )

        if registration is not None:
            fsavg.registration.project(registration)
        elif self.has_registration():
            fsavg.registration.project(self.registration)
        else:
            raise Exception("Spherical registration not set!")

        params_subject = fsavg.registration.resample(parameters_fsav)
        return params_subject

    def set_infra_supra_model(self, model_name: tuple, append=False):
        """Set infra-supra model, can append to existing models.

        Parameters
        ----------
        model_name : model name, tuple like ('equivolume', 'local', 'spherical')

        Returns
        -------

        """

        model_fsav = load_model(
            model_name[0], model_name[1], self.name, registration=model_name[2]
        )
        params = model_fsav[f"{model_name[0]}_{model_name[1]}"]

        if model_name[1] in "local":
            params = self._map_infra_supra_model_to_subject(params)

        if self.infra_supra_model is None or append is False:
            self.infra_supra_model = {}

        self.infra_supra_model[f"{model_name[0]}_{model_name[1]}"] = params

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
        f = cortech.utils.atleast_nd_append(f, 3)
        return np.squeeze((1 - f) * self.white.vertices + f * self.pial.vertices)

    def _predict_linear_model(
        self, parameters: npt.NDArray, curv: Curvature, clip_range=(0.01, 99.9)
    ):
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
        k1k2 = k1 * k2

        dummy = np.ones_like(k1)

        predictors = np.stack([dummy, k1, k2, k1k2]).T

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
        return_surface: bool = False,
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
        (n_frac, n_vertices, 3).

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
            layers = self._layer_from_distance_fraction(frac)
            if return_surface:
                if layers.ndim == 2:
                    layers = self.white.new_from(layers)
                else:
                    layers = tuple(self.white.new_from(v) for v in layers)
            return layers
        elif method == "laplace":
            raise NotImplementedError

    def save(
        self,
        out_dir,
        ext: str | None = None,
        *,
        white="white",
        pial="pial",
        sphere="sphere",
        registration="sphere.reg",
        inf="infra-supra",
    ):
        out_dir = Path(out_dir)
        if ext is None:
            ext = ""
        else:
            assert ext.startswith(".")

        self.white.save((out_dir / f"{self.name}.{white}").with_suffix(ext))
        self.pial.save((out_dir / f"{self.name}.{pial}").with_suffix(ext))
        if self.sphere is not None:
            self.sphere.save((out_dir / f"{self.name}.{sphere}").with_suffix(ext))
        if self.registration is not None:
            self.registration.save(
                (out_dir / f"{self.name}.{registration}").with_suffix(ext)
            )
        if self.inf is not None:
            self.inf.save((out_dir / f"{self.name}.{inf}").with_suffix(ext))


    def compute_surface_gradients(self, data: npt.NDArray[float], surface_name="white", neighborhood_size=1) -> None:

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
            neighborhood_size: int
                    Size of the neighborhood over which the gradient is computed.
        Returns:
        coeffs: np.array(float)
                    A nodes x 3 array that stores the first order
                    gradient of the data projected to the principal
                    curvature directions along with the bias or
                    constant term.
        """

        # Get the principal curvature directions
        surf_tmp = getattr(self, surface_name)
        _, E = surf_tmp.compute_principal_curvatures()

        # The unknowns (3x1), two gradients and the bias
        num_params = 2
        coeffs = np.zeros((surf_tmp.n_vertices, num_params))

        # number of neighbors
        knn, kr = surf_tmp.k_ring_neighbors(neighborhood_size)
        m = np.array([x.size for x in knn])
        muq = np.unique(m)


        # Loop over the nodes with the same amount of neighbors
        # This way we can solve all the linear systems with the
        # same size at the same time.
        for mm in muq:
            i = np.where(m == mm)[0]

            # Get the neighbor indices
            nid = np.array([knn[j][kr[j,1]:kr[j,2]] for j in i])

            # Get the coordinates of the neighbors in MRI space
            coords_neighbors = surf_tmp.vertices[nid, ...]
            coords_neighbors = np.moveaxis(coords_neighbors, 0, 1)

            # Get the coordinates of the center nodes
            coords_centers = surf_tmp.vertices[i, ...]

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

    @classmethod
    def from_freesurfer_subject_dir(
        cls,
        sub_dir: Path | str,
        hemi: str,
        white: str | None = "white",
        pial: str | None = "pial",
        sphere: str | None = None,
        registration: str | None = None,
        inf: str | None = None,
        infra_supra_model_tuple=None,
        # thickness="thickness",
        # curv="avg_curv",
    ):
        assert hemi in {"lh", "rh"}

        white_surf = Surface.from_freesurfer_subject_dir(sub_dir, f"{hemi}.{white}")

        # The pial surfaces (?h.pial) are symlinks to either ?h.pial.T1 or
        # ?h.pial.T2 depending on whether the `-T2pial` flag was used when
        # invoking recon-all. Symlinks created in WSL on Windows do not
        # seem to work currently, hence this workaround
        try:
            pial_surf = Surface.from_freesurfer_subject_dir(sub_dir, f"{hemi}.{pial}")
        except OSError:  # invalid argument
            try:
                pial_surf = Surface.from_freesurfer_subject_dir(
                    sub_dir, f"{hemi}.{pial}.T2"
                )
            except FileNotFoundError:  # -T2pial was not used
                pial_surf = Surface.from_freesurfer_subject_dir(
                    sub_dir, f"{hemi}.{pial}.T1"
                )

        # spherical representation
        if sphere is None:
            sphere_surf = None
        else:
            sphere_surf = Sphere.from_freesurfer_subject_dir(
                sub_dir, f"{hemi}.{sphere}"
            )

        # spherical registration
        if registration is None:
            reg_surf = None
        else:
            reg_surf = Sphere.from_freesurfer_subject_dir(
                sub_dir, f"{hemi}.{registration}"
            )

        if inf is None:
            inf_surf = None
        else:
            inf_surf = Surface.from_freesurfer_subject_dir(sub_dir, f"{hemi}.{inf}")

        infra_supra_model = {}
        if infra_supra_model_tuple is not None:
            if type(infra_supra_model_tuple) is not tuple:
                raise Exception(
                    "Infra-supra model name needs to be tuple like ('equivolume', 'local', 'spherical')"
                )

            model_fsav = load_model(
                infra_supra_model_tuple[0], infra_supra_model_tuple[1], hemi, registration=infra_supra_model_tuple[2]
            )
            params_subject = Hemisphere._map_infra_supra_model_to_subject(
                model_fsav[f"{infra_supra_model_tuple[0]}_{infra_supra_model_tuple[1]}"], registration=reg_surf
            )
            infra_supra_model = {f"{infra_supra_model_tuple[0]}_{infra_supra_model_tuple[1]}": params_subject}

            if not infra_supra_model:
                print(
                    "Model loading failed, using the default (equivolume with fraction 0.5)"
                )
                infra_supra_model["equivolume_global"] = 0.5

        return cls(
            hemi,
            white_surf,
            pial_surf,
            sphere_surf,
            reg_surf,
            inf_surf,
            infra_supra_model=infra_supra_model,
        )


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
        @functools.wraps(method)
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

    def __repr__(self):
        s = "\n".join(f"{h} : {i}" for h, i in zip(("lh", "rh"), self))
        return s

    def __str__(self):
        s = "\n".join(f"{h} : {i}" for h, i in zip(("lh", "rh"), self))
        return s
