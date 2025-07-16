from collections import namedtuple
import functools
from pathlib import Path

import numpy as np
import numpy.typing as npt

import cortech.utils
from cortech.surface import Surface, Sphere
from cortech.constants import Curvature


class Hemisphere:
    """A class containing surfaces delineating the white-gray matter boundary
    and the gray matter-CSF (pial) boundary.

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
    ) -> None:
        assert name in {"lh", "rh"}
        self.name = name
        self.white = white
        self.pial = pial
        self.sphere = sphere
        self.registration = registration
        self.inf = inf

        self._surfaces = [self.white, self.pial]
        if self.sphere is not None:
            self._surfaces.append(self.sphere)
        if self.registration is not None:
            self._surfaces.append(self.registration)
        if self.inf is not None:
            self._surfaces.append(self.inf)

    def has_registration(self):
        return self.registration is not None

    def compute_thickness(self) -> None:
        """Calculate thickness at each vertex of node-matched surfaces."""

        # FIXME this should be a better estimate than simple vertex-to-vertex
        # distance?
        return np.linalg.norm(self.pial.vertices - self.white.vertices, axis=1)

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
        distance is `thickness`) which yields the desired volume fraction at each
        position.

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
        sub_dir: Path | str,
        hemi: str,
        white: str | None = "white",
        pial: str | None = "pial",
        sphere: str | None = None,
        registration: str | None = None,
        inf: str | None = None,
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
                sub_dir, f"{hemi}.{registration}"
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

        return cls(hemi, white_surf, pial_surf, sphere_surf, reg_surf, inf_surf)


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
