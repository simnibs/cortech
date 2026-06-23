import numpy as np
import numpy.typing as npt

from cortech.cgal.convex_hull_3 import convex_hull


def cart_to_sph(points) -> npt.NDArray:
    """

    physics/ISO convention

    https://en.wikipedia.org/wiki/Spherical_coordinate_system

    points : x, y, z in columns

    Returns
    -------
    spherical_coordinates: npt.NDArray
        Spherical coordinates of the form (r, theta, phi) where

        r : radius
        theta [  0; pi]
            Inclination (elevation) angle going from north (= 0) to south (= pi)
        phi [-pi; pi]
            Azimuth angle (going around equator)

    """
    zero = np.isclose(points, 0.0)

    r = np.linalg.norm(points, axis=-1)
    theta = np.acos(points[..., 2] / r)
    # atan2 chooses the correct quadrant
    phi = np.atan2(points[..., 1], points[..., 0])

    # corner cases
    # x = y = 0 (phi is nonsensical)
    phi[zero[..., :2].all(-1)] = -np.pi
    # x = y = z = 0
    undefined = zero.all(-1)
    r[undefined] = 0.0
    theta[undefined] = 0.0
    phi[undefined] = 0.0

    return np.stack((r, theta, phi), axis=-1)


def sph_to_cart(
    r: float | npt.NDArray, theta: npt.NDArray, phi: npt.NDArray
) -> npt.NDArray:
    """

    Parameters
    ----------

    Returns
    -------
    euclidean_coordinates: npt.NDArray
    """
    theta_sin = np.sin(theta)
    x = r * theta_sin * np.cos(phi)
    y = r * theta_sin * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=-1)


def fibonacci_points(n_points: int, radius: float = 1.0):
    """


    Parameters
    ----------
    n_points: int
    radius: float

    Returns
    -------
    vertices: ndarray
    faces : ndarray

    References
    ----------
    http://extremelearning.com.au/evenly-distributing-points-on-a-sphere/
        See fig. 1 for choice of 'best' epsilon (maximizer of minimum distance)

    """

    if n_points < 30:
        epsilon = 0.0
    elif n_points < 150:
        epsilon = 2.0
    else:
        epsilon = 2.5

    phi = 0.5 * (1.0 + np.sqrt(5.0))  # the golden ratio
    i = np.arange(0, n_points, dtype=float)

    # Fibonacci grid
    x2 = (i + 0.5 + epsilon) / (n_points + 2 * epsilon)
    y2 = i * phi
    if n_points >= 30:
        x2[0], y2[0] = 0, 0
        x2[-1], y2[-1] = 1, 0

    # Fibonacci sphere
    # phi   : latitude (from pole to pole, 0 <= phi <= pi)
    # theta : longitude (around sphere, 0 <= theta <= 2*pi)

    # spherical coordinates (r = 1 is implicit because it is unit sphere)
    theta = np.arccos(1 - 2 * x2)
    phi = 2 * np.pi * y2

    # cartesian coordinates
    # x3 = np.cos(theta) * np.sin(phi)
    # y3 = np.sin(theta) * np.sin(phi)
    # z3 = np.cos(phi)

    cart = sph_to_cart(1.0, theta, phi)

    return radius * cart  # np.array([x3, y3, z3]).T


def fibonacci_sphere(n_points: int, radius: float = 1.0):
    """Generates a triangulated sphere with N vertices and radius r centered on
    (0,0,0).

    Parameters
    ----------
    n_points: int
    radius: float

    Returns
    -------
    vertices: ndarray
    faces : ndarray
    """
    pts = fibonacci_points(n_points, radius)
    return convex_hull(pts)
