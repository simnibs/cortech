import numpy as np
import scipy.sparse

from cortech.utils import map_values

def edge_soup_to_polylines(edges):
    """Extract a number of polylines from `edges`.

    Preconditions
        A vertex can maximally be connected to two other vertices; otherwise
        the polyline is arbitrary.

    Parameters
    ----------
    edges :
        Edge soup of shape (n, 2) where each row defines the edge between two
        vertices.


    """
    # edge adjacency matrix
    n = len(edges)
    source = np.unique(edges)
    target = np.arange(n)
    assert len(source) == len(target)
    edges_re = map_values(source, target, edges.ravel())
    edges_re = edges_re.reshape(-1,2)
    rows = edges_re.T.ravel()
    cols = edges_re[:,::-1].T.ravel()
    data = np.ones(len(rows))
    A = scipy.sparse.csr_array((data, (rows, cols)))
    assert np.all(A.sum(1) <= 2), "One or more vertices has more than two connections; polylines are not unique"

    # split polylines
    ncc, cc = scipy.sparse.csgraph.connected_components(A, directed=False)
    polylines = []
    for i in range(ncc):
        start = np.flatnonzero(cc == i)[0]
        polyline = scipy.sparse.csgraph.depth_first_order(
            A, start, directed=False, return_predecessors=False)
        polyline = map_values(target, source, polyline)
        polylines.append(polyline)

    return polylines[0] if ncc == 1 else polylines
