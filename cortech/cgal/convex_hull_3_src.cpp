#include <vector>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/convex_hull_3.h>

#include <cgal_helpers.h>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = K::Point_3;
using Surface_mesh = CGAL::Surface_mesh<Point_3>;

std::pair<CGAL_t::vecvec<float>, CGAL_t::vecvec<int>> convex_hull_3(
    CGAL_t::vecvec<float> vertices
)
{
    int n_vertices = vertices.size();
    std::vector<Point_3> points(n_vertices);

    // Construct array of points
    for (int i = 0; i < n_vertices; ++i){
        points[i] = Point_3(vertices[i][0], vertices[i][1], vertices[i][2]);
    }

    // define polyhedron to hold convex hull
    // Polyhedron_3 poly;
    Surface_mesh mesh;

    // compute convex hull of non-collinear points
    CGAL::convex_hull_3(points.begin(), points.end(), mesh);

    auto pair = CGAL_sm::extract_vertices_and_faces(mesh);

    return pair;
}
