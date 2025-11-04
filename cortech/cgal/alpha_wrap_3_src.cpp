#include <vector>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/alpha_wrap_3.h>

#include <cgal_helpers.h>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point3 = K::Point_3;
using Surface_mesh = CGAL::Surface_mesh<K::Point_3>;

std::pair<CGAL_t::vecvec<float>, CGAL_t::vecvec<int>> aw3_alpha_wrap_3(
    CGAL_t::vecvec<float> points, const double alpha, const double offset)
{

    int n_points = points.size();

    std::vector<Point3> vertices(n_points);
    for (int i = 0; i < n_points; ++i)
    {
        vertices[i] = Point3(points[i][0], points[i][1], points[i][2]);
    }

    Surface_mesh wrap;
    CGAL::alpha_wrap_3(vertices, alpha, offset, wrap);

    auto pair = CGAL_sm::extract_vertices_and_faces(wrap);

    return pair;
}
