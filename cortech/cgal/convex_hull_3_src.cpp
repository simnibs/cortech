#include <vector>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/convex_hull_3.h>

#include <cgal_helpers.h>

using std::vector;

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Surface_mesh = CGAL::Surface_mesh<K::Point_3>;

std::pair<vector<vector<float>>, vector<vector<int>>> convex_hull_3(
    vector<vector<float>> vertices)
{
    vector<K::Point_3> points = cortech::vertices_to_point3(vertices);

    // compute convex hull of non-collinear points
    Surface_mesh mesh;
    CGAL::convex_hull_3(points.begin(), points.end(), mesh);
    auto pair = cortech::extract_vertices_and_faces(mesh);
    return pair;
}
