#include <vector>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/alpha_wrap_3.h>

#include <cgal_helpers.h>

using std::vector;

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Surface_mesh = CGAL::Surface_mesh<K::Point_3>;

cortech::SurfaceMesh aw3_alpha_wrap_3_points(
    vector<vector<float>> points,
    double alpha,
    double offset)
{
    vector<K::Point_3> vertices = cortech::vertices_to_point3(points);
    Surface_mesh wrap;
    CGAL::alpha_wrap_3(vertices, alpha, offset, wrap);
    return cortech::extract_vertices_and_faces(wrap);
}


cortech::SurfaceMesh aw3_alpha_wrap_3_surface(
    vector<vector<float>> vertices,
    vector<vector<int>> faces,
    double alpha,
    double offset)
{
    Surface_mesh mesh = cortech::from_polygon_soup(vertices, faces);
    Surface_mesh wrap;
    CGAL::alpha_wrap_3(mesh, alpha, offset, wrap);
    return cortech::extract_vertices_and_faces(wrap);
}
