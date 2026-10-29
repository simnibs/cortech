#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/mesh_segmentation.h>
#include <CGAL/property_map.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Surface_mesh.h>

#include <cgal_helpers.h>

using std::vector;

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Surface_mesh = CGAL::Surface_mesh<K::Point_3>;
using Edge_index = Surface_mesh::Edge_index;
using Face_index = Surface_mesh::Face_index;
using Halfedge_index = Surface_mesh::Halfedge_index;
using Vertex_index = Surface_mesh::Vertex_index;

namespace PMP = CGAL::Polygon_mesh_processing;

vector<float> ss_signed_distance_function(
    vector<vector<float>> vertices,
    vector<vector<int>> faces,
    double cone_angle = 2.0 / 3.0 * CGAL_PI,
    std::size_t number_of_rays = 25)
{
    Surface_mesh mesh = cortech::from_polygon_soup(vertices, faces);

    using Facet_double_map = Surface_mesh::Property_map<Face_index, double>;
    Facet_double_map sdf_property_map = mesh.add_property_map<Face_index, double>("f:sdf").first;;

    // Any other scalar values can be used instead of using SDF values computed using the CGAL function
    auto [sdf_min, sdf_max] = CGAL::sdf_values(
        mesh, sdf_property_map, cone_angle, number_of_rays, true);

    double sdf_range = sdf_max - sdf_min;

    for (Face_index f : mesh.faces())
    {
        sdf_property_map[f] = sdf_property_map[f] * sdf_range + sdf_min;
    }
    // interpolate to vertices
    vector<float> v_sdf_raw;
    v_sdf_raw.reserve(mesh.number_of_vertices());
    double sum, wsum;
    for (Vertex_index v : mesh.vertices()){
        sum = 0.0;
        wsum = 0.0;
        for (Face_index f : CGAL::faces_around_target(mesh.halfedge(v), mesh))
        {
            if (f == Surface_mesh::null_face()) continue; // only matters on open meshes
            // face angle instead ?
            const double w = PMP::face_area(f, mesh);
            sum  += w * sdf_property_map[f];
            wsum += w;
        }
        v_sdf_raw.push_back(static_cast<float>((wsum > 0.0) ? sum / wsum : 0.0));
    }
    return v_sdf_raw;
}