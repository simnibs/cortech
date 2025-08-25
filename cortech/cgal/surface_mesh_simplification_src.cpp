#include <vector>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Surface_mesh_simplification/edge_collapse.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Face_count_stop_predicate.h>

#include <cgal_helpers.h>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Surface_mesh = CGAL::Surface_mesh<K::Point_3>;
using edges_size_type = boost::graph_traits<Surface_mesh>::edges_size_type;
using vertex_descriptor = Surface_mesh::Vertex_index;
using face_descriptor = boost::graph_traits<Surface_mesh>::face_descriptor;

namespace SMS = CGAL::Surface_mesh_simplification;

std::pair<std::vector<std::vector<float>>, std::vector<std::vector<int>>> sms_simplify(
    std::vector<std::vector<float>> vertices,
    std::vector<std::vector<int>> faces,
    int stop_face_count)
{
    Surface_mesh mesh = CGAL_sm::build(vertices, faces);

    SMS::Face_count_stop_predicate<Surface_mesh> stop((edges_size_type)stop_face_count);

    // Garland&Heckbert simplification policies

    // typedef typename GHPolicies::Get_cost                                        GH_cost;
    // typedef typename GHPolicies::Get_placement                                   GH_placement;
    // typedef SMS::Bounded_normal_change_placement<GH_placement>                   Bounded_GH_placement;
    // GHPolicies gh_policies(mesh);
    // const GH_cost& gh_cost = gh_policies.get_cost();
    // const GH_placement& gh_placement = gh_policies.get_placement();
    // Bounded_GH_placement placement(gh_placement);
    // CGAL::parameters::get_cost(gh_cost).get_placement(placement)
    SMS::edge_collapse(mesh, stop);
    mesh.collect_garbage();

    auto pair = CGAL_sm::extract_vertices_and_faces(mesh);

    return pair;
}