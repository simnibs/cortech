#include <vector>
#include <unordered_map>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/extract_mean_curvature_flow_skeleton.h>

#include <cgal_helpers.h>

using std::vector;

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Surface_mesh = CGAL::Surface_mesh<K::Point_3>;
using edges_size_type = boost::graph_traits<Surface_mesh>::edges_size_type;
using vertex_descriptor = Surface_mesh::Vertex_index;
using face_descriptor = boost::graph_traits<Surface_mesh>::face_descriptor;

using Skeletonization = CGAL::Mean_curvature_flow_skeletonization<Surface_mesh>;
using Skeleton = Skeletonization::Skeleton;

typedef Skeleton::vertex_descriptor Skeleton_vertex;
typedef Skeleton::edge_descriptor Skeleton_edge;

cortech::SurfaceMesh smskel_skeletonize(
    vector<vector<float>> vertices,
    vector<vector<int>> faces)
{
    int i;

    Surface_mesh mesh = cortech::from_polygon_soup(vertices, faces);
    Skeleton skeleton;
    CGAL::extract_mean_curvature_flow_skeleton(mesh, skeleton);

    // edges
    std::unordered_map<Skeleton_vertex, int> v2v;
    i = 0;
    for (Skeleton_vertex v : CGAL::make_range(boost::vertices(skeleton)))
    {
        v2v[v] = i++;
    }

    vector<vector<int>> skel_edges(boost::num_edges(skeleton));
    i = 0;
    for (Skeleton_edge e : CGAL::make_range(boost::edges(skeleton)))
    {
        skel_edges[i++] = {v2v[source(e, skeleton)], v2v[target(e, skeleton)]};
    }

    // vertices
    vector<vector<float>> skel_vertices(boost::num_vertices(skeleton));
    i = 0;
    for (Skeleton_vertex v : CGAL::make_range(boost::vertices(skeleton)))
    {
        K::Point_3 &p = skeleton[v].point;
        skel_vertices[i++] = {(float)p.x(), (float)p.y(), (float)p.z()};
    }
    // Output skeleton points and the corresponding surface points
    // for (vertex_descriptor vd : skeleton[v].vertices)
    //     output << "2 " << skeleton[v].point << " "
    //            << get(CGAL::vertex_point, tmesh, vd) << "\n";

    return {skel_vertices, skel_edges};
}