#include <fstream>
#include <vector>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/extract_mean_curvature_flow_skeleton.h>
#include <CGAL/boost/graph/split_graph_into_polylines.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Surface_mesh_simplification/edge_collapse.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Face_count_stop_predicate.h>

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

std::pair<vector<vector<float>>, vector<vector<int>>> smskel_skeletonize(
    vector<vector<float>> vertices,
    vector<vector<int>> faces)
{

    Surface_mesh mesh = cortech::build(vertices, faces);
    Skeleton skeleton;

    CGAL::extract_mean_curvature_flow_skeleton(mesh, skeleton);

    std::cout << "Number of vertices of the skeleton: " << boost::num_vertices(skeleton) << "\n";
    std::cout << "Number of edges of the skeleton: " << boost::num_edges(skeleton) << "\n";

    // Output all the edges of the skeleton.
    std::ofstream output("skel-poly.polylines.txt");
    Display_polylines display(skeleton, output);
    CGAL::split_graph_into_polylines(skeleton, display);
    output.close();

    // Output skeleton points and the corresponding surface points
    output.open("correspondence-poly.polylines.txt");
    for (Skeleton_vertex v : CGAL::make_range(vertices(skeleton)))
        for (vertex_descriptor vd : skeleton[v].vertices)
            output << "2 " << skeleton[v].point << " "
                   << get(CGAL::vertex_point, tmesh, vd) << "\n";

    return EXIT_SUCCESS;
}