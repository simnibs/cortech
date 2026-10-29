#include <vector>
#include <algorithm>

#include <CGAL/boost/graph/border.h>
#include <CGAL/boost/graph/selection.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>

#include <cgal_helpers.h>

using std::vector;

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Surface_mesh = CGAL::Surface_mesh<K::Point_3>;
using Edge_index = Surface_mesh::Edge_index;
using Face_index = Surface_mesh::Face_index;
using Halfedge_index = Surface_mesh::Halfedge_index;
using Vertex_index = Surface_mesh::Vertex_index;


vector<vector<int>> graph_find_border_edges(
    vector<vector<float>> vertices,
    vector<vector<int>> faces)
{
    Surface_mesh mesh = cortech::from_polygon_soup(vertices, faces);

    // we don't know how many edges are border edges but preallocate too much
    // memory and resize later
    int n_edges = mesh.number_of_edges();
    vector<vector<int>> edges(n_edges, vector<int>(2));
    int i = 0;
    for (Edge_index e : mesh.edges())
    {
        if (mesh.is_border(e))
        {
            for (int j = 0; j < 2; j++)
                edges[i][j] = (int)mesh.vertex(e, j);
            i++;
        }
    }
    edges.resize(i);
    return edges;
}

vector<bool> graph_expand_face_selection_for_removal(
    vector<vector<float>> vertices,
    vector<vector<int>> faces,
    vector<int> selection)
{
    auto [mesh, v2v, f2f] = cortech::from_polygon_soup_with_vertex_and_face_map(vertices, faces);

    vector<Face_index> selection_as_face_index;
    selection_as_face_index.reserve(selection.size());
    for (int f : selection)
        selection_as_face_index.push_back(f2f[f]);

    auto is_selected = mesh.add_property_map<Face_index, bool>("f:is_selected", false).first;
    for (Face_index f : selection_as_face_index)
        put(is_selected, f, true);

    CGAL::expand_face_selection_for_removal(selection_as_face_index, mesh, is_selected);

    vector<bool> result;
    result.reserve(mesh.number_of_faces());
    for (Face_index f : mesh.faces())
        result.push_back(is_selected[f]);

    return result;
}

/*
vector<int> graph_expand_face_selection(
    vector<vector<float>> vertices,
    vector<vector<int>> faces,
    vector<int> selection,
    unsigned int k)
{
    auto [mesh, v2v, f2f] = cortech::from_polygon_soup_with_vertex_and_face_map(vertices, faces);

    vector<Face_index> selection_as_face_index;
    selection_as_face_index.reserve(selection.size());
    for (int f : selection)
        selection_as_face_index.push_back(f2f[f]);

    auto is_selected = mesh.add_property_map<Face_index, bool>("f:is_selected", false).first;
    for (Face_index f : selection_as_face_index)
        put(is_selected, f, true);

        vector<Face_index> out;
    CGAL::expand_face_selection(
        selection_as_face_index, mesh, k, is_selected, std::back_inserter(out));

    vector<int> result;
    result.reserve(out.size());
    for (Face_index f : out)
        result.push_back(f.idx());
    // vector<int> out_as_int(out.begin(), out.end());
    // results;
    // for (f :  is_selected)
    //     result.push_back();
    // result.resize();
    return result;
}

vector<int> graph_expand_vertex_selection(
    vector<vector<float>> vertices,
    vector<vector<int>> faces,
    vector<int> selection,
    unsigned int k)
{
    auto [mesh, v2v] = cortech::from_polygon_soup_with_vertex_map(vertices, faces);

    vector<Vertex_index> selection_as_vertex_index;
    selection_as_vertex_index.reserve(selection.size());
    for (int v : selection)
        selection_as_vertex_index.push_back(v2v[v]);

    auto is_selected = mesh.add_property_map<Vertex_index, bool>("v:is_selected", false).first;
    vector<Vertex_index> out;
    CGAL::expand_vertex_selection(
        selection_as_vertex_index, mesh, k, is_selected, std::back_inserter(out));
    vector<int> out_as_int(out.begin(), out.end());
    return out_as_int;
}
    */