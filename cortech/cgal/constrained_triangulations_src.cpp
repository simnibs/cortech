#include <vector>
#include <unordered_map>
#include <iostream>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/make_conforming_constrained_Delaunay_triangulation_3.h>

#include <cgal_helpers.h>

using std::vector;

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using CCDTr = CGAL::Conforming_constrained_Delaunay_triangulation_3<K, CGAL::Default>;
using Tr = CCDTr::Triangulation;

struct MeshGeometry
{
    vector<vector<float>> vertices;
    // vector<vector<int>> faces;
    vector<vector<int>> cells;
};

// Get vertices
// template <typename C3T3, typename TR>
std::pair<vector<vector<float>>, std::unordered_map<Tr::Vertex_handle, int>> tr_get_vertices(
    const Tr &tr)
{
    int i;

    std::unordered_map<Tr::Vertex_handle, int> vertex_to_index;
    vector<vector<float>> vertices(tr.number_of_vertices(), vector<float>(3));
    i = 0;
    for (auto v : tr.finite_vertex_handles())
    {
        vertex_to_index[v] = i;
        auto p = tr.point(v);
        vertices[i][0] = (float)p.x();
        vertices[i][1] = (float)p.y();
        vertices[i][2] = (float)p.z();
        ++i;
    }
    return std::make_pair(vertices, vertex_to_index);
}

// // Get facets (triangles)
// // template <typename C3T3, typename TR>
// vector<vector<int>> tr_get_facets(
//     const CCDTr &tr,
//     std::unordered_map<Tr::Vertex_handle, int> vertex_to_index)
// {
//     int i, j;
//     int n_facets = tr.number_of_constrained_facets();
//     vector<vector<int>> facets(n_facets, vector<int>(3));
//     i = 0;
//     for (auto f : tr.constrained_facets())
//     {
//         // Get facet vertices in CCW order.
//         j = 0;
//         for (auto v : tr.vertices(f))
//         {
//             facets[i][j++] = vertex_to_index[v];
//         }
//         ++i;
//     }
//     return facets;
// }

// template <typename C3T3, typename TR>
vector<vector<int>> tr_get_cells(
    const Tr &tr,
    std::unordered_map<Tr::Vertex_handle, int> vertex_to_index)
{
    // Get cells (tetrahedra)
    int i, j;

    // tr.number_of_cells()                 domain cells, infinite cells, facets
    // tr.number_of_finite_cells()          domain cells, infinite cells
    int n_cells = tr.number_of_cells();
    vector<vector<int>> cells(n_cells, vector<int>(4));
    i = 0;
    for (auto c : tr.finite_cell_handles()) // iterator over cell *handles*
    {
        j = 0;
        for (auto v : tr.vertices(c))
            cells[i][j++] = vertex_to_index[v];
        ++i;
    }
    return cells;
}

MeshGeometry constrained_triangulation_make_ccdt(
    vector<vector<float>> vertices,
    vector<vector<int>> faces)
{
    Surface_mesh mesh = cortech::from_polygon_soup(vertices, faces);

    std::cout << "number of vertices " << vertices.size() << std::endl;
    std::cout << "number of faces    " << faces.size() << std::endl;
    std::cout << "-------------------------------------" << std::endl;

    auto ccdt = CGAL::make_conforming_constrained_Delaunay_triangulation_3(mesh);

    // std::ofstream out("sphere_out.medit");
    // out.precision(17);
    // CGAL::IO::write_MEDIT(out, ccdt);

    // extract triangulation
    Tr tr = ccdt.triangulation();

    std::cout << "number of vertices " << ccdt.number_of_vertices() << std::endl;
    std::cout << "number of cells    " << ccdt.number_of_cells() << std::endl;
    std::cout << "number of c facets " << ccdt.number_of_constrained_facets() << std::endl;

    std::cout << "number of vertices " << tr.number_of_vertices() << std::endl;
    std::cout << "number of cells    " << tr.number_of_cells() << std::endl;
    std::cout << "number of fin cells    " << tr.number_of_finite_cells() << std::endl;
    std::cout << "number of facets    " << tr.number_of_facets() << std::endl;
    std::cout << "number of fin facets    " << tr.number_of_finite_facets() << std::endl;

    auto vp = tr_get_vertices(tr);
    auto vertices_out = vp.first;
    auto vertex_to_index = vp.second;
    // auto faces_out = tr_get_facets(ccdt, vertex_to_index);
    auto cells = tr_get_cells(tr, vertex_to_index);

    // return {vertices_out, faces_out, cells};
    return {vertices_out, cells};
    // return 0;
}