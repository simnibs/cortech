#include <vector>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>

using std::vector;

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Surface_mesh = CGAL::Surface_mesh<K::Point_3>;

namespace cortech
{
    template <typename T>
    using vecvec = vector<vector<T>>;

    struct SurfaceMesh
    {
        vector<vector<float>> vertices;
        vector<vector<int>> faces;
    };

    struct SurfaceMeshWithPMaps
    {
        vector<vector<float>> vertices;
        vector<vector<int>> faces;
        vector<int> faces_pmap;
    };

    struct VolumeMesh
    {
        vector<vector<float>> vertices;
        vector<vector<int>> faces;
        vector<vector<int>> cells;
    };

    struct VolumeMeshWithPMaps
    {
        vector<vector<float>> vertices;
        vector<vector<int>> faces;
        vector<vector<int>> cells;
        vector<int> faces_pmap;
        vector<int> cells_pmap;
    };

    std::pair<Surface_mesh, vector<Surface_mesh::Vertex_index>> from_polygon_soup_with_vertex_map(
        const vector<vector<float>> &vertices,
        const vector<vector<int>> &faces);
    Surface_mesh from_polygon_soup(
        const vector<vector<float>> &vertices,
        const vector<vector<int>> &faces);
    Surface_mesh from_polygon_soup(
        const vector<K::Point_3> &points,
        const vector<vector<int>> &faces);
    vector<K::Point_3> vertices_to_point3(const vector<vector<float>> &vertices);
    vector<vector<float>> point3_to_vertices(const vector<K::Point_3> &points);
    vector<vector<float>> extract_vertices(const Surface_mesh &mesh);
    vector<vector<int>> extract_faces(const Surface_mesh &mesh);
    std::pair<vector<vector<float>>, vector<vector<int>>> extract_vertices_and_faces(
        const Surface_mesh &mesh);
}

// namespace CGAL_tr
// {
//     std::pair<vector<vector<float>>, std::unordered_map<Tr::Vertex_handle, int>> get_vertices<Tr>(Tr &tr);
//     vector<vector<int>> get_cells(Tr &tr, std::unordered_map<Tr::Vertex_handle, int> vertex_to_index);
// }