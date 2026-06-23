#include <iostream>
#include <stdexcept>
#include <vector>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Surface_mesh.h>

#include <cgal_helpers.h>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = K::Point_3;
using Surface_mesh = CGAL::Surface_mesh<Point_3>;
using Face_index = Surface_mesh::Face_index;
using Halfedge_index = Surface_mesh::Halfedge_index;
using Vertex_index = Surface_mesh::Vertex_index;

using std::vector;

namespace cortech
{
    // Build Surface_mesh manually and return the index to Vertex_index mapping
    std::pair<Surface_mesh, vector<Vertex_index>> from_polygon_soup_with_vertex_map(
        const vector<vector<float>> &vertices,
        const vector<vector<int>> &faces)
    {
        Surface_mesh mesh;
        bool is_valid_mesh = CGAL::Polygon_mesh_processing::is_polygon_soup_a_polygon_mesh(faces);
        if (!is_valid_mesh)
            throw std::runtime_error("Triangulation does not define a valid polygon mesh.");
        int n_vertices = vertices.size();
        int n_faces = faces.size();

        vector<Vertex_index> v2v(n_vertices); // index to Vertex_index
        for (int i = 0; i < n_vertices; ++i)
        {
            auto &v = vertices[i];
            v2v[i] = mesh.add_vertex(Point_3(v[0], v[1], v[2]));
        }
        vector<int> f(3);
        for (int i = 0; i < n_faces; ++i)
        {
            auto &f = faces[i];
            mesh.add_face(v2v[f[0]], v2v[f[1]], v2v[f[2]]);
        }
        return std::make_pair(mesh, v2v);
    }

    Surface_mesh from_polygon_soup(
        const vector<vector<float>> &vertices,
        const vector<vector<int>> &faces)
    {
        Surface_mesh mesh;
        bool is_valid_mesh = CGAL::Polygon_mesh_processing::is_polygon_soup_a_polygon_mesh(faces);
        if (!is_valid_mesh)
            throw std::runtime_error("Triangulation does not define a valid polygon mesh.");
        auto points = vertices_to_point3(vertices);
        CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh(points, faces, mesh);
        return mesh;
    }

    Surface_mesh from_polygon_soup(
        const vector<Point_3> &points,
        const vector<vector<int>> &faces)
    {
        Surface_mesh mesh;
        bool is_valid_mesh = CGAL::Polygon_mesh_processing::is_polygon_soup_a_polygon_mesh(faces);
        if (!is_valid_mesh)
            throw std::runtime_error("Triangulation does not define a valid polygon mesh.");
        CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh(points, faces, mesh);
        return mesh;
    }

    vector<Point_3> vertices_to_point3(const vector<vector<float>> &vertices)
    {
        vector<Point_3> points(vertices.size());
        int i = 0;
        for (auto v : vertices)
            points[i++] = Point_3(v[0], v[1], v[2]);
        return points;
    }

    vector<vector<float>> point3_to_vertices(const vector<Point_3> &points)
    {
        vector<vector<float>> vertices(points.size(), vector<float>(3));
        int i = 0;
        for (Point_3 p : points)
        {
            auto &v = vertices[i++];
            v[0] = (float)p.x();
            v[1] = (float)p.y();
            v[2] = (float)p.z();
        }
        return vertices;
    }

    vector<vector<float>> extract_vertices(const Surface_mesh &mesh)
    {
        // Extract vertices from a Surface_mesh into a vector of vectors.
        int n_vertices = mesh.number_of_vertices();
        vector<vector<float>> vertices(n_vertices, vector<float>(3));

        for (Vertex_index vi : mesh.vertices())
        {
            Point_3 p = mesh.point(vi);
            auto &v = vertices[vi];
            v[0] = (float)p.x();
            v[1] = (float)p.y();
            v[2] = (float)p.z();
        }
        return vertices;
    };

    vector<vector<int>> extract_faces(const Surface_mesh &mesh)
    {
        // Extract faces from a Surface_mesh into a vector of vectors.
        int n_faces = mesh.number_of_faces();
        vector<vector<int>> faces(n_faces, vector<int>(3));

        // for each face index, iterate over its halfedges and return all `target` vertices
        int i = 0;
        int j;
        for (Face_index fi : mesh.faces())
        {
            Halfedge_index h = mesh.halfedge(fi);
            auto &f = faces[i++];
            j = 0;
            for (Halfedge_index hi : mesh.halfedges_around_face(h))
                f[j++] = (int)mesh.target(hi); // Vertex_index to int
        }
        return faces;
    }

    std::pair<vector<vector<float>>, vector<vector<int>>> extract_vertices_and_faces(
        const Surface_mesh &mesh)
    {
        return std::make_pair(extract_vertices(mesh), extract_faces(mesh));
    }
}

// namespace CGAL_tr
// {
//     // Get vertices
//     template <typename TR>
//     std::pair<vector<vector<float>>, std::unordered_map<Tr::Vertex_handle, int>> get_vertices(Tr &tr)
//     {
//         int i;

//         std::unordered_map<Tr::Vertex_handle, int> vertex_to_index;
//         vector<vector<float>> vertices(tr.number_of_vertices(), vector<float>(3));
//         i = 0;
//         for (auto v : tr.finite_vertex_handles())
//         {
//             vertex_to_index[v] = i;
//             auto p = tr.point(v);
//             vertices[i][0] = (float)p.x();
//             vertices[i][1] = (float)p.y();
//             vertices[i][2] = (float)p.z();
//             ++i;
//         }
//         return std::make_pair(vertices, vertex_to_index);
//     }

//     template <typename TR>
//     vector<vector<int>> get_cells(
//         Tr &tr,
//         std::unordered_map<Tr::Vertex_handle, int> vertex_to_index)
//     {
//         // Get cells (tetrahedra)
//         int i, j;

//         // tr.number_of_cells()                 domain cells, infinite cells, facets
//         // tr.number_of_finite_cells()          domain cells, infinite cells
//         int n_cells = tr.number_of_cells();
//         vector<vector<int>> cells(n_cells, vector<int>(4));
//         i = 0;
//         for (auto c : tr.finite_cell_handles()) // iterator over cell *handles*
//         {
//             j = 0;
//             for (auto v : tr.vertices(c))
//                 cells[i][j++] = vertex_to_index[v];
//             ++i;
//         }
//         return cells;
//     }
// }