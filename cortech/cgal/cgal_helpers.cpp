
#include <vector>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>

#include <cgal_helpers.h>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = K::Point_3;
using Surface_mesh = CGAL::Surface_mesh<Point_3>;
using vertex_descriptor = Surface_mesh::Vertex_index;

namespace CGAL_sm
{
    Surface_mesh build(CGAL_t::vecvec<float> vertices, CGAL_t::vecvec<int> faces)
    {
        // Construct a Surface_mesh from vectors of vertices and faces.
        Surface_mesh mesh;
        int n_vertices = vertices.size();
        int n_faces = faces.size();

        std::vector<vertex_descriptor> vertex_indices(n_vertices);
        for (int i = 0; i < n_vertices; ++i)
        {
            vertex_indices[i] = mesh.add_vertex(
                Point_3(vertices[i][0], vertices[i][1], vertices[i][2]));
        }
        for (int i = 0; i < n_faces; ++i)
        {
            mesh.add_face(
                vertex_indices[faces[i][0]],
                vertex_indices[faces[i][1]],
                vertex_indices[faces[i][2]]);
        }
        assert(CGAL::is_triangle_mesh(mesh));

        return mesh;
    }

    CGAL_t::vecvec<float> extract_vertices(Surface_mesh mesh)
    {
        // Extract vertices from a Surface_mesh into a vector of vectors.
        int n_vertices = mesh.number_of_vertices();
        CGAL_t::vecvec<float> vertices(n_vertices, std::vector<float>(3));

        for (vertex_descriptor vi : mesh.vertices())
        {
            Point_3 p = mesh.point(vi);
            vertices[vi][0] = (float)p.x();
            vertices[vi][1] = (float)p.y();
            vertices[vi][2] = (float)p.z();
        }
        return vertices;
    };

    CGAL_t::vecvec<int> extract_faces(Surface_mesh mesh)
    {
        // Extract faces from a Surface_mesh into a vector of vectors.
        int n_faces = mesh.number_of_faces();
        CGAL_t::vecvec<int> faces(n_faces, std::vector<int>(3));

        // for each face index, iterate over its halfedges and return all `target` vertices
        int i = 0;
        for (Surface_mesh::Face_index fi : mesh.faces())
        {
            int j = 0;
            Surface_mesh::Halfedge_index h = mesh.halfedge(fi);
            for (Surface_mesh::Halfedge_index hi : mesh.halfedges_around_face(h))
            {
                vertex_descriptor vi = mesh.target(hi);
                faces[i][j] = (int)vi;
                j++;
            }
            j = 0;
            i++;
        }
        return faces;
    }

    std::pair<CGAL_t::vecvec<float>, CGAL_t::vecvec<int>> extract_vertices_and_faces(Surface_mesh mesh)
    {
        // Extract vertices and faces into a pair of vectors.
        auto pair = std::make_pair(extract_vertices(mesh), extract_faces(mesh));
        return pair;
    }
}
