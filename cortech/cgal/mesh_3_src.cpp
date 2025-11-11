#include <iostream>
#include <stack>
#include <unordered_map>
#include <vector>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>

#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/Mesh_triangulation_3.h>

#include <CGAL/Polyhedral_mesh_domain_3.h>
#include <CGAL/Polyhedral_mesh_domain_with_features_3.h>
#include <CGAL/make_mesh_3.h>
// #include <CGAL/3D_Mesh_3/Facets_in_complex_3_to_triangle_mesh.h>
#include <CGAL/remove_far_points_in_mesh_3.h>
#include <cgal_helpers.h>
// #include <CGAL/IO/File_medit.h>

#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polyhedron_incremental_builder_3.h>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Surface_mesh = CGAL::Surface_mesh<K::Point_3>;
using edge_descriptor = Surface_mesh::Edge_index;
using face_descriptor = Surface_mesh::Face_index;
using halfedge_descriptor = Surface_mesh::Halfedge_index;
using vertex_descriptor = Surface_mesh::Vertex_index;

#ifdef CGAL_CONCURRENT_MESH_3
typedef CGAL::Parallel_tag Concurrency_tag;
#else
typedef CGAL::Sequential_tag Concurrency_tag;
#endif

typedef CGAL::Polyhedron_3<K> Polyhedron;
typedef CGAL::Polyhedral_mesh_domain_3<Polyhedron, K> Mesh_domain;

// Triangulation
typedef CGAL::Mesh_triangulation_3<Mesh_domain, CGAL::Default, Concurrency_tag>::type Tr;
typedef CGAL::Mesh_complex_3_in_triangulation_3<Tr> C3t3;
// Criteria
typedef CGAL::Mesh_criteria_3<Tr> Mesh_criteria;

// typedef CGAL::Polyhedral_mesh_domain_with_features_3<K> Mesh_domain;
// Triangulation
// typedef CGAL::Mesh_triangulation_3<Mesh_domain, CGAL::Default, CGAL::Sequential_tag>::type Tr;
// typedef CGAL::Mesh_complex_3_in_triangulation_3<
//     Tr, Mesh_domain::Corner_index, Mesh_domain::Curve_index>
//     C3t3;

// // Criteria
// typedef CGAL::Mesh_criteria_3<Tr> Mesh_criteria;

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Polyhedron_3<K> Polyhedron;

struct V2FII
{
    std::vector<std::vector<float>> v0;
    std::vector<std::vector<int>> v1;
    std::vector<std::vector<int>> v2;
};

template <class HDS>
class Build_from_vectors : public CGAL::Modifier_base<HDS>
{
    const CGAL_t::vecvec<float> &points_;
    const CGAL_t::vecvec<int> &faces_;

public:
    Build_from_vectors(const CGAL_t::vecvec<float> &points,
                       const CGAL_t::vecvec<int> &faces)
        : points_(points), faces_(faces) {}

    void operator()(HDS &hds)
    {
        typedef CGAL::Polyhedron_incremental_builder_3<HDS> Builder;
        Builder builder(hds, true);
        builder.begin_surface(points_.size(), faces_.size());

        // Add vertices
        for (auto &p : points_)
        {
            K::Point_3 v = K::Point_3(p[0], p[1], p[2]);
            builder.add_vertex(v);
        }

        // Add faces
        for (auto &face : faces_)
        {
            builder.begin_facet();
            for (int idx : face)
                builder.add_vertex_to_facet(idx);
            builder.end_facet();
        }

        builder.end_surface();
    }
};

Polyhedron build_polyhedron(
    CGAL_t::vecvec<float> vertices,
    CGAL_t::vecvec<int> faces)
{
    Polyhedron poly;
    Build_from_vectors<Polyhedron::HalfedgeDS> builder(vertices, faces);
    poly.delegate(builder);
    if (poly.is_valid())
        std::cout << "Polyhedron built successfully.\n";
    return poly;
}

// void polyhedron_extract_vertices_and_faces(const Polyhedron &poly)
// {
//     CGAL_t::vecvec<float> faces(poly.size_of_facets(), std::vector<int>(3));
//     int i = 0;
//     for (auto f = poly.facets_begin(); f != poly.facets_end(); ++f, i++)
//     {
//         std::vector<int> face_indices(3);
//         auto h = f->halfedge();
//         do
//         {
//             int idx = (int)h->vertex()
//                           face_indices.push_back(idx);
//             h = h->next();
//         } while (h != f->halfedge());
//         faces[i] = face_indices;
//     }
// }

// CGAL_t::vecvec<float> polyhedron_extract_vertices(const Polyhedron &poly)
// {
//     CGAL_t::vecvec<float> vertices(poly.size_of_vertices(), std::vector<float>(3));

//     int i = 0;
//     for (auto v = poly.points())
//     {
//         K::Point_3 p = K::Point_3(v->point());
//         vertices[i][0] = (float)p.x();
//         vertices[i][1] = (float)p.y();
//         vertices[i][2] = (float)p.z();
//         i++;
//     }
//     return vertices;
// }

// template <typename T0, typename T1, typename T2>
// struct Three2DVectors
// {
//     CGAL_t::vecvec<T0> v0;
//     CGAL_t::vecvec<T1> v1;
//     CGAL_t::vecvec<T2> v2;
// };

V2FII get_vertices_and_tetrahedra(C3t3 c3t3)
{
    int i, j; // counters

    auto tr = c3t3.triangulation();

    // bool renumber_subdomain_indices = false;
    // bool show_patches = false;
    // bool all_c = false;
    // bool all_v = all_c || false;

    // // property maps
    // typedef CGAL::IO::Medit_pmap_generator<C3t3, renumber_subdomain_indices, show_patches> Generator;
    // typedef typename Generator::Cell_pmap Cell_pmap;
    // typedef typename Generator::Facet_pmap Facet_pmap;
    // typedef typename Generator::Facet_pmap_twice Facet_pmap_twice;
    // typedef typename Generator::Vertex_pmap Vertex_pmap;

    // Cell_pmap cell_pmap(c3t3);
    // Facet_pmap facet_pmap(c3t3, cell_pmap);
    // Facet_pmap_twice facet_pmap_twice(c3t3, cell_pmap);
    // Vertex_pmap vertex_pmap(c3t3, cell_pmap, facet_pmap);

    // VERTICES
    // ========
    i = 0;
    std::unordered_map<Tr::Vertex_handle, int> vertex_to_index;
    CGAL_t::vecvec<float> vertices(tr.number_of_vertices(), std::vector<float>(3));
    for (auto v : tr.finite_vertex_handles())
    {
        vertex_to_index[v] = i;
        auto p = tr.point(v);
        vertices[i][0] = (float)p.x();
        vertices[i][1] = (float)p.y();
        vertices[i][2] = (float)p.z();
        ++i;
    }

    // TRIANGLES
    // =========
    bool print_each_facet_twice = false;

    int number_of_triangles = c3t3.number_of_facets_in_complex();
    if (print_each_facet_twice)
        number_of_triangles += number_of_triangles;
    CGAL_t::vecvec<int> triangles(number_of_triangles, std::vector<int>(3));
    i = 0; // reset
    for (auto f : tr.finite_facets())
    {
        if (c3t3.is_in_complex(f))
        {

            auto [c, index] = f; // handle,
            // Apply priority among subdomains, to get consistent facet orientation per subdomain-pair interface.
            if (print_each_facet_twice)
            {
                auto mirror_facet = tr.mirror_facet(f);
                [[maybe_unused]] auto [c2, _] = mirror_facet;
                // NOTE: We mirror a facet when needed to make it consistent with Use_cell_indices_pmap.
                // if (get(cell_pmap, c) > get(cell_pmap, c2))
                // {
                //     f = mirror_facet;
                // }
            }

            // Get facet vertices in CCW order.
            j = 0;
            for (auto v : tr.vertices(f))
            {
                triangles[i][j++] = vertex_to_index[v];
            }
            // os << get(facet_pmap, f) << '\n';

            // Print triangle again if needed, with opposite orientation
            if (print_each_facet_twice)
            {
                i = 2;
                for (auto v : tr.vertices(f))
                {
                    triangles[i][j--] = vertex_to_index[v];
                }
                // os << get(facet_twice_pmap, f) << '\n';
            }
            ++i;
        }
    }

    // TETRAHEDRA
    // ==========
    // tr.number_of_cells()                 domain cells, infinite cells, facets
    // tr.number_of_finite_cells()          domain cells, infinite cells
    // c3t3.number_of_cells_in_complex()    domain cells
    CGAL_t::vecvec<int> tetrahedra(c3t3.number_of_cells_in_complex(), std::vector<int>(4));
    std::vector<int> subdomain_index(c3t3.number_of_cells_in_complex());
    i = 0;                                  // reset
    for (auto c : tr.finite_cell_handles()) // iterator over cell *handles*
    {
        if (c3t3.is_in_complex(c)) // only save cells in the domain
        {
            subdomain_index[i] = c.subdomain_index();
            j = 0;
            for (auto v : tr.vertices(c))
            {
                tetrahedra[i][j++] = vertex_to_index[v];
            }
            ++i;
        }
    }
    V2FII r = {vertices, triangles, tetrahedra};
    return r;
}

V2FII mesh3_make_mesh(
    CGAL_t::vecvec<float> vertices,
    CGAL_t::vecvec<int> faces)
{
    // make domain
    // Surface_mesh mesh = CGAL_sm::build(vertices, faces);
    Polyhedron mesh = build_polyhedron(vertices, faces);
    Mesh_domain domain(mesh);

    // get sharp features
    // domain.detect_features();

    // set meshing criteria
    Mesh_criteria criteria(
        CGAL::parameters::edge_size(0.025).facet_angle(25).facet_size(0.05).facet_distance(0.005).cell_radius_edge_ratio(3).cell_size(0.05));

    // Mesh generation
    C3t3 c3t3 = CGAL::make_mesh_3<C3t3>(
        domain, criteria, CGAL::parameters::no_perturb().no_exude());

    c3t3.remove_isolated_vertices();
    CGAL::remove_far_points_in_mesh_3(c3t3);

    // Output
    std::ofstream medit_file("/home/jesperdn/repositories/simnibs-cortech/out_1.mesh");
    CGAL::IO::write_MEDIT(medit_file, c3t3);
    medit_file.close();

    // Extract surface mesh from tetrahedral mesh boundary
    // Surface_mesh boundary_mesh;

    auto r = get_vertices_and_tetrahedra(c3t3);

    // std::cout << std::get<1>(t)[0][0] << " " << std::get<1>(t)[0][1] << " " << std::get<1>(t)[0][2] << std::endl;
    // std::cout << std::get<1>(t)[1][0] << " " << std::get<1>(t)[1][1] << " " << std::get<1>(t)[1][2] << std::endl;
    // std::cout << std::get<1>(t)[2][0] << " " << std::get<1>(t)[2][1] << " " << std::get<1>(t)[2][2] << std::endl;
    return r;
    // return std::make_pair(std::get<0>(t), std::get<2>(t));
}
