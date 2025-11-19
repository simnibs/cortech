#include <iostream>
// #include <stack>
#include <unordered_map>
#include <vector>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
// domains
#include <CGAL/Polyhedral_mesh_domain_3.h>
#include <CGAL/Polyhedral_complex_mesh_domain_3.h>
// #include <CGAL/Polyhedral_mesh_domain_with_features_3.h>

#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/make_mesh_3.h>

#include <CGAL/facets_in_complex_3_to_triangle_mesh.h>
#include <CGAL/remove_far_points_in_mesh_3.h>
#include <cgal_helpers.h>

// polyhedron
// #include <CGAL/Polyhedron_3.h>
#include <CGAL/Polyhedron_incremental_builder_3.h>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Surface_mesh = CGAL::Surface_mesh<K::Point_3>;

#ifdef CGAL_CONCURRENT_MESH_3
typedef CGAL::Parallel_tag Concurrency_tag;
#else
typedef CGAL::Sequential_tag Concurrency_tag;
#endif

// typedef CGAL::Polyhedron_3<K> Polyhedron;
// specialized version of Polyhedron from Mesh_3
typedef CGAL::Mesh_polyhedron_3<K>::type Polyhedron;

// simple...
typedef CGAL::Polyhedral_mesh_domain_3<Polyhedron, K> Mesh_domain;
// typedef CGAL::Polyhedral_mesh_domain_with_features_3<K> Mesh_domain;
// Triangulation
typedef CGAL::Mesh_triangulation_3<Mesh_domain, CGAL::Default, Concurrency_tag>::type Tr;
typedef CGAL::Mesh_complex_3_in_triangulation_3<Tr> C3t3;
// Criteria
typedef CGAL::Mesh_criteria_3<Tr> Mesh_criteria;

// complex
typedef CGAL::Polyhedral_complex_mesh_domain_3<K> complex_Mesh_domain;
typedef CGAL::Mesh_triangulation_3<complex_Mesh_domain, CGAL::Default, Concurrency_tag>::type complex_Tr;
typedef CGAL::Mesh_complex_3_in_triangulation_3<complex_Tr, complex_Mesh_domain::Corner_index, complex_Mesh_domain::Curve_index> complex_C3t3;
typedef CGAL::Mesh_criteria_3<complex_Tr> complex_Mesh_criteria;

struct V2FIIII
{
    std::vector<std::vector<float>> vertices;
    std::vector<std::vector<int>> faces;
    std::vector<std::vector<int>> cells;
    std::vector<int> faces_pmap;
    std::vector<int> cells_pmap;
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

// Get vertices
// template <typename C3T3, typename TR>
std::pair<CGAL_t::vecvec<float>, std::unordered_map<complex_C3t3::Vertex_handle, int>> c3t3_get_vertices(
    const complex_C3t3 &c3t3)
{
    int i;
    const auto &tr = c3t3.triangulation();

    std::unordered_map<complex_C3t3::Vertex_handle, int> vertex_to_index;
    CGAL_t::vecvec<float> vertices(tr.number_of_vertices(), std::vector<float>(3));
    i = 0;
    for (auto v : tr.finite_vertex_handles())
    // for (auto v : c3t3.vertices_in_complex())
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

// Get facets (triangles)
// template <typename C3T3, typename TR>
std::pair<CGAL_t::vecvec<int>, std::vector<int>> c3t3_get_facets(
    const complex_C3t3 &c3t3,
    std::unordered_map<complex_C3t3::Vertex_handle, int> vertex_to_index)
{
    int i, j;

    const auto &tr = c3t3.triangulation();

    bool print_each_facet_twice = false;

    int n_facets = c3t3.number_of_facets_in_complex();
    if (print_each_facet_twice)
        n_facets += n_facets;
    CGAL_t::vecvec<int> facets(n_facets, std::vector<int>(3));
    std::vector<int> facets_id(n_facets);
    i = 0;
    for (auto f : tr.finite_facets())
    {
        if (c3t3.is_in_complex(f))
        {
            auto [c, index] = f; // handle,
            facets_id[i] = c3t3.surface_patch_index(f);
            // auto sp_index = c3t3.surface_patch_index(f);
            // int c_domain = c3t3.subdomain_index(c);

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
                facets[i][j++] = vertex_to_index[v];

            // Print triangle again if needed, with opposite orientation
            if (print_each_facet_twice)
            {
                i = 2;
                for (auto v : tr.vertices(f))
                    facets[i][j--] = vertex_to_index[v];
                // os << get(facet_twice_pmap, f) << '\n';
            }
            ++i;
        }
    }
    return std::make_pair(facets, facets_id);
}

// template <typename C3T3, typename TR>
std::pair<CGAL_t::vecvec<int>, std::vector<int>> c3t3_get_cells(
    const complex_C3t3 &c3t3,
    std::unordered_map<complex_C3t3::Vertex_handle, int> vertex_to_index)
{
    // Get cells (tetrahedra)
    int i, j;
    const auto &tr = c3t3.triangulation();

    // tr.number_of_cells()                 domain cells, infinite cells, facets
    // tr.number_of_finite_cells()          domain cells, infinite cells
    // c3t3.number_of_cells_in_complex()    domain cells
    int n_cells = c3t3.number_of_cells_in_complex();
    CGAL_t::vecvec<int> cells(n_cells, std::vector<int>(4));
    std::vector<int> cells_id(n_cells);
    i = 0;
    for (auto c : tr.finite_cell_handles()) // iterator over cell *handles*
    {
        if (c3t3.is_in_complex(c)) // only save cells in the domain
        {
            cells_id[i] = c3t3.subdomain_index(c);
            j = 0;
            for (auto v : tr.vertices(c))
                cells[i][j++] = vertex_to_index[v];
            ++i;
        }
    }
    return std::make_pair(cells, cells_id);
}

// template <typename C3T3, typename TR>
V2FIIII c3t3_get_all(const complex_C3t3 &c3t3)
{

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

    auto vp = c3t3_get_vertices(c3t3);
    auto vertices = vp.first;
    auto vertex_to_index = vp.second;

    auto fp = c3t3_get_facets(c3t3, vertex_to_index);
    auto facets = fp.first;
    auto facets_id = fp.second;

    auto cp = c3t3_get_cells(c3t3, vertex_to_index);
    auto cells = cp.first;
    auto cells_id = cp.second;

    return {vertices, facets, cells, facets_id, cells_id};
}

// V2FIIII mesh3_make_mesh(
//     CGAL_t::vecvec<float> vertices,
//     CGAL_t::vecvec<int> faces,
//     float edge_size,
//     float cell_radius_edge_ratio,
//     float cell_size,
//     float facet_angle,
//     float facet_distance,
//     float facet_size)
// {
//     // Domain
//     Polyhedron mesh = build_polyhedron(vertices, faces);
//     Mesh_domain domain(mesh);
//     // domain.detect_features(); // get sharp features

//     // Meshing criteria
//     Mesh_criteria criteria(
//         CGAL::parameters::edge_size(edge_size)
//             .facet_angle(facet_angle)
//             .facet_size(facet_size)
//             .facet_distance(facet_distance)
//             .cell_radius_edge_ratio(cell_radius_edge_ratio)
//             .cell_size(cell_size));

//     // Mesh generation
//     std::cout << "CGAL :: Mesh_3 :: Generating mesh..." << std::endl;
//     C3t3 c3t3 = CGAL::make_mesh_3<C3t3>(
//         domain, criteria, CGAL::parameters::no_perturb().no_exude());

//     std::cout << "CGAL :: Mesh_3 :: Refining mesh..." << std::endl;
//     CGAL::refine_mesh_3(c3t3, domain, criteria);

//     c3t3.remove_isolated_vertices();
//     CGAL::remove_far_points_in_mesh_3(c3t3);

//     // Output
//     // std::ofstream medit_file("/home/jesperdn/repositories/simnibs-cortech/out_1.mesh");
//     // CGAL::IO::write_MEDIT(medit_file, c3t3);
//     // medit_file.close();

//     // Surface_mesh sm;
//     // // CGAL::facets_in_complex_3_to_triangle_mesh(c3t3, sm);

//     // Extract surface mesh from tetrahedral mesh boundary
//     return c3t3_get_all<C3t3, Tr>(c3t3);
// }

// V2FIIII mesh3_make_mesh_bounding(
//     CGAL_t::vecvec<float> v_inside,
//     CGAL_t::vecvec<int> f_inside,
//     CGAL_t::vecvec<float> v_bounding,
//     CGAL_t::vecvec<int> f_bounding,
//     float edge_size,
//     float cell_radius_edge_ratio,
//     float cell_size,
//     float facet_angle,
//     float facet_distance,
//     float facet_size)
// {
//     // Domain
//     Polyhedron mesh_inside = build_polyhedron(v_inside, f_inside);
//     Polyhedron mesh_bounding = build_polyhedron(v_bounding, f_bounding);
//     Mesh_domain domain(mesh_inside, mesh_bounding);
//     // domain.detect_features(); // get sharp features

//     // Meshing criteria
//     Mesh_criteria criteria(
//         CGAL::parameters::edge_size(edge_size)
//             .facet_angle(facet_angle)
//             .facet_size(facet_size)
//             .facet_distance(facet_distance)
//             .cell_radius_edge_ratio(cell_radius_edge_ratio)
//             .cell_size(cell_size));

//     // Mesh generation
//     std::cout << "CGAL :: Mesh_3 :: Generating mesh..." << std::endl;
//     C3t3 c3t3 = CGAL::make_mesh_3<C3t3>(
//         domain, criteria, CGAL::parameters::no_perturb().no_exude());

//     std::cout << "CGAL :: Mesh_3 :: Refining mesh..." << std::endl;
//     CGAL::refine_mesh_3(c3t3, domain, criteria);

//     c3t3.remove_isolated_vertices();
//     CGAL::remove_far_points_in_mesh_3(c3t3);

//     // Surface_mesh sm;
//     // // CGAL::facets_in_complex_3_to_triangle_mesh(c3t3, sm);

//     // Extract surface mesh from tetrahedral mesh boundary
//     return c3t3_get_all<C3t3, Tr>(c3t3);
// }

V2FIIII mesh3_make_mesh_complex(
    std::vector<CGAL_t::vecvec<float>> vertices,
    std::vector<CGAL_t::vecvec<int>> faces,
    std::vector<std::pair<int, int>> incident_subdomains,
    float edge_size,
    float cell_radius_edge_ratio,
    float cell_size,
    float facet_angle,
    float facet_distance,
    float facet_size)
{
    // Domain
    int n_patches = faces.size();
    assert(n_patches == vertices.size());
    assert(n_patches == incident_subdomains.size());

    std::vector<Polyhedron> patches(n_patches);
    for (int i = 0; i < n_patches; ++i)
    {
        patches[i] = build_polyhedron(vertices[i], faces[i]);
        if (patches[i].is_valid())
            std::cout << "Polyhedron " << i << " : " << "OK" << std::endl;
    }
    complex_Mesh_domain domain(
        patches.begin(),
        patches.end(),
        incident_subdomains.begin(),
        incident_subdomains.end());
    domain.detect_features(); // get borders and sharp features

    // Meshing criteria
    complex_Mesh_criteria criteria(
        CGAL::parameters::edge_size(edge_size)
            .facet_angle(facet_angle)
            .facet_size(facet_size)
            .facet_distance(facet_distance)
            .cell_radius_edge_ratio(cell_radius_edge_ratio)
            .cell_size(cell_size));

    // Mesh generation
    std::cout << "CGAL :: Mesh_3 :: Generating mesh..." << std::endl;
    complex_C3t3 c3t3 = CGAL::make_mesh_3<complex_C3t3>(
        domain, criteria, CGAL::parameters::no_perturb().no_exude());

    std::cout << "CGAL :: Mesh_3 :: Refining mesh..." << std::endl;
    CGAL::refine_mesh_3(c3t3, domain, criteria);

    c3t3.remove_isolated_vertices();
    CGAL::remove_far_points_in_mesh_3(c3t3);

    // Output
    // std::ofstream medit_file("/home/jesperdn/repositories/simnibs-cortech/out.mesh");
    // CGAL::IO::write_MEDIT(medit_file, c3t3);
    // medit_file.close();

    // Surface_mesh sm;
    // // CGAL::facets_in_complex_3_to_triangle_mesh(c3t3, sm);

    // return c3t3_get_all<complex_C3t3, complex_Tr>(c3t3);
    return c3t3_get_all(c3t3);
}
