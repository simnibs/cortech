#include <array>
#include <iostream>
#include <vector>

#include <boost/unordered_map.hpp>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/property_map.h>

#include <CGAL/Adaptive_remeshing_sizing_field.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/tetrahedral_remeshing.h>
#include <CGAL/Tetrahedral_remeshing/Remeshing_triangulation_3.h>
#include <CGAL/tetrahedron_soup_to_triangulation_3.h>

#include <cgal_helpers.h>

using std::vector;

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Tetrahedral_remeshing::Remeshing_triangulation_3<K> Remeshing_triangulation;
using Tr = Remeshing_triangulation;
using Vertex_handle = Tr::Vertex_handle;

// edges
// using Vertex_pair = std::pair<Vertex_handle, Vertex_handle>;
// using Constraints_set = bbost::unordered_set<Vertex_pair, boost::hash<Vertex_pair>>;
// using Constraints_pmap = CGAL::Boolean_property_map<Constraints_set>;

// facets
// using Facet = std::array<Vertex_handle, Vertex_handle, Vertex_handle;
// using Constraints_set = bbost::unordered_set<Facet, boost::hash<Vertex_pair>>;
// using Constraints_pmap = CGAL::Boolean_property_map<Constraints_set>;

template <class Tr, typename PointRange>
void build_vertices(Tr &tr,
                    const PointRange &points,
                    vector<typename Tr::Vertex_handle> &vertex_handle_vector)
{
    typedef typename Tr::Vertex_handle Vertex_handle;
    typedef typename Tr::Point Point;

    vertex_handle_vector[0] = tr.tds().create_vertex(); // creates the infinite vertex
    tr.set_infinite_vertex(vertex_handle_vector[0]);

    // build vertices
    int i = 1;
    // for (std::size_t i = 1; i < points.size() + 1; ++i)
    for (auto p : points)
    {
        Vertex_handle vh = tr.tds().create_vertex();
        vertex_handle_vector[i] = vh;
        // std::cout << i << std::endl;
        vh->set_point(Point(p[0], p[1], p[2]));
        ++i;
    }
}

bool build_triangulation(
    Tr &tr,
    vector<vector<float>> vertices,
    vector<vector<int>> faces,
    vector<vector<int>> cells,
    vector<int> faces_pmap,
    vector<int> cells_pmap)
{

    // using Point_3 = typename Tr::Point;
    using Subdomain_index = typename Tr::Cell::Subdomain_index;
    using Surface_patch_index = typename Tr::Cell::Surface_patch_index;

    using Facet = std::array<int, 3>;        // 3 = id
    using Tet_with_ref = std::array<int, 4>; // 4 = id

    typedef typename Tr::Vertex_handle Vertex_handle;
    typedef typename Tr::Cell_handle Cell_handle;
    typedef std::array<Vertex_handle, 3> Facet_vvv;

    // associate to a face the two (at most) incident tets and the id of the face in the cell
    typedef std::pair<Cell_handle, int> Incident_cell;
    typedef boost::unordered_map<Facet_vvv, vector<Incident_cell>> Incident_cells_map;
    Incident_cells_map incident_cells_map;

    int i;
    int n_vertices = vertices.size();
    int n_faces = faces.size();
    int n_cells = cells.size();

    // Remeshing_triangulation tr;
    tr.tds().clear(); // not tr.clear() since it calls tr.init(), which we don't want

    // POINTS
    // ======================
    // id to vertex_handle
    // index 0 is for infinite vertex; 1 to n for points in `points`
    vector<typename Tr::Vertex_handle> vertex_handle_vector(n_vertices + 1);
    build_vertices<Tr>(tr, vertices, vertex_handle_vector);
    for (auto vh : vertex_handle_vector)
    {
        vh->set_dimension(-1);
    }

    // FACETS
    // ======================
    boost::unordered_map<Facet, Surface_patch_index> border_facets(n_faces);
    // for (auto f : facets){
    Facet f;
    for (int i = 0; i < n_faces; ++i)
    {
        auto &tmp = faces[i];
        f[0] = tmp[0];
        f[1] = tmp[1];
        f[2] = tmp[2];
        border_facets.emplace(f, static_cast<Surface_patch_index>(faces_pmap[i]));
    }

    // CELLS
    // ======================
    vector<Tet_with_ref> finite_cells(n_cells);
    i = 0;
    for (auto c : cells)
    {
        finite_cells[i] = {c[0], c[1], c[2], c[3]};
        ++i;
    }

    vector<Subdomain_index> subdomains(n_cells);
    i = 0;
    for (auto cp : cells_pmap)
    {
        subdomains[i] = static_cast<Subdomain_index>(cp);
        ++i;
    }

    bool verbose = false;
    bool replace_domain_0 = false;
    bool allow_non_manifold = false;

    CGAL::SMDS_3::build_finite_cells<Tr>(
        tr,
        finite_cells,
        subdomains,
        vertex_handle_vector,
        incident_cells_map, border_facets, verbose, replace_domain_0);

    CGAL::SMDS_3::build_infinite_cells<Tr>(tr, incident_cells_map, verbose, allow_non_manifold);
    tr.tds().set_dimension(3);
    CGAL::SMDS_3::assign_neighbors<Tr>(tr, incident_cells_map, allow_non_manifold);

    // disabled because the TDS is not valid when cells do not cover the convex hull of vertices
    // return tr.tds().is_valid();
    return EXIT_SUCCESS;
}

// // Get vertices
// // template <typename TR>
// std::pair<vector<vector<float>>, boost::unordered_map<Tr::Vertex_handle, int>> tr_get_vertices(const Tr &tr)
// {
//     int i;

//     boost::unordered_map<Tr::Vertex_handle, int> vertex_to_index;
//     vector<vector<float>> vertices(tr.number_of_vertices(), vector<float>(3));
//     i = 0;
//     for (auto v : tr.finite_vertex_handles())
//     {
//         vertex_to_index[v] = i;
//         auto p = tr.point(v);
//         vertices[i][0] = (float)p.x();
//         vertices[i][1] = (float)p.y();
//         vertices[i][2] = (float)p.z();
//         ++i;
//     }
//     return std::make_pair(vertices, vertex_to_index);
// }

// // Get facets (triangles)
// // template <typename C3T3, typename TR>
// vector<vector<int>> tr_get_facets(
//     const Tr &tr,
//     boost::unordered_map<Tr::Vertex_handle, int> vertex_to_index)
// {
//     int i, j;
//     int n_facets = tr.number_of_finite_facets();
//     vector<vector<int>> facets(n_facets, vector<int>(3));
//     i = 0;
//     for (auto f : tr.finite_facets())
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

// // template <typename C3T3, typename TR>
// vector<vector<int>> tr_get_cells(
//     const Tr &tr,
//     boost::unordered_map<Tr::Vertex_handle, int> vertex_to_index)
// {
//     // Get cells (tetrahedra)
//     int i, j;

//     // tr.number_of_cells()                 domain cells, infinite cells, facets
//     // tr.number_of_finite_cells()          domain cells, infinite cells
//     int n_cells = tr.number_of_finite_cells();
//     vector<vector<int>> cells(n_cells, vector<int>(4));
//     i = 0;
//     for (auto c : tr.finite_cell_handles()) // iterator over cell *handles*
//     {
//         j = 0;
//         for (auto v : tr.vertices(c))
//             cells[i][j++] = vertex_to_index[v];
//         ++i;
//     }
//     return cells;
// }

// Get vertices
template <typename C3T3>
std::pair<vector<vector<float>>, boost::unordered_map<typename C3T3::Vertex_handle, int>> c3t3_get_vertices(
    const C3T3 &c3t3)
{
    int i;
    const auto &tr = c3t3.triangulation();

    boost::unordered_map<typename C3T3::Vertex_handle, int> vertex_to_index;
    vector<vector<float>> vertices(tr.number_of_vertices(), vector<float>(3));
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
template <typename C3T3>
std::pair<vector<vector<int>>, vector<int>> c3t3_get_facets(
    const C3T3 &c3t3,
    boost::unordered_map<typename C3T3::Vertex_handle, int> vertex_to_index)
{
    int i, j;

    const auto &tr = c3t3.triangulation();

    bool print_each_facet_twice = false;

    int n_facets = c3t3.number_of_facets_in_complex();
    if (print_each_facet_twice)
        n_facets += n_facets;
    vector<vector<int>> facets(n_facets, vector<int>(3));
    vector<int> facets_id(n_facets);
    i = 0;
    for (auto f : tr.finite_facets())
    {
        if (c3t3.is_in_complex(f))
        {
            // auto [c, index] = f; // handle,
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

template <typename C3T3>
std::pair<vector<vector<int>>, vector<int>> c3t3_get_cells(
    const C3T3 &c3t3,
    boost::unordered_map<typename C3T3::Vertex_handle, int> vertex_to_index)
{
    // Get cells (tetrahedra)
    int i, j;
    const auto &tr = c3t3.triangulation();

    // tr.number_of_cells()                 domain cells, infinite cells, facets
    // tr.number_of_finite_cells()          domain cells, infinite cells
    // c3t3.number_of_cells_in_complex()    domain cells
    int n_cells = c3t3.number_of_cells_in_complex();
    vector<vector<int>> cells(n_cells, vector<int>(4));
    vector<int> cells_id(n_cells);
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

template <typename C3T3>
cortech::VolumeMeshWithPMaps c3t3_get_all(const C3T3 &c3t3)
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

// template <typename Tr>
// struct Facets_pmap
// {
//     const int entry_map;

// public:
//     using key_type = typename std::array<int, 3>;
//     using value_type = bool;
//     using reference = bool;
//     using category = boost::read_write_property_map_tag;

//     friend value_type get(const Facets_pmap &map,
//                           const key_type &f)
//     {
//         return (map.entry_map.count(f) == 1);
//     }
//   friend void put(Cells_of_subdomain_pmap&,
//                   const key_type&,
//                   const value_type)
//   {
//     ; //nothing to do : subdomain indices are updated in remeshing
//   }
// };

cortech::VolumeMeshWithPMaps tetrahedral_remeshing_remesh(
    vector<vector<float>> vertices,
    vector<vector<int>> faces,
    vector<vector<int>> cells,
    vector<int> faces_pmap,
    vector<int> cells_pmap,
    vector<vector<int>> constrained_edges = {},
    // const vector<vector<int>> constrained_faces = {},
    std::string sizing_field_type = "uniform",
    float target_edge_length = 1.0,
    bool remesh_boundaries = true,
    int n_iterations = 1,
    bool check_triangulation = true)
{
    // vector<bool> cell_is_selected(vertices.size(), true);
    // vector<vector<int>> edge_is_constrained;
    // bool smooth_constrained_edges = false;

    vector<Tr::Point> points(vertices.size());
    int i = 0;
    for (auto v : vertices)
    {
        points[i] = Tr::Point(v[0], v[1], v[2]);
        ++i;
    }

    using Surface_patch_index = typename Tr::Cell::Surface_patch_index;
    using Facet = std::array<int, 3>; // 3 = id
    Facet f;
    boost::unordered_map<Facet, Surface_patch_index> border_facets(faces.size());
    for (int i = 0; i < faces.size(); ++i)
    {
        auto &tmp = faces[i];
        f[0] = tmp[0];
        f[1] = tmp[1];
        f[2] = tmp[2];
        std::sort(f.begin(), f.end());
        border_facets.emplace(f, static_cast<Surface_patch_index>(faces_pmap[i]));
    }

    Tr tr = CGAL::tetrahedron_soup_to_triangulation_3<Tr>(
        points, cells,
        CGAL::parameters::surface_facets(border_facets).subdomain_indices(std::cref(cells_pmap)));

    boost::unordered_map<std::size_t, Vertex_handle> vertex_mapper(tr.number_of_vertices());
    i = 0;
    for (Vertex_handle vh : tr.finite_vertex_handles())
    {
        vertex_mapper[i] = vh;
    }

    // edge constraints
    using Vertex_pair = std::pair<Vertex_handle, Vertex_handle>;
    using Constraints_set = std::unordered_set<Vertex_pair, boost::hash<Vertex_pair>>;
    using Constraints_pmap = CGAL::Boolean_property_map<Constraints_set>;
    Constraints_set constraints;
    for (auto e : constrained_edges)
    {
        constraints.emplace(CGAL::make_sorted_pair(vertex_mapper[e[0]], vertex_mapper[e[1]]));
    }

    // using Constraints_set = std::unordered_set<Tr::Facet, boost::hash<Tr::Facet>>;
    // using Constraints_pmap = CGAL::Boolean_property_map<Constraints_set>;
    // Constraints_pmap fc_map;

    // for (Tr::Cell_handle c : tr.finite_cell_handles())
    // {
    //     std::cout << c->surface_patch_index(0) << " " << c->surface_patch_index(1) << " " << c->surface_patch_index(2) << " " << c->surface_patch_index(3) << std::endl;
    // }

    // auto this_stuff = border_facets.find(f);
    // if (this_stuff != border_facets.end())
    // {
    //     std::cout << this_stuff->second << std::endl;
    // }

    // auto patch_id = f.first;
    // std::cout << CGAL::IO::oformat(patch_id) << std::endl;
    // if (patch_id > 2)
    // {
    //     std::cout << patch_id << std::endl;
    //     //     std::cout << "constraining facet " << counter << std::endl;
    //     //     ++counter;
    //     //     put(fc_map, f, true);
    // }
    // }

    // using Constraints_pmap = CGAL::Boolean_property_map<Constraints_set>;
    // for (int cfi : constrained_facets)
    // {
    //     auto &tmp = faces[cfi];
    //     f[0] = tmp[0];
    //     f[1] = tmp[1];
    //     f[2] = tmp[2];
    //     constrained_facets_map[f] = true;
    // }
    // Facets_pmap<Tr> facets_pmap(constrained_facets_map);

    // I think is_valid is also called as part of c3t3.set_triangulation()
    if (check_triangulation)
    {
        std::cout << "Checking triangulation... " << std::endl;
        if (!tr.is_valid(true))
        {
            i = 0;
            for (auto c : tr.all_cell_handles())
            {
                if (!tr.tds().is_valid(c))
                {
                    std::cerr << "invalid cell " << CGAL::IO::oformat(c) << std::endl;
                    for (int j = 0; j < 4; j++)
                    {
                        Tr::Cell_handle n = c->neighbor(j);
                        if (n == Tr::Cell_handle() || tr.tds().cells().is_used(n) == false)
                        {
                            std::cerr << "\tneighbor " << j << " == nullptr " << std::endl;
                        }
                    }
                    std::cerr << "with vertices" << std::endl;
                    for (int j = 0; j < 4; ++j)
                    {
                        Tr::Vertex_handle vh = c->vertex(j);
                        Tr::Point p = vh->point();
                        std::cout << "\t[" << p.x() << ", " << p.y() << ", " << p.z() << "]," << std::endl;
                    }
                }
                ++i;
            }
        }
    }

    std::cout << "number of vertices     : " << tr.number_of_vertices() << std::endl;
    std::cout << "number of faces        : " << tr.number_of_facets() << " " << tr.number_of_finite_facets() << std::endl;
    std::cout << "number of finite cells : " << tr.number_of_cells() << " " << tr.number_of_finite_cells() << std::endl;

    std::cout << "remeshing ..." << std::endl;

    using C3t3 = CGAL::Mesh_complex_3_in_triangulation_3<Tr, int, int>;
    C3t3 c3t3;
    c3t3.set_triangulation(tr);

    // this modifies the underlying tr of c3t3
    if (faces.size() > 0)
    {
        // create a vertex handle to vertex index mapping
        boost::unordered_map<Tr::Vertex_handle, int> vh_to_index;
        i = 0;
        for (Tr::Vertex_handle vh : tr.finite_vertex_handles())
        {
            vh_to_index[vh] = i++; // post increment
        }

        // find the surface facets in triangulation (where finite_facets
        // enumerates all cell facets)
        Facet f_index;
        for (Tr::Facet f : tr.finite_facets())
        {
            Tr::Cell_handle c = f.first; // cell handle
            int index = f.second;        // index of face in cell
            if ((index & 1) == 0)        // even
            {
                f_index[0] = vh_to_index[c->vertex((index + 2) & 3)]; // & 3 does % 4
                f_index[1] = vh_to_index[c->vertex((index + 1) & 3)];
                f_index[2] = vh_to_index[c->vertex((index + 3) & 3)];
            }
            else
            {
                f_index[0] = vh_to_index[c->vertex((index + 1) & 3)];
                f_index[1] = vh_to_index[c->vertex((index + 2) & 3)];
                f_index[2] = vh_to_index[c->vertex((index + 3) & 3)];
            }
            std::sort(f_index.begin(), f_index.end());

            auto it = border_facets.find(f_index);
            if (it != border_facets.end()) // exists
            {
                // std::cout << "found a facet with index " << it->second << std::endl;
                c3t3.add_to_complex(f, Tr::Cell::Surface_patch_index(it->second));
            }
        }
    }

    std::cout << "number of explicit facets " << faces.size() << std::endl;
    std::cout << "number of facets in complex c3t3 : " << c3t3.number_of_facets_in_complex() << std::endl;

    // CGAL::parameters np = CGAL::parameters::remesh_boundaries(remesh_boundaries).number_of_iterations(n_iterations);
    if (sizing_field_type == "uniform")
    {
        std::cout << "sizing field: uniform = " << target_edge_length << std::endl;

        CGAL::tetrahedral_isotropic_remeshing(
            tr,
            target_edge_length,
            // CGAL::parameters::remesh_boundaries(remesh_boundaries).number_of_iterations(n_iterations) /*.facet_is_constrained_map(fc_map)*/);
            CGAL::parameters::remesh_boundaries(remesh_boundaries).number_of_iterations(n_iterations).edge_is_constrained_map(Constraints_pmap(constraints)));
    }
    else if (sizing_field_type == "adaptive")
    {
        std::cout << "sizing field: adaptive" << std::endl;
        CGAL::tetrahedral_isotropic_remeshing(
            tr,
            CGAL::create_adaptive_remeshing_sizing_field(tr),
            CGAL::parameters::remesh_boundaries(remesh_boundaries).number_of_iterations(n_iterations)
            /*.facet_is_constrained_map(facets_pmap)*/);
    }
    // elif (sizing_field_type == "custom"){
    //     size = }

    std::cout << "remeshing done" << std::endl;

    // CGAL::parameters::cell_is_selected_map(cell_is_selected)
    //     .remesh_boundaries(remesh_boundaries)
    //     .number_of_iterations(n_iterations)
    //     .smooth_constrained_edges(smooth_constrained_edges)

    c3t3.set_triangulation(tr);

    return c3t3_get_all<C3t3>(c3t3);
}
