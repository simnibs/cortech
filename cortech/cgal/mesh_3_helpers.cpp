#include <iostream>
#include <unordered_map>
#include <vector>

#include <cgal_helpers.h>

// Get vertices
template <typename Tr>
std::pair<CGAL_t::vecvec<float>, std::unordered_map<typename Tr::Vertex_handle, int>> tr_get_vertices(Tr tr)
{
    int i;

    std::unordered_map<typename Tr::Vertex_handle, int> vertex_to_index;
    CGAL_t::vecvec<float> vertices(tr.number_of_vertices(), std::vector<float>(3));
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

// Get facets (triangles)
template <typename C3T3, typename Tr>
std::pair<CGAL_t::vecvec<int>, std::vector<int>> c3t3_get_facets(
    C3T3 c3t3,
    std::unordered_map<Tr::Vertex_handle, int> vertex_to_index)
{
    int i, j;

    auto tr = c3t3.triangulation();

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
            // auto sp_index = c3t3.surface_patch_index(f);
            // facets_id[i] = sp_index.second; // e.g., 0 and 1
            // std::cout << x.first << " " << x.second << std::endl;
            // int c_domain = c3t3.subdomain_index(c);
            facets_id[i] = c3t3.surface_patch_index(f);

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
                if (auto search = vertex_to_index.find(v); search != vertex_to_index.end())
                    facets[i][j++] = vertex_to_index.at(v);
                else
                    std::cout << "Unable to find facet " << i << std::endl;
            }
            // facets[i][j++] = vertex_to_index[v];
            // facets[i][j++] = vertex_to_index.at(v);

            // Print triangle again if needed, with opposite orientation
            if (print_each_facet_twice)
            {
                i = 2;
                for (auto v : tr.vertices(f))
                {
                    if (auto<typename C3T3, typename Tr> search = vertex_to_index.find(v); search != vertex_to_index.end())
                        facets[i][j--] = vertex_to_index.at(v);
                    else
                        std::cout << "Unable to find facet " << i << std::endl;
                }
                // facets[i][j--] = vertex_to_index[v];
                // facets[i][j--] = vertex_to_index.at(v);
                // os << get(facet_twice_pmap, f) << '\n';
            }
            ++i;
        }
    }
    return std::make_pair(facets, facets_id);
}

template <typename C3T3, typename Tr>
std::pair<CGAL_t::vecvec<int>, std::vector<int>> c3t3_get_cells(
    C3T3 c3t3,
    std::unordered_map<typename Tr::Vertex_handle, int> vertex_to_index)
{
    // Get cells (tetrahedra)

    int i, j;
    // tr.number_of_cells()                 domain cells, infinite cells, facets
    // tr.number_of_finite_cells()          domain cells, infinite cells
    // c3t3.number_of_cells_in_complex()    domain cells
    auto tr = c3t3.triangulation();
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
            // cells[i][j++] = vertex_to_index[v];
            {
                if (auto search = vertex_to_index.find(v); search != vertex_to_index.end())
                    cells[i][j++] = vertex_to_index[v];
                else
                    std::cout << "Unable to find cell " << i << std::endl;
            }
            ++i;
        }
    }
    return std::make_pair(cells, cells_id);
}

template <typename C3T3, typename Tr>
V2FIIII c3t3_get_all(C3T3 c3t3)
{
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
    auto vp = tr_get_vertices<typename Tr>(tr);
    auto vertices = vp.first;
    auto vertex_to_index = vp.second;

    auto fp = c3t3_get_facets<typename C3T3, typename Tr>(c3t3, vertex_to_index);
    auto facets = fp.first;
    auto facets_id = fp.second;

    auto cp = c3t3_get_cells<typename C3T3, typename Tr>(c3t3, vertex_to_index);
    auto cells = cp.first;
    auto cells_id = cp.second;

    return {vertices, facets, cells, facets_id, cells_id};
}
