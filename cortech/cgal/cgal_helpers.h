
#include <vector>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Surface_mesh = CGAL::Surface_mesh<K::Point_3>;

// convenience types
namespace CGAL_t
{
    template<typename T>
    using vecvec = std::vector<std::vector<T>>;
}

// Surface_mesh helper functions
namespace CGAL_sm
{
    Surface_mesh build(
        CGAL_t::vecvec<float> vertices, CGAL_t::vecvec<int> faces
    );
    CGAL_t::vecvec<float> extract_vertices(Surface_mesh mesh);

    CGAL_t::vecvec<int> extract_faces(Surface_mesh mesh);

    std::pair<CGAL_t::vecvec<float>, CGAL_t::vecvec<int>> extract_vertices_and_faces(
        Surface_mesh mesh
    );
}
