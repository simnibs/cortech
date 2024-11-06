// #include <iostream>
#include <vector>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
// #include <CGAL/Real_timer.h>
#include <CGAL/Surface_mesh.h>
// #include <CGAL/Polygon_mesh_processing/angle_and_area_smoothing.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
// #include <CGAL/Polygon_mesh_processing/corefinement.h>
// #include <CGAL/Polygon_mesh_processing/fair.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/repair_self_intersections.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/smooth_shape.h>
#include <CGAL/Side_of_triangle_mesh.h>

#include <cgal_helpers.h>


using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = K::Point_3;
using Surface_mesh = CGAL::Surface_mesh<Point_3>;
using vertex_descriptor = Surface_mesh::Vertex_index;
using face_descriptor = boost::graph_traits<Surface_mesh>::face_descriptor;

namespace PMP = CGAL::Polygon_mesh_processing;

std::vector<bool> pmp_points_inside_surface(
    CGAL_t::vecvec<float> vertices,
    CGAL_t::vecvec<int> faces,
    CGAL_t::vecvec<float> points,
    bool on_boundary_is_inside = true
)
{
    Surface_mesh mesh = CGAL_sm::build(vertices, faces);
    CGAL::Side_of_triangle_mesh<Surface_mesh, K> inside(mesh);

    std::size_t n_points = mesh.number_of_vertices();
    std::vector<bool> is_inside(n_points, false);

    for (std::size_t i = 0; i < n_points; ++i)
    {
        auto p = Point_3(points[i][0], points[i][1], points[i][2]);

        CGAL::Bounded_side res = inside(p);

        if (res == CGAL::ON_BOUNDED_SIDE) {
            is_inside[i] = true;
        }
        // point is *on* the boundary
        else if (res == CGAL::ON_BOUNDARY && on_boundary_is_inside) {
            is_inside[i] = true;
        }
        // else {
        //     is_inside[i] = false;
        // }
    }
    return is_inside;
}

std::pair<CGAL_t::vecvec<float>, CGAL_t::vecvec<int>> pmp_remove_self_intersections(
    CGAL_t::vecvec<float> vertices, CGAL_t::vecvec<int> faces
)
{
    Surface_mesh mesh = CGAL_sm::build(vertices, faces);
    PMP::experimental::remove_self_intersections(mesh.faces(), mesh); // optional args
    mesh.collect_garbage();
    auto pair = CGAL_sm::extract_vertices_and_faces(mesh);

    return pair;
}

CGAL_t::vecvec<int> pmp_self_intersections(CGAL_t::vecvec<float> vertices, CGAL_t::vecvec<int> faces)
{
    Surface_mesh mesh = CGAL_sm::build(vertices, faces);

    // CGAL::Real_timer timer;
    // timer.start();

    // std::cout << "Using parallel mode? " << std::is_same<CGAL::Parallel_if_available_tag, CGAL::Parallel_tag>::value << std::endl;

    std::vector<std::pair<face_descriptor, face_descriptor>> intersecting_tris;
    PMP::self_intersections<CGAL::Parallel_if_available_tag>(mesh.faces(), mesh, std::back_inserter(intersecting_tris));
    // std::cout << intersecting_tris.size() << " pairs of triangles intersect." << std::endl;

    int n_intersections = intersecting_tris.size();
    CGAL_t::vecvec<int> intersecting_faces(n_intersections, std::vector<int>(2));
    for (int i = 0; i < n_intersections; i++)
    {
        intersecting_faces[i][0] = (int)intersecting_tris[i].first;
        intersecting_faces[i][1] = (int)intersecting_tris[i].second;
    }

    // std::cout << "Elapsed time (self intersections): " << timer.time() << std::endl;

    return intersecting_faces;
}


std::pair<std::vector<int>,std::vector<int>> pmp_connected_components(
    CGAL_t::vecvec<float> vertices,
    CGAL_t::vecvec<int> faces,
    std::vector<int> constrained_faces
)
{
    Surface_mesh mesh = CGAL_sm::build(vertices, faces);

    // should be enough to just construct a facelistgraph
    // Surface_mesh mesh = construct_FaceListGraph(faces);

    // CGAL::Real_timer timer;
    // timer.start();

    // Extract *all* edges of `constrained_faces` and use these as constraints
    // std::set<Surface_mesh::Edge_index> indices;
    // for (auto fi : constrained_faces)
    // {
    //     Surface_mesh::Halfedge_index h = mesh.halfedge((Surface_mesh::Face_index)fi);
    //     for (Surface_mesh::Halfedge_index hi : mesh.halfedges_around_face(h))
    //     {
    //         indices.insert(mesh.edge(hi));
    //     }
    // }
    // CGAL::Boolean_property_map<std::set<Surface_mesh::Edge_index>> constrained_edges_map(indices);
    // std::cout << "constraining " << indices.size() << " edges" << std::endl;

    // Extract the *outer* edges of `constrained_faces` and use these as constraints
    std::map<Surface_mesh::Edge_index, int> indices_with_counts;
    for (auto fi : constrained_faces)
    {
        Surface_mesh::Halfedge_index h = mesh.halfedge((Surface_mesh::Face_index)fi);
        for (Surface_mesh::Halfedge_index hi : mesh.halfedges_around_face(h))
        {
            auto edge = mesh.edge(hi);
            if (indices_with_counts.count(edge) == 0){
                // new edge
                indices_with_counts[edge] = 1;
            } else {
                // already seen edge
                indices_with_counts[edge]++;
            }
        }
    }
    // Keep only edges which occur once (i.e., "outer" edges)
    std::set<Surface_mesh::Edge_index> indices;
    for ( const auto &pair : indices_with_counts ) {
        if (pair.second == 1){
            indices.insert(pair.first);
        }
    }
    CGAL::Boolean_property_map<std::set<Surface_mesh::Edge_index>> constrained_edges_map(indices);
    // std::cout << "constraining " << indices.size() << " edges" << std::endl;

    // face component map (output)
    Surface_mesh::Property_map<face_descriptor, std::size_t> fccmap = mesh.add_property_map<face_descriptor, std::size_t>("f:CC").first;

    std::size_t num = PMP::connected_components(
        mesh,
        fccmap,
        CGAL::parameters::edge_is_constrained_map(constrained_edges_map));

    // std::cerr << "- The graph has " << num << " connected components (face connectivity)" << std::endl;

    typedef std::map<std::size_t /*index of CC*/, unsigned int /*nb*/> Components_size;
    Components_size nb_per_cc;
    for (face_descriptor f : mesh.faces())
    {
        nb_per_cc[fccmap[f]]++;
    }
    // for (const Components_size::value_type &cc : nb_per_cc)
    // {
    //     std::cout << "\t CC #" << cc.first
    //               << " is made of " << cc.second << " faces" << std::endl;
    // }
    // std::cout << "Elapsed time (connected components): " << timer.time() << std::endl;

    std::vector<int> cc(mesh.number_of_faces());
    std::vector<int> cc_size(num);
    for (face_descriptor f : mesh.faces())
    {
        cc[f] = (int)fccmap[f];
        cc_size[fccmap[f]]++;
    }
    auto pair = std::make_pair(cc, cc_size);

    return pair;
}

// std::pair<std::vector<int>,std::vector<int>> pmp_volume_connected_components(
//     CGAL_t::vecvec<int> faces,
//     bool do_orientation_tests = false,
//     bool do_self_intersection_tests = false)
// {
//     Surface_mesh mesh = construct_FaceListGraph(faces);

//     CGAL::Real_timer timer;
//     timer.start();

//     // face component map (output)
//     Surface_mesh::Property_map<face_descriptor, std::size_t> fccmap = mesh.add_property_map<face_descriptor, std::size_t>("f:CC").first;

//     std::size_t num = PMP::volume_connected_components(
//         mesh,
//         fccmap,
//         CGAL::parameters::do_orientation_tests(do_orientation_tests).do_self_intersection_tests(do_self_intersection_tests));

//     // CGAL::parameters::do_orientation_tests(true).do_self_intersection_tests(true).is_cc_outward_oriented(true)

//     std::cerr << "- The graph has " << num << " connected components (face connectivity)" << std::endl;

//     typedef std::map<std::size_t /*index of CC*/, unsigned int /*nb*/> Components_size;
//     Components_size nb_per_cc;
//     for (face_descriptor f : mesh.faces())
//     {
//         nb_per_cc[fccmap[f]]++;
//     }
//     for (const Components_size::value_type &cc : nb_per_cc)
//     {
//         std::cout << "\t CC #" << cc.first
//                   << " is made of " << cc.second << " faces" << std::endl;
//     }
//     std::cout << "Elapsed time (connected components): " << timer.time() << std::endl;

//     std::vector<int> cc(mesh.number_of_faces());
//     std::vector<int> cc_size(num);
//     for (face_descriptor f : mesh.faces())
//     {
//         cc[f] = (int)fccmap[f];
//         cc_size[fccmap[f]]++;
//     }
//     auto pair = std::make_pair(cc, cc_size);

//     return pair;
// }

CGAL_t::vecvec<float> pmp_smooth_shape(
    CGAL_t::vecvec<float> vertices,
    CGAL_t::vecvec<int> faces,
    std::vector<int> constrained_vertices,
    const double time,
    const unsigned int nb_iterations)
{
    Surface_mesh mesh = CGAL_sm::build(vertices, faces);

    // CGAL::Real_timer timer;
    // timer.start();

    std::set<vertex_descriptor> indices;
    for (int i : constrained_vertices)
    {
        indices.insert((vertex_descriptor)i);
    }
    // std::cout << "constraining " << indices.size() << " vertices" << std::endl;
    CGAL::Boolean_property_map<std::set<vertex_descriptor>> vcmap(indices);

    PMP::smooth_shape(
        mesh,
        time,
        CGAL::parameters::number_of_iterations(nb_iterations)
            .vertex_is_constrained_map(vcmap));

    // std::cout << "Elapsed time (smoothing): " << timer.time() << std::endl;

    auto vertices_out = CGAL_sm::extract_vertices(mesh);

    return vertices_out;
}

// CGAL_t::vecvec<float> pmp_angle_and_area_smoothing(
//     CGAL_t::vecvec<float> vertices,
//     CGAL_t::vecvec<int> faces,
//     std::vector<int> constrained_vertices,
//     const unsigned int nb_iterations,
//     bool use_angle_smoothing = true,
//     bool use_area_smoothing = true,
//     bool use_delaunay_flips = true,
//     bool use_safety_constraints = false)
// {
//     Surface_mesh mesh = CGAL_sm::build(vertices, faces);

//     CGAL::Real_timer timer;
//     timer.start();

//     std::set<vertex_descriptor> indices;
//     for (int i : constrained_vertices)
//     {
//         indices.insert((vertex_descriptor)i);
//     }
//     std::cout << "constraining " << indices.size() << " vertices" << std::endl;
//     CGAL::Boolean_property_map<std::set<vertex_descriptor>> vcmap(indices);

//     PMP::angle_and_area_smoothing(mesh, CGAL::parameters::number_of_iterations(nb_iterations)
//                                             .use_angle_smoothing(use_angle_smoothing)
//                                             .use_area_smoothing(use_area_smoothing)
//                                             .use_Delaunay_flips(use_delaunay_flips)
//                                             .use_safety_constraints(use_safety_constraints)
//                                             .vertex_is_constrained_map(vcmap));

//     std::cout << "Elapsed time (smoothing): " << timer.time() << std::endl;

//     auto vertices_out = CGAL_sm::extract_vertices(mesh);

//     return vertices_out;
// }

// CGAL_t::vecvec<float> pmp_fair(
//     CGAL_t::vecvec<float> vertices,
//     CGAL_t::vecvec<int> faces,
//     std::vector<int> indices)
// {
//     Surface_mesh mesh = CGAL_sm::build(vertices, faces);

//     CGAL::Real_timer timer;
//     timer.start();

//     std::set<vertex_descriptor> vertex_indices;
//     for (int i : indices)
//     {
//         vertex_indices.insert((vertex_descriptor)i);
//     }
//     std::cout << "fairing " << vertex_indices.size() << " vertices" << std::endl;
//     CGAL::Boolean_property_map<std::set<vertex_descriptor>> vcmap(vertex_indices);

//     auto success = PMP::fair(mesh, vertex_indices);

//     std::cout << "Fairing : " << (success ? "succeeded" : "failed") << std::endl;
//     std::cout << "Elapsed time (fairing): " << timer.time() << std::endl;

//     auto vertices_faired = CGAL_sm::extract_vertices(mesh);

//     return vertices_faired;
// }


std::pair<CGAL_t::vecvec<float>, CGAL_t::vecvec<int>> pmp_isotropic_remeshing(
    CGAL_t::vecvec<float> vertices,
    CGAL_t::vecvec<int> faces,
    const double target_edge_length,
    const int n_iterations)
{
    Surface_mesh mesh = CGAL_sm::build(vertices, faces);

    // CGAL::Real_timer timer;
    // timer.start();

    PMP::isotropic_remeshing(
        mesh.faces(),
        target_edge_length,
        mesh,
        CGAL::parameters::number_of_iterations(n_iterations)
    );

    // std::cout << "Elapsed time (isotropic remeshing): " << timer.time() << std::endl;

    // explicit garbage collection needed as vertices are only *marked* as removed
    //
    //   https://github.com/CGAL/cgal/discussions/6625
    //   https://doc.cgal.org/latest/Surface_mesh/index.html#sectionSurfaceMesh_memory
    mesh.collect_garbage();

    auto pair = CGAL_sm::extract_vertices_and_faces(mesh);

    return pair;
}


// // Compute union between two meshes and refine.
// std::pair<CGAL_t::vecvec<float>, CGAL_t::vecvec<int>> pmp_corefine_and_union(
//     CGAL_t::vecvec<float> vertices1,
//     CGAL_t::vecvec<int> faces1,
//     CGAL_t::vecvec<float> vertices2,
//     CGAL_t::vecvec<int> faces2)
// {
//     Surface_mesh mesh1 = CGAL_sm::build(vertices1, faces1);
//     Surface_mesh mesh2 = CGAL_sm::build(vertices2, faces2);
//     Surface_mesh mesh_union;

//     bool valid_union = PMP::corefine_and_compute_union(mesh1, mesh2, mesh_union);
//     if (valid_union)
//     {
//         std::cout << "Union was successfully computed\n";
//         auto pair = CGAL_sm::extract_vertices_and_faces(mesh_union);
//         return pair;
//     }
// }
