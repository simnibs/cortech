// #include <iostream>
#include <vector>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>

#include <CGAL/Polygon_mesh_processing/angle_and_area_smoothing.h>
#include <CGAL/Polygon_mesh_processing/clip.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
// #include <CGAL/Polygon_mesh_processing/corefinement.h>
// #include <CGAL/Polygon_mesh_processing/fair.h>
#include <CGAL/Polygon_mesh_processing/intersection.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/repair_self_intersections.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/smooth_shape.h>
#include <CGAL/Polygon_mesh_processing/tangential_relaxation.h>
#include <CGAL/Polygon_mesh_processing/interpolated_corrected_curvatures.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>

#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/Polygon_mesh_processing/border.h>

// #include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>
#include <CGAL/Side_of_triangle_mesh.h>

#include <cgal_helpers.h>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Surface_mesh = CGAL::Surface_mesh<K::Point_3>;
using vertex_descriptor = Surface_mesh::Vertex_index;
using face_descriptor = boost::graph_traits<Surface_mesh>::face_descriptor;
using halfedge_descriptor = boost::graph_traits<Surface_mesh>::halfedge_descriptor;

namespace PMP = CGAL::Polygon_mesh_processing;

// struct Array_traits
// {
//     struct Equal_3
//     {
//         bool operator()(const std::array<K::FT, 3> &p, const std::array<K::FT, 3> &q) const
//         {
//             return (p == q);
//         }
//     };
//     struct Less_xyz_3
//     {
//         bool operator()(const std::array<K::FT, 3> &p, const std::array<K::FT, 3> &q) const
//         {
//             return std::lexicographical_compare(p.begin(), p.end(), q.begin(), q.end());
//         }
//     };
//     Equal_3 equal_3_object() const { return Equal_3(); }
//     Less_xyz_3 less_xyz_3_object() const { return Less_xyz_3(); }
// };

// std::pair<CGAL_t::vecvec<float>, CGAL_t::vecvec<int>> pmp_repair_mesh(
//     CGAL_t::vecvec<float> vertices,
//     CGAL_t::vecvec<int> faces)
// {
//     // int n_vertices = vertices.size();

//     std::vector<std::array<K::FT, 3>> points;
//     for (int i = 0; i < vertices.size(); ++i)
//     {
//         points[i] = CGAL::make_array<K::FT>(vertices[i][0], vertices[i][1], vertices[i][2]);
//         // points[i] = K::Point_3(vertices[i][0], vertices[i][1], vertices[i][2]);
//     }

//     std::vector<std::array<std::size_t, 3>> polygons;
//     for (int i = 0; i < faces.size(); i++)
//     {
//         polygons[i] = CGAL::make_array<std::size_t>(
//             faces[i][0],
//             faces[i][1],
//             faces[i][2]);
//     }

//     PMP::repair_polygon_soup(points, polygons, CGAL::parameters::geom_traits(Array_traits()));

//     // Surface_mesh mesh;
//     // PMP::orient_polygon_soup(points, polygons);
//     // PMP::polygon_soup_to_polygon_mesh(points, polygons, mesh);

//     CGAL_t::vecvec<float> outpoints;
//     for (int i = 0; i < points.size(); ++i)
//     {
//         outpoints[i] = {points[i][0], points[i][1], points[i][2]};
//     }

//     CGAL_t::vecvec<int> outpolygons;
//     for (int i = 0; i < polygons.size(); ++i)
//     {
//         outpolygons[i] = {polygons[i][0], polygons[i][1], polygons[i][2]};
//     }

//     return std::make_pair(outpoints, outpolygons);
// }

std::pair<CGAL_t::vecvec<float>, CGAL_t::vecvec<int>> pmp_hole_fill_refine_fair(
    CGAL_t::vecvec<float> vertices,
    CGAL_t::vecvec<int> faces)
{
    Surface_mesh mesh = CGAL_sm::build(vertices, faces);

    unsigned int nb_holes = 0;

    // collect one halfedge per boundary cycle
    std::vector<halfedge_descriptor> border_cycles;
    PMP::extract_boundary_cycles(mesh, std::back_inserter(border_cycles));

    for (halfedge_descriptor h : border_cycles)
    {
        // if(max_hole_diam > 0 && max_num_hole_edges > 0 &&
        //     !is_small_hole(h, mesh, max_hole_diam, max_num_hole_edges))
        // continue;

        std::vector<face_descriptor> patch_facets;
        std::vector<vertex_descriptor> patch_vertices;
        bool success = std::get<0>(PMP::triangulate_refine_and_fair_hole(
            mesh,
            h,
            CGAL::parameters::face_output_iterator(std::back_inserter(patch_facets)).vertex_output_iterator(std::back_inserter(patch_vertices))));

        std::string status = (success) ? "success" : "failed";
        std::cout << "Hole " << nb_holes << std::endl;
        std::cout << "  n faces    : " << patch_facets.size() << std::endl;
        std::cout << "  n vertices : " << patch_vertices.size() << std::endl;
        std::cout << "  status     : " << status << std::endl;
        ++nb_holes;
    }

    std::cout << std::endl;
    std::cout << nb_holes << " holes have been filled" << std::endl;
    auto pair = CGAL_sm::extract_vertices_and_faces(mesh);

    return pair;
}

std::pair<CGAL_t::vecvec<float>, CGAL_t::vecvec<int>> pmp_split(
    CGAL_t::vecvec<float> vertices,
    CGAL_t::vecvec<int> faces,
    std::vector<float> plane_origin,
    std::vector<float> plane_direction)
{
    Surface_mesh mesh = CGAL_sm::build(vertices, faces);

    // plane
    K::Point_3 origin = K::Point_3(plane_origin[0], plane_origin[1], plane_origin[2]);
    K::Vector_3 direction = K::Vector_3(plane_direction[0], plane_direction[1], plane_direction[2]);
    K::Plane_3 plane = K::Plane_3(origin, direction);

    PMP::split(mesh, plane);
    auto pair = CGAL_sm::extract_vertices_and_faces(mesh);
    return pair;
}

std::pair<CGAL_t::vecvec<float>, CGAL_t::vecvec<int>> pmp_clip(
    CGAL_t::vecvec<float> vertices,
    CGAL_t::vecvec<int> faces,
    std::vector<float> plane_origin,
    std::vector<float> plane_direction)
{
    Surface_mesh mesh = CGAL_sm::build(vertices, faces);

    // plane
    K::Point_3 origin = K::Point_3(plane_origin[0], plane_origin[1], plane_origin[2]);
    K::Vector_3 direction = K::Vector_3(plane_direction[0], plane_direction[1], plane_direction[2]);
    K::Plane_3 plane = K::Plane_3(origin, direction);

    // bool is_manifold =
    PMP::clip(mesh, plane, CGAL::parameters::clip_volume(true));
    mesh.collect_garbage();
    auto pair = CGAL_sm::extract_vertices_and_faces(mesh);
    return pair;
}

CGAL_t::vecvec<int> pmp_intersecting_meshes(
    CGAL_t::vecvec<float> vertices0,
    CGAL_t::vecvec<int> faces0,
    CGAL_t::vecvec<float> vertices1,
    CGAL_t::vecvec<int> faces1)
{
    Surface_mesh mesh0 = CGAL_sm::build(vertices0, faces0);
    Surface_mesh mesh1 = CGAL_sm::build(vertices1, faces1);

    auto np1 = CGAL::parameters::default_values();
    auto np2 = CGAL::parameters::default_values();

    std::vector<std::pair<face_descriptor, face_descriptor>> intersecting_tris;
    PMP::internal::compute_face_face_intersection(mesh0, mesh1, std::back_inserter(intersecting_tris), np1, np2);
    // PMP::self_intersections<CGAL::Parallel_if_available_tag>(mesh.faces(), mesh, std::back_inserter(intersecting_tris));

    int n_intersections = intersecting_tris.size();
    CGAL_t::vecvec<int> intersecting_faces(n_intersections, std::vector<int>(2));
    for (int i = 0; i < n_intersections; i++)
    {
        intersecting_faces[i][0] = (int)intersecting_tris[i].first;
        intersecting_faces[i][1] = (int)intersecting_tris[i].second;
    }

    return intersecting_faces;
}

CGAL_t::vecvec<float> pmp_interpolated_corrected_curvatures(CGAL_t::vecvec<float> vertices, CGAL_t::vecvec<int> faces)
{
    Surface_mesh mesh = CGAL_sm::build(vertices, faces);

    // define property map to store curvature value and directions
    Surface_mesh::Property_map<vertex_descriptor, K::FT> mean_curv_map, gaussian_curv_map;
    Surface_mesh::Property_map<vertex_descriptor, PMP::Principal_curvatures_and_directions<K>> principal_curv_and_dir_map;

    // creating and tying surface mesh property maps for curvatures (with defaults = 0)
    bool created = false;
    boost::tie(mean_curv_map, created) = mesh.add_property_map<vertex_descriptor, K::FT>("v:mean_curv_map", 0);
    assert(created);
    boost::tie(gaussian_curv_map, created) = mesh.add_property_map<vertex_descriptor, K::FT>("v:gaussian_curv_map", 0);
    assert(created);
    boost::tie(principal_curv_and_dir_map, created) = mesh.add_property_map<vertex_descriptor, PMP::Principal_curvatures_and_directions<K>>("v:principal_curv_and_dir_map", {0, 0, K::Vector_3(0, 0, 0), K::Vector_3(0, 0, 0)});
    assert(created);

    PMP::orient(mesh); // ensure outwards pointing normals
    PMP::interpolated_corrected_curvatures(mesh,
                                           CGAL::parameters::vertex_mean_curvature_map(mean_curv_map)
                                               .vertex_Gaussian_curvature_map(gaussian_curv_map)
                                               .vertex_principal_curvatures_and_directions_map(principal_curv_and_dir_map)
                                           // uncomment to use an expansion ball radius of 0.5 to estimate the curvatures
                                           //                 .ball_radius(0.5)
    );

    std::vector<float> zero_vector = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<std::vector<float>> curv(mesh.number_of_vertices(), zero_vector);
    int i = 0;
    for (auto v : mesh.vertices())
    {
        auto PC = get(principal_curv_and_dir_map, v);
        curv[i][0] = (float)PC.max_curvature;          // k1
        curv[i][1] = (float)PC.min_curvature;          // k2
        curv[i][2] = (float)get(mean_curv_map, v);     // H
        curv[i][3] = (float)get(gaussian_curv_map, v); // K
        curv[i][4] = (float)PC.max_direction[0];
        curv[i][5] = (float)PC.max_direction[1];
        curv[i][6] = (float)PC.max_direction[2];
        curv[i][7] = (float)PC.min_direction[0];
        curv[i][8] = (float)PC.min_direction[1];
        curv[i][9] = (float)PC.min_direction[2];
        i++;
    }
    return curv;
}

std::vector<bool> pmp_points_inside_surface(
    CGAL_t::vecvec<float> vertices,
    CGAL_t::vecvec<int> faces,
    CGAL_t::vecvec<float> points,
    bool on_boundary_is_inside = true)
{
    Surface_mesh mesh = CGAL_sm::build(vertices, faces);
    CGAL::Side_of_triangle_mesh<Surface_mesh, K> inside(mesh);

    std::size_t n_points = mesh.number_of_vertices();
    std::vector<bool> is_inside(n_points, false);

    for (std::size_t i = 0; i < n_points; ++i)
    {
        auto p = K::Point_3(points[i][0], points[i][1], points[i][2]);

        CGAL::Bounded_side res = inside(p);

        if (res == CGAL::ON_BOUNDED_SIDE)
        {
            is_inside[i] = true;
        }
        // point is *on* the boundary
        else if (res == CGAL::ON_BOUNDARY && on_boundary_is_inside)
        {
            is_inside[i] = true;
        }
        // else {
        //     is_inside[i] = false;
        // }
    }
    return is_inside;
}

std::pair<CGAL_t::vecvec<float>, CGAL_t::vecvec<int>> pmp_remove_self_intersections(
    CGAL_t::vecvec<float> vertices, CGAL_t::vecvec<int> faces)
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

    std::vector<std::pair<face_descriptor, face_descriptor>> intersecting_tris;
    PMP::self_intersections<CGAL::Parallel_if_available_tag>(
        mesh, std::back_inserter(intersecting_tris));

    int n_intersections = intersecting_tris.size();
    CGAL_t::vecvec<int> intersecting_faces(n_intersections, std::vector<int>(2));
    for (int i = 0; i < n_intersections; i++)
    {
        intersecting_faces[i][0] = (int)intersecting_tris[i].first;
        intersecting_faces[i][1] = (int)intersecting_tris[i].second;
    }

    return intersecting_faces;
}

std::pair<std::vector<int>, std::vector<int>> pmp_connected_components(
    CGAL_t::vecvec<float> vertices,
    CGAL_t::vecvec<int> faces,
    std::vector<int> constrained_faces)
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
            if (indices_with_counts.count(edge) == 0)
            {
                // new edge
                indices_with_counts[edge] = 1;
            }
            else
            {
                // already seen edge
                indices_with_counts[edge]++;
            }
        }
    }
    // Keep only edges which occur once (i.e., "outer" edges)
    std::set<Surface_mesh::Edge_index> indices;
    for (const auto &pair : indices_with_counts)
    {
        if (pair.second == 1)
        {
            indices.insert(pair.first);
        }
    }
    CGAL::Boolean_property_map<std::set<Surface_mesh::Edge_index>> constrained_edges_map(indices);

    // face component map (output)
    Surface_mesh::Property_map<face_descriptor, std::size_t> fccmap = mesh.add_property_map<face_descriptor, std::size_t>("f:CC").first;

    std::size_t num = PMP::connected_components(
        mesh,
        fccmap,
        CGAL::parameters::edge_is_constrained_map(constrained_edges_map));

    // typedef std::map<std::size_t /*index of CC*/, unsigned int /*nb*/> Components_size;
    // Components_size nb_per_cc;
    // for (face_descriptor f : mesh.faces())
    // {
    //     nb_per_cc[fccmap[f]]++;
    // }
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

CGAL_t::vecvec<float> pmp_tangential_relaxation(
    CGAL_t::vecvec<float> vertices,
    CGAL_t::vecvec<int> faces,
    std::vector<int> constrained_vertices,
    const unsigned int nb_iterations)
{
    Surface_mesh mesh = CGAL_sm::build(vertices, faces);

    std::set<vertex_descriptor> indices;
    for (int i : constrained_vertices)
    {
        indices.insert((vertex_descriptor)i);
    }
    CGAL::Boolean_property_map<std::set<vertex_descriptor>> vcmap(indices);

    PMP::tangential_relaxation(
        mesh,
        CGAL::parameters::number_of_iterations(nb_iterations)
            .vertex_is_constrained_map(vcmap));

    auto vertices_out = CGAL_sm::extract_vertices(mesh);

    return vertices_out;
}

CGAL_t::vecvec<float> pmp_smooth_shape(
    CGAL_t::vecvec<float> vertices,
    CGAL_t::vecvec<int> faces,
    std::vector<int> constrained_vertices,
    const double time,
    const unsigned int nb_iterations)
{
    Surface_mesh mesh = CGAL_sm::build(vertices, faces);

    std::set<vertex_descriptor> indices;
    for (int i : constrained_vertices)
    {
        indices.insert((vertex_descriptor)i);
    }
    CGAL::Boolean_property_map<std::set<vertex_descriptor>> vcmap(indices);

    PMP::smooth_shape(
        mesh,
        time,
        CGAL::parameters::number_of_iterations(nb_iterations)
            .vertex_is_constrained_map(vcmap));

    auto vertices_out = CGAL_sm::extract_vertices(mesh);

    return vertices_out;
}

CGAL_t::vecvec<float> pmp_smooth_angle_and_area(
    CGAL_t::vecvec<float> vertices,
    CGAL_t::vecvec<int> faces,
    std::vector<int> constrained_vertices,
    const unsigned int nb_iterations,
    bool use_angle_smoothing = true,
    bool use_area_smoothing = true,
    bool use_delaunay_flips = true,
    bool use_safety_constraints = false)
{
    Surface_mesh mesh = CGAL_sm::build(vertices, faces);

    std::set<vertex_descriptor> indices;
    for (int i : constrained_vertices)
    {
        indices.insert((vertex_descriptor)i);
    }
    CGAL::Boolean_property_map<std::set<vertex_descriptor>> vcmap(indices);

    PMP::angle_and_area_smoothing(mesh, CGAL::parameters::number_of_iterations(nb_iterations)
                                            .use_angle_smoothing(use_angle_smoothing)
                                            .use_area_smoothing(use_area_smoothing)
                                            .use_Delaunay_flips(use_delaunay_flips)
                                            .use_safety_constraints(use_safety_constraints)
                                            .vertex_is_constrained_map(vcmap));

    auto vertices_out = CGAL_sm::extract_vertices(mesh);

    return vertices_out;
}

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

    PMP::isotropic_remeshing(
        mesh.faces(),
        target_edge_length,
        mesh,
        CGAL::parameters::number_of_iterations(n_iterations));

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
