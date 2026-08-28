#include <iostream>
#include <numeric>
#include <queue>
#include <vector>


#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/Surface_mesh.h>

#include <CGAL/boost/graph/border.h>

#include <CGAL/Polygon_mesh_processing/Adaptive_sizing_field.h>
#include <CGAL/Polygon_mesh_processing/angle_and_area_smoothing.h>
// #include <CGAL/Polygon_mesh_processing/autorefinement.h>
#include <CGAL/Polygon_mesh_processing/clip.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Polygon_mesh_processing/fair.h>
#include <CGAL/Polygon_mesh_processing/interpolated_corrected_curvatures.h>
#include <CGAL/Polygon_mesh_processing/intersection.h>
// #include <CGAL/Polygon_mesh_processing/manifoldness.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup_extension.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/refine.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/repair_degeneracies.h>
#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/repair_self_intersections.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/smooth_shape.h>
#include <CGAL/Polygon_mesh_processing/stitch_borders.h>
#include <CGAL/Polygon_mesh_processing/tangential_relaxation.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
// #include <CGAL/Polygon_mesh_processing/internal/Snapping/snap.h>

#include <cgal_helpers.h>
#include <pmp_custom_remesher.h>
#include <pmp_custom_sizing_field.h>

using std::vector;

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Surface_mesh = CGAL::Surface_mesh<K::Point_3>;
using Edge_index = Surface_mesh::Edge_index;
using Face_index = Surface_mesh::Face_index;
using Halfedge_index = Surface_mesh::Halfedge_index;
using Vertex_index = Surface_mesh::Vertex_index;

namespace PMP = CGAL::Polygon_mesh_processing;

// struct MeshOutput
// {
//     vector<vector<float>> vertices;
//     vector<vector<int>> faces;
// };

// struct MeshWithPMaps
// {
//     vector<vector<float>> vertices;
//     vector<vector<int>> faces;
//     vector<int> vertices_pmap;
//     vector<int> faces_pmap;
// };

// Surface_mesh::Property_map<Vertex_index, int> make_vertex_id_map(
//     Surface_mesh mesh, std::string name)
// {
//     Surface_mesh::Property_map<Vertex_index, int> v_id;
//     bool created;
//     boost::tie(v_id, created) = mesh.add_property_map<Vertex_index, int>(name, -1);
//     int id = 0;
//     for (auto v : mesh.vertices())
//         v_id[v] = id++;
//     return v_id;
// }

Surface_mesh::Property_map<Face_index, int> make_face_id_map(
    Surface_mesh mesh, std::string name)
{
    Surface_mesh::Property_map<Face_index, int> f_id;
    bool created;
    boost::tie(f_id, created) = mesh.add_property_map<Face_index, int>(name, -1);
    int id = 0;
    for (auto v : mesh.faces())
        f_id[v] = id++;
    return f_id;
}

vector<int> vertex_property_map_to_vector(
    Surface_mesh mesh, std::string name)
{
    // pmap is std::optional< ... >
    auto pmap = mesh.property_map<Vertex_index, int>(name);
    assert(pmap);
    auto pmap_value = pmap.value();
    vector<int> v_map_vec(mesh.number_of_vertices());
    int i = 0;
    for (auto v : mesh.vertices())
        v_map_vec[i++] = pmap_value[v];
    return v_map_vec;
}
vector<int> face_property_map_to_vector(
    Surface_mesh mesh,
    std::string name)
{
    auto pmap = mesh.property_map<Face_index, int>(name);
    assert(pmap);
    auto pmap_value = pmap.value();
    vector<int> f_map_vec(mesh.number_of_faces());
    int i = 0;
    for (auto f : mesh.faces())
        f_map_vec[i++] = pmap_value[f];
    return f_map_vec;
}

void add_property_map_face_id(Surface_mesh &mesh, std::string name = "f:original_id"){
    // Surface_mesh::Property_map<Face_index, int> orig_f_id;
    // .second is a boolean indicate status of "creation"
    auto orig_f_id = mesh.add_property_map<Face_index, int>(name, -1).first;
    int i = 0;
    for (auto f : mesh.faces())
        orig_f_id[f] = i++;
}

Surface_mesh::Property_map<Face_index, int> add_property_map_face_patch_id(Surface_mesh &mesh, vector<int> &face_id, std::string name = "f:patch_id"){
    auto face_patch_map = mesh.add_property_map<Face_index, int>(name, -1).first;
    if (!face_id.empty()){
        int i = 0;
        for (Face_index f : mesh.faces())
            face_patch_map[f] = face_id[i++];
    }
    return face_patch_map;
}

void add_property_map_vertex_id(Surface_mesh &mesh, std::string name = "v:original_id"){
    // Surface_mesh::Property_map<Vertex_index, int> orig_v_id;
    // .second is a boolean indicate status of "creation"
    auto orig_v_id = mesh.add_property_map<Vertex_index, int>(name, -1).first;
    int i = 0;
    for (auto v : mesh.vertices())
        orig_v_id[v] = i++;
}

vector<Face_index> index_vector_to_face_range(vector<int> face_is_selected){
    vector<Face_index> face_range;
    face_range.reserve(face_is_selected.size());
    for (int f : face_is_selected)
        face_range.push_back(Face_index(f));
    return face_range;
}

std::set<Edge_index> make_edge_set(
    const Surface_mesh &mesh,
    const vector<vector<int>> &constrained_edges,
    const vector<Vertex_index> &v2v)
{
    std::set<Edge_index> edge_indices; // constrained_edges.size()
    for (auto ei : constrained_edges)
    {
        // find a halfedge containing the two Vertex_index
        Halfedge_index h = mesh.halfedge(v2v[ei[0]], v2v[ei[1]]);
        Edge_index e = mesh.edge(h);
        edge_indices.emplace(e);
    }
    return edge_indices;
}

std::set<Halfedge_index> make_halfedge_set(
    const Surface_mesh &mesh,
    const vector<vector<int>> &edges,
    const vector<Vertex_index> &v2v)
{
    std::set<Halfedge_index> set; // constrained_edges.size()
    for (auto e : edges)
    {
        // find a halfedge containing the two Vertex_index
        Vertex_index v0 = v2v[e[0]];
        Vertex_index v1 = v2v[e[1]];
        Halfedge_index h = mesh.halfedge(v0, v1);
        // ensure halfedge is pointing towards v1
        if (mesh.target(h) != v1)
            h = mesh.opposite(h);
        set.emplace(h);
    }
    return set;
}

template<typename Container>
Container make_halfedge_container(
    const Surface_mesh &mesh,
    const vector<vector<int>> &edges,
    const vector<Vertex_index> &v2v)
{
    Container container;
    for (auto e : edges)
    {
        // find a halfedge containing the two Vertex_index
        Vertex_index v0 = v2v[e[0]];
        Vertex_index v1 = v2v[e[1]];
        Halfedge_index h = mesh.halfedge(v0, v1);
        // ensure halfedge is pointing towards v1
        if (mesh.target(h) != v1)
            h = mesh.opposite(h);
        container.emplace(h);
    }
    return container;
}


CGAL::Boolean_property_map<std::set<Edge_index>> make_edge_is_constrained_map(
    const Surface_mesh &mesh,
    const vector<vector<int>> &constrained_edges,
    const vector<Vertex_index> &v2v)
{
    auto ecs = make_edge_set(mesh, constrained_edges, v2v);
    CGAL::Boolean_property_map<std::set<Edge_index>> ecm(ecs);
    return ecm;
}

std::set<Vertex_index> make_vertex_is_constrained_set(
    const vector<int> &constrained_vertices,
    const vector<Vertex_index> &v2v)
{
    std::set<Vertex_index> indices;
    for (auto v : constrained_vertices)
        indices.emplace(v2v[v]);
    return indices;
}

CGAL::Boolean_property_map<std::set<Vertex_index>> make_vertex_is_constrained_map(
    const vector<int> &constrained_vertices,
    const vector<Vertex_index> &v2v)
{
    auto vcs = make_vertex_is_constrained_set(constrained_vertices, v2v);
    CGAL::Boolean_property_map<std::set<Vertex_index>> vcm(vcs);
    return vcm;
}

// struct Array_traits
// {
//     struct Equal_3
//     {
//         bool operator()(std::array<K::FT, 3> &p, std::array<K::FT, 3> &q) const
//         {
//             return (p == q);
//         }
//     };
//     struct Less_xyz_3
//     {
//         bool operator()(std::array<K::FT, 3> &p, std::array<K::FT, 3> &q) const
//         {
//             return std::lexicographical_compare(p.begin(), p.end(), q.begin(), q.end());
//         }
//     };
//     Equal_3 equal_3_object() { return Equal_3(); }
//     Less_xyz_3 less_xyz_3_object() { return Less_xyz_3(); }
// };

// cortech::SurfaceMesh pmp_repair_mesh(
//     vector<vector<float>> vertices,
//     vector<vector<int>> faces)
// {
//     // int n_vertices = vertices.size();

//     vector<std::array<K::FT, 3>> points;
//     for (int i = 0; i < vertices.size(); ++i)
//     {
//         points[i] = CGAL::make_array<K::FT>(vertices[i][0], vertices[i][1], vertices[i][2]);
//         // points[i] = K::Point_3(vertices[i][0], vertices[i][1], vertices[i][2]);
//     }

//     vector<std::array<std::size_t, 3>> polygons;
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

//     vector<vector<float>> outpoints;
//     for (int i = 0; i < points.size(); ++i)
//     {
//         outpoints[i] = {points[i][0], points[i][1], points[i][2]};
//     }

//     vector<vector<int>> outpolygons;
//     for (int i = 0; i < polygons.size(); ++i)
//     {
//         outpolygons[i] = {polygons[i][0], polygons[i][1], polygons[i][2]};
//     }

//     return {outpoints, outpolygons};
// }

// cortech::SurfaceMesh pmp_snap_borders(
//     vector<vector<float>> vertices,
//     vector<vector<int>> faces)
// {
//     Surface_mesh mesh = cortech::build(vertices, faces);
//     PMP::experimental::snap_borders(mesh);
//     return = cortech::extract_vertices_and_faces(mesh);
// }

// cortech::SurfaceMesh pmp_autorefine_triangle_soup(
//     vector<vector<float>> vertices,
//     vector<vector<int>> faces,
//     bool apply_iterative_snap_rounding = true,
//     unsigned int n_iter = 5)
// {
//     auto points = cortech::vertices_to_point3(vertices);
//     PMP::autorefine_triangle_soup(
//         points,
//         faces,
//         CGAL::parameters::concurrency_tag(CGAL::Parallel_if_available_tag())
//             .apply_iterative_snap_rounding(apply_iterative_snap_rounding)
//             .number_of_iterations(n_iter));
//     auto vo = cortech::point3_to_vertices(points);
//     return {vo, faces};
// }

cortech::SurfaceMesh pmp_clip(
    vector<vector<float>> vertices,
    vector<vector<int>> faces,
    vector<float> plane_origin,
    vector<float> plane_direction)
{
    Surface_mesh mesh = cortech::from_polygon_soup(vertices, faces);

    // plane
    K::Point_3 origin = K::Point_3(plane_origin[0], plane_origin[1], plane_origin[2]);
    K::Vector_3 direction = K::Vector_3(plane_direction[0], plane_direction[1], plane_direction[2]);
    K::Plane_3 plane = K::Plane_3(origin, direction);

    // bool is_manifold =
    PMP::clip(mesh, plane, CGAL::parameters::clip_volume(true));
    mesh.collect_garbage();
    return cortech::extract_vertices_and_faces(mesh);
}

std::pair<vector<int>, vector<int>> pmp_connected_components(
    vector<vector<float>> vertices,
    vector<vector<int>> faces,
    vector<vector<int>> constrained_edges = {})
{
    // Surface_mesh mesh = cortech::from_polygon_soup(vertices, faces);
    auto p = cortech::from_polygon_soup_with_vertex_map(vertices, faces);
    Surface_mesh &mesh = p.first;
    vector<Vertex_index> &v2v = p.second;

    // Extract the *outer* edges of `constrained_faces` and use these as constraints

    // std::map<Edge_index, int> indices_with_counts;
    // for (auto fi : constrained_faces)
    // {
    //     Surface_mesh::Halfedge_index h = mesh.halfedge((Surface_mesh::Face_index)fi);
    //     for (Surface_mesh::Halfedge_index hi : mesh.halfedges_around_face(h))
    //     {
    //         auto edge = mesh.edge(hi);
    //         if (indices_with_counts.count(edge) == 0)
    //             indices_with_counts[edge] = 1; // new edge
    //         else
    //             indices_with_counts[edge]++; // already seen edge
    //     }
    // }

    // // Keep only edges which occur once (i.e., "outer" edges)
    // std::set<Surface_mesh::Edge_index> indices;
    // for (auto &pair : indices_with_counts)
    // {
    //     if (pair.second == 1)
    //     {
    //         indices.insert(pair.first);
    //     }
    // }
    // CGAL::Boolean_property_map<std::set<Surface_mesh::Edge_index>> constrained_edges_map(indices);

    auto ecs = make_edge_set(mesh, constrained_edges, v2v);
    CGAL::Boolean_property_map<std::set<Edge_index>> ecm(ecs);
    // face component map (output)
    Surface_mesh::Property_map<Face_index, int> fccmap = mesh.add_property_map<Face_index, int>("f:CC").first;

    std::size_t num = PMP::connected_components(
        mesh,
        fccmap,
        CGAL::parameters::edge_is_constrained_map(ecm));

    vector<int> cc(mesh.number_of_faces());
    vector<int> cc_size(num);
    for (Face_index f : mesh.faces())
    {
        cc[f] = fccmap[f];
        cc_size[fccmap[f]]++;
    }
    return {cc, cc_size};
}

std::pair<
    std::pair<cortech::SurfaceMeshWithPMaps, cortech::SurfaceMeshWithPMaps>,
    std::pair<vector<vector<int>>, vector<vector<int>>>
> pmp_corefine(
    vector<vector<float>> v0,
    vector<vector<int>> f0,
    vector<vector<float>> v1,
    vector<vector<int>> f1,
    bool return_intersection_edges = false)
{
    Surface_mesh m0 = cortech::from_polygon_soup(v0, f0);
    Surface_mesh m1 = cortech::from_polygon_soup(v1, f1);

    add_property_map_face_id(m0);
    add_property_map_face_id(m1);

    add_property_map_vertex_id(m0);
    add_property_map_vertex_id(m1);

    auto ecm0 = m0.add_property_map<Edge_index,bool>(
        "e:is_constrained", false).first;
    auto ecm1 = m1.add_property_map<Edge_index,bool>(
        "e:is_constrained", false).first;

    PMP::corefine(m0, m1,
        CGAL::parameters::edge_is_constrained_map(ecm0),
        CGAL::parameters::edge_is_constrained_map(ecm1));

    vector<vector<int>> edges0, edges1;
    if (return_intersection_edges){
        int i;
        i = 0;
        edges0.resize(m0.number_of_edges(), vector<int>(2));
        for (Edge_index e : m0.edges())
        {
            if (ecm0[e])
                edges0[i++] = {(int)m0.vertex(e, 0), (int)m0.vertex(e, 1)};
        }
        edges0.resize(i);

        i = 0;
        edges1.resize(m1.number_of_edges(), vector<int>(2));
        for (Edge_index e : m1.edges())
        {
            if (ecm1[e])
                edges1[i++] = {(int)m1.vertex(e, 0), (int)m1.vertex(e, 1)};
        }
        edges1.resize(i);
    } else {
        edges0 = {};
        edges1 = {};
    }

    auto outm0 = cortech::extract_vertices_and_faces(m0);
    auto outm1 = cortech::extract_vertices_and_faces(m1);
    auto vertex_id0 = vertex_property_map_to_vector(m0, "v:original_id");
    auto face_id0 = face_property_map_to_vector(m0, "f:original_id");
    auto vertex_id1 = vertex_property_map_to_vector(m1, "v:original_id");
    auto face_id1 = face_property_map_to_vector(m1, "f:original_id");
    cortech::SurfaceMeshWithPMaps outm0pm = {outm0.vertices, outm0.faces, vertex_id0, face_id0};
    cortech::SurfaceMeshWithPMaps outm1pm = {outm1.vertices, outm1.faces, vertex_id1, face_id1};
    auto out0 = std::make_pair(outm0pm, outm1pm);
    auto out1 = std::make_pair(edges0, edges1);
    return {out0, out1};
}

cortech::SurfaceMesh pmp_duplicate_non_manifold_edges_in_polygon_soup(
    vector<vector<float>> vertices,
    vector<vector<int>> faces)
{
    auto points = cortech::vertices_to_point3(vertices);
    PMP::duplicate_non_manifold_edges_in_polygon_soup(points, faces);
    auto vo = cortech::point3_to_vertices(points);
    return {vo, faces};
}

cortech::SurfaceMesh pmp_duplicate_non_manifold_vertices(
    vector<vector<float>> vertices,
    vector<vector<int>> faces)
{
    Surface_mesh mesh = cortech::from_polygon_soup(vertices, faces, false);
    PMP::duplicate_non_manifold_vertices(mesh);
    auto out = cortech::extract_vertices_and_faces(mesh);
    return out;
}

vector<vector<int>> pmp_extract_boundary_cycles(
    vector<vector<float>> vertices,
    vector<vector<int>> faces)
{
    Surface_mesh mesh = cortech::from_polygon_soup(vertices, faces);
    vector<Halfedge_index> boundary_cycles;
    CGAL::extract_boundary_cycles(mesh, std::back_inserter(boundary_cycles));

    // cycle is a vector of halfedges forming one boundary loop
    vector<vector<int>> boundary_cycles_indices;
    // for (auto cycle : boundary_cycles)
    // {
    //     // vector<int> boundary_cycles_index(cycle.size());
    //     // for (Halfedge_index h : cycle)
    //     // {
    //     //     //     Vertex_index v0 = mesh.source(h);
    //     //     //     boundary_cycles_index.push_back((int)v0);
    //     // }
    //     // boundary_cycles_indices.push_back(boundary_cycles_index);
    // }
    return boundary_cycles_indices;
}

// vector<vector<int>> faces(n_faces, vector<int>(3));

// // for each face index, iterate over its halfedges and return all `target` vertices
// int i = 0;
// for (Surface_mesh::Face_index fi : mesh.faces())
// {
//     int j = 0;
//     Surface_mesh::Halfedge_index h = mesh.halfedge(fi);
//     for (Surface_mesh::Halfedge_index hi : mesh.halfedges_around_face(h))
//     {
//         Vertex_index vi = mesh.target(hi);
//         faces[i][j] = (int)vi;
//         j++;
//     }
//     j = 0;
//     i++;
// }
// return faces;

vector<vector<float>> pmp_fair(
    vector<vector<float>> vertices,
    vector<vector<int>> faces,
    vector<int> indices,
    int continuity = 1)
{
    Surface_mesh mesh = cortech::from_polygon_soup(vertices, faces);

    std::set<Vertex_index> vertex_indices;
    for (int i : indices)
    {
        vertex_indices.insert(Vertex_index(i));
    }
    CGAL::Boolean_property_map<std::set<Vertex_index>> vcmap(vertex_indices);

    PMP::fair(mesh, vertex_indices, CGAL::parameters::fairing_continuity(continuity));
    auto vertices_faired = cortech::extract_vertices(mesh);

    return vertices_faired;
}

cortech::SurfaceMesh pmp_refine(
    vector<vector<float>> vertices,
    vector<vector<int>> faces,
    vector<int> faces_to_refine = {},
    double density = 2.0)
{
    Surface_mesh mesh = cortech::from_polygon_soup(vertices, faces);
    // vector<boost::graph_traits<Surface_mesh>::vertex_descriptor> new_vertices;
    // vector<boost::graph_traits<Surface_mesh>::face_descriptor> new_faces;
    vector<Vertex_index> new_vertices;
    vector<Face_index> new_faces;

    if (faces_to_refine.empty())
    {
        PMP::refine(
            mesh,
            mesh.faces(),
            std::back_inserter(new_faces),
            std::back_inserter(new_vertices),
            CGAL::parameters::density_control_factor(density));
    }
    else
    {
        vector<Face_index> selected_faces(faces_to_refine.size());
        for (std::size_t i = 0; i < faces_to_refine.size(); i++)
            selected_faces[i] = Face_index(faces_to_refine[i]);
        PMP::refine(
            mesh,
            selected_faces,
            std::back_inserter(new_faces),
            std::back_inserter(new_vertices),
            CGAL::parameters::density_control_factor(density));
    }
    return cortech::extract_vertices_and_faces(mesh);
}

cortech::SurfaceMesh pmp_hole_fill_refine_fair(
    vector<vector<float>> vertices,
    vector<vector<int>> faces)
{
    Surface_mesh mesh = cortech::from_polygon_soup(vertices, faces);

    unsigned int nb_holes = 0;

    // collect one halfedge per boundary cycle
    vector<Halfedge_index> border_cycles;
    CGAL::extract_boundary_cycles(mesh, std::back_inserter(border_cycles));

    for (Halfedge_index h : border_cycles)
    {
        // if(max_hole_diam > 0 && max_num_hole_edges > 0 &&
        //     !is_small_hole(h, mesh, max_hole_diam, max_num_hole_edges))
        // continue;

        vector<Face_index> patch_facets;
        vector<Vertex_index> patch_vertices;
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
    return cortech::extract_vertices_and_faces(mesh);
}

vector<vector<int>> pmp_intersecting_meshes(
    vector<vector<float>> vertices0,
    vector<vector<int>> faces0,
    vector<vector<float>> vertices1,
    vector<vector<int>> faces1)
{
    auto points0 = cortech::vertices_to_point3(vertices0);
    auto points1 = cortech::vertices_to_point3(vertices1);

    PMP::duplicate_non_manifold_edges_in_polygon_soup(points0, faces0);
    PMP::duplicate_non_manifold_edges_in_polygon_soup(points1, faces1);

    Surface_mesh mesh0 = cortech::from_polygon_soup(points0, faces0);
    Surface_mesh mesh1 = cortech::from_polygon_soup(points1, faces1);

    auto np1 = CGAL::parameters::default_values();
    auto np2 = CGAL::parameters::default_values();

    vector<std::pair<Face_index, Face_index>> intersecting_tris;
    PMP::internal::compute_face_face_intersection(mesh0, mesh1, std::back_inserter(intersecting_tris), np1, np2);
    // PMP::self_intersections<CGAL::Parallel_if_available_tag>(mesh.faces(), mesh, std::back_inserter(intersecting_tris));

    int n_intersections = intersecting_tris.size();
    vector<vector<int>> intersecting_faces(n_intersections, vector<int>(2));
    for (int i = 0; i < n_intersections; i++)
    {
        intersecting_faces[i][0] = (int)intersecting_tris[i].first;
        intersecting_faces[i][1] = (int)intersecting_tris[i].second;
    }

    return intersecting_faces;
}

vector<vector<float>> pmp_interpolated_corrected_curvatures(
    vector<vector<float>> vertices,
    vector<vector<int>> faces)
{
    Surface_mesh mesh = cortech::from_polygon_soup(vertices, faces);

    // define property map to store curvature value and directions
    Surface_mesh::Property_map<Vertex_index, K::FT> mean_curv_map, gaussian_curv_map;
    Surface_mesh::Property_map<Vertex_index, PMP::Principal_curvatures_and_directions<K>> principal_curv_and_dir_map;

    // creating and tying surface mesh property maps for curvatures (with defaults = 0)
    bool created = false;
    boost::tie(mean_curv_map, created) = mesh.add_property_map<Vertex_index, K::FT>("v:mean_curv_map", 0);
    assert(created);
    boost::tie(gaussian_curv_map, created) = mesh.add_property_map<Vertex_index, K::FT>("v:gaussian_curv_map", 0);
    assert(created);
    boost::tie(principal_curv_and_dir_map, created) = mesh.add_property_map<Vertex_index, PMP::Principal_curvatures_and_directions<K>>("v:principal_curv_and_dir_map", {0, 0, K::Vector_3(0, 0, 0), K::Vector_3(0, 0, 0)});
    assert(created);

    // PMP::orient(mesh); // ensure outwards pointing normals

    PMP::interpolated_corrected_curvatures(mesh,
                                           CGAL::parameters::vertex_mean_curvature_map(mean_curv_map)
                                               .vertex_Gaussian_curvature_map(gaussian_curv_map)
                                               .vertex_principal_curvatures_and_directions_map(principal_curv_and_dir_map)
                                           // uncomment to use an expansion ball radius of 0.5 to estimate the curvatures
                                           //                 .ball_radius(0.5)
    );

    vector<float> zero_vector = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    vector<vector<float>> curv(mesh.number_of_vertices(), zero_vector);
    int i = 0;
    for (auto v : mesh.vertices())
    {
        auto PC = get(principal_curv_and_dir_map, v);
        auto &c = curv[i];
        c[0] = (float)PC.max_curvature;          // k1
        c[1] = (float)PC.min_curvature;          // k2
        c[2] = (float)get(mean_curv_map, v);     // H
        c[3] = (float)get(gaussian_curv_map, v); // K
        c[4] = (float)PC.max_direction[0];
        c[5] = (float)PC.max_direction[1];
        c[6] = (float)PC.max_direction[2];
        c[7] = (float)PC.min_direction[0];
        c[8] = (float)PC.min_direction[1];
        c[9] = (float)PC.min_direction[2];
        i++;
    }
    return curv;
}

bool pmp_is_polygon_soup_a_polygon_mesh(vector<vector<int>> faces)
{
    return CGAL::Polygon_mesh_processing::is_polygon_soup_a_polygon_mesh(faces);
}

struct Allow_no_surface_crossing
{
  CGAL::Side_of_triangle_mesh<Surface_mesh, K> m_side_of_tmesh;

  Allow_no_surface_crossing(const Surface_mesh& mesh)
    : m_side_of_tmesh(mesh)
  {}

  bool operator()(Vertex_index, K::Point_3 src, K::Point_3 tgt) const
  {
    const CGAL::Bounded_side s_src = m_side_of_tmesh(src);
    const CGAL::Bounded_side s_tgt = m_side_of_tmesh(tgt);
    return (s_src == s_tgt);
  }
};

cortech::SurfaceMeshWithFaceidAndPMaps pmp_adaptive_remeshing(
    vector<vector<float>> vertices,
    vector<vector<int>> faces,
    double error_tol,
    double edge_length_min,
    double edge_length_max,
    int n_iterations = 1,
    bool protect_constraints = false,
    bool collapse_constraints = true,
    bool do_split = true,
    bool do_collapse = true,
    bool do_flip = true,
    int number_of_relaxation_steps = 1,
    bool disallow_surface_crossing = false,
    vector<int> face_id = {},
    vector<int> face_is_selected = {},
    vector<int> vertex_is_constrained = {},
    vector<vector<int>> edge_is_constrained = {})
{
    auto [mesh, v2v] = cortech::from_polygon_soup_with_vertex_map(vertices, faces);

    add_property_map_face_id(mesh);
    add_property_map_vertex_id(mesh);
    auto face_patch_map = add_property_map_face_patch_id(mesh, face_id);
    // auto vcm = make_vertex_is_constrained_map(vertex_is_constrained, v2v);
    // auto ecm = make_edge_is_constrained_map(mesh, edge_is_constrained, v2v);
    auto ecs = make_edge_set(mesh, edge_is_constrained, v2v);
    CGAL::Boolean_property_map<std::set<Edge_index>> ecm(ecs);
    auto vcs = make_vertex_is_constrained_set(vertex_is_constrained, v2v);
    CGAL::Boolean_property_map<std::set<Vertex_index>> vcm(vcs);

    auto make_np = [&]() {
        return CGAL::parameters::number_of_iterations(n_iterations)
        .edge_is_constrained_map(ecm)
        .vertex_is_constrained_map(vcm)
        .protect_constraints(protect_constraints)
        .collapse_constraints(collapse_constraints)
        .face_patch_map(face_patch_map)
        .do_split(do_split)
        .do_collapse(do_collapse)
        .do_flip(do_flip)
        .number_of_relaxation_steps(number_of_relaxation_steps);
    };

    const std::pair min_max_length{edge_length_min, edge_length_max};

    if (face_is_selected.empty()){
        PMP::Adaptive_sizing_field<Surface_mesh> sizing_field(
            error_tol, min_max_length, mesh.faces(), mesh);

        if (disallow_surface_crossing){
            Allow_no_surface_crossing shall_move(mesh);
            PMP::isotropic_remeshing(mesh.faces(), sizing_field, mesh, make_np().allow_move_functor(shall_move));
        }
        else
            PMP::isotropic_remeshing(mesh.faces(), sizing_field, mesh, make_np());
    }
    else
    {
        auto faces_to_remesh = index_vector_to_face_range(face_is_selected);

        PMP::Adaptive_sizing_field<Surface_mesh> sizing_field(
            error_tol, min_max_length, faces_to_remesh, mesh);

        if (disallow_surface_crossing){
            Allow_no_surface_crossing shall_move(mesh);
            PMP::isotropic_remeshing(faces_to_remesh, sizing_field, mesh, make_np().allow_move_functor(shall_move));
        }
        else
            PMP::isotropic_remeshing(faces_to_remesh, sizing_field, mesh, make_np());
    }

    // explicit garbage collection needed as vertices are only *marked* as removed
    //
    //   https://github.com/CGAL/cgal/discussions/6625
    //   https://doc.cgal.org/latest/Surface_mesh/index.html#sectionSurfaceMesh_memory
    mesh.collect_garbage();

    auto out = cortech::extract_vertices_and_faces(mesh);

    auto orig_v_id = vertex_property_map_to_vector(mesh, "v:original_id");
    auto orig_f_id = face_property_map_to_vector(mesh, "f:original_id");
    auto out_face_id = face_property_map_to_vector(mesh, "f:patch_id");

    return {out.vertices, out.faces, out_face_id, orig_v_id, orig_f_id};
}


cortech::SurfaceMeshWithFaceidAndPMaps pmp_custom_remeshing(
    vector<vector<float>> vertices,
    vector<vector<int>> faces,
    vector<float> sizing,
    int n_iterations = 1,
    bool protect_constraints = false,
    bool collapse_constraints = true,
    bool do_split = true,
    bool do_collapse = true,
    bool do_flip = true,
    int number_of_relaxation_steps = 1,
    bool disallow_surface_crossing = false,
    vector<int> face_id = {},
    vector<int> face_is_selected = {},
    vector<int> vertex_is_constrained = {},
    vector<vector<int>> edge_is_constrained = {})
{
    auto [mesh, v2v] = cortech::from_polygon_soup_with_vertex_map(vertices, faces);
    // auto [sizing_mesh, v2v] = cortech::from_polygon_soup_with_vertex_map(vertices, faces);
    // auto sizing_mesh = cortech::from_polygon_soup(vertices, faces);

    add_property_map_face_id(mesh);
    add_property_map_vertex_id(mesh);
    auto face_patch_map = add_property_map_face_patch_id(mesh, face_id);
    // auto vcm = make_vertex_is_constrained_map(vertex_is_constrained, v2v);
    // auto ecm = make_edge_is_constrained_map(mesh, edge_is_constrained, v2v);
    auto ecs = make_edge_set(mesh, edge_is_constrained, v2v);
    CGAL::Boolean_property_map<std::set<Edge_index>> ecm(ecs);
    auto vcs = make_vertex_is_constrained_set(vertex_is_constrained, v2v);
    CGAL::Boolean_property_map<std::set<Vertex_index>> vcm(vcs);

    auto make_np = [&]() {
        return CGAL::parameters::number_of_iterations(n_iterations)
        .edge_is_constrained_map(ecm)
        .vertex_is_constrained_map(vcm)
        .protect_constraints(protect_constraints)
        .collapse_constraints(collapse_constraints)
        .face_patch_map(face_patch_map)
        .do_split(do_split)
        .do_collapse(do_collapse)
        .do_flip(do_flip)
        .number_of_relaxation_steps(number_of_relaxation_steps);
    };

    PMP::Precomputed_sizing_field<Surface_mesh> sizing_field(sizing, mesh);

    if (face_is_selected.empty()){
        if (disallow_surface_crossing){
            Allow_no_surface_crossing shall_move(mesh);
            PMP::isotropic_remeshing(mesh.faces(), sizing_field, mesh, make_np().allow_move_functor(shall_move));
        }
        else
            PMP::isotropic_remeshing(mesh.faces(), sizing_field, mesh, make_np());
    }
    else
    {
        auto faces_to_remesh = index_vector_to_face_range(face_is_selected);
        if (disallow_surface_crossing){
            Allow_no_surface_crossing shall_move(mesh);
            PMP::isotropic_remeshing(faces_to_remesh, sizing_field, mesh, make_np().allow_move_functor(shall_move));
        }
        else
            PMP::isotropic_remeshing(faces_to_remesh, sizing_field, mesh, make_np());
    }

    // explicit garbage collection needed as vertices are only *marked* as removed
    //
    //   https://github.com/CGAL/cgal/discussions/6625
    //   https://doc.cgal.org/latest/Surface_mesh/index.html#sectionSurfaceMesh_memory
    mesh.collect_garbage();

    auto out = cortech::extract_vertices_and_faces(mesh);

    auto orig_v_id = vertex_property_map_to_vector(mesh, "v:original_id");
    auto orig_f_id = face_property_map_to_vector(mesh, "f:original_id");
    auto out_face_id = face_property_map_to_vector(mesh, "f:patch_id");

    return {out.vertices, out.faces, out_face_id, orig_v_id, orig_f_id};
}

cortech::SurfaceMeshWithFaceidAndPMaps pmp_isotropic_remeshing(
    vector<vector<float>> vertices,
    vector<vector<int>> faces,
    double target_edge_length,
    int n_iterations = 1,
    bool protect_constraints = false,
    bool collapse_constraints = true,
    bool do_split = true,
    bool do_collapse = true,
    bool do_flip = true,
    int number_of_relaxation_steps = 1,
    bool disallow_surface_crossing = false,
    vector<int> face_id = {},
    vector<int> face_is_selected = {},
    vector<int> vertex_is_constrained = {},
    vector<vector<int>> edge_is_constrained = {})
{
    auto [mesh, v2v] = cortech::from_polygon_soup_with_vertex_map(vertices, faces);

    add_property_map_face_id(mesh);
    add_property_map_vertex_id(mesh);
    auto face_patch_map = add_property_map_face_patch_id(mesh, face_id);
    // auto vcm = make_vertex_is_constrained_map(vertex_is_constrained, v2v);
    // auto ecm = make_edge_is_constrained_map(mesh, edge_is_constrained, v2v);
    auto ecs = make_edge_set(mesh, edge_is_constrained, v2v);
    CGAL::Boolean_property_map<std::set<Edge_index>> ecm(ecs);
    auto vcs = make_vertex_is_constrained_set(vertex_is_constrained, v2v);
    CGAL::Boolean_property_map<std::set<Vertex_index>> vcm(vcs);

    auto make_np = [&]() {
        return CGAL::parameters::number_of_iterations(n_iterations)
        .edge_is_constrained_map(ecm)
        .vertex_is_constrained_map(vcm)
        .protect_constraints(protect_constraints)
        .collapse_constraints(collapse_constraints)
        .face_patch_map(face_patch_map)
        .do_split(do_split)
        .do_collapse(do_collapse)
        .do_flip(do_flip)
        .number_of_relaxation_steps(number_of_relaxation_steps);
    };

    if (face_is_selected.empty()){
        if (disallow_surface_crossing){
            Allow_no_surface_crossing shall_move(mesh);
            PMP::isotropic_remeshing(mesh.faces(), target_edge_length, mesh, make_np().allow_move_functor(shall_move));
        }
        else
            PMP::isotropic_remeshing(mesh.faces(), target_edge_length, mesh, make_np());
    }
    else
    {
        auto faces_to_remesh = index_vector_to_face_range(face_is_selected);
        if (disallow_surface_crossing){
            Allow_no_surface_crossing shall_move(mesh);
            PMP::isotropic_remeshing(faces_to_remesh, target_edge_length, mesh, make_np().allow_move_functor(shall_move));
        }
        else
            PMP::isotropic_remeshing(faces_to_remesh, target_edge_length, mesh, make_np());
    }

    // explicit garbage collection needed as vertices are only *marked* as removed
    //
    //   https://github.com/CGAL/cgal/discussions/6625
    //   https://doc.cgal.org/latest/Surface_mesh/index.html#sectionSurfaceMesh_memory
    mesh.collect_garbage();

    auto out = cortech::extract_vertices_and_faces(mesh);

    auto orig_v_id = vertex_property_map_to_vector(mesh, "v:original_id");
    auto orig_f_id = face_property_map_to_vector(mesh, "f:original_id");
    auto out_face_id = face_property_map_to_vector(mesh, "f:patch_id");

    return {out.vertices, out.faces, out_face_id, orig_v_id, orig_f_id};
}


vector<int> pmp_non_manifold_vertices(
    vector<vector<float>> vertices,
    vector<vector<int>> faces)
{
    Surface_mesh mesh = cortech::from_polygon_soup(vertices, faces, false);

    vector<Halfedge_index> out;
    PMP::non_manifold_vertices(mesh, std::back_inserter(out));

    vector<int> non_manifold_vertices;
    non_manifold_vertices.reserve(out.size());
    for (Halfedge_index h : out)
        non_manifold_vertices.push_back(static_cast<int>(mesh.target(h).idx()));
    return non_manifold_vertices;
}

cortech::SurfaceMesh pmp_merge_duplicate_points_in_polygon_soup(
    vector<vector<float>> vertices,
    vector<vector<int>> faces)
{
    vector<K::Point_3> points = cortech::vertices_to_point3(vertices);
    PMP::merge_duplicate_points_in_polygon_soup(points, faces);
    vector<vector<float>> vertices_out = cortech::point3_to_vertices(points);
    return {vertices_out, faces};
}

cortech::SurfaceMesh pmp_merge_duplicate_polygons_in_polygon_soup(
    vector<vector<float>> vertices,
    vector<vector<int>> faces)
{
    vector<K::Point_3> points = cortech::vertices_to_point3(vertices);
    PMP::merge_duplicate_polygons_in_polygon_soup(points, faces);
    vector<vector<float>> vertices_out = cortech::point3_to_vertices(points);
    return {vertices_out, faces};
}

cortech::SurfaceMesh pmp_orient(
    vector<vector<float>> vertices,
    vector<vector<int>> faces,
    bool outward_orientation = true)
{
    Surface_mesh mesh = cortech::from_polygon_soup(vertices, faces);
    PMP::orient(
        mesh, CGAL::parameters::outward_orientation(outward_orientation));
    return cortech::extract_vertices_and_faces(mesh);
}

std::pair<bool, cortech::SurfaceMesh> pmp_orient_polygon_soup(
    vector<vector<float>> vertices,
    vector<vector<int>> faces)
{
    // Consistently orient edges in polygon soup
    auto points = cortech::vertices_to_point3(vertices);
    // status == true
    //  operation succeeded
    // status == false
    //  some points were duplicated thus producing a combinatorically manifold
    // but self-intersecting mesh
    bool status = PMP::orient_polygon_soup(points, faces);
    vector<vector<float>> vertices_out = cortech::point3_to_vertices(points);
    return {status, {vertices_out, faces}};
}

vector<bool> pmp_points_inside(
    vector<vector<float>> vertices,
    vector<vector<int>> faces,
    vector<vector<float>> points,
    bool on_boundary_is_inside = true)
{
    Surface_mesh mesh = cortech::from_polygon_soup(vertices, faces);

    CGAL::Side_of_triangle_mesh<Surface_mesh, K> inside(mesh);

    std::size_t n_points = points.size();
    vector<bool> is_inside(n_points, false);
    for (std::size_t i = 0; i < n_points; i++)
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

cortech::SurfaceMeshWithPMaps pmp_remove_almost_degenerate_faces(
    vector<vector<float>> vertices,
    vector<vector<int>> faces,
    vector<int> face_is_selected = {},
    double cap_threshold = -0.9396926207859083, // cos(160 degrees)
    double needle_threshold = 4.0, // longest/shortest edge
    vector<int> vertex_is_constrained = {},
    vector<vector<int>> edge_is_constrained = {})
{
    auto [mesh, v2v] = cortech::from_polygon_soup_with_vertex_map(vertices, faces);
    add_property_map_face_id(mesh);
    add_property_map_vertex_id(mesh);
    // auto vcm = make_vertex_is_constrained_map(vertex_is_constrained, v2v);
    // auto ecm = make_edge_is_constrained_map(mesh, edge_is_constrained, v2v);

    auto np = CGAL::parameters::cap_threshold(cap_threshold)
        .needle_threshold(needle_threshold);
        // .edge_is_constrained_map(ecm)
        // .vertex_is_constrained_map(vcm);

    if (face_is_selected.empty())
        PMP::remove_almost_degenerate_faces(mesh.faces(), mesh, np);
    else {
        auto face_range = index_vector_to_face_range(face_is_selected);
        PMP::remove_almost_degenerate_faces(face_range, mesh, np);
    }
    mesh.collect_garbage();
    auto v_orig_id = vertex_property_map_to_vector(mesh, "v:original_id");
    auto f_orig_id = face_property_map_to_vector(mesh, "f:original_id");
    auto out_mesh = cortech::extract_vertices_and_faces(mesh);
    return {out_mesh.vertices, out_mesh.faces, v_orig_id, f_orig_id};
}


cortech::SurfaceMeshWithPMaps pmp_remove_self_intersections(
    vector<vector<float>> vertices,
    vector<vector<int>> faces,
    vector<int> face_is_selected = {})
{
    Surface_mesh mesh = cortech::from_polygon_soup(vertices, faces);
    add_property_map_face_id(mesh);
    add_property_map_vertex_id(mesh);
    bool success;
    if (face_is_selected.empty())
        success = PMP::experimental::remove_self_intersections(mesh.faces(), mesh);
    else {
        auto face_range = index_vector_to_face_range(face_is_selected);
        success = PMP::experimental::remove_self_intersections(face_range, mesh);
    }
    if (!success)
        throw std::runtime_error("Failed to remove all self-intersections");

    mesh.collect_garbage();
    auto v_orig_id = vertex_property_map_to_vector(mesh, "v:original_id");
    auto f_orig_id = face_property_map_to_vector(mesh, "f:original_id");
    auto out_mesh = cortech::extract_vertices_and_faces(mesh);
    return {out_mesh.vertices, out_mesh.faces, v_orig_id, f_orig_id};
}

vector<vector<int>> pmp_self_intersections(
    vector<vector<float>> vertices,
    vector<vector<int>> faces)
{
    // Surface_mesh mesh = cortech::from_polygon_soup(vertices, faces);
    auto points = cortech::vertices_to_point3(vertices);
    PMP::duplicate_non_manifold_edges_in_polygon_soup(points, faces);
    Surface_mesh mesh = cortech::from_polygon_soup(points, faces);

    vector<std::pair<Face_index, Face_index>> intersecting_tris;
    PMP::self_intersections<CGAL::Parallel_if_available_tag>(
        mesh, std::back_inserter(intersecting_tris));

    int n_intersections = intersecting_tris.size();
    vector<vector<int>> intersecting_faces(n_intersections, vector<int>(2));
    for (int i = 0; i < n_intersections; i++)
    {
        intersecting_faces[i][0] = (int)intersecting_tris[i].first;
        intersecting_faces[i][1] = (int)intersecting_tris[i].second;
    }

    return intersecting_faces;
}


cortech::SurfaceMeshWithFaceidAndPMaps pmp_collapse_halfedges(
    vector<vector<float>> vertices,
    vector<vector<int>> faces,
    vector<vector<int>> edges,
    vector<int> face_id = {})
{
    auto [mesh, v2v] = cortech::from_polygon_soup_with_vertex_map(vertices, faces);

    add_property_map_face_id(mesh);
    add_property_map_vertex_id(mesh);
    auto face_patch_map = add_property_map_face_patch_id(mesh, face_id);

    // auto hecs = make_halfedge_set(mesh, edges, v2v);
    auto hecs = make_halfedge_container<std::queue<Halfedge_index>>(mesh, edges, v2v);

    PMP::collapse_halfedges(hecs, mesh, CGAL::parameters::face_patch_map(face_patch_map));

    mesh.collect_garbage();

    auto out = cortech::extract_vertices_and_faces(mesh);
    auto orig_v_id = vertex_property_map_to_vector(mesh, "v:original_id");
    auto orig_f_id = face_property_map_to_vector(mesh, "f:original_id");
    auto out_face_id = face_property_map_to_vector(mesh, "f:patch_id");

    return {out.vertices, out.faces, out_face_id, orig_v_id, orig_f_id};
}


cortech::SurfaceMeshWithFaceidAndPMaps pmp_collapse_short_edges(
    vector<vector<float>> vertices,
    vector<vector<int>> faces,
    double target_edge_length,
    vector<int> face_id = {},
    vector<int> face_is_selected = {},
    vector<int> vertex_is_constrained = {},
    vector<vector<int>> edge_is_constrained = {})
{
    auto [mesh, v2v] = cortech::from_polygon_soup_with_vertex_map(vertices, faces);

    add_property_map_face_id(mesh);
    add_property_map_vertex_id(mesh);
    auto face_patch_map = add_property_map_face_patch_id(mesh, face_id);
    // auto vcm = make_vertex_is_constrained_map(vertex_is_constrained, v2v);
    // auto ecm = make_edge_is_constrained_map(mesh, edge_is_constrained, v2v);
    auto ecs = make_edge_set(mesh, edge_is_constrained, v2v);
    CGAL::Boolean_property_map<std::set<Edge_index>> ecm(ecs);
    auto vcs = make_vertex_is_constrained_set(vertex_is_constrained, v2v);
    CGAL::Boolean_property_map<std::set<Vertex_index>> vcm(vcs);

    auto np = CGAL::parameters::number_of_iterations(1)
        .edge_is_constrained_map(ecm)
        .vertex_is_constrained_map(vcm)
        .protect_constraints(false)
        .collapse_constraints(true)
        .face_patch_map(face_patch_map)
        .do_split(false)
        .do_collapse(true)
        .do_flip(false)
        .number_of_relaxation_steps(0);

    PMP::Uniform_sizing_field_strict_short sizing(target_edge_length, mesh);

    if (face_is_selected.empty())
        PMP::isotropic_remeshing(mesh.faces(), sizing, mesh, np);
    else
    {
        auto faces_to_remesh = index_vector_to_face_range(face_is_selected);
        PMP::isotropic_remeshing(faces_to_remesh, sizing, mesh, np);
    }

    mesh.collect_garbage();

    auto out = cortech::extract_vertices_and_faces(mesh);
    auto orig_v_id = vertex_property_map_to_vector(mesh, "v:original_id");
    auto orig_f_id = face_property_map_to_vector(mesh, "f:original_id");
    auto out_face_id = face_property_map_to_vector(mesh, "f:patch_id");

    return {out.vertices, out.faces, out_face_id, orig_v_id, orig_f_id};
}


// cortech::SurfaceMeshWithFaceid pmp_collapse_short_edges(
//     vector<vector<float>> vertices,
//     vector<vector<int>> faces,
//     double sizing,
//     bool collapse_constraint = true,
//     vector<int> face_id = {},
//     vector<vector<int>> edge_is_constrained = {})
// {
//     int i;

//     // Surface_mesh mesh = cortech::from_polygon_soup(vertices, faces);
//     auto p = cortech::from_polygon_soup_with_vertex_map(vertices, faces);
//     Surface_mesh &mesh = p.first;
//     vector<Vertex_index> &v2v = p.second;

//     auto face_patch_map = add_property_map_face_patch_id(mesh, face_id);
//     add_property_map_vertex_id(mesh);
//     add_property_map_face_id(mesh);

//     // auto ecmap = make_edge_is_constrained_map(mesh, edge_is_constrained, v2v);

//     auto np = CGAL::parameters::face_patch_map(face_patch_map);
//         // .edge_is_constrained_map(ecmap);

//     PMP::collapse_short_edges(sizing, mesh, np);

//     cortech::SurfaceMesh out = cortech::extract_vertices_and_faces(mesh);
//     vector<int> out_face_id = face_property_map_to_vector(mesh, "f:patch_id");

//     return {out.vertices, out.faces, out_face_id};
// }

cortech::SurfaceMeshWithFaceidAndPMaps pmp_split_edges(
    vector<vector<float>> vertices,
    vector<vector<int>> faces,
    vector<vector<int>> edges,
    vector<int> face_id = {})
{
    auto [mesh, v2v] = cortech::from_polygon_soup_with_vertex_map(vertices, faces);

    add_property_map_face_id(mesh);
    add_property_map_vertex_id(mesh);
    auto face_patch_map = add_property_map_face_patch_id(mesh, face_id);

    auto ecs = make_edge_set(mesh, edges, v2v);

    PMP::split_edges(ecs, mesh, CGAL::parameters::face_patch_map(face_patch_map));

    mesh.collect_garbage();

    auto out = cortech::extract_vertices_and_faces(mesh);
    auto orig_v_id = vertex_property_map_to_vector(mesh, "v:original_id");
    auto orig_f_id = face_property_map_to_vector(mesh, "f:original_id");
    auto out_face_id = face_property_map_to_vector(mesh, "f:patch_id");

    return {out.vertices, out.faces, out_face_id, orig_v_id, orig_f_id};
}


cortech::SurfaceMeshWithFaceidAndPMaps pmp_flip_edges(
    vector<vector<float>> vertices,
    vector<vector<int>> faces,
    vector<vector<int>> edges,
    vector<int> face_id = {})
{
    auto [mesh, v2v] = cortech::from_polygon_soup_with_vertex_map(vertices, faces);

    add_property_map_face_id(mesh);
    add_property_map_vertex_id(mesh);
    auto face_patch_map = add_property_map_face_patch_id(mesh, face_id);

    auto ecs = make_edge_set(mesh, edges, v2v);

    PMP::flip_edges(ecs, mesh, CGAL::parameters::face_patch_map(face_patch_map));

    mesh.collect_garbage();

    auto out = cortech::extract_vertices_and_faces(mesh);
    auto orig_v_id = vertex_property_map_to_vector(mesh, "v:original_id");
    auto orig_f_id = face_property_map_to_vector(mesh, "f:original_id");
    auto out_face_id = face_property_map_to_vector(mesh, "f:patch_id");

    return {out.vertices, out.faces, out_face_id, orig_v_id, orig_f_id};
}


cortech::SurfaceMeshWithFaceidAndPMaps pmp_split_long_edges(
    vector<vector<float>> vertices,
    vector<vector<int>> faces,
    double sizing,
    vector<int> face_id = {},
    vector<vector<int>> edges = {})
{
    // Surface_mesh mesh = cortech::from_polygon_soup(vertices, faces);
    auto p = cortech::from_polygon_soup_with_vertex_map(vertices, faces);
    Surface_mesh &mesh = p.first;
    vector<Vertex_index> &v2v = p.second;

    add_property_map_face_id(mesh);
    add_property_map_vertex_id(mesh);

    auto face_patch_map = add_property_map_face_patch_id(mesh, face_id);
    auto np = CGAL::parameters::face_patch_map(face_patch_map);

    if (edges.empty())
        PMP::split_long_edges(mesh.edges(), sizing, mesh, np);
    else {
        auto ecs = make_edge_set(mesh, edges, v2v);
        PMP::split_long_edges(ecs, sizing, mesh, np);
    }

    cortech::SurfaceMesh out = cortech::extract_vertices_and_faces(mesh);
    vector<int> out_face_id = face_property_map_to_vector(mesh, "f:patch_id");
    vector<int> orig_vid = vertex_property_map_to_vector(mesh, "v:original_id");
    vector<int> orig_fid = face_property_map_to_vector(mesh, "f:original_id");

    return {out.vertices, out.faces, out_face_id, orig_vid, orig_fid};
}

bool pmp_does_triangle_soup_self_intersect(vector<vector<float>> vertices, vector<vector<int>> faces)
{
    auto points = cortech::vertices_to_point3(vertices);
    return PMP::does_triangle_soup_self_intersect(points, faces);
}

vector<vector<float>> pmp_smooth_angle_and_area(
    vector<vector<float>> vertices,
    vector<vector<int>> faces,
    vector<int> constrained_vertices,
    unsigned int nb_iterations = 1,
    bool use_angle_smoothing = true,
    bool use_area_smoothing = true,
    bool use_delaunay_flips = true,
    bool use_safety_constraints = false)
{
    Surface_mesh mesh = cortech::from_polygon_soup(vertices, faces);

    std::set<Vertex_index> indices;
    for (int i : constrained_vertices)
    {
        indices.insert((Vertex_index)i);
    }
    CGAL::Boolean_property_map<std::set<Vertex_index>> vcmap(indices);

    PMP::angle_and_area_smoothing(mesh, CGAL::parameters::number_of_iterations(nb_iterations)
                                            .use_angle_smoothing(use_angle_smoothing)
                                            .use_area_smoothing(use_area_smoothing)
                                            .use_Delaunay_flips(use_delaunay_flips)
                                            .use_safety_constraints(use_safety_constraints)
                                            .vertex_is_constrained_map(vcmap));

    auto vertices_out = cortech::extract_vertices(mesh);

    return vertices_out;
}

vector<vector<float>> pmp_smooth_shape(
    vector<vector<float>> vertices,
    vector<vector<int>> faces,
    vector<int> constrained_vertices,
    double time,
    unsigned int nb_iterations = 1)
{
    Surface_mesh mesh = cortech::from_polygon_soup(vertices, faces);

    std::set<Vertex_index> indices;
    for (int i : constrained_vertices)
    {
        indices.insert((Vertex_index)i);
    }
    CGAL::Boolean_property_map<std::set<Vertex_index>> vcmap(indices);

    PMP::smooth_shape(
        mesh,
        time,
        CGAL::parameters::number_of_iterations(nb_iterations)
            .vertex_is_constrained_map(vcmap));

    auto vertices_out = cortech::extract_vertices(mesh);

    return vertices_out;
}

vector<vector<float>> pmp_smooth_shape_by_curvature_threshold(
    vector<vector<float>> vertices,
    vector<vector<int>> faces,
    double time,
    unsigned int nb_iterations = 1,
    double curv_threshold = 0.0,
    bool apply_above_curv_threshold = true,
    double ball_radius = -1.0)
{
    // use an expansion ball radius of `ball_radius` to estimate the curvatures
    // -1.0 disables curvature smoothing
    Surface_mesh mesh = cortech::from_polygon_soup(vertices, faces);

    // define property map to store curvature value and directions
    Surface_mesh::Property_map<Vertex_index, K::FT> mean_curv_map;
    // creating and tying surface mesh property maps for curvatures (with defaults = 0)
    bool created = false;
    boost::tie(mean_curv_map, created) = mesh.add_property_map<Vertex_index, K::FT>("v:mean_curv_map", 0);
    assert(created);

    // PMP::orient(mesh); // ensure outwards pointing normals

    for (std::size_t i = 0; i < nb_iterations; i++)
    {
        PMP::interpolated_corrected_curvatures(
            mesh,
            CGAL::parameters::vertex_mean_curvature_map(mean_curv_map)
                .ball_radius(ball_radius));

        // constrain the relevant vertices
        std::set<Vertex_index> indices;
        for (auto v : mesh.vertices())
        {
            float H = (float)get(mean_curv_map, v);
            if ((apply_above_curv_threshold && (H < curv_threshold)) || (!apply_above_curv_threshold && (H > curv_threshold)))
            {
                indices.insert(v);
            }
        }

        CGAL::Boolean_property_map<std::set<Vertex_index>> vcmap(indices);
        PMP::smooth_shape(
            mesh,
            time,
            CGAL::parameters::vertex_is_constrained_map(vcmap));

        // if (remesh_nb_iterations > 0){
        //     PMP::isotropic_remeshing(
        //         mesh.faces(),
        //         remesh_edge_length,
        //         mesh,
        //         CGAL::parameters::number_of_iterations(remesh_nb_iterations));

        //     // explicit garbage collection needed as vertices are only *marked* as removed
        //     //
        //     //   https://github.com/CGAL/cgal/discussions/6625
        //     //   https://doc.cgal.org/latest/Surface_mesh/index.html#sectionSurfaceMesh_memory
        //     mesh.collect_garbage();
        // }
    }

    auto vertices_out = cortech::extract_vertices(mesh);

    return vertices_out;
}

cortech::SurfaceMesh pmp_split_with_plane(
    vector<vector<float>> vertices,
    vector<vector<int>> faces,
    vector<float> plane_origin,
    vector<float> plane_direction)
{
    Surface_mesh mesh = cortech::from_polygon_soup(vertices, faces);

    // plane
    K::Point_3 origin = K::Point_3(plane_origin[0], plane_origin[1], plane_origin[2]);
    K::Vector_3 direction = K::Vector_3(plane_direction[0], plane_direction[1], plane_direction[2]);
    K::Plane_3 plane = K::Plane_3(origin, direction);

    PMP::split(mesh, plane);
    return cortech::extract_vertices_and_faces(mesh);
}

std::pair<cortech::SurfaceMesh, cortech::SurfaceMesh> pmp_split_with_surface(
    vector<vector<float>> mesh_v,
    vector<vector<int>> mesh_f,
    vector<vector<float>> splitter_v,
    vector<vector<int>> splitter_f)
{
    Surface_mesh mesh = cortech::from_polygon_soup(mesh_v, mesh_f);
    Surface_mesh splitter = cortech::from_polygon_soup(splitter_v, splitter_f);

    PMP::split(mesh, splitter);
    auto mesh_out = cortech::extract_vertices_and_faces(mesh);
    auto splitter_out = cortech::extract_vertices_and_faces(splitter);
    return {mesh_out, splitter_out};
}

cortech::SurfaceMeshWithPMaps pmp_stitch_borders(
    vector<vector<float>> vertices,
    vector<vector<int>> faces)
{
    Surface_mesh mesh = cortech::from_polygon_soup(vertices, faces);

    // auto orig_v_id = make_vertex_id_map(mesh);
    // auto orig_f_id = make_face_id_map(mesh);

    int i, id;

    Surface_mesh::Property_map<Vertex_index, int> orig_v_id;
    Surface_mesh::Property_map<Face_index, int> orig_f_id;
    bool created;
    boost::tie(orig_v_id, created) = mesh.add_property_map<Vertex_index, int>("v:original_id", -1);
    boost::tie(orig_f_id, created) = mesh.add_property_map<Face_index, int>("f:original_id", -1);
    id = 0;
    for (auto v : mesh.vertices())
    {
        orig_v_id[v] = id++;
    }
    id = 0;
    for (auto f : mesh.faces())
    {
        orig_f_id[f] = id++;
    }

    PMP::stitch_borders(mesh);
    mesh.collect_garbage();
    auto out = cortech::extract_vertices_and_faces(mesh);

    vector<int> original_vertex_index(mesh.number_of_vertices());
    i = 0;
    for (auto v : mesh.vertices())
        original_vertex_index[i++] = orig_v_id[v];

    vector<int> original_face_index(mesh.number_of_faces());
    i = 0;
    for (auto f : mesh.faces())
        original_face_index[i++] = orig_f_id[f];

    // auto original_vertex_index = vertex_property_map_to_vector(mesh, orig_v_id);
    // auto original_face_index = face_property_map_to_vector(mesh, orig_f_id);
    return {out.vertices, out.faces, original_vertex_index, original_face_index};
}

vector<vector<float>> pmp_tangential_relaxation(
    vector<vector<float>> vertices,
    vector<vector<int>> faces,
    vector<int> constrained_vertices,
    unsigned int nb_iterations = 1)
{
    Surface_mesh mesh = cortech::from_polygon_soup(vertices, faces);

    std::set<Vertex_index> indices;
    for (int i : constrained_vertices)
    {
        indices.insert((Vertex_index)i);
    }
    CGAL::Boolean_property_map<std::set<Vertex_index>> vcmap(indices);

    PMP::tangential_relaxation(
        mesh,
        CGAL::parameters::number_of_iterations(nb_iterations)
            .vertex_is_constrained_map(vcmap));

    auto vertices_out = cortech::extract_vertices(mesh);

    return vertices_out;
}

// std::pair<vector<int>,vector<int>> pmp_volume_connected_components(
//     vector<vector<int>> faces,
//     bool do_orientation_tests = false,
//     bool do_self_intersection_tests = false)
// {
//     Surface_mesh mesh = construct_FaceListGraph(faces);

//     CGAL::Real_timer timer;
//     timer.start();

//     // face component map (output)
//     Surface_mesh::Property_map<Face_index, std::size_t> fccmap = mesh.add_property_map<Face_index, std::size_t>("f:CC").first;

//     std::size_t num = PMP::volume_connected_components(
//         mesh,
//         fccmap,
//         CGAL::parameters::do_orientation_tests(do_orientation_tests).do_self_intersection_tests(do_self_intersection_tests));

//     // CGAL::parameters::do_orientation_tests(true).do_self_intersection_tests(true).is_cc_outward_oriented(true)

//     std::cerr << "- The graph has " << num << " connected components (face connectivity)" << std::endl;

//     typedef std::map<std::size_t /*index of CC*/, unsigned int /*nb*/> Components_size;
//     Components_size nb_per_cc;
//     for (Face_index f : mesh.faces())
//     {
//         nb_per_cc[fccmap[f]]++;
//     }
//     for (Components_size::value_type &cc : nb_per_cc)
//     {
//         std::cout << "\t CC #" << cc.first
//                   << " is made of " << cc.second << " faces" << std::endl;
//     }
//     std::cout << "Elapsed time (connected components): " << timer.time() << std::endl;

//     vector<int> cc(mesh.number_of_faces());
//     vector<int> cc_size(num);
//     for (Face_index f : mesh.faces())
//     {
//         cc[f] = (int)fccmap[f];
//         cc_size[fccmap[f]]++;
//     }
//     auto pair = std::make_pair(cc, cc_size);

//     return pair;
// }
