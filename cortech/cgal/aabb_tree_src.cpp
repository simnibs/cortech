#include <cmath>

#include <CGAL/Simple_cartesian.h>

#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h> // _3
#include <CGAL/AABB_triangle_primitive.h> // _3

#include <cgal_helpers.h>

// #include <CGAL/Real_timer.h>

using KS = CGAL::Simple_cartesian<double>;

using Point = KS::Point_3;
using Triangle = KS::Triangle_3;
using Iterator = std::vector<Triangle>::iterator;
using Primitive = CGAL::AABB_triangle_primitive<KS, Iterator>;
using AABB_triangle_traits = CGAL::AABB_traits<KS, Primitive>;
using Tree = CGAL::AABB_tree<AABB_triangle_traits>;
using Point_and_primitive_id = Tree::Point_and_primitive_id;

std::vector<double> aabb_distance(
    CGAL_t::vecvec<double> vertices,
    CGAL_t::vecvec<int> faces,
    CGAL_t::vecvec<double> query_points,
    CGAL_t::vecvec<double> query_hints,
    bool accelerate_distance_queries = true
){
    int n_vertices = vertices.size();
    int n_faces = faces.size();
    int n_qp = query_points.size();

    // check hints
    bool use_hints;
    if (query_hints.size() == n_qp){
        accelerate_distance_queries = false; // force this
        use_hints = true;
    }
    else if (query_hints.size() == 1 && query_hints[0].empty()) {
        // input like this: [[]]
        use_hints = false;
    }
    else {
        throw std::invalid_argument("`query_hints` should either be empty or contain one hint per query point.");
    }

    std::vector<Point> vp(n_vertices);
    std::vector<Point> qp(n_qp);
    std::vector<Triangle> triangles(n_faces);
    std::vector<double> distance(n_qp);

    // CGAL::Real_timer timer;
    // timer.start();

    // build vertices as Points
    for (int i = 0; i < n_vertices; ++i)
    {
        vp[i] = Point(vertices[i][0], vertices[i][1], vertices[i][2]);
    }
    // build query point as Points
    for (int i = 0; i < n_qp; ++i)
    {
        qp[i] = Point(query_points[i][0], query_points[i][1], query_points[i][2]);
    }
    // build faces as Triangles
    for (int i = 0; i < n_faces; ++i)
        {
            triangles.push_back(Triangle(vp[faces[i][0]], vp[faces[i][1]], vp[faces[i][2]]));
        }

    // build tree
    Tree tree(triangles.begin(), triangles.end());
    tree.build();
    // std::cout << "Elapsed time (build tree): " << timer.time() << std::endl;
    if (accelerate_distance_queries){
        tree.accelerate_distance_queries();
    }
    else {
        tree.do_not_accelerate_distance_queries();
    }
    // std::cout << "Elapsed time (accelerate): " << timer.time() << std::endl;

    // query distances
    if (use_hints){
        for (int i = 0; i < n_qp; ++i)
        {
            Point query_hint = Point(query_hints[i][0], query_hints[i][1], query_hints[i][2]);
            KS::FT squared_distance = tree.squared_distance(qp[i], query_hint);
            distance[i] = sqrt((double)squared_distance);
        }
    }
    else {
        for (int i = 0; i < n_qp; ++i)
        {
            KS::FT squared_distance = tree.squared_distance(qp[i]);
            distance[i] = sqrt((double)squared_distance);
        }
    }
    // std::cout << "Elapsed time (query): " << timer.time() << std::endl;

    return distance;
}

// std::pair<CGAL_t::vecvec<double>,std::vector<int>> aabb_closest_point_and_primitive(
//     CGAL_t::vecvec<double> vertices,
//     CGAL_t::vecvec<int> faces,
//     CGAL_t::vecvec<double> query_points,
//     CGAL_t::vecvec<double> query_hints_point,
//     // CGAL_t::vecvec<int> query_hints_primitive,
//     bool accelerate_distance_queries = true
// ){
//     int n_vertices = vertices.size();
//     int n_faces = faces.size();
//     int n_qp = query_points.size();

//     // check hints
//     bool use_hints;
//     if (query_hints_point.size() == n_qp ){ //&& query_hints_primitive.size() == n_qp
//         accelerate_distance_queries = false; // force this
//         use_hints = true;
//     }
//     else if (query_hints_point.size() == 1 && query_hints_point[0].empty()) {
//         // input like this: [[]]
//         use_hints = false;
//     }
//     else {
//         throw std::invalid_argument("`query_hints_point` should either be empty or contain one hint per query point.");
//     }

//     std::vector<Point> vp(n_vertices);
//     std::vector<Point> qp(n_qp);
//     std::vector<Triangle> triangles(n_faces);
//     CGAL_t::vecvec<double> closest_point(n_qp);
//     std::vector<int> closest_primitive(n_qp);

//     // CGAL::Real_timer timer;
//     // timer.start();

//     Surface_mesh mesh = CGAL_sm::build(vertices, faces);

//     // build vertices as Points
//     // for (int i = 0; i < n_vertices; ++i)
//     // {
//     //     vp[i] = Point(vertices[i][0], vertices[i][1], vertices[i][2]);
//     // }
//     // build query point as Points
//     for (int i = 0; i < n_qp; ++i)
//     {
//         qp[i] = Point(query_points[i][0], query_points[i][1], query_points[i][2]);
//     }
//     // build faces as Triangles
//     // for (int i = 0; i < n_faces; ++i)
//     //     {
//     //         triangles.push_back(Triangle(vp[faces[i][0]], vp[faces[i][1]], vp[faces[i][2]]));
//     //     }

//     // build tree
//     // Tree tree(triangles.begin(), triangles.end());
//     Tree tree(faces(mesh).first, faces(mesh).second(), mesh);
//     tree.build();
//     // std::cout << "Elapsed time (build tree): " << timer.time() << std::endl;
//     if (accelerate_distance_queries){
//         tree.accelerate_distance_queries();
//     }
//     else {
//         tree.do_not_accelerate_distance_queries();
//     }
//     // std::cout << "Elapsed time (accelerate): " << timer.time() << std::endl;

//     // query distances
//     if (use_hints){
//         for (int i = 0; i < n_qp; ++i)
//         {
//             // Point_and_primitive_id query_hint =

//             // Point(query_hints[i][0], query_hints[i][1], query_hints[i][2]);
//             // Primitive(triangles[query_hints_primitive[i]]);

//             // Point_and_primitive_id pp = tree.closest_point_and_primitive(qp[i], query_hint);
//             // Point cp = pp.first;
//             // closest_point[i][0] = (double)cp.x();
//             // closest_point[i][1] = (double)cp.y();
//             // closest_point[i][2] = (double)cp.z();
//             // // Polyhedron::Face_handle primitive_id = pp.second; // closest primitive id
//             // closest_primitive[i] = (int)id(pp.second); // closest primitive id
//         }
//     }
//     else {
//         for (int i = 0; i < n_qp; ++i)
//         {
//             Point_and_primitive_id pp = tree.closest_point_and_primitive(qp[i]);
//             Point cp = pp.first;
//             closest_point[i][0] = cp.x();
//             closest_point[i][1] = cp.y();
//             closest_point[i][2] = cp.z();
//             Surface_mesh::Face_handle primitive_id = pp.second; // closest primitive id
//             // Triangle p = triangles[pp.second];
//             // std::cout << "primi: " << p.vertex(0) << std::endl;
//             // closest_primitive[i] = (int)(p); // closest primitive id
//         }
//     }
//     // std::cout << "Elapsed time (query): " << timer.time() << std::endl;

//     return std::pair(closest_point, closest_primitive);
// }

