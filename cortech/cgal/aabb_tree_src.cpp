#include <cmath>

#include <CGAL/Simple_cartesian.h>

#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h> // _3
#include <CGAL/AABB_triangle_primitive.h> // _3

// #include <CGAL/Surface_Mesh.h>

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

// using Segment =  K::Segment_3;
// using Segment_intersection = std::optional<Tree::Intersection_and_primitive_id<Segment>::Type>;
// using Intersection_and_primitive_id = AABBTraits::Intersection_and_primitive_id<Segment>


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
    for (int i = 0; i < n_vertices; i++)
    {
        vp[i] = Point(vertices[i][0], vertices[i][1], vertices[i][2]);
    }
    // build query point as Points
    for (int i = 0; i < n_qp; i++)
    {
        qp[i] = Point(query_points[i][0], query_points[i][1], query_points[i][2]);
    }
    // build faces as Triangles
    for (int i = 0; i < n_faces; i++)
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
        for (int i = 0; i < n_qp; i++)
        {
            Point query_hint = Point(query_hints[i][0], query_hints[i][1], query_hints[i][2]);
            KS::FT squared_distance = tree.squared_distance(qp[i], query_hint);
            distance[i] = sqrt((double)squared_distance);
        }
    }
    else {
        for (int i = 0; i < n_qp; i++)
        {
            KS::FT squared_distance = tree.squared_distance(qp[i]);
            distance[i] = sqrt((double)squared_distance);
        }
    }
    // std::cout << "Elapsed time (query): " << timer.time() << std::endl;

    return distance;
}


// std::tuple<std::vector<int>, std::vector<int>, CGAL_t::vecvec<float>> aabb_all_segment_intersections(
//     CGAL_t::vecvec<float> vertices,
//     CGAL_t::vecvec<int> faces,
//     CGAL_t::vecvec<float> segment_start,
//     CGAL_t::vecvec<float> segment_end
// ){

//     int n_segments = segment_start.size()
//     std::vector<float> intersect_point(3);

//     // output
//     std::vector<int> intersection_ind;   // segment index of intersection
//     std::vector<int> intersection_prim;  // primitive (face) index of intersection
//     CGAL_t::vecvec<float> intersection_pos; // position of intersection


//     for (int i = 0; i < n_segments; i++){
//         a = Point(segment_start[i][0], segment_start[i][1], segment_start[i][2])
//         b = Point(segment_end[i][0], segment_end[i][1], segment_end[i][2])
//         Segment segment_query(a,b);
//         // computes all intersections with segment query (as pairs object - primitive_id)
//         std::list<Segment_intersection> intersections;
//         tree.all_intersections(segment_query, std::back_inserter(intersections));

//         int n_intersections = intersections.size()
//         // for (std::list<Segment_intersection>::iterator it=intersections.begin(); it != intersections.end(); ++it){
//         for (j = 0; j < n_intersections; j++){
//             intersection = intersections[j]

//             intersection_ind.push_back(i);

//             // first is the position of the intersection (point or segment)
//             if (Point p = std::get<Point>(&(intersection->first))){
//                 std::cout << "intersection object is a point" << std::endl;
//             }
//             else if (Segment s = std::get<Segment>(&(intersection->first))){
//                 std::cout << "intersection object is a segment" << std::endl;
//                 // for (int k=0; k<3; k++){
//                 // we use the source point (starting point of segment) as point
//                 // of intersection
//                 Point p = s->source();
//             }
//             else {
//                 return EXIT_FAILURE;
//             }

//             for (k = 0; k<3; k++){
//                 intersect_point[k] = (float) *v;
//             }
//             intersection_pos.push_back(pp)

//             // second is the primitive in which the intersection occurred
//             int prim = (int) *(intersection->second);
//             intersection_prim.push_back(prim);
//         }

//     }
//     return std::make_tuple(intersection_ind, intersection_prim, intersection_pos);
// }

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
//     // for (int i = 0; i < n_vertices; i++)
//     // {
//     //     vp[i] = Point(vertices[i][0], vertices[i][1], vertices[i][2]);
//     // }
//     // build query point as Points
//     for (int i = 0; i < n_qp; i++)
//     {
//         qp[i] = Point(query_points[i][0], query_points[i][1], query_points[i][2]);
//     }
//     // build faces as Triangles
//     // for (int i = 0; i < n_faces; i++)
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
//         for (int i = 0; i < n_qp; i++)
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
//         for (int i = 0; i < n_qp; i++)
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

