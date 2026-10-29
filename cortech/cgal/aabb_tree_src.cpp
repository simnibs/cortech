#include <cmath>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_triangle_primitive_3.h>

#include <cgal_helpers.h>

using std::vector;

using KS = CGAL::Simple_cartesian<double>;

using Point = KS::Point_3;
using Triangle = KS::Triangle_3;
using Tetrahedron = KS::Tetrahedron_3;
using Iterator = vector<Triangle>::iterator;
using Primitive = CGAL::AABB_triangle_primitive_3<KS, Iterator>;
using Tree = CGAL::AABB_tree<CGAL::AABB_traits_3<KS, Primitive>>;
using Point_and_primitive_id = Tree::Point_and_primitive_id;

vector<Point> build_points(vector<vector<double>> vertices)
{
    int n_vertices = vertices.size();
    vector<Point> vp(n_vertices);
    for (int i = 0; i < n_vertices; i++)
    {
        auto &p = vertices[i];
        vp[i] = Point(p[0], p[1], p[2]);
    }
    return vp;
}

vector<Triangle> build_triangles(vector<Point> points, vector<vector<int>> faces)
{
    int n_faces = faces.size();
    vector<Triangle> triangles(n_faces);
    for (int i = 0; i < n_faces; i++)
    {
        auto &f = faces[i];
        triangles[i] = Triangle(points[f[0]], points[f[1]], points[f[2]]);
    }
    return triangles;
}

vector<double> aabb_distance(
    vector<vector<double>> vertices,
    vector<vector<int>> faces,
    vector<vector<double>> query_points,
    vector<vector<double>> query_hints = {},
    bool accelerate_distance_queries = true)
{
    std::size_t n_qp = query_points.size();

    // check hints
    bool use_hints;
    if (query_hints.size() == n_qp)
    {
        accelerate_distance_queries = false; // force this
        use_hints = true;
    }
    else if (query_hints.size() == 1 && query_hints[0].empty())
        use_hints = false; // input like this: [[]]
    else
        throw std::invalid_argument("`query_hints` should either be empty or contain one hint per query point.");

    auto vp = build_points(vertices);
    auto qp = build_points(query_points);
    auto triangles = build_triangles(vp, faces);

    Tree tree(triangles.begin(), triangles.end());
    tree.build();
    if (accelerate_distance_queries)
        tree.accelerate_distance_queries();
    else
        tree.do_not_accelerate_distance_queries();

    vector<double> distance(n_qp);
    if (use_hints)
    {
        for (int i = 0; i < n_qp; i++)
        {
            auto qh = query_hints[i];
            Point query_hint = Point(qh[0], qh[1], qh[2]);
            KS::FT squared_distance = tree.squared_distance(qp[i], query_hint);
            distance[i] = sqrt((double)squared_distance);
        }
    }
    else
    {
        for (int i = 0; i < n_qp; i++)
        {
            KS::FT squared_distance = tree.squared_distance(qp[i]);
            distance[i] = sqrt((double)squared_distance);
        }
    }

    return distance;
}

// std::pair<vector<vector<double>>,vector<int>> aabb_closest_point_and_primitive(
//     vector<vector<double>> vertices,
//     vector<vector<int>> faces,
//     vector<vector<double>> query_points,
//     vector<vector<double>> query_hints_point,
//     // vector<vector<int>> query_hints_primitive,
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

//     vector<Point> vp(n_vertices);
//     vector<Point> qp(n_qp);
//     vector<Triangle> triangles(n_faces);
//     vector<vector<double>> closest_point(n_qp);
//     vector<int> closest_primitive(n_qp);

//     auto vp = build_points(vertices);
//     auto qp = build_points(query_points);
//     auto triangles = build_triangles(vp, faces);

//     Tree tree(triangles.begin(), triangles.end());
//     tree.build();
//     if (accelerate_distance_queries)
//         tree.accelerate_distance_queries();
//     else
//         tree.do_not_accelerate_distance_queries();

//     // // build tree
//     // // Tree tree(triangles.begin(), triangles.end());
//     // Surface_mesh mesh = cortech::build(vertices, faces);
//     // Tree tree(faces(mesh).first, faces(mesh).second(), mesh);
//     // tree.build();
//     // // std::cout << "Elapsed time (build tree): " << timer.time() << std::endl;
//     // if (accelerate_distance_queries){
//     //     tree.accelerate_distance_queries();
//     // }
//     // else {
//     //     tree.do_not_accelerate_distance_queries();
//     // }

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
//             closest_point[i] = {cp.x(), cp.y(), cp.z()};
//             // Surface_mesh::Face_handle primitive_id = pp.second; // closest primitive id
//             // Triangle p = triangles[pp.second];
//             // std::cout << "primi: " << p.vertex(0) << std::endl;
//             closest_primitive[i] = (int)(p); // closest primitive id
//         }
//     }
//     // std::cout << "Elapsed time (query): " << timer.time() << std::endl;

//     return std::pair(closest_point, closest_primitive);
// }

// std::tuple<vector<int>, vector<int>, vector<vector<float>>> aabb_all_segment_intersections(
//     vector<vector<float>> vertices,
//     vector<vector<int>> faces,
//     vector<vector<float>> segment_start,
//     vector<vector<float>> segment_end
// ){

//     int n_segments = segment_start.size()
//     vector<float> intersect_point(3);

//     // output
//     vector<int> intersection_ind;   // segment index of intersection
//     vector<int> intersection_prim;  // primitive (face) index of intersection
//     vector<vector<float>> intersection_pos; // position of intersection

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
