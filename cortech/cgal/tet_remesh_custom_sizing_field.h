// Copyright (c) 2023 GeometryFactory (France).
// All rights reserved.
//
// This file is part of CGAL (www.cgal.org).
//
// $URL$
// $Id$
// SPDX-License-Identifier: GPL-3.0-or-later OR LicenseRef-Commercial
//
//
// Author(s)     : Jane Tournois
//
//******************************************************************************
// File Description : Defines a sizing field adapted to a triangulation
//******************************************************************************

#include <CGAL/license/Tetrahedral_remeshing.h>

#include <CGAL/Tetrahedral_remeshing_sizing_field.h>

#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>

#include <CGAL/boost/graph/named_params_helper.h>

#include <CGAL/property_map.h>

#include <CGAL/Tetrahedral_remeshing/internal/tetrahedral_remeshing_helpers.h>
#include <CGAL/Tetrahedral_remeshing/internal/property_maps.h>

#include <array>
#include <limits>
#include <set>
#include <vector>


namespace CGAL
{

    template <typename Tr>
class Precomputed_sizing_field
  : public Tetrahedral_remeshing_sizing_field<typename Tr::Geom_traits>
{
  // Types
public:
  typedef typename Tr::Geom_traits              GT;
  typedef typename GT::FT                       FT;
  typedef typename Tr::Geom_traits::Point_3     Point_3; //Bare_point
  typedef typename Tr::Vertex::Index            Index;

private:
  typedef typename Tr::Point                    Tr_point;
  typedef typename Tr::Facet                    Facet;
  typedef typename Tr::Edge                     Edge;
  typedef typename Tr::Vertex_handle            Vertex_handle;
  typedef typename Tr::Cell_handle              Cell_handle;
  typedef typename Tr::Cell::Surface_patch_index Surface_patch_index;
  typedef typename std::unordered_map<Vertex_handle, FT> Size_map;

  struct Point_with_info
  {
    Point_3 p;
    FT size;
    int dimension;
  };

private:
  struct Point_property_map
  {
    using Self = Point_property_map;
    using value_type = Point_3;
    using reference = value_type; //TODO : why can't that be value_type& ?
    using key_type = Point_with_info;
    using category = boost::readable_property_map_tag;

    const value_type operator[](const key_type& pwi) const { return pwi.p; }
    friend const value_type get(const Self&, const key_type& pwi) { return pwi.p; }
  };

private:
    using Kd_traits = CGAL::Search_traits_adapter<Point_with_info,
                                                Point_property_map,
                                                CGAL::Search_traits_3<GT> >;
    using Neighbor_search = CGAL::Orthogonal_k_neighbor_search<Kd_traits>;
    using Kd_tree = typename Neighbor_search::Tree;
    using Distance = typename Neighbor_search::Distance;
    using Splitter = typename Neighbor_search::Splitter;

public:
    Precomputed_sizing_field(
    const Tr& tr,
    const Size_map& size_map,
    const Size_map& size_map_dim0,
    int nb_neighbors_dim3 = 30,
    int nb_neighbors_dim2 = 6
)
    : m_kd_tree_3(
        points_with_info_dim3(tr, size_map),
        Splitter(),
        Kd_traits(Point_property_map())
    )
    , m_kd_tree_2(
        points_with_info_dim2(tr, size_map),
        Splitter(),
        Kd_traits(Point_property_map())
    )
    , m_kd_tree_0(
        points_with_info_from_size_map(tr, size_map_dim0),
        Splitter(),
        Kd_traits(Point_property_map())
    )
    , m_nb_neighbors_dim3()
    , m_nb_neighbors_dim2()
    {
        m_kd_tree_3.build();
        m_kd_tree_2.build();
        m_kd_tree_0.build();
        m_nb_neighbors_dim3 = nb_neighbors_dim3;
        m_nb_neighbors_dim2 = nb_neighbors_dim2;
        // m_facet_patch_is_constrained = std::get<int>(facet_patch_is_constrained);
    }

private:
    std::vector<Point_with_info> points_with_info_dim3(
        const Tr& tr, const Size_map& size_map) const
    {
        // dim = 3 : inside volume
        namespace Tet_remeshing = CGAL::Tetrahedral_remeshing;
        auto cp = tr.geom_traits().construct_point_3_object();

        std::size_t n = tr.number_of_vertices() + tr.number_of_finite_cells();

        std::vector<Point_with_info> points;
        points.reserve(n);
        for (const Vertex_handle v : tr.finite_vertex_handles())
        {
            FT size = size_map.at(v);
            if (CGAL::is_zero(size))
                continue;
            points.push_back(Point_with_info{ cp(tr.point(v)), size, v->in_dimension() });
        }

        // add internal points
        for (const Cell_handle c : tr.finite_cell_handles())
        {
            // inside cells
            const FT size = average_around_cell(tr, c, size_map);
            if (CGAL::is_zero(size))
                continue;
            points.push_back(Point_with_info{ centroid(tr.tetrahedron(c)), size, 3 });

        }
        std::cout << "kdtree of dim = 3 has size " << points.size() << std::endl;

        return points;
        }

    std::vector<Point_with_info> points_with_info_dim2(
        const Tr& tr, const Size_map& size_map) const
    {
        // dim = 3 : inside volume
        namespace Tet_remeshing = CGAL::Tetrahedral_remeshing;
        auto cp = tr.geom_traits().construct_point_3_object();

        std::vector<Point_with_info> points;
        // points.reserve(tr.number_of_vertices());
        for (const Vertex_handle v : tr.finite_vertex_handles())
        {
            if (v->in_dimension() < 3)
            {
                FT size = size_map.at(v);
                if (CGAL::is_zero(size))
                    continue;
                points.push_back(Point_with_info{ cp(tr.point(v)), size, v->in_dimension() });
            }
        }

        // add points on surface facets
        for (const Cell_handle c : tr.finite_cell_handles())
        {
            for (int i = 0; i < 4; ++i)
            {
                const Cell_handle cn = c->neighbor(i);
                if(c->is_facet_on_surface(i))
                {
                    FT size = average_around_facet(tr, Facet(c, i), size_map);
                    if (CGAL::is_zero(size))
                        continue;
                    points.push_back(Point_with_info{ centroid(tr.triangle(c, i)), size, 2 });
                }
            }
        }
        std::cout << "kdtree of dim = 2 has size " << points.size() << std::endl;
        return points;
    }

    std::vector<Point_with_info> points_with_info_from_size_map(
        const Tr& tr, const Size_map& size_map) const
    {
        // dim = 0 : "corners"
        namespace Tet_remeshing = CGAL::Tetrahedral_remeshing;
        auto cp = tr.geom_traits().construct_point_3_object();
        auto midpt = tr.geom_traits().construct_midpoint_3_object();

        std::vector<Point_with_info> points;
        // points.reserve(size_map.size());
        for (auto const& [v, size] : size_map)
        {
            if (CGAL::is_zero(size))
                continue;
            points.push_back(Point_with_info{ cp(tr.point(v)), FT(size), 0});
        }

        for (const Cell_handle c : tr.finite_cell_handles())
        {
            // on complex edges
            for (const Edge& e : Tet_remeshing::cell_edges(c, tr))
            {
            auto vp = Tet_remeshing::make_vertex_pair(e);
            if (auto a = size_map.find(vp.first); a != size_map.end())
            {
                if (auto b = size_map.find(vp.second); b != size_map.end())
                {
                    FT size = 0.5 * (a->second + b->second);
                    // if(get(ecmap, Tet_remeshing::make_vertex_pair(e)))
                    if (CGAL::is_zero(size))
                        continue;
                    points.push_back(Point_with_info{midpt(tr.segment(e)), size, 0});
                }
            }
            }
        }
        std::cout << "kdtree of size map has size = " << points.size() << std::endl;
        return points;
    }

    // std::vector<Point_with_info> points_with_info_dim0(
    //     const Tr& tr, const Size_map& size_map) const
    // {
    //     // dim = 0 : "corners"
    //     namespace Tet_remeshing = CGAL::Tetrahedral_remeshing;
    //     auto cp = tr.geom_traits().construct_point_3_object();

    //     std::vector<Point_with_info> points;
    //     // points.reserve(tr.number_of_vertices());
    //     for (const Vertex_handle v : tr.finite_vertex_handles())
    //     {
    //         if (v->in_dimension() == 0)
    //         {
    //             FT size = size_map.at(v);
    //             if (CGAL::is_zero(size))
    //                 continue;
    //             points.push_back(Point_with_info{ cp(tr.point(v)), size, 0 });
    //         }
    //     }
    //     // points.resize();
    //     std::cout << "kdtree of dim = 0 has size " << points.size() << std::endl;
    //     return points;
    // }


    std::vector<Point_with_info> points_with_info(
        const Tr& tr,
        const int dim,
        const Size_map& size_map) const
    {
    namespace Tet_remeshing = CGAL::Tetrahedral_remeshing;
    auto cp = tr.geom_traits().construct_point_3_object();

    std::size_t n = tr.number_of_vertices() + tr.number_of_finite_cells();

    std::vector<Point_with_info> points;
    points.reserve(n);
    for (const Vertex_handle v : tr.finite_vertex_handles())
    {
      if(  (dim == 3 && dim == v->in_dimension())//inside volume
        || (dim < 3  && v->in_dimension() < 3))  //on surface
      {
        FT size = size_map.at(v);
        if (CGAL::is_zero(size))
          continue;
        points.push_back(Point_with_info{ cp(tr.point(v)), size, v->in_dimension() });
      }
    }

    // add internal points
    for (const Cell_handle c : tr.finite_cell_handles())
    {
        // inside cells
        if(dim == 3)
        {
            const FT size = average_around_cell(tr, c, size_map);
            if (CGAL::is_zero(size))
            continue;
            points.push_back(Point_with_info{ centroid(tr.tetrahedron(c)), size, 3 });
        }
        // else

        //   {
        //     // on surface facets
        //     for (int i = 0; i < 4; ++i)
        //     {
        //       const Cell_handle cn = c->neighbor(i);
        //       if(  get(cell_selector, c) != get(cell_selector, cn)
        //         || get(fcmap, Facet(c, i))
        //         || c->is_facet_on_surface(i) )
        //       {
        //         const FT size = average_edge_length_2(c, i, tr, size_map);
        //         if (CGAL::is_zero(size))
        //           continue;
        //         points.push_back(Point_with_info{ centroid(tr.triangle(c, i)), size, 2 });
        //       }
        //     }
        //     // on complex edges
        //     for (const Edge& e : Tet_remeshing::cell_edges(c, tr))
        //     {
        //       if(get(ecmap, Tet_remeshing::make_vertex_pair(e)))
        //       {
        //         const FT size = Tet_remeshing::approximate_edge_length(e, tr);
        //         if (CGAL::is_zero(size))
        //           continue;
        //         points.push_back(Point_with_info{midpt(tr.segment(e)), size, 1 });
        //       }
        //     }
        //   }
    }
    return points;
    }

    const Kd_tree& kd_tree(const int dim) const
    {
        if (dim == 3)
            return m_kd_tree_3;
        else
            return m_kd_tree_2;
    }

    int nb_neighbors(const int dim) const
    {
        if (dim == 3)
            return m_nb_neighbors_dim3;
        else
            return m_nb_neighbors_dim2;
    }

public:
    /**
     * Returns size at point `p`, assumed to be included in the input
     * subcomplex with dimension `dim` and index `index`.
     */
    // FT operator()(const Point_3& p, const int& dim, const Index& ) const
    FT operator()(const Point_3& p, const int& dim, const Index& index) const
    {

        if (dim == 0){
            // return FT(1e6);
        // }

            // search only for a single nearest neighbor (no interpolation)
            // in the dim 0 kdtree
            Point_property_map pp_map;
            Distance dist(pp_map);

            // std::cout << "dim = " << dim << " :: index = " << std::get<int>(index) << std::endl;
            const auto sqd = GT().compute_squared_distance_3_object();

            Neighbor_search search(m_kd_tree_0, p, 1, 0.0, true, dist);

            for (const auto& neighbor : search)
            {
                [[maybe_unused]] const auto& [n, size, dimension] = neighbor.first;
                FT distance = sqd(p, n);
                if( CGAL::is_zero(distance)){
                    // std::cout << "  OK returning size " << size << std::endl;
                    return FT(size);
                }
                // else {
                //     std::cout << "dim 0 : squared distance = " << distance << std::endl;
                //     std::cout << "  p : " << p.x() << " " << p.y() << " " << p.z() << std::endl;
                //     std::cout << "  n : " << n.x() << " " << n.y() << " " << n.z() << std::endl;
                //     std::cout << "  [ next kdtree ]" << std::endl;
                // }
            }
        }

        // else go on and do a proper search because this is not one of the points in this kdtree

        // // check if the point is on a constrained domain
        // if (dim == 2) {
        //     // std::cout << "dim = " << dim << " :: index = " << std::get<int>(index) << std::endl;
        //     // if (std::get<int>(index) == 1020){
        //     //     std::cout << "dim = " << dim << " :: index = " << std::get<int>(index) << " " << m_facet_patch_is_constrained << std::endl;
        //     // }

        //     // index is std::variant (Subdomain_index/Surface_patch_index/Curve_index/Corner_index)
        //     // "get" to get the actual integer associated with index
        // }

        // otherwise, perform kdtree search

        Point_property_map pp_map;
        Distance dist(pp_map);

        // Find nearest vertex and local size before remeshing
        Neighbor_search search(
            kd_tree(dim),
            p, //query point
            nb_neighbors(dim), //nb nearest neighbors
            0, //epsilon
            true, //search nearest
            dist
        );

        FT max_size = 0.;
        std::vector<Point_with_info> neighbors;

        for (const auto& neighbor : search)
        {
            [[maybe_unused]] const auto& [pi, size, dimension] = neighbor.first;

            max_size = (std::max)(max_size, size);

            if( (dim == 3 && dimension == dim) //volume point
                || (dim < 3  && dimension < 3) )  //surface point
            {
                neighbors.push_back(neighbor.first);
            }
        }

        if (neighbors.empty())
            return max_size;

        return interpolate_on_n_vertices(p, neighbors);
    }

private:

    FT average_around_cell(
        const Tr& tr, const Cell_handle c, const Size_map& size_map) const
    {
        FT sum = 0.0;
        for (Vertex_handle v : tr.vertices(c))
            sum += size_map.at(v);
        return sum / FT(4.0);
    }

    FT average_around_facet(
        const Tr& tr, const Facet f, const Size_map& size_map) const
    {
        FT sum = 0.0;
        for (Vertex_handle v : tr.vertices(f))
            sum += size_map.at(v);
        return sum / FT(3.0);
    }

    /**
     * Returns size at point `p`, by interpolation among neighbors
     */
    FT interpolate_on_n_vertices(
        const Point_3& p,
        const std::vector<Point_with_info>& points_with_info) const
    {
        // Interpolate value using values at vertices
        const auto sqd = GT().compute_squared_distance_3_object();

        FT sum_weights = 0.;
        FT sum_sizes = 0.;

        for (const auto pwi : points_with_info)
        {
            const FT size = pwi.size;
            const FT sqdist = sqd(pwi.p, p);

            if (is_zero(sqdist))
                return size;

            const FT weight = 1. / CGAL::approximate_sqrt(sqdist);
            sum_weights += weight;
            sum_sizes += weight * size;
        }

        CGAL_assertion(sum_weights > 0);
        CGAL_assertion(sum_sizes > 0);
        return sum_sizes / sum_weights;
    }

private:
    Kd_tree m_kd_tree_3; //volumes
    Kd_tree m_kd_tree_2; //surfaces
    Kd_tree m_kd_tree_0; //corners
    int m_nb_neighbors_dim3;
    int m_nb_neighbors_dim2;
};//end of class Precomputed_sizing_field
} //namespace CGAL
