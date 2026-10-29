// Copyright (c) 2015 GeometryFactory (France).
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


#include <CGAL/license/Polygon_mesh_processing/meshing_hole_filling.h>

#include <CGAL/Polygon_mesh_processing/repair_degeneracies.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Polygon_mesh_processing/shape_predicates.h>

#include <CGAL/Polygon_mesh_processing/internal/Isotropic_remeshing/remesh_impl.h

#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_triangle_primitive_3.h>
#include <CGAL/boost/graph/border.h>
#include <CGAL/boost/graph/Euler_operations.h>
#include <CGAL/boost/graph/properties.h>
#include <CGAL/property_map.h>
#include <CGAL/Dynamic_property_map.h>
#include <CGAL/iterator.h>
#include <CGAL/tags.h>

#include <boost/bimap.hpp>
#include <boost/bimap/multiset_of.hpp>
#include <boost/bimap/set_of.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/range.hpp>
#include <boost/range/join.hpp>
#include <memory>
#include <boost/container/flat_set.hpp>
#include <boost/property_map/function_property_map.hpp>

#include <map>
#include <list>
#include <vector>
#include <iterator>
#include <fstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <optional>

namespace CGAL {

namespace Polygon_mesh_processing {

template <typename PM, typename VPMap> class Uniform_sizing_field;

namespace internal {

//   enum Halfedge_status {
//     PATCH,       //h and hopp belong to the patch to be remeshed
//     PATCH_BORDER,//h belongs to the patch, hopp is MESH
//     MESH,        //h and hopp belong to the mesh, not the patch
//     MESH_BORDER, //h belongs to the mesh, face(hopp, pmesh) == null_face()
//     ISOLATED_CONSTRAINT //h is constrained, and incident to faces that do not belong to a patch
//   };

//   // A property map
//   template <typename PM, typename FaceIndexMap>
//   struct Border_constraint_pmap
//   {
//     typedef typename boost::graph_traits<PM>::halfedge_descriptor halfedge_descriptor;
//     typedef typename boost::graph_traits<PM>::edge_descriptor edge_descriptor;
//     typedef FaceIndexMap FIMap;

//     std::shared_ptr< std::set<edge_descriptor> > border_edges_ptr;
//     const PM* pmesh_ptr_;

//   public:
//     typedef edge_descriptor                     key_type;
//     typedef bool                                value_type;
//     typedef value_type                          reference;
//     typedef boost::read_write_property_map_tag  category;

//     Border_constraint_pmap()
//       : border_edges_ptr(new std::set<edge_descriptor>() )
//       , pmesh_ptr_(nullptr)
//     {}

//     template <class FaceRange>
//     Border_constraint_pmap(const PM& pmesh
//                          , const FaceRange& faces
//                          , const FIMap& fimap)
//       : border_edges_ptr(new std::set<edge_descriptor>() )
//       , pmesh_ptr_(&pmesh)
//     {
//       std::vector<halfedge_descriptor> border;
//       border_halfedges(faces, *pmesh_ptr_, std::back_inserter(border), parameters::face_index_map(fimap));

//       for(halfedge_descriptor h : border)
//         border_edges_ptr->insert(edge(h, *pmesh_ptr_));
//     }

//     friend bool get(const Border_constraint_pmap<PM, FIMap>& map,
//                     const edge_descriptor& e)
//     {
//       CGAL_assertion(map.pmesh_ptr_!=nullptr);
//       return map.border_edges_ptr->count(e)!=0;
//     }

//     friend void put(Border_constraint_pmap<PM, FIMap>& map,
//                     const edge_descriptor& e,
//                     const bool is)
//     {
//       CGAL_assertion(map.pmesh_ptr_ != nullptr);
//       if (is)
//         map.border_edges_ptr->insert(e);
//       else
//         map.border_edges_ptr->erase(e);
//     }
//   };


//   template <typename PM,
//             typename FaceIndexMap>
//   struct Connected_components_pmap
//   {
//     typedef typename boost::graph_traits<PM>::face_descriptor   face_descriptor;
//     typedef std::size_t                                         Patch_id;
//     typedef FaceIndexMap                                        FIMap;
//     typedef Connected_components_pmap<PM, FIMap>                CCMap;

//     typedef CGAL::dynamic_face_property_t<Patch_id> Face_property_tag;
//     typedef typename boost::property_map<PM, Face_property_tag >::type Patch_ids_map;
//     Patch_ids_map patch_ids_map;
//     std::size_t nb_cc;

//     template <class Range>
//     bool same_range(const Range& r1, const Range& r2)
//     {
//       return std::begin(r1)==std::begin(r2) &&
//              std::end(r1)==std::end(r2);
//     }

//     template <class Range1, class Range2>
//     bool same_range(const Range1& r1, const Range2& r2)
//     {
//       return std::distance(std::begin(r1), std::end(r1)) ==
//              std::distance(std::begin(r2), std::end(r2));
//     }

//   public:
//     typedef face_descriptor                     key_type;
//     typedef Patch_id                            value_type;
//     typedef Patch_id                            reference;
//     typedef boost::read_write_property_map_tag  category;

//     //note pmesh is a non-const ref because properties are added and removed
//     //modify the mesh data structure, but not the mesh itself
//     template <class FaceRange, class EdgeIsConstrainedMap>
//     Connected_components_pmap(const FaceRange& face_range
//                             , PM& pmesh
//                             , EdgeIsConstrainedMap ecmap
//                             , FIMap fimap
//                             , const bool do_init = true)
//     {
//       patch_ids_map = get(Face_property_tag(),pmesh);
//       if (do_init)
//       {
// #ifdef CGAL_PMP_REMESHING_VERBOSE
//         std::cout << "Compute connected components property map." << std::endl;
// #endif
//         if ( same_range(face_range, (faces(pmesh))) )
//         {
//           // applied on the whole mesh
//           nb_cc = connected_components(pmesh, patch_ids_map,
//                                        parameters::edge_is_constrained_map(ecmap)
//                                                   .face_index_map(fimap));
//         }
//         else
//         {
//           // applied on a subset of the mesh
//           nb_cc = connected_components(
//                     pmesh, patch_ids_map,
//                     parameters::edge_is_constrained_map(
//                                   make_OR_property_map(ecmap,
//                                                        internal::Border_constraint_pmap<PM, FIMap>(pmesh, face_range, fimap)))
//                                .face_index_map(fimap));
//         }
//       }
//       else
//         nb_cc=0; // default value
//     }


//     friend value_type get(const CCMap& m, const key_type& f)
//     {
//       if (m.nb_cc == 1)
//         return 0;
//       return get(m.patch_ids_map, f);
//     }
//     friend void put(CCMap& m, const key_type& f, const value_type i)
//     {
//       if (m.nb_cc != 1)
//         put(m.patch_ids_map, f, i);
//     }
//   };

//   template<typename PM,
//            typename EdgeConstraintMap,
//            typename FacePatchMap,
//            typename SizingFunction>
//   bool constraints_are_short_enough(const PM& pmesh,
//                                     EdgeConstraintMap ecmap,
//                                     const FacePatchMap& fpm,
//                                     const SizingFunction& sizing)
//   {
//     typedef typename boost::graph_traits<PM>::halfedge_descriptor halfedge_descriptor;
//     typedef typename boost::graph_traits<PM>::edge_descriptor     edge_descriptor;
//     for(edge_descriptor e : edges(pmesh))
//     {
//       halfedge_descriptor h = halfedge(e, pmesh);
//       if (  is_border(e, pmesh) ||
//             get(ecmap, e) ||
//             get(fpm, face(h,pmesh))!=get(fpm, face(opposite(h,pmesh),pmesh)) )
//       {
//         if (sizing.is_too_long(source(h, pmesh), target(h, pmesh), pmesh))
//         {
//           return false;
//         }
//       }
//     }
//     return true;
//   }

  template<typename PolygonMesh
         , typename VertexPointMap
         , typename GeomTraits
         , typename EdgeIsConstrainedMap
         , typename VertexIsConstrainedMap
         , typename FacePatchMap
         , typename FaceIndexMap
  >
  class Edge_range_collapser
  {
    typedef PolygonMesh PM;
    typedef typename boost::graph_traits<PM>::halfedge_descriptor halfedge_descriptor;
    typedef typename boost::graph_traits<PM>::edge_descriptor     edge_descriptor;
    typedef typename boost::graph_traits<PM>::vertex_descriptor   vertex_descriptor;
    typedef typename boost::graph_traits<PM>::face_descriptor     face_descriptor;

    typedef typename GeomTraits::FT         FT;
    typedef typename GeomTraits::Point_3    Point;
    typedef typename GeomTraits::Vector_3   Vector_3;
    typedef typename GeomTraits::Plane_3    Plane_3;
    typedef typename GeomTraits::Triangle_3 Triangle_3;

    typedef Edge_range_collapser<PM, VertexPointMap
                               , GeomTraits
                               , EdgeIsConstrainedMap
                               , VertexIsConstrainedMap
                               , FacePatchMap
                               , FaceIndexMap
                               > Self;

  private:
    typedef typename boost::property_traits<FacePatchMap>::value_type Patch_id;
    typedef std::vector<Triangle_3>                      Triangle_list;
    typedef std::vector<Patch_id>                        Patch_id_list;
    typedef std::map<Patch_id,std::size_t>               Patch_id_to_index_map;

    typedef CGAL::AABB_triangle_primitive_3<GeomTraits,
                     typename Triangle_list::iterator>    AABB_primitive;
    typedef CGAL::AABB_traits_3<GeomTraits, AABB_primitive> AABB_traits;
    typedef CGAL::AABB_tree<AABB_traits>                  AABB_tree;

    typedef typename boost::property_map<
      PM, CGAL::dynamic_halfedge_property_t<Halfedge_status> >::type Halfedge_status_pmap;

  public:
    Edge_range_collapser(PolygonMesh& pmesh
                       , VertexPointMap& vpmap
                       , const GeomTraits& gt
                       , const bool protect_constraints
                       , EdgeIsConstrainedMap ecmap
                       , VertexIsConstrainedMap vcmap
                       , FacePatchMap fpmap
                       , FaceIndexMap fimap
                       , const bool build_tree = true)//built by the remesher
      : mesh_(pmesh)
      , vpmap_(vpmap)
      , gt_(gt)
      , build_tree_(build_tree)
      , has_border_(false)
      , input_triangles_()
      , input_patch_ids_()
      , protect_constraints_(protect_constraints)
      , patch_ids_map_(fpmap)
      , ecmap_(ecmap)
      , vcmap_(vcmap)
      , fimap_(fimap)
    {
      halfedge_status_pmap_ = get(CGAL::dynamic_halfedge_property_t<Halfedge_status>(),
                                  pmesh);
      CGAL_warning_code(input_mesh_is_valid_ = CGAL::is_valid_polygon_mesh(pmesh));
      CGAL_warning_msg(input_mesh_is_valid_,
        "The input mesh is not a valid polygon mesh. "
        "It could lead PMP::isotropic_remeshing() to fail.");
    }

    ~Edge_range_collapser()
    {
      if (build_tree_){
        for(std::size_t i=0; i < trees.size();++i){
          delete trees[i];
        }
      }
    }

    template<typename FaceRange>
    void init_remeshing(const FaceRange& face_range)
    {
      tag_halfedges_status(face_range); //called first

      for(face_descriptor f : face_range)
      {
        CGAL_assertion(is_triangle(halfedge(f, mesh_), mesh_));
        if(is_degenerate_triangle_face(f, mesh_, parameters::vertex_point_map(vpmap_)
                                                            .geom_traits(gt_)))
          continue;

        Patch_id pid = get_patch_id(f);
        input_triangles_.push_back(triangle(f));
        input_patch_ids_.push_back(pid);
        std::pair<typename Patch_id_to_index_map::iterator, bool>
          res = patch_id_to_index_map.insert(std::make_pair(pid,0));
        if(res.second){
          res.first->second =  patch_id_to_index_map.size()-1;
        }
      }
      CGAL_assertion(input_triangles_.size() == input_patch_ids_.size());

      if (!build_tree_)
        return;
      trees.resize(patch_id_to_index_map.size());
      for(std::size_t i=0; i < trees.size(); ++i){
        trees[i] = new AABB_tree();
      }
      typename Triangle_list::iterator it;
      typename Patch_id_list::iterator pit;
      for(it = input_triangles_.begin(), pit = input_patch_ids_.begin();
          it != input_triangles_.end();
          ++it, ++pit){
        trees[patch_id_to_index_map[*pit]]->insert(it);
      }
      for(std::size_t i=0; i < trees.size(); ++i){
        trees[i]->build();
      }
    }

    template<typename EdgeRange>
//     template<typename EdgeRange, typename SizingFunction>
    void collapse_edges(
       const EdgeRange& edge_range,
       const bool collapse_constraints)
    {
      typedef boost::bimap<
        boost::bimaps::set_of<halfedge_descriptor>,
        boost::bimaps::multiset_of<FT, std::less<FT> > >          Boost_bimap;
      typedef typename Boost_bimap::value_type                    short_edge;

//       Boost_bimap short_edges;
//       for(edge_descriptor e : edges(mesh_))
//       {
//         std::optional<FT> sqlen = sizing.is_too_short(halfedge(e, mesh_), mesh_);
//         if(sqlen != std::nullopt
//           && is_collapse_allowed(e, collapse_constraints))
//           short_edges.insert(short_edge(halfedge(e, mesh_), sqlen.value()));
//       }
      Boost_bimap short_edges;
      for(edge_descriptor e : edge_range)
      {
       if(is_collapse_allowed(e, collapse_constraints))
          short_edges.insert(short_edge(halfedge(e, mesh_), 0.0));
      }

      while (!short_edges.empty())
//       while (!short_edges.empty())
      {
        //the edge with shortest length
        typename Boost_bimap::right_map::iterator eit = short_edges.right.begin();
        halfedge_descriptor he = eit->second;
        short_edges.right.erase(eit);

        edge_descriptor e = edge(he, mesh_);
        if (!is_collapse_allowed(e, collapse_constraints))
          continue; //situation could have changed since it was added to the bimap

        //handle the boundary case :
        //a PATCH_BORDER edge can be collapsed,
        //and an edge incident to PATCH_BORDER can be collapsed,
        //but only if the boundary vertex is kept,
        //so re-insert opposite(he) to collapse it
        if (!is_on_patch(he))
        {
          CGAL_assertion(!protect_constraints_);//is_collapse_allowed returned false
          if (is_on_border(he) || is_on_mesh(he))
          {
            he = opposite(he, mesh_); //he now is PATCH_BORDER
            e = edge(he, mesh_);
            CGAL_assertion(is_on_patch_border(he));
          }
        }//end if(not on PATCH)

        //let's try to collapse he into vb
        vertex_descriptor va = source(he, mesh_);
        vertex_descriptor vb = target(he, mesh_);

        bool is_va_constrained = is_constrained(va) || is_corner(va);
        bool is_vb_constrained = is_constrained(vb) || is_corner(vb);

        // do not collapse edge with two constrained vertices
        if (is_va_constrained && is_vb_constrained) continue;

        bool can_swap = !is_vb_constrained;

        //do not collapse an edge connecting two different constrained polylines
        if (is_va_constrained && is_vb_constrained && !is_on_patch_border(he))
          continue;

        bool is_va_on_constrained_polyline = is_on_patch_border(va);
        bool is_vb_on_constrained_polyline = is_on_patch_border(vb);

        // swap if vb is not constrained and va is constrained or the only vertex on a constrained polyline
        if (can_swap && (is_va_constrained || (is_va_on_constrained_polyline && !is_vb_on_constrained_polyline)))
        {
          he = opposite(he, mesh_);
          e=edge(he, mesh_);
          std::swap(va, vb);
          // no need to swap is_vX_on_constrained_polyline and is_vX_constrained
          can_swap=false;
        }

        if(collapse_would_invert_face(he))
        {
          if (can_swap//if swap allowed (no constrained vertices)
              && (!is_vb_on_constrained_polyline || is_va_on_constrained_polyline)
              && !collapse_would_invert_face(opposite(he, mesh_)))
          {
            he = opposite(he, mesh_);
            e=edge(he, mesh_);
            std::swap(va, vb);
          }
          else
            continue;//both directions invert a face
        }
        CGAL_assertion(!collapse_would_invert_face(he));
        CGAL_assertion(is_collapse_allowed(e, collapse_constraints));

        if (!CGAL::Euler::does_satisfy_link_condition(e, mesh_))//necessary to collapse
          continue;

        // before collapsing va into vb, check that it does not break a corner
        // or a constrained vertex
        if (collapse_ok)
        {
          //"collapse va into vb along e"
          // remove edges incident to va and vb, because their lengths will change
          for(halfedge_descriptor ha : halfedges_around_target(va, mesh_))
          {
            short_edges.left.erase(ha);
            short_edges.left.erase(opposite(ha, mesh_));
          }
          for(halfedge_descriptor hb : halfedges_around_target(vb, mesh_))
          {
            short_edges.left.erase(hb);
            short_edges.left.erase(opposite(hb, mesh_));
          }

          //before collapse
          halfedge_descriptor he_opp= opposite(he, mesh_);
          bool mesh_border_case     = is_on_border(he);
          bool mesh_border_case_opp = is_on_border(he_opp);
          halfedge_descriptor ep_p  = prev(he_opp, mesh_);
          halfedge_descriptor en    = next(he, mesh_);
          halfedge_descriptor ep    = prev(he, mesh_);
          halfedge_descriptor en_p  = next(he_opp, mesh_);

          // merge halfedge_status to keep the more important on both sides
          //do it before collapse is performed to be sure everything is valid
          if (!mesh_border_case)
            merge_and_update_status(en, ep);
          if (!mesh_border_case_opp)
            merge_and_update_status(en_p, ep_p);

          if (!protect_constraints_)
            put(ecmap_, e, false);
          else
            CGAL_assertion( !get(ecmap_, e) );

          //perform collapse
          CGAL_assertion(target(halfedge(e, mesh_), mesh_) == vb);
          vertex_descriptor vkept = CGAL::Euler::collapse_edge(e, mesh_, ecmap_);
          CGAL_assertion(is_valid(mesh_));
          CGAL_assertion(vkept == vb);//is the constrained point still here
          //fix constrained case
          CGAL_assertion((is_constrained(vkept) || is_corner(vkept) || is_on_patch_border(vkept)) ==
                         (is_va_constrained || is_vb_constrained || is_va_on_constrained_polyline || is_vb_on_constrained_polyline));
          if (fix_degenerate_faces(vkept, short_edges, sizing, collapse_constraints))
          {
            //insert new/remaining short edges
            for (halfedge_descriptor ht : halfedges_around_target(vkept, mesh_))
            {
              std::optional<FT> sqlen = sizing.is_too_short(ht, mesh_);
              if (sqlen != std::nullopt
                && is_collapse_allowed(edge(ht, mesh_), collapse_constraints))
                short_edges.insert(short_edge(ht, sqlen.value()));
            }
          }
        }//end if(collapse_ok)
      }

private:
  Patch_id get_patch_id(const face_descriptor& f) const
  {
    if (f == boost::graph_traits<PM>::null_face())
      return Patch_id(-1);
    return get(patch_ids_map_, f);
  }

  void set_patch_id(const face_descriptor& f, const Patch_id& i)
  {
    put(patch_ids_map_, f, i);
  }

  struct Patch_id_property_map
  {
    typedef boost::readable_property_map_tag       category;
    typedef Patch_id                               value_type;
    typedef Patch_id                               reference;
    typedef typename Triangle_list::const_iterator key_type;

    const Self* remesher_ptr_;

    Patch_id_property_map()
      : remesher_ptr_(nullptr) {}
    Patch_id_property_map(const Self& remesher)
      : remesher_ptr_(&remesher) {}

    friend value_type get(const Patch_id_property_map& m, key_type tr_it)
    {
      //tr_it is an iterator from triangles_
      std::size_t id_in_vec = std::distance(
        m.remesher_ptr_->input_triangles().begin(), tr_it);

      CGAL_assertion(id_in_vec < m.remesher_ptr_->input_patch_ids().size());
      CGAL_assertion(*tr_it == m.remesher_ptr_->input_triangles()[id_in_vec]);

      return m.remesher_ptr_->input_patch_ids()[id_in_vec];
    }
  };

  private:
    Triangle_3 triangle(face_descriptor f) const
    {
      halfedge_descriptor h = halfedge(f, mesh_);
      vertex_descriptor v1  = target(h, mesh_);
      vertex_descriptor v2  = target(next(h, mesh_), mesh_);
      vertex_descriptor v3  = target(next(next(h, mesh_), mesh_), mesh_);
      return Triangle_3(get(vpmap_, v1), get(vpmap_, v2), get(vpmap_, v3));
    }

    bool is_constrained(const edge_descriptor& e) const
    {
      return is_on_border(e) || is_on_patch_border(e);
    }

    bool is_collapse_allowed(const edge_descriptor& e
                           , const bool collapse_constraints) const
    {
      halfedge_descriptor he = halfedge(e, mesh_);
      halfedge_descriptor hopp = opposite(he, mesh_);

      if (is_on_mesh(he) && is_on_mesh(hopp))
        return false;

      if (is_an_isolated_constraint(he) || is_an_isolated_constraint(hopp))
        return false;

      if ( (protect_constraints_ || !collapse_constraints) && is_constrained(e))
        return false;
      if (is_on_patch(he)) //hopp is also on patch
      {
        CGAL_assertion(is_on_patch(hopp));
        if (is_on_patch_border(target(he, mesh_)) && is_on_patch_border(source(he, mesh_)))
          return false;//collapse would induce pinching the selection
        else
          return (is_collapse_allowed_on_patch(he)
               && is_collapse_allowed_on_patch(hopp));
      }
      else if (is_on_patch_border(he))
        return is_collapse_allowed_on_patch_border(he);
      else if (is_on_patch_border(hopp))
        return is_collapse_allowed_on_patch_border(hopp);
      return false;
    }

    bool is_collapse_allowed_on_patch(const halfedge_descriptor& he) const
    {
      halfedge_descriptor hopp = opposite(he, mesh_);

      if (is_on_patch_border(next(he, mesh_)) && is_on_patch_border(prev(he, mesh_)))
        return false;//too many cases to be handled
      if (is_on_patch_border(next(hopp, mesh_)) && is_on_patch_border(prev(hopp, mesh_)))
        return false;//too many cases to be handled
      if (is_on_patch_border(next(he, mesh_)))
      {
        //avoid generation of degenerate faces, and self-intersections
        if (source(he, mesh_) ==
          target(next(next_on_patch_border(next(he, mesh_)), mesh_), mesh_))
          return false;
      }
      if (is_on_patch_border(prev(hopp, mesh_)))
      {
        //avoid generation of degenerate faces, and self-intersections
        if (target(hopp, mesh_) ==
          source(prev(prev_on_patch_border(prev(hopp, mesh_)), mesh_), mesh_))
          return false;
      }
      return true;
    }

    bool is_collapse_allowed_on_patch_border(const halfedge_descriptor& h) const
    {
      CGAL_precondition(is_on_patch_border(h));
      halfedge_descriptor hopp = opposite(h, mesh_);

      if (is_on_patch_border(next(h, mesh_)) && is_on_patch_border(prev(h, mesh_)))
        return false;

      if (is_on_patch_border(hopp))
      {
        if (is_on_patch_border(next(hopp, mesh_)) && is_on_patch_border(prev(hopp, mesh_)))
          return false;
        else if (next_on_patch_border(h) == hopp && prev_on_patch_border(h) == hopp)
          return false; //isolated patch border
        else
          return true;
      }
      CGAL_assertion(is_on_mesh(hopp) || is_on_border(hopp));
      return true;//we already checked we're not pinching a hole in the patch
    }

    bool is_flip_topologically_allowed(const edge_descriptor& e) const
    {
      halfedge_descriptor h=halfedge(e, mesh_);
      return !halfedge(target(next(h, mesh_), mesh_),
               target(next(opposite(h, mesh_), mesh_), mesh_),
               mesh_).second;
    }


    halfedge_descriptor next_on_patch_border(const halfedge_descriptor& h) const
    {
      CGAL_precondition(is_on_patch_border(h));
      CGAL_assertion_code(const Patch_id& pid = get_patch_id(face(h, mesh_)));

      halfedge_descriptor end = opposite(h, mesh_);
      halfedge_descriptor nxt = next(h, mesh_);
      do
      {
        if (is_on_patch_border(nxt))
        {
          CGAL_assertion(get_patch_id(face(nxt, mesh_)) == pid);
          return nxt;
        }
        nxt = next(opposite(nxt, mesh_), mesh_);
      }
      while (end != nxt);

      CGAL_assertion(get_patch_id(face(nxt, mesh_)) == pid);
      CGAL_assertion(is_on_patch_border(end));
      return end;
    }

    halfedge_descriptor prev_on_patch_border(const halfedge_descriptor& h) const
    {
      CGAL_precondition(is_on_patch_border(h));
      CGAL_assertion_code(const Patch_id& pid = get_patch_id(face(h, mesh_)));

      halfedge_descriptor end = opposite(h, mesh_);
      halfedge_descriptor prv = prev(h, mesh_);
      do
      {
        if (is_on_patch_border(prv))
        {
          CGAL_assertion(get_patch_id(face(prv, mesh_)) == pid);
          return prv;
        }
        prv = prev(opposite(prv, mesh_), mesh_);
      }
      while (end != prv);

      CGAL_assertion(is_on_patch_border(end));
      CGAL_assertion(get_patch_id(face(prv, mesh_)) == pid);
      return end;
    }

    bool collapse_would_invert_face(const halfedge_descriptor& h) const
    {
      vertex_descriptor tv = target(h, mesh_);
      typename boost::property_traits<VertexPointMap>::reference
        s = get(vpmap_, source(h, mesh_)); //s for source
      typename boost::property_traits<VertexPointMap>::reference
        t = get(vpmap_, target(h, mesh_)); //t for target

      //check if collapsing the edge [src; tgt] towards tgt
      //would inverse the normal to the considered face
      //src and tgt are the endpoints of the edge to be collapsed
      //p and q are the vertices that form the face to be tested
      //along with src before collapse, and with tgt after collapse
      for(halfedge_descriptor hd :
          halfedges_around_target(opposite(h, mesh_), mesh_))
      {
        if (face(hd, mesh_) == boost::graph_traits<PM>::null_face())
          continue;

        vertex_descriptor tnhd = target(next(hd, mesh_), mesh_);
        vertex_descriptor tnnhd = target(next(next(hd, mesh_), mesh_), mesh_);
        typename boost::property_traits<VertexPointMap>::reference
          p = get(vpmap_, tnhd);
        typename boost::property_traits<VertexPointMap>::reference
          q = get(vpmap_, tnnhd);

        if((tv == tnnhd) || (tv == tnhd))
          continue;

        if ( GeomTraits().collinear_3_object()(s, p, q)
          || GeomTraits().collinear_3_object()(t, p, q))
          continue;

        typename GeomTraits::Construct_cross_product_vector_3 cross_product
          = GeomTraits().construct_cross_product_vector_3_object();

        if(CGAL::sign(cross_product(Vector_3(s, p), Vector_3(s, q))
                    * cross_product(Vector_3(t, p), Vector_3(t, q)))
          != CGAL::POSITIVE)
          return true;
      }
      return false;
    }

    bool is_constrained(const vertex_descriptor& v) const
    {
      return get(vcmap_, v);
    }

    bool is_corner(const vertex_descriptor& v) const
    {
      if(! has_border_){
        return false;
      }
      unsigned int nb_incident_features = 0;
      for(halfedge_descriptor h : halfedges_around_target(v, mesh_))
      {
        halfedge_descriptor hopp = opposite(h, mesh_);
        if ( is_on_border(h) || is_on_patch_border(h)
          || is_on_border(hopp) || is_on_patch_border(hopp)
          || is_an_isolated_constraint(h))
          ++nb_incident_features;
        if (nb_incident_features > 2)
          return true;
      }
      return (nb_incident_features == 1);
    }

    template<typename FaceRange>
    void tag_halfedges_status(const FaceRange& face_range)
    {
      //init halfedges as:
      //  - MESH,        //h and hopp belong to the mesh, not the patch
      //  - MESH_BORDER  //h belongs to the mesh, face(hopp, pmesh) == null_face()
      for(halfedge_descriptor h : halfedges(mesh_))
      {
        //being part of the border of the mesh is predominant
        if (is_border(h, mesh_)){
          set_status(h, MESH_BORDER); //erase previous value if exists
          has_border_ = true;
        } else {
          set_status(h, MESH);
        }
      }

      //tag PATCH,       //h and hopp belong to the patch to be remeshed
      std::vector<halfedge_descriptor> patch_halfedges;
      for(face_descriptor f : face_range)
      {
        for(halfedge_descriptor h :
            halfedges_around_face(halfedge(f, mesh_), mesh_))
        {
          set_status(h, PATCH);
          patch_halfedges.push_back(h);
        }
      }

      // tag patch border halfedges
      for(halfedge_descriptor h : patch_halfedges)
      {
        CGAL_assertion(status(h) == PATCH);
        if( status(opposite(h, mesh_)) != PATCH
         || get_patch_id(face(h, mesh_)) != get_patch_id(face(opposite(h, mesh_), mesh_)))
        {
          set_status(h, PATCH_BORDER);
          has_border_ = true;
        }
      }

      // update status using constrained edge map
      if (!std::is_same<EdgeIsConstrainedMap,
                          Static_boolean_property_map<edge_descriptor, false> >::value)
      {
        for(edge_descriptor e : edges(mesh_))
        {
          if (get(ecmap_, e))
          {
            //deal with h and hopp for borders that are sharp edges to be preserved
            halfedge_descriptor h = halfedge(e, mesh_);
            Halfedge_status hs = status(h);
            if (hs == PATCH) {
              set_status(h, PATCH_BORDER);
              hs = PATCH_BORDER;
              has_border_ = true;
            }

            halfedge_descriptor hopp = opposite(h, mesh_);
            Halfedge_status hsopp = status(hopp);
            if (hsopp == PATCH) {
              set_status(hopp, PATCH_BORDER);
              hsopp = PATCH_BORDER;
              has_border_ = true;
            }

            if (hs != PATCH_BORDER && hsopp != PATCH_BORDER)
            {
              if(hs != MESH_BORDER)
                set_status(h, ISOLATED_CONSTRAINT);
              if(hsopp != MESH_BORDER)
                set_status(hopp, ISOLATED_CONSTRAINT);
            }
          }
        }
      }
    }

    Halfedge_status status(const halfedge_descriptor& h) const
    {
      return get(halfedge_status_pmap_,h);
    }

    void set_status(const halfedge_descriptor& h,
                    const Halfedge_status& s)
    {
      put(halfedge_status_pmap_,h,s);
    }

    void merge_and_update_status(halfedge_descriptor en,
                                 halfedge_descriptor ep)
    {

      halfedge_descriptor eno = opposite(en, mesh_);
      halfedge_descriptor epo = opposite(ep, mesh_);
      Halfedge_status s_eno = status(eno);
      Halfedge_status s_epo = status(epo);

      Halfedge_status s_ep = status(ep);
      if(s_epo == MESH_BORDER
        || s_ep == MESH_BORDER
        || s_epo == PATCH_BORDER
        || s_ep == PATCH_BORDER)
      {
        set_status(en, s_epo);
        set_status(eno, s_ep);
      }
      else
      {
        Halfedge_status s_en = status(en);
        if(s_eno == MESH_BORDER
          || s_en == MESH_BORDER
          || s_eno == PATCH_BORDER
          || s_en == PATCH_BORDER)
        {
          set_status(ep, s_epo);
          set_status(epo, s_ep);
        }
      }
      // else keep current status for en and eno
    }

    void remove_border_face(const halfedge_descriptor h)
    {
      CGAL_assertion(is_border(opposite(h, mesh_), mesh_));
      for (halfedge_descriptor hf : halfedges_around_face(h, mesh_))
      {
        set_status(hf, MESH_BORDER); //only 1 or 2 of the listed halfedges
                                     //will survive face removal, but status will be correct
        set_status(opposite(hf, mesh_), PATCH_BORDER); //idem
                                     //some of them will not survive but setting status
                                     //is cheaper then checking which should be set
      }
      CGAL::Euler::remove_face(h, mesh_);
    }

    template<typename Bimap, typename SizingFunction>
    bool fix_degenerate_faces(const vertex_descriptor& v,
                              Bimap& short_edges,
                              const SizingFunction& sizing,
                              const bool collapse_constraints)
    {
      std::unordered_set<halfedge_descriptor> degenerate_faces;
      for(halfedge_descriptor h :
          halfedges_around_target(halfedge(v, mesh_), mesh_))
      {
        if(!is_border(h, mesh_) &&
           is_triangle(h, mesh_) &&
           is_degenerate_triangle_face(face(h, mesh_), mesh_,
                                       parameters::vertex_point_map(vpmap_)
                                                   .geom_traits(gt_)))
          degenerate_faces.insert(h);
      }

      if(degenerate_faces.empty())
        return true;

      bool done = false;

      while(!degenerate_faces.empty())
      {
        halfedge_descriptor h = *(degenerate_faces.begin());
        degenerate_faces.erase(degenerate_faces.begin());

        if(is_border(opposite(h, mesh_), mesh_))
        {
          remove_border_face(h);
          continue;
        }

        for(halfedge_descriptor hf :
            halfedges_around_face(h, mesh_))
        {
          halfedge_descriptor hfo = opposite(hf, mesh_);

          if(is_border(hfo, mesh_))
          {
            remove_border_face(h);
            break;
          }
          vertex_descriptor vc = target(hf, mesh_);
          vertex_descriptor va = target(next(hf, mesh_), mesh_);
          vertex_descriptor vb = target(next(next(hf, mesh_), mesh_), mesh_);
          Vector_3 ab(get(vpmap_,va), get(vpmap_,vb));
          Vector_3 ac(get(vpmap_,va), get(vpmap_,vc));
          if (ab * ac < 0)
          {
            halfedge_descriptor h_ab = prev(hf, mesh_);
            halfedge_descriptor h_ca = next(hf, mesh_);

            short_edges.left.erase(hf);
            short_edges.left.erase(hfo);

            CGAL_assertion( !get(ecmap_, edge(hf, mesh_)) );

            if (!is_flip_topologically_allowed(edge(hf, mesh_)))
              continue;

            // geometric condition for flip --> do not create new degenerate face
            vertex_descriptor vd = target(next(hfo, mesh_), mesh_);
            if ( collinear( get(vpmap_, va), get(vpmap_, vb), get(vpmap_, vd) ) ||
                 collinear( get(vpmap_, va), get(vpmap_, vc), get(vpmap_, vd) ) )  continue;

            // remove opposite face from the queue (if degenerate)
            degenerate_faces.erase(hfo);
            degenerate_faces.erase(next(hfo, mesh_));
            degenerate_faces.erase(prev(hfo, mesh_));

            CGAL::Euler::flip_edge(hf, mesh_);
            done = true;

            //update status
            set_status(h_ab, merge_status(h_ab, hf, hfo));
            set_status(h_ca, merge_status(h_ca, hf, hfo));
            if (is_on_patch(h_ca) || is_on_patch_border(h_ca))
            {
              set_status(hf, PATCH);
              set_status(hfo, PATCH);
            }

            //insert new edges in 'short_edges'
            if (is_collapse_allowed(edge(hf, mesh_), collapse_constraints))
            {
              std::optional<FT> sqlen = sizing.is_too_short(hf, mesh_);
              if (sqlen != std::nullopt)
                short_edges.insert(typename Bimap::value_type(hf, sqlen.value()));
            }

            break;
          }
        }
      }
      return done;
    }

    bool is_on_patch(const halfedge_descriptor& h) const
    {
      bool res =(status(h) == PATCH);
      CGAL_assertion(res == (status(opposite(h, mesh_)) == PATCH));
      return res;
    }

    bool is_on_patch(const face_descriptor& f) const
    {
      for(halfedge_descriptor h :
          halfedges_around_face(halfedge(f, mesh_), mesh_))
      {
        if (is_on_patch(h) || is_on_patch_border(h))
          return true;
      }
      return false;
    }

    bool is_on_patch(const vertex_descriptor& v) const
    {
      if(! has_border_){
        return true;
      }
      for(halfedge_descriptor h :
          halfedges_around_target(v, mesh_))
      {
        if (!is_on_patch(h))
          return false;
      }
      return true;
    }

public:
    bool is_on_patch_border(const halfedge_descriptor& h) const
    {
      bool res = (status(h) == PATCH_BORDER);
      if (res)
      {
        CGAL_assertion_code(Halfedge_status hs = status(opposite(h, mesh_)));
        CGAL_assertion(hs == MESH_BORDER
                    || hs == MESH
                    || hs == PATCH_BORDER);//when 2 incident patches are remeshed
      }
      return res;
    }
    bool is_on_patch_border(const edge_descriptor& e) const
    {
      return is_on_patch_border(halfedge(e,mesh_))
          || is_on_patch_border(opposite(halfedge(e, mesh_), mesh_));
    }
    bool is_on_patch_border(const vertex_descriptor& v) const
    {
      if(! has_border_){
        return false;
      }
      for(halfedge_descriptor h : halfedges_around_target(v, mesh_))
      {
        if (is_on_patch_border(h) || is_on_patch_border(opposite(h, mesh_)))
          return true;
      }
      return false;
    }

    bool is_on_border(const halfedge_descriptor& h) const
    {
      bool res = (status(h) == MESH_BORDER);
      CGAL_assertion(res == is_border(h, mesh_));
      CGAL_assertion(res == is_border(next(h, mesh_), mesh_));
      return res;
    }

    bool is_on_border(const edge_descriptor& e) const
    {
      return is_on_border(halfedge(e, mesh_))
          || is_on_border(opposite(halfedge(e, mesh_), mesh_));
    }

    bool is_on_mesh(const halfedge_descriptor& h) const
    {
      return status(h) == MESH;
    }

    bool is_an_isolated_constraint(const halfedge_descriptor& h) const
    {
      bool res = (status(h) == ISOLATED_CONSTRAINT);
      CGAL_assertion_code(Halfedge_status so = status(opposite(h, mesh_)));
      CGAL_assertion(!res || so == ISOLATED_CONSTRAINT || so == MESH_BORDER);
      return res;
    }

private:
  public:
    const Triangle_list& input_triangles() const {
      return input_triangles_;
    }

    const Patch_id_list& input_patch_ids() const {
      return input_patch_ids_;
    }

  private:
    PolygonMesh& mesh_;
    VertexPointMap& vpmap_;
    const GeomTraits& gt_;
    bool build_tree_;
    bool has_border_;
    std::vector<AABB_tree*> trees;
    Patch_id_to_index_map patch_id_to_index_map;
    Triangle_list input_triangles_;
    Patch_id_list input_patch_ids_;
    Halfedge_status_pmap halfedge_status_pmap_;
    bool protect_constraints_;
    FacePatchMap patch_ids_map_;
    EdgeIsConstrainedMap ecmap_;
    VertexIsConstrainedMap vcmap_;
    FaceIndexMap fimap_;
    CGAL_assertion_code(bool input_mesh_is_valid_;)

  };//end class Edge_range_collapser
}//end namespace internal

template<typename PolygonMesh
       , typename EdgeRange
       , typename NamedParameters = parameters::Default_named_parameters
       >
void collapse_edges(const EdgeRange& edges
                    , PolygonMesh& pmesh
                    , const NamedParameters& np = parameters::default_values())
{
  typedef PolygonMesh PM;
  typedef typename boost::graph_traits<PM>::edge_descriptor edge_descriptor;
  typedef typename boost::graph_traits<PM>::vertex_descriptor vertex_descriptor;
  using parameters::choose_parameter;
  using parameters::get_parameter;

  typedef typename GetGeomTraits<PM, NamedParameters>::type GT;
  GT gt = choose_parameter<GT>(get_parameter(np, internal_np::geom_traits));

  typedef typename GetVertexPointMap<PM, NamedParameters>::type VPMap;
  VPMap vpmap = choose_parameter(get_parameter(np, internal_np::vertex_point),
                                 get_property_map(vertex_point, pmesh));

  typedef typename GetInitializedFaceIndexMap<PolygonMesh, NamedParameters>::type FIMap;
  FIMap fimap = CGAL::get_initialized_face_index_map(pmesh, np);

  typedef typename internal_np::Lookup_named_param_def <
        internal_np::edge_is_constrained_t,
        NamedParameters,
        Static_boolean_property_map<edge_descriptor, false> // default (no constraint pmap)
      > ::type ECMap;
  ECMap ecmap = choose_parameter<Static_boolean_property_map<edge_descriptor, false>>(get_parameter(np, internal_np::edge_is_constrained));

  typedef typename internal_np::Lookup_named_param_def <
      internal_np::face_patch_t,
      NamedParameters,
      internal::Connected_components_pmap<PM, FIMap>//default
    > ::type FPMap;
  FPMap fpmap = choose_parameter(
    get_parameter(np, internal_np::face_patch),
    internal::Connected_components_pmap<PM, FIMap>(faces(pmesh), pmesh, ecmap, fimap, false));

  typename internal::Incremental_remesher<PM, VPMap, GT, ECMap,
    Static_boolean_property_map<vertex_descriptor, false>, // no constraint pmap
    FPMap,FIMap
  >
    remesher(pmesh, vpmap, gt, false/*protect constraints*/, ecmap,
             Static_boolean_property_map<vertex_descriptor, false>(),
             fpmap,
             fimap,
             false/*need aabb_tree*/);
     remesher.collapse_edges(edges, sizing);
}
}
}//end namespace Polygon_mesh_processing
}//end namespace CGAL
