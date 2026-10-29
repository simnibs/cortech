// These are adaptive versions of CGAL's original implementations (Uniform and Adaptive)

#include <cmath>
#include <optional>

#include <CGAL/license/Polygon_mesh_processing/meshing_hole_filling.h>
#include <CGAL/Polygon_mesh_processing/internal/Sizing_field_base.h>
#include <CGAL/number_utils.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/locate.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>


namespace CGAL
{
namespace Polygon_mesh_processing
{

/*
A sizing field that samples pre-computed values from a supplied property map.
*/

template <class PolygonMesh,
          class VPMap =  typename boost::property_map<PolygonMesh, CGAL::vertex_point_t>::const_type>
class Precomputed_sizing_field
#ifndef DOXYGEN_RUNNING
: public internal::Sizing_field_base<PolygonMesh, VPMap>
#endif
{
private:
  typedef internal::Sizing_field_base<PolygonMesh, VPMap> Base;
  typedef typename CGAL::dynamic_vertex_property_t<typename Base::FT> Vertex_property_tag;
  typedef typename boost::property_map<PolygonMesh, Vertex_property_tag>::type VertexSizingMap;
public:
  typedef typename Base::K          K;
  typedef typename Base::FT         FT;
  typedef typename Base::Point_3    Point_3;
  typedef typename Base::face_descriptor     Face_index;
  typedef typename Base::halfedge_descriptor Halfedge_index;
  typedef typename Base::vertex_descriptor   Vertex_index;

  // Constructor

  Precomputed_sizing_field(const std::vector<float> sizing, PolygonMesh& pmesh)
  : m_pmesh(pmesh)
  , m_vertex_sizing_map(get(Vertex_property_tag(), pmesh))
  , m_vpmap(get(CGAL::vertex_point, pmesh))
  {
    for (Vertex_index v : vertices(pmesh))
      put(m_vertex_sizing_map, v, sizing[v.idx()]);
  }

private:
  FT sqlength(const Vertex_index va, const Vertex_index vb) const
  {
    return FT(squared_distance(get(m_vpmap, va), get(m_vpmap, vb)));
  }

  FT sqlength(const Halfedge_index& h, const PolygonMesh& pmesh) const
  {
    return sqlength(target(h, pmesh), source(h, pmesh));
  }

  // FT sizing_at_vertex(const Vertex_index v) const
  // {
  //   auto p = get(m_vpmap, v);
  //   Face_location<Base, FT> loc = locate_with_AABB_tree(p, m_tree, m_pmesh);
  //   Face_index f = loc.first;
  //   const auto& bc = loc.second;              // barycentric coords (w0,w1,w2)

  //   Halfedge_index h = halfedge(f, m_pmesh);
  //   Vertex_index v0 = target(h, m_pmesh);
  //   Vertex_index v1 = target(next(h, m_pmesh), m_pmesh);
  //   Vertex_index v2 = target(next(next(h, m_pmesh), m_pmesh), m_pmesh);

  //   return bc[0]*m_size_map[v0] + bc[1]*m_size_map[v1] + bc[2]*m_size_map[v2];
  // }

public:
  FT at(const Vertex_index v, const PolygonMesh& /* pmesh */) const
  {
    CGAL_assertion(get(m_vertex_sizing_map, v) > 0);
    return get(m_vertex_sizing_map, v);
  }

  std::optional<FT> is_too_long(
    const Vertex_index va,
    const Vertex_index vb,
    const PolygonMesh& pmesh) const
  {
    const FT sqlen = sqlength(va, vb);
    FT sqtarg_len = CGAL::square(4./3. * (CGAL::min)(get(m_vertex_sizing_map, va),
                                                     get(m_vertex_sizing_map, vb)));
    CGAL_assertion(get(m_vertex_sizing_map, va) > 0);
    CGAL_assertion(get(m_vertex_sizing_map, vb) > 0);
    if (sqlen > sqtarg_len)
      return sqlen / sqtarg_len;
    else
      return std::nullopt;
  }

  std::optional<FT> is_too_short(const Halfedge_index h, const PolygonMesh& pmesh) const
  {
    const FT sqlen = sqlength(h, pmesh);
    FT sqtarg_len = CGAL::square(4./5. * (CGAL::min)(get(m_vertex_sizing_map, source(h, pmesh)),
                                                     get(m_vertex_sizing_map, target(h, pmesh))));
    CGAL_assertion(get(m_vertex_sizing_map, source(h, pmesh)) > 0);
    CGAL_assertion(get(m_vertex_sizing_map, target(h, pmesh)) > 0);

    if (sqlen < sqtarg_len)
      return sqlen / sqtarg_len;
    else
      return std::nullopt;
  }


  Point_3 split_placement(const Halfedge_index h, const PolygonMesh& pmesh) const
  {
    return midpoint(get(m_vpmap, target(h, pmesh)),
                    get(m_vpmap, source(h, pmesh)));
  }

  // void register_split_vertex(const Vertex_index, const PolygonMesh&) const
  // {
  //   // nothing to do — at()/is_too_long()/is_too_short() re-query the field
  //   // from the current geometric position, so new vertices are handled for free.
  // }

  void register_split_vertex(const Vertex_index v, const PolygonMesh& pmesh)
  {
    // calculating it as the average of two vertices on other ends
    // of halfedges as updating is done during an edge split
    FT vertex_size = 0;
    CGAL_assertion(CGAL::halfedges_around_target(v, pmesh).size() == 2);
    for (Halfedge_index ha: CGAL::halfedges_around_target(v, pmesh))
    {
      vertex_size += get(m_vertex_sizing_map, source(ha, pmesh));
    }
    vertex_size /= FT(CGAL::halfedges_around_target(v, pmesh).size());

    put(m_vertex_sizing_map, v, vertex_size);
  }

private:
  const PolygonMesh m_pmesh;
  const VertexSizingMap m_vertex_sizing_map;
  const VPMap m_vpmap;
};

/*!
* Features:
* - Edges shorter than the target edge length will be collapsed
* - Edges are never split
*
*/
template <class PolygonMesh,
          class VPMap =  typename boost::property_map<PolygonMesh, CGAL::vertex_point_t>::const_type>
class Uniform_sizing_field_strict_short
#ifndef DOXYGEN_RUNNING
: public internal::Sizing_field_base<PolygonMesh, VPMap>
#endif
{
private:
  typedef internal::Sizing_field_base<PolygonMesh, VPMap> Base;

public:
  typedef typename Base::FT         FT;
  typedef typename Base::Point_3    Point_3;
  typedef typename Base::halfedge_descriptor halfedge_descriptor;
  typedef typename Base::vertex_descriptor   vertex_descriptor;

  /// \name Creation
  /// @{

  /*!
  * Constructor.
  * \param size the target edge length for isotropic remeshing. If set to 0,
  *        the criterion for edge length is ignored and edges are neither split nor collapsed.
  * \param vpmap is the input vertex point map that associates points to the vertices of
  *        the input mesh.
  */
  Uniform_sizing_field_strict_short(const FT size, const VPMap& vpmap)
    : m_size(size)
    , m_sq_short(CGAL::square(1.0 * size))
    // , m_sq_long(  CGAL::square(4./3. * size))
    , m_sq_long(CGAL::square(INFINITY))
    , m_vpmap(vpmap)
  {}

  /*!
  * Constructor using internal vertex point map of the input polygon mesh.
  *
  * @param size the target edge length for isotropic remeshing. If set to 0,
  *        the criterion for edge length is ignored and edges are neither split nor collapsed.
  * @param pmesh a polygon mesh with triangulated surface patches to be remeshed. The default
  *        vertex point map of `pmesh` is used to construct the class.
  */
  Uniform_sizing_field_strict_short(const FT size, const PolygonMesh& pmesh)
    : Uniform_sizing_field_strict_short(size, get(CGAL::vertex_point, pmesh))
  {}

  /// @}

private:
  FT sqlength(const vertex_descriptor va,
              const vertex_descriptor vb) const
  {
    return FT(squared_distance(get(m_vpmap, va), get(m_vpmap, vb)));
  }

  FT sqlength(const halfedge_descriptor& h, const PolygonMesh& pmesh) const
  {
    return sqlength(target(h, pmesh), source(h, pmesh));
  }

public:
  FT at(const vertex_descriptor /* v */, const PolygonMesh& /* pmesh */) const
  {
    return m_size;
  }

  std::optional<FT> is_too_long(const vertex_descriptor va, const vertex_descriptor vb, const PolygonMesh& /* pmesh */) const
  {
    return std::nullopt; // never too long
  }

  std::optional<FT> is_too_short(const halfedge_descriptor h, const PolygonMesh& pmesh) const
  {
    const FT sqlen = sqlength(h, pmesh);
    if (sqlen < m_sq_short)
      //no need to return the ratio for the uniform field
      return sqlen;
    else
      return std::nullopt;
  }

  Point_3 split_placement(const halfedge_descriptor h, const PolygonMesh& pmesh) const
  {
    return midpoint(get(m_vpmap, target(h, pmesh)),
                    get(m_vpmap, source(h, pmesh)));
  }

  void register_split_vertex(const vertex_descriptor /* v */, const PolygonMesh& /* pmesh */)
  {}

private:
  const FT m_size;
  const FT m_sq_short;
  const FT m_sq_long;
  const VPMap m_vpmap;
};

}//end namespace Polygon_mesh_processing
}//end namespace CGAL