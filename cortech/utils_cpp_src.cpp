#include <algorithm>
#include <cmath>
#include <queue>
#include <vector>
#include <unordered_map>

/*
Map the values in `values` from `source` to the corresponding values in `target`.
*/
std::vector<int> map_values_from_source_to_target(
    std::vector<int> source_values,
    std::vector<int> target_values,
    std::vector<int> values)
{
    // build map
    std::unordered_map<int, int> map(source_values.size());
    for (std::size_t i = 0; i < source_values.size(); ++i)
        map[source_values[i]] = target_values[i];
    // apply map
    std::vector<int> values_out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i)
        values_out[i] = map.at(values[i]);
    return values_out;
}


// P: vertex positions, T: tets (4 vertex indices), h: sizing values
std::vector<float> sizing_field_gradation(
    const std::vector<std::vector<float>>& points,
    const std::vector<std::vector<unsigned int>>& cells,
    // const std::vector<std::array<float,3>>& points,
    // const std::vector<std::array<int,4>>& cells,
    std::vector<float> h,
    float g)
{
    std::size_t n = points.size();

    // Build a unique edge adjacency list from the tets (6 edges per tet)
    std::vector<std::vector<unsigned int>> adjacency(n);
    static const unsigned int edge_pairs[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
    for (const auto& cell : cells)
    {
        for (auto& e : edge_pairs)
        {
            adjacency[cell[e[0]]].push_back(cell[e[1]]);
            adjacency[cell[e[1]]].push_back(cell[e[0]]);
        }
    }
    // keep unique edges only
    for (auto& a : adjacency)
    {
        std::sort(a.begin(), a.end());
        a.erase(std::unique(a.begin(), a.end()), a.end());
    }

    auto dist = [&](unsigned int a, unsigned int b)
    {
        float dx = points[a][0]-points[b][0];
        float dy = points[a][1]-points[b][1];
        float dz = points[a][2]-points[b][2];
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    };

    using h_vert = std::pair<float, unsigned int>; // (h value, vertex)
    std::priority_queue<h_vert, std::vector<h_vert>, std::greater<h_vert>> pq;
    for (unsigned int v = 0; v < n; ++v)
        pq.push({h[v], v});

    std::vector<char> done(n, 0);
    while (!pq.empty()) {
        auto [hv, v] = pq.top();
        pq.pop();
        if (done[v] || hv > h[v])
            continue; // stale entry (lazy deletion)
        done[v] = 1;
        for (unsigned int u : adjacency[v])
        {
            float cand = h[v] + g * dist(u, v);
            if (cand < h[u])
            {
                h[u] = cand;
                pq.push({cand, u});
            }
        }
    }
    return h;
}