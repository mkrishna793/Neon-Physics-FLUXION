# The HAPR Algorithm

**HAPR (Hierarchical Attention Placement with Routing)** avoids flattening a million gates. Instead, it places clusters using a camera zoom-in approach.

## 7 Steps
1. **PARTITION**: Split circuit into hierarchy tree (1M gates → 1000 clusters → 100 subclusters → gates)
2. **ATTEND**: Each subcluster attends to its neighbors using `attention(A→B) = shared_wires / total_wires`.
3. **COARSE PLACE**: Place top-level clusters using spectral embedding of the attention graph.
4. **FINE PLACE**: Place subclusters inside their parent region.
5. **GATE PLACE**: Force-directed placement of gates within leaf clusters (GPU accelerated).
6. **ROUTE MAP**: Predict routing congestion using bounding-box crossing counts on a grid (GPU accelerated).
7. **REFINE**: Iteratively move gates away from hot cells and greedily swap to optimize HPWL.
