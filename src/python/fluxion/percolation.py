import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

from .particle_system import FluxionParticleSystem, FluxionParticle


@dataclass
class PercolationResult:
    """Result of a thermal percolation analysis."""
    is_percolating: bool
    max_cluster_size: int
    max_span_ratio: float
    hot_gate_count: int
    total_clusters: int
    percolation_risk: float  # 0.0 to 1.0


class ThermalPercolationChecker:
    """
    Thermal Percolation Checker
    
    Evaluates if high-power gates have clustered together to form a
    contiguous 'thermal wall' that could block heat dissipation or
    create dangerous thermal gradients across the chip.
    """
    
    def __init__(self, power_threshold_percentile: float = 75.0, 
                 connection_radius: float = 15.0,
                 critical_span_ratio: float = 0.5):
        """
        Initialize the percolation checker.
        
        Args:
            power_threshold_percentile: Percentile of power above which a gate is considered 'hot'.
            connection_radius: Distance within which hot gates are considered thermally connected.
            critical_span_ratio: Ratio of chip dimension a cluster must span to be considered percolating.
        """
        self.power_threshold_percentile = power_threshold_percentile
        self.connection_radius = connection_radius
        self.critical_span_ratio = critical_span_ratio

    def analyze(self, system: FluxionParticleSystem) -> PercolationResult:
        """
        Analyze the current particle system for thermal percolation.
        
        Args:
            system: The particle system to analyze
            
        Returns:
            PercolationResult containing metrics and risk assessment
        """
        particles = list(system.particles.values())
        if not particles:
            return PercolationResult(False, 0, 0.0, 0, 0, 0.0)

        # 1. Identify "hot" gates
        powers = [p.power_pw for p in particles]
        threshold = np.percentile(powers, self.power_threshold_percentile)
        
        hot_gates = [p for p in particles if p.power_pw >= threshold]
        
        if not hot_gates:
            return PercolationResult(False, 0, 0.0, 0, 0, 0.0)

        # 2. Build adjacency list of thermally connected hot gates
        n_hot = len(hot_gates)
        adj_list = {i: [] for i in range(n_hot)}
        
        # KDTree for O(N log N) distance queries instead of O(N^2) square distance matrix memory explosion
        if n_hot > 0:
            coords = np.array([[p.x, p.y] for p in hot_gates])
            try:
                from scipy.spatial import KDTree
                tree = KDTree(coords)
                pairs = tree.query_pairs(self.connection_radius)
                for i, j in pairs:
                    adj_list[i].append(j)
                    adj_list[j].append(i)
            except ImportError:
                for i in range(n_hot):
                    p1_coord = coords[i]
                    diffs = coords[i+1:] - p1_coord
                    dists_sq = np.sum(diffs**2, axis=1)
                    connected_indices = np.where(dists_sq <= self.connection_radius**2)[0] + i + 1
                    for j in connected_indices:
                        adj_list[i].append(int(j))
                        adj_list[int(j)].append(i)

        # 3. Find connected components (clusters)
        visited = set()
        clusters = []

        for i in range(n_hot):
            if i not in visited:
                cluster = []
                # BFS/DFS
                stack = [i]
                while stack:
                    curr = stack.pop()
                    if curr not in visited:
                        visited.add(curr)
                        cluster.append(curr)
                        stack.extend(adj_list[curr])
                clusters.append(cluster)

        # 4. Evaluate clusters for percolation spans
        max_span_ratio = 0.0
        max_cluster_size = 0
        die_w = max(system.die_width, 1.0)
        die_h = max(system.die_height, 1.0)
        
        for cluster_indices in clusters:
            cluster_size = len(cluster_indices)
            if cluster_size > max_cluster_size:
                max_cluster_size = cluster_size
                
            xs = [hot_gates[i].x for i in cluster_indices]
            ys = [hot_gates[i].y for i in cluster_indices]
            
            span_x = (max(xs) - min(xs)) / die_w
            span_y = (max(ys) - min(ys)) / die_h
            
            cluster_span = max(span_x, span_y)
            if cluster_span > max_span_ratio:
                max_span_ratio = cluster_span

        is_percolating = max_span_ratio >= self.critical_span_ratio
        
        # Risk is a combination of span and size
        risk = min(max_span_ratio / self.critical_span_ratio, 1.0)
        
        return PercolationResult(
            is_percolating=is_percolating,
            max_cluster_size=max_cluster_size,
            max_span_ratio=max_span_ratio,
            hot_gate_count=n_hot,
            total_clusters=len(clusters),
            percolation_risk=risk
        )
