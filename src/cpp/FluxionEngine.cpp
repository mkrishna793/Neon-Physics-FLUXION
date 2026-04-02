// -*- mode: C++; c-file-style: "cc-mode" -*-
//*************************************************************************
// DESCRIPTION: FLUXION C++ Physics Engine
//*************************************************************************

#include "FluxionEngine.h"
#include "BarnesHutTree.h"

#include <cmath>
#include <random>
#include <iostream>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fluxion {

FluxionEngine::FluxionEngine() : m_die_width(1000), m_die_height(1000) {}

FluxionEngine::~FluxionEngine() {}

void FluxionEngine::setDieSize(double width, double height) {
    m_die_width = width;
    m_die_height = height;
}

void FluxionEngine::setForceWeights(double wire, double thermal, double timing, double topo, double density) {
    w_wire = wire;
    w_thermal = thermal;
    w_timing = timing;
    w_topo = topo;
    w_density = density;
}

void FluxionEngine::loadFromGraph(const FluxionGraph& graph) {
    m_particles.clear();
    m_connections.clear();
    
    m_die_width = graph.dieWidth;
    m_die_height = graph.dieHeight;
    
    // Create reverse ID map
    std::map<uint64_t, int> id_to_idx;
    
    m_particles.reserve(graph.nodes.size());
    int idx = 0;
    for (const auto& pair : graph.nodes) {
        const auto& n = pair.second;
        Particle p;
        p.id = n.id;
        p.x = n.x;
        p.y = n.y;
        p.power = n.power;
        p.area = n.area;
        p.delay = n.delay;
        p.logic_level = n.level;
        
        // Estimate width/height from area
        p.width = std::sqrt(n.area) * 2.0;
        p.height = std::sqrt(n.area) * 0.5;
        
        m_particles.push_back(p);
        id_to_idx[n.id] = idx++;
    }
    
    m_connections.reserve(graph.connections.size());
    for (const auto& c : graph.connections) {
        if (id_to_idx.count(c.sourceId) && id_to_idx.count(c.destId)) {
            Connection conn;
            conn.src_idx = id_to_idx[c.sourceId];
            conn.dst_idx = id_to_idx[c.destId];
            conn.is_critical = c.isCriticalPath;
            conn.expected_delay = 100.0; // Place holder
            m_connections.push_back(conn);
        }
    }
    
    // Resize working memory
    fx.resize(m_particles.size(), 0.0);
    fy.resize(m_particles.size(), 0.0);
    
    std::cout << "[Fluxion Engine] Loaded " << m_particles.size() << " particles and " 
              << m_connections.size() << " connections." << std::endl;
}

void FluxionEngine::calculateForces() {
    int n = m_particles.size();
    
    // Zero out forces
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        fx[i] = 0.0;
        fy[i] = 0.0;
    }
    
    if (w_wire > 0) applyWireTension();
    if (w_thermal > 0) applyThermalRepulsion();
    if (w_timing > 0) applyTimingGravity();
    
    // Add density bounds check
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        Particle& p = m_particles[i];
        
        // Boundary repulsion (soft walls)
        double wall_margin = 10.0;
        double wall_k = 100.0;
        
        if (p.x < wall_margin) fx[i] += wall_k * (wall_margin - p.x);
        if (p.x > m_die_width - wall_margin) fx[i] -= wall_k * (p.x - (m_die_width - wall_margin));
        if (p.y < wall_margin) fy[i] += wall_k * (wall_margin - p.y);
        if (p.y > m_die_height - wall_margin) fy[i] -= wall_k * (p.y - (m_die_height - wall_margin));
    }
}

void FluxionEngine::applyWireTension() {
    double spring_k = 1.0 * w_wire;
    
    // Note: To parallelize this safely without data races on fx/fy, we either need
    // atomic updates, or compute per-particle instead of per-connection.
    // For simplicity we use serial or atomic here.
    
    #pragma omp parallel for
    for (size_t i = 0; i < m_connections.size(); ++i) {
        const auto& c = m_connections[i];
        const Particle& p1 = m_particles[c.src_idx];
        const Particle& p2 = m_particles[c.dst_idx];
        
        double dx = p2.x - p1.x;
        double dy = p2.y - p1.y;
        
        double force_x = spring_k * dx;
        double force_y = spring_k * dy;
        
        #pragma omp atomic
        fx[c.src_idx] += force_x;
        #pragma omp atomic
        fy[c.src_idx] += force_y;
        
        #pragma omp atomic
        fx[c.dst_idx] -= force_x;
        #pragma omp atomic
        fy[c.dst_idx] -= force_y;
    }
}

void FluxionEngine::applyThermalRepulsion() {
    int n = m_particles.size();
    if (n < 2) return;
    
    double k = 100.0 * w_thermal;
    
    if (n > 2000) {
        // Use Barnes-Hut for large N
        BarnesHutTree tree(0.5); // theta = 0.5
        
        // Extract plain double arrays for BH tree
        std::vector<double> px(n), py(n), q(n);
        
        #pragma omp parallel for
        for (int i=0; i<n; ++i) {
            px[i] = m_particles[i].x;
            py[i] = m_particles[i].y;
            q[i] = std::sqrt(m_particles[i].power + 1.0);
        }
        
        tree.build(px.data(), py.data(), q.data(), n);
        
        #pragma omp parallel for
        for (int i=0; i<n; ++i) {
            double fnx, fny;
            tree.computeForcesForParticle(i, px[i], py[i], q[i], k, 5.0, fnx, fny);
            fx[i] += fnx;
            fy[i] += fny;
        }
    } else {
        // O(N^2) for small N
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            double local_fx = 0.0;
            double local_fy = 0.0;
            
            const Particle& p1 = m_particles[i];
            double q1 = std::sqrt(p1.power + 1.0);
            
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                
                const Particle& p2 = m_particles[j];
                double q2 = std::sqrt(p2.power + 1.0);
                
                double dx = p1.x - p2.x;
                double dy = p1.y - p2.y;
                double dist_sq = dx*dx + dy*dy;
                double dist = std::sqrt(dist_sq);
                
                if (dist < 5.0) dist = 5.0; // cap minimum distance
                
                double force_mag = (k * q1 * q2) / (dist_sq);
                
                local_fx += force_mag * (dx / dist);
                local_fy += force_mag * (dy / dist);
            }
            
            fx[i] += local_fx;
            fy[i] += local_fy;
        }
    }
}

void FluxionEngine::applyTimingGravity() {
    double g = 0.1 * w_timing;
    
    // Find max level
    int max_level = 1;
    for (const auto& p : m_particles) {
        if (p.logic_level > max_level) max_level = p.logic_level;
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < m_particles.size(); ++i) {
        const Particle& p = m_particles[i];
        
        // Target X based on level: inputs left, outputs right
        double target_x = (static_cast<double>(p.logic_level) / max_level) * m_die_width;
        
        // Pull towards target X
        fx[i] += g * (target_x - p.x);
    }
}

void FluxionEngine::optimize(int steps, double initial_temp) {
    if (m_particles.empty()) return;
    
    std::cout << "[Fluxion Engine] Commencing Physical Synthesis." << std::endl;
    std::cout << "  - Threads: " 
#ifdef _OPENMP
              << omp_get_max_threads()
#else
              << "1 (OpenMP not enabled)"
#endif
              << std::endl;
              
    std::mt19random_engine rng(42);
    
    int n = m_particles.size();
    std::vector<double> velocities_x(n, 0.0);
    std::vector<double> velocities_y(n, 0.0);
    
    double temp = initial_temp;
    double cooling_rate = std::pow(0.01 / initial_temp, 1.0 / steps); // Target 0.01 final temp
    
    double dt = 0.05; // Time step
    double damping = 0.8; // Friction
    
    for (int step = 0; step < steps; ++step) {
        calculateForces();
        
        // Update positions using Verlet-style integration with random thermal noise
        std::normal_distribution<double> noise(0.0, temp);
        
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            Particle& p = m_particles[i];
            
            double mass = std::max(p.area, 1.0);
            
            double ax = fx[i] / mass;
            double ay = fy[i] / mass;
            
            velocities_x[i] = (velocities_x[i] + ax * dt) * damping + noise(rng);
            velocities_y[i] = (velocities_y[i] + ay * dt) * damping + noise(rng);
            
            p.x += velocities_x[i] * dt;
            p.y += velocities_y[i] * dt;
            
            // Hard boundary clamping
            if (p.x < 0) { p.x = 0; velocities_x[i] = -velocities_x[i] * 0.5; }
            if (p.x > m_die_width) { p.x = m_die_width; velocities_x[i] = -velocities_x[i] * 0.5; }
            if (p.y < 0) { p.y = 0; velocities_y[i] = -velocities_y[i] * 0.5; }
            if (p.y > m_die_height) { p.y = m_die_height; velocities_y[i] = -velocities_y[i] * 0.5; }
        }
        
        temp *= cooling_rate;
        
        if (step > 0 && step % (steps / 10) == 0) {
            std::cout << "  - Step " << step << "/" << steps 
                      << " | Temp: " << temp << std::endl;
        }
    }
    
    std::cout << "[Fluxion Engine] Optimization complete." << std::endl;
}

} // namespace fluxion
