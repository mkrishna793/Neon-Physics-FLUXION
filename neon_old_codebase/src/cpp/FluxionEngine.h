// -*- mode: C++; c-file-style: "cc-mode" -*-
//*************************************************************************
// DESCRIPTION: FLUXION C++ Physics Engine
//
// Standalone high-performance implementation of the Thermodynamic Placement
// Engine. Compiles as a library to be called from the Verilator C++ pass.
// Allows us to perform exact Physical Synthesis without dropping to Python.
//*************************************************************************

#ifndef FLUXION_ENGINE_H_
#define FLUXION_ENGINE_H_

#include "V3FluxionExport.h" // For FluxionGraph, FluxionNode
#include <vector>

namespace fluxion {

// Simplified struct for tight memory packing and SIMD
struct Particle {
    uint64_t id;
    double x;
    double y;
    double width;
    double height;
    
    // Physics properties
    double power;
    double area;
    double delay;
    
    // Abstract targets
    int logic_level;
};

// Simplified connection struct
struct Connection {
    int src_idx;  // Index into particles vector, not raw ID
    int dst_idx;
    bool is_critical;
    double expected_delay;
};

// Main placement engine class
class FluxionEngine {
public:
    FluxionEngine();
    ~FluxionEngine();

    // Data Loaders
    void loadFromGraph(const FluxionGraph& graph);

    // Physics Engine Configuration
    void setDieSize(double width, double height);
    void setForceWeights(double wire, double thermal, double timing, double topo, double density);
    
    // Main execution
    void optimize(int steps = 5000, double initial_temp = 100.0);

    // Results retrieval
    const std::vector<Particle>& getParticles() const { return m_particles; }

private:
    std::vector<Particle> m_particles;
    std::vector<Connection> m_connections;
    
    double m_die_width;
    double m_die_height;
    
    // Weights
    double w_wire = 1.0;
    double w_thermal = 0.5;
    double w_timing = 0.8;
    double w_topo = 0.3;
    double w_density = 0.0;
    
    // Working memory for forces
    std::vector<double> fx;
    std::vector<double> fy;

    // Internal force calculators
    void calculateForces();
    void applyWireTension();
    void applyThermalRepulsion();
    void applyTimingGravity();
};

} // namespace fluxion

#endif // FLUXION_ENGINE_H_
