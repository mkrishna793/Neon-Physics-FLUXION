// -*- mode: C++; c-file-style: "cc-mode" -*-
//*************************************************************************
// DESCRIPTION: Verilator: FLUXION Physics-Native Placement Export Pass
//
// Implementation of the FLUXION export pass that converts Verilator's
// internal circuit representation into a particle system for thermodynamic
// placement optimization.
//
// SPDX-FileCopyrightText: 2025 FLUXION Project
// SPDX-License-Identifier: LGPL-3.0-only OR Artistic-2.0
//
//*************************************************************************

#include "V3FluxionExport.h"

#include "V3Ast.h"
#include "V3Global.h"
#include "V3Stats.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

//=============================================================================
// JSON Utilities
//=============================================================================

namespace {

std::string escapeJson(const std::string& s) {
    std::string result;
    result.reserve(s.length());
    for (char c : s) {
        switch (c) {
        case '"': result += "\\\""; break;
        case '\\': result += "\\\\"; break;
        case '\b': result += "\\b"; break;
        case '\f': result += "\\f"; break;
        case '\n': result += "\\n"; break;
        case '\r': result += "\\r"; break;
        case '\t': result += "\\t"; break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                std::ostringstream oss;
                oss << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<int>(c);
                result += oss.str();
            } else {
                result += c;
            }
        }
    }
    return result;
}

std::string toJson(const std::string& key, const std::string& value) {
    return "\"" + escapeJson(key) + "\": \"" + escapeJson(value) + "\"";
}

std::string toJson(const std::string& key, int value) {
    return "\"" + escapeJson(key) + "\": " + std::to_string(value);
}

std::string toJson(const std::string& key, uint64_t value) {
    return "\"" + escapeJson(key) + "\": " + std::to_string(value);
}

std::string toJson(const std::string& key, double value) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "\"" << escapeJson(key) << "\": " << value;
    return oss.str();
}

std::string toJson(const std::string& key, bool value) {
    return "\"" + escapeJson(key) + "\": " + (value ? "true" : "false");
}

template<typename T>
std::string toJsonArray(const std::string& key, const std::vector<T>& arr) {
    std::ostringstream oss;
    oss << "\"" << escapeJson(key) << "\": [";
    for (size_t i = 0; i < arr.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << arr[i];
    }
    oss << "]";
    return oss.str();
}

std::string indent(int level) {
    return std::string(level * 2, ' ');
}

} // anonymous namespace

//=============================================================================
// FluxionGraph Implementation
//=============================================================================

uint64_t FluxionGraph::addNode(const FluxionNode& node) {
    uint64_t id = node.id;
    nodes[id] = node;
    totalGates += node.gateCount;
    totalPower += node.power;
    totalArea += node.area;
    return id;
}

void FluxionGraph::addConnection(const FluxionConnection& conn) {
    connections.push_back(conn);
    totalNets++;
}

void FluxionGraph::computeStatistics() {
    maxLevel = 0;
    totalPower = 0;
    totalArea = 0;
    totalGates = 0;

    for (auto& pair : nodes) {
        FluxionNode& node = pair.second;
        totalGates += node.gateCount;
        totalPower += node.power;
        totalArea += node.area;
        if (node.level > maxLevel) {
            maxLevel = node.level;
        }
    }
}

void FluxionGraph::identifyCriticalPaths() {
    // Identify timing-critical paths using BFS from primary inputs
    // to primary outputs, tracking delay accumulation

    std::map<uint64_t, double> arrivalTime;
    std::map<uint64_t, uint64_t> predecessor;

    // Initialize with primary inputs (nodes with no inputs)
    for (auto& pair : nodes) {
        FluxionNode& node = pair.second;
        if (node.inputs.empty()) {
            arrivalTime[node.id] = 0;
        }
    }

    // Propagate delays using topological order
    bool changed = true;
    int iterations = 0;
    const int maxIterations = nodes.size() + 1000; // Prevent infinite loops

    while (changed && iterations < maxIterations) {
        changed = false;
        iterations++;

        for (auto& pair : nodes) {
            FluxionNode& node = pair.second;

            // Skip primary inputs
            if (node.inputs.empty()) continue;

            // Calculate maximum arrival time at this node
            double maxArrival = 0;
            uint64_t maxPred = 0;

            for (uint64_t inputId : node.inputs) {
                auto it = arrivalTime.find(inputId);
                if (it != arrivalTime.end()) {
                    // Find connection delay
                    double connDelay = 0;
                    for (const auto& conn : connections) {
                        if (conn.sourceId == inputId && conn.destId == node.id) {
                            connDelay = conn.isCriticalPath ? 1.5 : 1.0; // Weighted delay
                            break;
                        }
                    }

                    double arrival = it->second + node.delay + connDelay;
                    if (arrival > maxArrival) {
                        maxArrival = arrival;
                        maxPred = inputId;
                    }
                }
            }

            auto currentIt = arrivalTime.find(node.id);
            double currentArrival = (currentIt != arrivalTime.end()) ? currentIt->second : 0;

            if (maxArrival > currentArrival) {
                arrivalTime[node.id] = maxArrival;
                predecessor[node.id] = maxPred;
                changed = true;
            }
        }
    }

    // Find critical path ending at primary outputs (nodes with no outputs)
    double criticalPathDelay = 0;
    uint64_t criticalSink = 0;

    for (auto& pair : nodes) {
        FluxionNode& node = pair.second;
        if (node.outputs.empty()) {
            auto it = arrivalTime.find(node.id);
            if (it != arrivalTime.end() && it->second > criticalPathDelay) {
                criticalPathDelay = it->second;
                criticalSink = node.id;
            }
        }
    }

    // Reconstruct critical path
    if (criticalSink != 0) {
        FluxionCriticalPath path;
        path.totalDelay = criticalPathDelay;

        uint64_t current = criticalSink;
        while (current != 0) {
            path.nodeIds.push_back(current);
            auto predIt = predecessor.find(current);
            if (predIt != predecessor.end()) {
                current = predIt->second;
            } else {
                break;
            }
        }

        // Reverse to get path from input to output
        std::reverse(path.nodeIds.begin(), path.nodeIds.end());

        // Mark connections on critical path
        for (size_t i = 0; i < path.nodeIds.size() - 1; ++i) {
            uint64_t src = path.nodeIds[i];
            uint64_t dst = path.nodeIds[i + 1];
            for (auto& conn : connections) {
                if (conn.sourceId == src && conn.destId == dst) {
                    conn.isCriticalPath = true;
                }
            }
        }

        criticalPaths.push_back(path);
    }
}

std::string FluxionGraph::toJson() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);

    oss << "{\n";
    oss << indent(1) << "\"fluxion_version\": \"1.0.0\",\n";
    oss << indent(1) << "\"format\": \"circuit_particles\",\n";
    oss << indent(1) << "\"module\": \"" << escapeJson(moduleName) << "\",\n";
    oss << indent(1) << "\"statistics\": {\n";
    oss << indent(2) << "\"total_gates\": " << totalGates << ",\n";
    oss << indent(2) << "\"total_nets\": " << totalNets << ",\n";
    oss << indent(2) << "\"total_power_pw\": " << totalPower << ",\n";
    oss << indent(2) << "\"total_area_um2\": " << totalArea << ",\n";
    oss << indent(2) << "\"max_logic_level\": " << maxLevel << ",\n";
    oss << indent(2) << "\"critical_paths\": " << criticalPaths.size() << "\n";
    oss << indent(1) << "},\n";

    // Physical constraints
    oss << indent(1) << "\"physical_constraints\": {\n";
    oss << indent(2) << "\"die_width_um\": " << dieWidth << ",\n";
    oss << indent(2) << "\"die_height_um\": " << dieHeight << ",\n";
    oss << indent(2) << "\"target_clock_period_ps\": " << targetClockPeriod << "\n";
    oss << indent(1) << "},\n";

    // Nodes (particles)
    oss << indent(1) << "\"nodes\": [\n";
    {
        bool first = true;
        for (const auto& pair : nodes) {
            const FluxionNode& node = pair.second;
            if (!first) oss << ",\n";
            first = false;

            oss << indent(2) << "{\n";
            oss << indent(3) << toJson("id", node.id) << ",\n";
            oss << indent(3) << toJson("name", node.name) << ",\n";
            oss << indent(3) << toJson("type", node.type) << ",\n";
            oss << indent(3) << toJson("gate_count", node.gateCount) << ",\n";
            oss << indent(3) << toJson("power_pw", node.power) << ",\n";
            oss << indent(3) << toJson("area_um2", node.area) << ",\n";
            oss << indent(3) << toJson("x", node.x) << ",\n";
            oss << indent(3) << toJson("y", node.y) << ",\n";
            oss << indent(3) << toJson("delay_ps", node.delay) << ",\n";
            oss << indent(3) << toJson("level", node.level) << ",\n";
            oss << indent(3) << toJson("thermal_resistance", node.thermalResistance) << ",\n";
            oss << indent(3) << toJson("heat_generation", node.heatGeneration) << ",\n";

            // Inputs array
            oss << indent(3) << "\"inputs\": [";
            for (size_t i = 0; i < node.inputs.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << node.inputs[i];
            }
            oss << "],\n";

            // Outputs array
            oss << indent(3) << "\"outputs\": [";
            for (size_t i = 0; i < node.outputs.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << node.outputs[i];
            }
            oss << "]\n";

            oss << indent(2) << "}";
        }
    }
    oss << "\n" << indent(1) << "],\n";

    // Connections (nets)
    oss << indent(1) << "\"connections\": [\n";
    {
        bool first = true;
        for (const FluxionConnection& conn : connections) {
            if (!first) oss << ",\n";
            first = false;

            oss << indent(2) << "{\n";
            oss << indent(3) << toJson("source_id", conn.sourceId) << ",\n";
            oss << indent(3) << toJson("dest_id", conn.destId) << ",\n";
            oss << indent(3) << toJson("name", conn.name) << ",\n";
            oss << indent(3) << toJson("bit_width", conn.bitWidth) << ",\n";
            oss << indent(3) << toJson("is_critical_path", conn.isCriticalPath) << "\n";
            oss << indent(2) << "}";
        }
    }
    oss << "\n" << indent(1) << "],\n";

    // Critical paths
    oss << indent(1) << "\"critical_paths\": [\n";
    {
        bool first = true;
        for (const FluxionCriticalPath& path : criticalPaths) {
            if (!first) oss << ",\n";
            first = false;

            oss << indent(2) << "{\n";
            oss << indent(3) << toJson("total_delay_ps", path.totalDelay) << ",\n";
            oss << indent(3) << toJson("slack_ps", path.slack) << ",\n";
            oss << indent(3) << "\"node_ids\": [";
            for (size_t i = 0; i < path.nodeIds.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << path.nodeIds[i];
            }
            oss << "]\n";
            oss << indent(2) << "}";
        }
    }
    oss << "\n" << indent(1) << "]\n";

    oss << "}\n";

    return oss.str();
}

void FluxionGraph::toJsonFile(const std::string& filename) const {
    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Error: Cannot open output file: " << filename << std::endl;
        return;
    }
    outFile << toJson();
    outFile.close();
}

//=============================================================================
// V3FluxionGraphVisitor Implementation
//=============================================================================

void V3FluxionGraphVisitor::visit(AstNetlist* nodep) {
    // Process top-level netlist
    m_graph.moduleName = "top";

    // Visit all modules
    for (AstNodeModule* modp = nodep->modulesp(); modp;
         modp = VN_CAST(modp->nextp(), NodeModule)) {
        iterate(modp);
    }

    // Build connectivity after all modules are processed
    buildConnectivity();

    // Compute statistics
    m_graph.computeStatistics();

    // Identify critical paths
    m_graph.identifyCriticalPaths();
}

void V3FluxionGraphVisitor::visit(AstModule* nodep) {
    // Skip already processed modules
    std::string modName = nodep->name();
    if (m_processedModules.count(modName)) {
        return;
    }
    m_processedModules.insert(modName);

    // Set module name if this is the top module
    if (nodep->isTop()) {
        m_graph.moduleName = modName;
    }

    // Extract module hierarchy
    extractModuleHierarchy(nodep);

    // Extract combinational logic
    extractCombinationalLogic(nodep);

    // Extract sequential logic (flip-flops, latches)
    extractSequentialLogic(nodep);

    // Visit children
    iterateChildren(nodep);
}

void V3FluxionGraphVisitor::visit(AstCell* nodep) {
    // Process cell instance
    std::string instName = nodep->name();
    std::string modType = nodep->modName();

    // Create node for this cell instance
    FluxionNode node;
    node.id = m_graph.nodes.size() + 1;
    node.name = instName;
    node.type = modType;
    node.gateCount = 1; // Will be refined based on module complexity

    // Estimate physical properties based on module type
    if (modType.find("DFF") != std::string::npos ||
        modType.find("dff") != std::string::npos) {
        node.delay = 50.0;      // ps - typical DFF delay
        node.area = 10.0;       // um² - typical DFF area
        node.power = 100.0;     // pW - typical DFF power
        node.thermalResistance = 0.5;
    } else if (modType.find("AND") != std::string::npos ||
               modType.find("and") != std::string::npos) {
        node.delay = 10.0;
        node.area = 2.0;
        node.power = 5.0;
        node.thermalResistance = 1.0;
    } else if (modType.find("OR") != std::string::npos ||
               modType.find("or") != std::string::npos) {
        node.delay = 10.0;
        node.area = 2.0;
        node.power = 5.0;
        node.thermalResistance = 1.0;
    } else if (modType.find("NAND") != std::string::npos ||
               modType.find("nand") != std::string::npos) {
        node.delay = 8.0;
        node.area = 1.5;
        node.power = 4.0;
        node.thermalResistance = 1.0;
    } else if (modType.find("NOR") != std::string::npos ||
               modType.find("nor") != std::string::npos) {
        node.delay = 8.0;
        node.area = 1.5;
        node.power = 4.0;
        node.thermalResistance = 1.0;
    } else if (modType.find("XOR") != std::string::npos ||
               modType.find("xor") != std::string::npos) {
        node.delay = 15.0;
        node.area = 3.0;
        node.power = 8.0;
        node.thermalResistance = 0.9;
    } else if (modType.find("MUX") != std::string::npos ||
               modType.find("mux") != std::string::npos) {
        node.delay = 20.0;
        node.area = 5.0;
        node.power = 15.0;
        node.thermalResistance = 0.7;
    } else if (modType.find("ADD") != std::string::npos ||
               modType.find("add") != std::string::npos) {
        node.delay = 30.0;
        node.area = 8.0;
        node.power = 25.0;
        node.thermalResistance = 0.6;
    } else {
        // Default estimates for unknown modules
        node.delay = 20.0;
        node.area = 4.0;
        node.power = 10.0;
        node.thermalResistance = 1.0;
    }

    // Initialize position (will be randomized before placement)
    node.x = 0.0;
    node.y = 0.0;
    node.level = 0;
    node.heatGeneration = node.power;

    m_graph.addNode(node);
    m_nodeMap[nodep] = node.id;

    iterateChildren(nodep);
}

void V3FluxionGraphVisitor::visit(AstVar* nodep) {
    // Process variable declarations
    iterateChildren(nodep);
}

void V3FluxionGraphVisitor::visit(AstNodeAssign* nodep) {
    // Process assignments to build connectivity
    iterateChildren(nodep);
}

void V3FluxionGraphVisitor::extractModuleHierarchy(AstModule* modulep) {
    // Extract hierarchical structure
    for (AstNode* nodep = modulep->stmtsp(); nodep; nodep = nodep->nextp()) {
        // Process hierarchical cells
        if (AstCell* cellp = VN_CAST(nodep, Cell)) {
            iterate(cellp);
        }
    }
}

void V3FluxionGraphVisitor::extractCombinationalLogic(AstModule* modulep) {
    // Extract combinational logic gates
    // This processes always_comb blocks and continuous assignments

    for (AstNode* nodep = modulep->stmtsp(); nodep; nodep = nodep->nextp()) {
        // Process assignments and logic blocks
        if (AstNodeAssign* assignp = VN_CAST(nodep, NodeAssign)) {
            iterate(assignp);
        }
    }
}

void V3FluxionGraphVisitor::extractSequentialLogic(AstModule* modulep) {
    // Extract sequential logic (flip-flops, latches)
    // This processes always_ff blocks

    for (AstNode* nodep = modulep->stmtsp(); nodep; nodep = nodep->nextp()) {
        // Look for sequential elements
        iterate(nodep);
    }
}

void V3FluxionGraphVisitor::buildConnectivity() {
    // Build connection graph between nodes
    // This establishes the input/output relationships

    std::map<std::string, uint64_t> signalToDriver;

    // First pass: identify signal drivers
    for (auto& pair : m_graph.nodes) {
        FluxionNode& node = pair.second;

        // For each output, record the driving node
        for (uint64_t outputId : node.outputs) {
            // Map signal name to driving node
        }
    }

    // Second pass: establish connections based on signal names
    for (auto& pair : m_graph.nodes) {
        FluxionNode& destNode = pair.second;

        for (uint64_t srcId : destNode.inputs) {
            FluxionConnection conn;
            conn.sourceId = srcId;
            conn.destId = destNode.id;
            conn.bitWidth = 1; // Will be refined
            conn.isCriticalPath = false;

            m_graph.addConnection(conn);
        }
    }
}

//=============================================================================
// V3FluxionExport Implementation
//=============================================================================

void V3FluxionExport::exportCircuit(AstNetlist* netlistp, const std::string& outputFile) {
    UINFO(2, "FLUXION: Exporting circuit to " << outputFile << std::endl);

    // Create graph
    FluxionGraph graph;

    // Create visitor and traverse AST
    V3FluxionGraphVisitor visitor(graph);
    visitor.visit(netlistp);

    // Compute statistics
    graph.computeStatistics();

    // Identify critical paths
    graph.identifyCriticalPaths();

    // Randomize initial positions
    std::srand(42); // Deterministic for reproducibility
    for (auto& pair : graph.nodes) {
        FluxionNode& node = pair.second;
        node.x = (static_cast<double>(std::rand()) / RAND_MAX) * graph.dieWidth;
        node.y = (static_cast<double>(std::rand()) / RAND_MAX) * graph.dieHeight;
    }

    // Export to JSON
    graph.toJsonFile(outputFile);

    UINFO(2, "FLUXION: Exported " << graph.nodes.size() << " nodes, "
          << graph.connections.size() << " connections" << std::endl);

    // Statistics
    V3Stats::addStat("Fluxion", "Nodes", graph.nodes.size());
    V3Stats::addStat("Fluxion", "Connections", graph.connections.size());
    V3Stats::addStat("Fluxion", "CriticalPaths", graph.criticalPaths.size());
}

//=============================================================================
// Gate Property Estimation
//=============================================================================

double V3FluxionExport::estimateGateDelay(const std::string& type) {
    // Return estimated delay in picoseconds based on gate type
    static const std::map<std::string, double> delays = {
        {"AND", 12.0}, {"NAND", 8.0}, {"OR", 12.0}, {"NOR", 8.0},
        {"XOR", 15.0}, {"XNOR", 15.0}, {"NOT", 5.0}, {"BUF", 5.0},
        {"MUX", 20.0}, {"DFF", 50.0}, {"LAT", 30.0}, {"ADD", 35.0},
        {"SUB", 35.0}, {"MUL", 100.0}, {"DIV", 150.0}
    };

    std::string upperType = type;
    std::transform(upperType.begin(), upperType.end(), upperType.begin(), ::toupper);

    for (const auto& pair : delays) {
        if (upperType.find(pair.first) != std::string::npos) {
            return pair.second;
        }
    }

    return 20.0; // Default delay
}

double V3FluxionExport::estimateGateArea(const std::string& type) {
    // Return estimated area in square micrometers
    static const std::map<std::string, double> areas = {
        {"AND", 2.5}, {"NAND", 2.0}, {"OR", 2.5}, {"NOR", 2.0},
        {"XOR", 4.0}, {"XNOR", 4.0}, {"NOT", 1.5}, {"BUF", 1.5},
        {"MUX", 6.0}, {"DFF", 12.0}, {"LAT", 8.0}, {"ADD", 15.0},
        {"SUB", 15.0}, {"MUL", 50.0}, {"DIV", 80.0}
    };

    std::string upperType = type;
    std::transform(upperType.begin(), upperType.end(), upperType.begin(), ::toupper);

    for (const auto& pair : areas) {
        if (upperType.find(pair.first) != std::string::npos) {
            return pair.second;
        }
    }

    return 5.0; // Default area
}

double V3FluxionExport::estimateGatePower(const std::string& type) {
    // Return estimated power in picowatts
    static const std::map<std::string, double> powers = {
        {"AND", 8.0}, {"NAND", 6.0}, {"OR", 8.0}, {"NOR", 6.0},
        {"XOR", 12.0}, {"XNOR", 12.0}, {"NOT", 3.0}, {"BUF", 3.0},
        {"MUX", 20.0}, {"DFF", 150.0}, {"LAT", 100.0}, {"ADD", 40.0},
        {"SUB", 40.0}, {"MUL", 200.0}, {"DIV", 300.0}
    };

    std::string upperType = type;
    std::transform(upperType.begin(), upperType.end(), upperType.begin(), ::toupper);

    for (const auto& pair : powers) {
        if (upperType.find(pair.first) != std::string::npos) {
            return pair.second;
        }
    }

    return 15.0; // Default power
}

//=============================================================================
// Integration with Verilator Pipeline
//=============================================================================

// This function integrates with Verilator's processing pipeline
// It should be called after V3Order and before V3EmitC
void V3FluxionExport_process(AstNetlist* netlistp, const std::string& outputFile) {
    V3FluxionExport::exportCircuit(netlistp, outputFile);
}