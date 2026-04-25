// -*- mode: C++; c-file-style: "cc-mode" -*-
//*************************************************************************
// DESCRIPTION: Verilator: FLUXION Physics-Native Placement Export Pass
//
// FLUXION Export Pass: Converts Verilator's internal AST/circuit graph
// into a particle system JSON representation for thermodynamic placement.
//
// Code available from: https://github.com/fluxion-project
// SPDX-FileCopyrightText: 2025 FLUXION Project
// SPDX-License-Identifier: LGPL-3.0-only OR Artistic-2.0
//
//*************************************************************************

#ifndef VERILATOR_V3FLUXION_EXPORT_H_
#define VERILATOR_V3FLUXION_EXPORT_H_

#include "config_build.h"
#include "verilatedos.h"

#include "V3Ast.h"
#include "V3Graph.h"

#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// Forward declarations
class V3Graph;
class V3GraphVertex;

//=============================================================================
// FluxionNode - Represents a gate/module as a physical particle
//=============================================================================

struct FluxionNode {
    std::string name;           // Instance name
    std::string type;           // Module/gate type (e.g., "AND", "OR", "DFF")
    uint64_t id;                // Unique identifier
    int gateCount;              // Number of primitive gates represented
    double power;               // Power consumption estimate (pW)
    double area;                // Area estimate (um²)

    // Position (initial random, then optimized)
    double x;
    double y;

    // Timing information
    double delay;               // Gate delay (ps)
    int level;                  // Logic level from primary inputs

    // Connections
    std::vector<uint64_t> inputs;   // IDs of driving nodes
    std::vector<uint64_t> outputs;  // IDs of driven nodes

    // Thermal data
    double thermalResistance;    // Thermal resistance (K/W)
    double heatGeneration;       // Power dissipation

    FluxionNode() : id(0), gateCount(1), power(0), area(0),
                    x(0), y(0), delay(0), level(0),
                    thermalResistance(1.0), heatGeneration(0) {}
};

//=============================================================================
// FluxionConnection - Represents a wire between gates
//=============================================================================

struct FluxionConnection {
    uint64_t sourceId;          // Source node ID
    uint64_t destId;            // Destination node ID
    std::string name;           // Signal name
    int bitWidth;               // Bit width of the connection
    double estimatedLength;    // Estimated wire length (um)
    double capacitance;         // Wire capacitance (fF)
    bool isCriticalPath;        // Is this on a critical timing path?

    FluxionConnection() : sourceId(0), destId(0), bitWidth(1),
                          estimatedLength(0), capacitance(0), isCriticalPath(false) {}
};

//=============================================================================
// FluxionCriticalPath - Represents a timing-critical path
//=============================================================================

struct FluxionCriticalPath {
    std::vector<uint64_t> nodeIds;  // Nodes on the path in order
    double totalDelay;              // Total path delay (ps)
    double slack;                   // Timing slack (ps)
    std::string startClock;         // Source clock domain
    std::string endClock;           // Destination clock domain
};

//=============================================================================
// FluxionGraph - The complete circuit graph for physics simulation
//=============================================================================

class FluxionGraph {
public:
    std::string moduleName;                   // Top module name
    std::map<uint64_t, FluxionNode> nodes;    // All nodes indexed by ID
    std::vector<FluxionConnection> connections; // All connections
    std::vector<FluxionCriticalPath> criticalPaths; // Timing critical paths

    // Design statistics
    double totalPower;           // Total power estimate (pW)
    double totalArea;            // Total area estimate (um²)
    int totalGates;              // Total gate count
    int totalNets;               // Total nets
    int maxLevel;                // Maximum logic level

    // Physical constraints
    double dieWidth;             // Die width (um)
    double dieHeight;            // Die height (um)
    double targetClockPeriod;    // Target clock period (ps)

    FluxionGraph() : totalPower(0), totalArea(0), totalGates(0),
                     totalNets(0), maxLevel(0), dieWidth(1000),
                     dieHeight(1000), targetClockPeriod(1000) {}

    // JSON export
    std::string toJson() const;
    void toJsonFile(const std::string& filename) const;

    // Statistics
    void computeStatistics();
    void identifyCriticalPaths();

    // Node operations
    uint64_t addNode(const FluxionNode& node);
    void addConnection(const FluxionConnection& conn);
};

//=============================================================================
// V3FluxionExport - Main export pass visitor
//=============================================================================

class V3FluxionExport final {
public:
    // Main entry point - called from Verilator pipeline
    static void exportCircuit(AstNetlist* netlistp, const std::string& outputFile);

private:
    // Internal state
    std::unique_ptr<FluxionGraph> m_graphp;
    std::map<std::string, uint64_t> m_nameToId;
    uint64_t m_nextId = 1;

    // AST traversal methods
    void visitNode(AstNode* nodep);
    void visitModule(AstModule* modulep);
    void visitCell(AstCell* cellp);
    void visitVar(AstVar* varp);
    void visitAssign(AstNodeAssign* assignp);

    // Graph construction
    void processNetlist(AstNetlist* netlistp);
    void buildConnections();
    void computeLogicLevels();
    void estimatePhysicalProperties();

    // Helpers
    uint64_t getOrCreateNodeId(const std::string& name, const std::string& type);
    double estimateGateDelay(const std::string& type);
    double estimateGateArea(const std::string& type);
    double estimateGatePower(const std::string& type);

    V3FluxionExport() : m_graphp(std::make_unique<FluxionGraph>()) {}
};

//=============================================================================
// V3FluxionGraphVisitor - AST visitor for circuit extraction
//=============================================================================

class V3FluxionGraphVisitor final : public VNVisitor {
private:
    FluxionGraph& m_graph;
    std::map<AstNode*, uint64_t> m_nodeMap;
    std::set<std::string> m_processedModules;

public:
    explicit V3FluxionGraphVisitor(FluxionGraph& graph) : m_graph(graph) {}

    void visit(AstNetlist* nodep) override;
    void visit(AstModule* nodep) override;
    void visit(AstCell* nodep) override;
    void visit(AstVar* nodep) override;
    void visit(AstNodeAssign* nodep) override;
    void visit(AstNode* nodep) override { iterateChildren(nodep); }

private:
    void extractModuleHierarchy(AstModule* modulep);
    void extractCombinationalLogic(AstModule* modulep);
    void extractSequentialLogic(AstModule* modulep);
    void buildConnectivity();
};

#endif // VERILATOR_V3FLUXION_EXPORT_H_