// -*- mode: C++; c-file-style: "cc-mode" -*-
//*************************************************************************
// DESCRIPTION: FLUXION Barnes-Hut Tree (C++)
//*************************************************************************

#include "BarnesHutTree.h"
#include <cmath>
#include <algorithm>

namespace fluxion {

BarnesHutTree::BarnesHutTree(double theta) : m_theta(theta), m_px(nullptr), m_py(nullptr), m_charges(nullptr) {}

BarnesHutTree::~BarnesHutTree() {}

void BarnesHutTree::build(const double* px, const double* py, const double* charges, int n) {
    m_px = px;
    m_py = py;
    m_charges = charges;
    m_root.reset();
    
    if (n == 0) return;
    
    double min_x = px[0], max_x = px[0];
    double min_y = py[0], max_y = py[0];
    
    for (int i = 1; i < n; ++i) {
        if (px[i] < min_x) min_x = px[i];
        if (px[i] > max_x) max_x = px[i];
        if (py[i] < min_y) min_y = py[i];
        if (py[i] > max_y) max_y = py[i];
    }
    
    double margin = 1.0;
    min_x -= margin; max_x += margin;
    min_y -= margin; max_y += margin;
    
    double size = std::max(max_x - min_x, max_y - min_y);
    double cx = (min_x + max_x) / 2.0;
    double cy = (min_y + max_y) / 2.0;
    
    m_root = std::make_unique<BHNode>();
    m_root->x_min = cx - size/2;
    m_root->x_max = cx + size/2;
    m_root->y_min = cy - size/2;
    m_root->y_max = cy + size/2;
    
    for (int i = 0; i < n; ++i) {
        insert(m_root.get(), i, px[i], py[i], charges[i]);
    }
}

void BarnesHutTree::insert(BHNode* node, int idx, double px, double py, double charge) {
    if (node->count == 0) {
        node->particle_idx = idx;
        node->cx = px;
        node->cy = py;
        node->total_charge = charge;
        node->count = 1;
        return;
    }
    
    if (node->isLeaf()) {
        if (node->size() < 1e-8) {
            // Very identical coordinates, simply merge
            double tc = node->total_charge + charge;
            node->cx = (node->cx * node->total_charge + px * charge) / tc;
            node->cy = (node->cy * node->total_charge + py * charge) / tc;
            node->total_charge = tc;
            node->count++;
            return;
        }
        
        int old_idx = node->particle_idx;
        double old_px = m_px[old_idx];
        double old_py = m_py[old_idx];
        double old_charge = m_charges[old_idx];
        
        node->particle_idx = -1;
        
        insertIntoChild(node, old_idx, old_px, old_py, old_charge);
        insertIntoChild(node, idx, px, py, charge);
        
        double tc = old_charge + charge;
        node->cx = (old_px * old_charge + px * charge) / tc;
        node->cy = (old_py * old_charge + py * charge) / tc;
        node->total_charge = tc;
        node->count = 2;
    } else {
        insertIntoChild(node, idx, px, py, charge);
        
        double tc = node->total_charge + charge;
        node->cx = (node->cx * node->total_charge + px * charge) / tc;
        node->cy = (node->cy * node->total_charge + py * charge) / tc;
        node->total_charge = tc;
        node->count++;
    }
}

void BarnesHutTree::insertIntoChild(BHNode* node, int idx, double px, double py, double charge) {
    double mx = (node->x_min + node->x_max) / 2;
    double my = (node->y_min + node->y_max) / 2;
    
    int quad = 0;
    double bounds[4];
    
    if (px <= mx) {
        if (py <= my) {
            quad = 2; // SW
            bounds[0] = node->x_min; bounds[1] = node->y_min; bounds[2] = mx; bounds[3] = my;
        } else {
            quad = 0; // NW
            bounds[0] = node->x_min; bounds[1] = my; bounds[2] = mx; bounds[3] = node->y_max;
        }
    } else {
        if (py <= my) {
            quad = 3; // SE
            bounds[0] = mx; bounds[1] = node->y_min; bounds[2] = node->x_max; bounds[3] = my;
        } else {
            quad = 1; // NE
            bounds[0] = mx; bounds[1] = my; bounds[2] = node->x_max; bounds[3] = node->y_max;
        }
    }
    
    if (!node->children[quad]) {
        node->children[quad] = std::make_unique<BHNode>();
        node->children[quad]->x_min = bounds[0];
        node->children[quad]->y_min = bounds[1];
        node->children[quad]->x_max = bounds[2];
        node->children[quad]->y_max = bounds[3];
    }
    
    insert(node->children[quad].get(), idx, px, py, charge);
}

void BarnesHutTree::computeForcesForParticle(int p_idx, double px, double py, double charge, 
                                             double force_k, double min_dist,
                                             double& out_fx, double& out_fy) const {
    out_fx = 0; out_fy = 0;
    if (m_root) {
        computeForceRec(m_root.get(), p_idx, px, py, charge, force_k, min_dist, out_fx, out_fy);
    }
}

void BarnesHutTree::computeForceRec(BHNode* node, int p_idx, double px, double py, double charge,
                                    double force_k, double min_dist,
                                    double& fx, double& fy) const {
    if (node->count == 0) return;
    if (node->isLeaf() && node->particle_idx == p_idx) return;
    
    double dx = px - node->cx;
    double dy = py - node->cy;
    double dist_sq = dx*dx + dy*dy;
    double dist = std::sqrt(dist_sq);
    if (dist < min_dist) dist = min_dist;
    
    if (node->isLeaf() || (node->size() / dist < m_theta)) {
        double f = (force_k * charge * node->total_charge) / (dist * dist);
        fx += f * (dx / dist);
        fy += f * (dy / dist);
    } else {
        for (int i = 0; i < 4; ++i) {
            if (node->children[i]) {
                computeForceRec(node->children[i].get(), p_idx, px, py, charge, force_k, min_dist, fx, fy);
            }
        }
    }
}

} // namespace fluxion
