// -*- mode: C++; c-file-style: "cc-mode" -*-
//*************************************************************************
// DESCRIPTION: FLUXION Barnes-Hut Tree (C++)
//*************************************************************************

#ifndef FLUXION_BARNES_HUT_H_
#define FLUXION_BARNES_HUT_H_

#include <vector>
#include <memory>

namespace fluxion {

struct BHNode {
    double x_min, y_min, x_max, y_max;
    double cx = 0, cy = 0;
    double total_charge = 0;
    int count = 0;
    int particle_idx = -1;
    
    std::unique_ptr<BHNode> children[4];
    
    bool isLeaf() const {
        return !children[0] && !children[1] && !children[2] && !children[3];
    }
    
    double size() const {
        return std::max(x_max - x_min, y_max - y_min);
    }
};

class BarnesHutTree {
public:
    explicit BarnesHutTree(double theta = 0.5);
    ~BarnesHutTree();

    void build(const double* px, const double* py, const double* charges, int n);
    
    void computeForcesForParticle(int p_idx, double px, double py, double charge, 
                                  double force_k, double min_dist,
                                  double& out_fx, double& out_fy) const;

private:
    double m_theta;
    std::unique_ptr<BHNode> m_root;
    
    const double* m_px;
    const double* m_py;
    const double* m_charges;

    void insert(BHNode* node, int idx, double px, double py, double charge);
    void insertIntoChild(BHNode* node, int idx, double px, double py, double charge);
    
    void computeForceRec(BHNode* node, int p_idx, double px, double py, double charge,
                         double force_k, double min_dist,
                         double& acc_fx, double& acc_fy) const;
};

} // namespace fluxion

#endif // FLUXION_BARNES_HUT_H_
