#include "lm_impl.h"

#include <iostream>
namespace poselib {

void print_iteration(const BundleStats &stats, RobustLoss *loss_fn) {
    if (stats.iterations == 0) {
        std::cout << "initial_cost=" << stats.initial_cost << "\n";
    }
    std::cout << "iter=" << stats.iterations << ", cost=" << stats.cost << ", step=" << stats.step_norm
              << ", grad=" << stats.grad_norm << ", lambda=" << stats.lambda << "\n";
}

} // namespace poselib