#pragma once
#include "network.hpp"
#include "trainer.hpp"
#include <vector>
#include <string>
#include <functional>
#include <map>

namespace libAI {

struct HyperparameterResult {
    std::map<std::string, float> params;
    float val_loss;
    float val_acc;
};

class HyperparameterTuner {
public:
    std::vector<HyperparameterResult> results;

    std::map<std::string, float> grid_search(
        const Tensor& x_train, const Tensor& y_train,
        const Tensor& x_val, const Tensor& y_val,
        const std::map<std::string, std::vector<float>>& param_grid,
        std::function<std::pair<NeuralNetwork*, Optimizer*>(const std::map<std::string, float>&)> network_builder,
        int epochs = 50, int batch_size = 32);
};

} // namespace libAI
