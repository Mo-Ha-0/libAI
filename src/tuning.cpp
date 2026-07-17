#include "../include/libAI/tuning.hpp"
#include <iostream>
#include <algorithm>

namespace libAI {

std::map<std::string, float> HyperparameterTuner::grid_search(
    const Tensor& x_train, const Tensor& y_train,
    const Tensor& x_val, const Tensor& y_val,
    const std::map<std::string, std::vector<float>>& param_grid,
    std::function<std::pair<NeuralNetwork*, Optimizer*>(const std::map<std::string, float>&)> network_builder,
    int epochs, int batch_size)
{
    results.clear();

    std::vector<std::map<std::string, float>> combinations = {{}};
    for (auto& [key, values] : param_grid) {
        std::vector<std::map<std::string, float>> new_combinations;
        for (auto& combo : combinations) {
            for (float val : values) {
                auto new_combo = combo;
                new_combo[key] = val;
                new_combinations.push_back(new_combo);
            }
        }
        combinations = std::move(new_combinations);
    }

    std::cout << "Grid search over " << combinations.size() << " combinations..." << std::endl;

    float best_val_acc = -1.0f;
    std::map<std::string, float> best_params;

    for (size_t i = 0; i < combinations.size(); i++) {
        std::cout << "Combination " << i + 1 << "/" << combinations.size() << ": ";
        for (auto& [k, v] : combinations[i]) {
            std::cout << k << "=" << v << " ";
        }
        std::cout << std::endl;

        auto [network, optimizer] = network_builder(combinations[i]);

        NeuralNetwork* net = network;
        Optimizer* opt = optimizer;

        Trainer trainer(net, opt);
        trainer.fit(x_train, y_train, x_val, y_val, epochs, batch_size, false, 0);

        float val_acc = trainer.val_acc_history.empty() ? 0.0f : trainer.val_acc_history.back();
        float val_loss = trainer.val_loss_history.empty() ? 0.0f : trainer.val_loss_history.back();

        HyperparameterResult result;
        result.params = combinations[i];
        result.val_loss = val_loss;
        result.val_acc = val_acc;
        results.push_back(result);

        std::cout << "  -> val_loss: " << val_loss << ", val_acc: " << val_acc << std::endl;

        if (val_acc > best_val_acc) {
            best_val_acc = val_acc;
            best_params = combinations[i];
        }

        delete net;
        delete opt;
    }

    std::sort(results.begin(), results.end(),
              [](const HyperparameterResult& a, const HyperparameterResult& b) {
                  return a.val_acc > b.val_acc;
              });

    std::cout << "Best params: ";
    for (auto& [k, v] : best_params) {
        std::cout << k << "=" << v << " ";
    }
    std::cout << "(val_acc=" << best_val_acc << ")" << std::endl;

    return best_params;
}

} // namespace libAI
