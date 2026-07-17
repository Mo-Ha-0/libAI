#pragma once
#include "core/layer.hpp"
#include "layers/dense.hpp"
#include <vector>
#include <memory>
#include <string>

namespace libAI {

class NeuralNetwork {
public:
    std::vector<std::unique_ptr<Layer>> layers;
    std::unique_ptr<LossLayer> loss_layer;

    template<typename L, typename... Args>
    void add_layer(Args&&... args) {
        layers.push_back(std::make_unique<L>(std::forward<Args>(args)...));
    }

    template<typename L, typename... Args>
    void set_loss(Args&&... args) {
        loss_layer = std::make_unique<L>(std::forward<Args>(args)...);
    }

    Tensor predict(const Tensor& x, bool training = false);
    float compute_loss(const Tensor& x, const Tensor& y, bool training = true);
    float accuracy(const Tensor& x, const Tensor& y);
    std::vector<std::pair<std::unordered_map<std::string, Tensor>*,
                          const std::unordered_map<std::string, Tensor>*>> gradient();
    void save(const std::string& filepath) const;
    void load(const std::string& filepath);
};

} // namespace libAI
