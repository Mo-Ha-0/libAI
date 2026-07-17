#pragma once
#include "../core/layer.hpp"

namespace libAI {

class Dropout : public Layer {
public:
    float drop_rate;
    Dropout(float drop_rate = 0.5f);
    Tensor forward(const Tensor& x, bool training = true) override;
    Tensor backward(const Tensor& dout) override;
};

class BatchNormalization : public Layer {
public:
    size_t num_features;
    float momentum;
    float eps;
    Tensor running_mean;
    Tensor running_var;

    BatchNormalization(size_t num_features, float momentum = 0.9f, float eps = 1e-5f);
    Tensor forward(const Tensor& x, bool training = true) override;
    Tensor backward(const Tensor& dout) override;
};

} // namespace libAI
