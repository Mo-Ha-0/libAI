#pragma once
#include "../core/layer.hpp"

namespace libAI {

class MeanSquaredError : public LossLayer {
public:
    float forward(const Tensor& x, const Tensor& y, bool training = true) override;
    Tensor backward(float dout = 1.0f) override;
};

class SoftmaxCrossEntropy : public LossLayer {
public:
    float forward(const Tensor& x, const Tensor& y, bool training = true) override;
    Tensor backward(float dout = 1.0f) override;
};

} // namespace libAI
