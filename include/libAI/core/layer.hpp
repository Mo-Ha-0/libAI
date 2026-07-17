#pragma once
#include "tensor.hpp"
#include <unordered_map>
#include <string>
#include <memory>

namespace libAI {

class Layer {
public:
    std::unordered_map<std::string, Tensor> params;
    std::unordered_map<std::string, Tensor> grads;
    std::unordered_map<std::string, Tensor> cache;

    virtual ~Layer() = default;
    virtual Tensor forward(const Tensor& x, bool training = true) = 0;
    virtual Tensor backward(const Tensor& dout) = 0;
};

class LossLayer {
public:
    std::unordered_map<std::string, Tensor> cache;
    virtual ~LossLayer() = default;
    virtual float forward(const Tensor& x, const Tensor& y, bool training = true) = 0;
    virtual Tensor backward(float dout = 1.0f) = 0;
};

} // namespace libAI
