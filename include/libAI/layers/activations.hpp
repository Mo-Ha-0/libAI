#pragma once
#include "../core/layer.hpp"

namespace libAI {

class Linear : public Layer {
public:
    Tensor forward(const Tensor& x, bool training = true) override;
    Tensor backward(const Tensor& dout) override;
};

class ReLU : public Layer {
public:
    Tensor forward(const Tensor& x, bool training = true) override;
    Tensor backward(const Tensor& dout) override;
};

class Sigmoid : public Layer {
public:
    Tensor forward(const Tensor& x, bool training = true) override;
    Tensor backward(const Tensor& dout) override;
};

class Tanh : public Layer {
public:
    Tensor forward(const Tensor& x, bool training = true) override;
    Tensor backward(const Tensor& dout) override;
};

} // namespace libAI
