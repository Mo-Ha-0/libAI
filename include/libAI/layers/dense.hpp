#pragma once
#include "../core/layer.hpp"

namespace libAI {

/**
 * A fully connected (dense) layer for a neural network.
 * This layer performs a linear transformation on the input data, followed by an optional activation function.
 * It maintains weights and biases as parameters, and computes gradients during backpropagation.
 */
class Dense : public Layer {
public:
    size_t input_size;
    size_t output_size;

    Dense(size_t input_size, size_t output_size);
    Tensor forward(const Tensor& x, bool training = true) override;
    Tensor backward(const Tensor& dout) override;
};

} // namespace libAI
