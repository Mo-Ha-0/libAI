#include "../../include/libAI/layers/dense.hpp"
#include <cmath>

namespace libAI {

Dense::Dense(size_t input_size, size_t output_size)
    : input_size(input_size), output_size(output_size)
{
    float scale = std::sqrt(2.0f / static_cast<float>(input_size));
    params["W"] = Tensor::random_normal({input_size, output_size}, 0.0f, scale);
    params["b"] = Tensor::zeros({output_size});
    grads["W"] = Tensor::zeros({input_size, output_size});
    grads["b"] = Tensor::zeros({output_size});
}

Tensor Dense::forward(const Tensor& x, bool training) {
    cache["x"] = x;
    Tensor out = x.matmul(params["W"]).add(params["b"]);
    return out;
}

Tensor Dense::backward(const Tensor& dout) {
    const Tensor& x = cache["x"];
    grads["W"] = x.T().matmul(dout);
    grads["b"] = dout.sum(0);
    Tensor dx = dout.matmul(params["W"].T());
    return dx;
}

} // namespace libAI
