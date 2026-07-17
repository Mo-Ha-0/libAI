#include "../../include/libAI/layers/activations.hpp"
#include <cmath>

namespace libAI {

Tensor Linear::forward(const Tensor& x, bool training) {
    return x;
}

Tensor Linear::backward(const Tensor& dout) {
    return dout;
}

Tensor ReLU::forward(const Tensor& x, bool training) {
    cache["x"] = x;
    return x.max(0.0f);
}

Tensor ReLU::backward(const Tensor& dout) {
    const Tensor& x = cache["x"];
    Tensor mask = x.gt(0.0f);
    return dout.mul(mask);
}

Tensor Sigmoid::forward(const Tensor& x, bool training) {
    Tensor clipped = x.clip(-500.0f, 500.0f);
    Tensor out = clipped.neg().exp().add(1.0f).pow(-1.0f);
    cache["out"] = out;
    return out;
}

Tensor Sigmoid::backward(const Tensor& dout) {
    const Tensor& out = cache["out"];
    return dout.mul(out).mul(out.neg().add(1.0f));
}

Tensor Tanh::forward(const Tensor& x, bool training) {
    Tensor out = x.mul(2.0f).clip(-500.0f, 500.0f).exp().sub(1.0f).div(
        x.mul(2.0f).clip(-500.0f, 500.0f).exp().add(1.0f)
    );
    cache["out"] = out;
    return out;
}

Tensor Tanh::backward(const Tensor& dout) {
    const Tensor& out = cache["out"];
    return dout.mul(out.pow(2.0f).neg().add(1.0f));
}

} // namespace libAI
