#include "../../include/libAI/layers/regularization.hpp"
#include <cmath>
#include <random>

namespace libAI {

Dropout::Dropout(float drop_rate) : drop_rate(drop_rate) {}

Tensor Dropout::forward(const Tensor& x, bool training) {
    if (!training) {
        return x;
    }
    size_t N = x.size();
    Tensor mask(x.shape_);
    float scale = 1.0f / (1.0f - drop_rate);

    #pragma omp parallel
    {
        unsigned tid = static_cast<unsigned>(omp_get_thread_num());
        std::mt19937 gen(static_cast<unsigned>(std::random_device{}()) ^ tid);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        #pragma omp for
        for (size_t i = 0; i < N; i++) {
            mask.data_[i] = (dist(gen) > drop_rate) ? scale : 0.0f;
        }
    }

    cache["mask"] = mask;
    return x.mul(mask);
}

Tensor Dropout::backward(const Tensor& dout) {
    const Tensor& mask = cache["mask"];
    return dout.mul(mask);
}

BatchNormalization::BatchNormalization(size_t num_features, float momentum, float eps)
    : num_features(num_features), momentum(momentum), eps(eps)
{
    params["gamma"] = Tensor::ones({num_features});
    params["beta"] = Tensor::zeros({num_features});
    grads["gamma"] = Tensor::zeros({num_features});
    grads["beta"] = Tensor::zeros({num_features});
    running_mean = Tensor::zeros({num_features});
    running_var = Tensor::ones({num_features});
}

Tensor BatchNormalization::forward(const Tensor& x, bool training) {
    size_t N = x.rows();
    size_t D = x.cols();

    if (!training) {
        Tensor x_norm = x.sub(running_mean).div(running_var.sqrt());
        return x_norm.mul(params["gamma"]).add(params["beta"]);
    }

    Tensor batch_mean = x.mean(0);
    Tensor x_centered = x.sub(batch_mean);
    Tensor batch_var = x_centered.pow(2.0f).mean(0);

    running_mean = running_mean.mul(momentum).add(batch_mean.mul(1.0f - momentum));
    running_var = running_var.mul(momentum).add(batch_var.mul(1.0f - momentum));

    Tensor x_norm = x_centered.div(batch_var.add(eps).sqrt());
    Tensor out = x_norm.mul(params["gamma"]).add(params["beta"]);

    cache["x_norm"] = x_norm;
    cache["x_centered"] = x_centered;
    cache["batch_var"] = batch_var;
    cache["gamma"] = params["gamma"];

    return out;
}

Tensor BatchNormalization::backward(const Tensor& dout) {
    size_t N = static_cast<float>(dout.rows());
    const Tensor& x_norm = cache["x_norm"];
    const Tensor& x_centered = cache["x_centered"];
    const Tensor& batch_var = cache["batch_var"];
    const Tensor& gamma = cache["gamma"];

    grads["gamma"] = dout.mul(x_norm).sum(0);
    grads["beta"] = dout.sum(0);

    Tensor dx_norm = dout.mul(gamma);
    Tensor var_eps = batch_var.add(eps);
    Tensor inv_std = var_eps.pow(-0.5f);
    Tensor inv_var = var_eps.pow(-1.0f);

    Tensor dx1 = dx_norm;
    Tensor dx2 = dx_norm.sum(0).div(N * -1.0f);
    Tensor dx3 = x_centered.mul(
        dx_norm.mul(x_centered).sum(0).mul(inv_var).mul(-1.0f / N)
    );

    return dx1.add(dx2).add(dx3).mul(inv_std);
}

} // namespace libAI
