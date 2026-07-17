#include "../../include/libAI/layers/losses.hpp"
#include <cmath>
#include <algorithm>

namespace libAI {

float MeanSquaredError::forward(const Tensor& x, const Tensor& y, bool training) {
    Tensor diff = x.sub(y);
    Tensor squared = diff.pow(2.0f);
    Tensor row_sums = squared.sum(1);
    float loss = row_sums.mean();
    cache["x"] = x;
    cache["y"] = y;
    return loss;
}

Tensor MeanSquaredError::backward(float dout) {
    const Tensor& x = cache["x"];
    const Tensor& y = cache["y"];
    float N = static_cast<float>(x.rows());
    return x.sub(y).mul(2.0f / N);
}

float SoftmaxCrossEntropy::forward(const Tensor& x, const Tensor& y, bool training) {
    size_t N = x.rows();
    size_t C = x.cols();

    Tensor max_x = Tensor({N, 1});
    for (size_t i = 0; i < N; i++) {
        float mx = x.data_[i * C];
        for (size_t j = 1; j < C; j++) {
            mx = std::max(mx, x.data_[i * C + j]);
        }
        max_x.data_[i] = mx;
    }

    Tensor exps(x.shape_);
    Tensor sum_exps({N, 1}, 0.0f);
    #pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
        float row_sum = 0.0f;
        size_t base = i * C;
        for (size_t j = 0; j < C; j++) {
            float val = std::exp(x.data_[base + j] - max_x.data_[i]);
            exps.data_[base + j] = val;
            row_sum += val;
        }
        sum_exps.data_[i] = row_sum;
    }

    Tensor probs(x.shape_);
    #pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
        size_t base = i * C;
        float inv_sum = 1.0f / (sum_exps.data_[i] + 1e-10f);
        for (size_t j = 0; j < C; j++) {
            probs.data_[base + j] = exps.data_[base + j] * inv_sum;
        }
    }

    float loss = 0.0f;
    bool label_mode = (y.ndim() == 1 || (y.ndim() == 2 && y.cols() == 1));
    if (label_mode) {
        #pragma omp parallel for reduction(+:loss)
        for (size_t i = 0; i < N; i++) {
            int label = static_cast<int>(y.data_[i * y.cols()]);
            loss += -std::log(probs.data_[i * C + label] + 1e-10f);
        }
    } else {
        #pragma omp parallel for reduction(+:loss)
        for (size_t i = 0; i < N; i++) {
            size_t base = i * C;
            for (size_t j = 0; j < C; j++) {
                loss += -y.data_[base + j] * std::log(probs.data_[base + j] + 1e-10f);
            }
        }
    }
    loss /= static_cast<float>(N);

    cache["probs"] = probs;
    cache["y"] = y;
    return loss;
}

Tensor SoftmaxCrossEntropy::backward(float dout) {
    const Tensor& probs = cache["probs"];
    const Tensor& y = cache["y"];
    size_t N = probs.rows();
    size_t C = probs.cols();

    Tensor dx(probs.shape_);
    bool label_mode = (y.ndim() == 1 || (y.ndim() == 2 && y.cols() == 1));
    if (label_mode) {
        for (size_t i = 0; i < N; i++) {
            int label = static_cast<int>(y.data_[i * y.cols()]);
            size_t base = i * C;
            for (size_t j = 0; j < C; j++) {
                dx.data_[base + j] = probs.data_[base + j];
            }
            dx.data_[base + label] -= 1.0f;
        }
    } else {
        for (size_t i = 0; i < N; i++) {
            size_t base = i * C;
            for (size_t j = 0; j < C; j++) {
                dx.data_[base + j] = probs.data_[base + j] - y.data_[base + j];
            }
        }
    }
    return dx.div(static_cast<float>(N));
}

} // namespace libAI
