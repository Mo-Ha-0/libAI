#include "../../include/libAI/optimizers/optimizers.hpp"
#include <cmath>
#include <string>

namespace libAI {

SGD::SGD(float learning_rate) : Optimizer(learning_rate) {}

void SGD::update(std::unordered_map<std::string, Tensor>& params,
                 const std::unordered_map<std::string, Tensor>& grads) {
    for (auto& [key, param] : params) {
        auto it = grads.find(key);
        if (it != grads.end()) {
            param = param.sub(it->second.mul(learning_rate));
        }
    }
}

Momentum::Momentum(float learning_rate, float momentum)
    : Optimizer(learning_rate), momentum_coeff(momentum) {}

void Momentum::update(std::unordered_map<std::string, Tensor>& params,
                      const std::unordered_map<std::string, Tensor>& grads) {
    for (auto& [key, param] : params) {
        auto it = grads.find(key);
        if (it == grads.end()) continue;
        std::string k = opt_key(params, key);

        if (velocities.find(k) == velocities.end()) {
            velocities[k] = std::unordered_map<std::string, Tensor>();
        }

        Tensor grad = it->second;

        if (velocities[k].find("v") == velocities[k].end()) {
            velocities[k]["v"] = Tensor(grad.shape_, 0.0f);
        }

        Tensor& v = velocities[k]["v"];
        v = v.mul(momentum_coeff).sub(grad.mul(learning_rate));
        param = param.add(v);
    }
}

AdaGrad::AdaGrad(float learning_rate, float eps)
    : Optimizer(learning_rate), eps(eps) {}

void AdaGrad::update(std::unordered_map<std::string, Tensor>& params,
                     const std::unordered_map<std::string, Tensor>& grads) {
    for (auto& [key, param] : params) {
        auto it = grads.find(key);
        if (it == grads.end()) continue;
        std::string k = opt_key(params, key);

        if (cache_.find(k) == cache_.end()) {
            cache_[k] = std::unordered_map<std::string, Tensor>();
        }

        Tensor grad = it->second;

        if (cache_[k].find("h") == cache_[k].end()) {
            cache_[k]["h"] = Tensor(grad.shape_, 0.0f);
        }

        Tensor& h = cache_[k]["h"];
        h = h.add(grad.mul(grad));

        param = param.sub(grad.mul(learning_rate).div(h.sqrt().add(eps)));
    }
}

Adam::Adam(float learning_rate, float beta1, float beta2, float eps)
    : Optimizer(learning_rate), beta1(beta1), beta2(beta2), eps(eps), t(0) {}

void Adam::update(std::unordered_map<std::string, Tensor>& params,
                  const std::unordered_map<std::string, Tensor>& grads) {
    t++;

    for (auto& [key, param] : params) {
        auto it = grads.find(key);
        if (it == grads.end()) continue;
        std::string k = opt_key(params, key);

        if (m.find(k) == m.end()) {
            m[k] = std::unordered_map<std::string, Tensor>();
            v[k] = std::unordered_map<std::string, Tensor>();
        }

        Tensor grad = it->second;

        if (m[k].find("m") == m[k].end()) {
            m[k]["m"] = Tensor(grad.shape_, 0.0f);
            v[k]["v"] = Tensor(grad.shape_, 0.0f);
        }

        Tensor& m_t = m[k]["m"];
        Tensor& v_t = v[k]["v"];

        m_t = m_t.mul(beta1).add(grad.mul(1.0f - beta1));
        v_t = v_t.mul(beta2).add(grad.mul(grad).mul(1.0f - beta2));

        float m_hat_denom = 1.0f - std::pow(beta1, static_cast<float>(t));
        float v_hat_denom = 1.0f - std::pow(beta2, static_cast<float>(t));

        Tensor m_hat = m_t.div(m_hat_denom);
        Tensor v_hat = v_t.div(v_hat_denom);

        param = param.sub(m_hat.mul(learning_rate).div(v_hat.sqrt().add(eps)));
    }
}

} // namespace libAI
