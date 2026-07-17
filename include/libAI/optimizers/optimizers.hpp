#pragma once
#include "../core/tensor.hpp"
#include <unordered_map>
#include <string>
#include <memory>

namespace libAI {

class Optimizer {
public:
    float learning_rate;
    explicit Optimizer(float lr) : learning_rate(lr) {}
    virtual ~Optimizer() = default;
    virtual void update(std::unordered_map<std::string, Tensor>& params,
                        const std::unordered_map<std::string, Tensor>& grads) = 0;
};

class SGD : public Optimizer {
public:
    explicit SGD(float learning_rate = 0.01f);
    void update(std::unordered_map<std::string, Tensor>& params,
                const std::unordered_map<std::string, Tensor>& grads) override;
};

inline std::string opt_key(const std::unordered_map<std::string, Tensor>& params, const std::string& name) {
    return std::to_string(reinterpret_cast<uintptr_t>(&params)) + "." + name;
}

class Momentum : public Optimizer {
public:
    float momentum_coeff;
    std::unordered_map<std::string, std::unordered_map<std::string, Tensor>> velocities;

    Momentum(float learning_rate = 0.01f, float momentum = 0.9f);
    void update(std::unordered_map<std::string, Tensor>& params,
                const std::unordered_map<std::string, Tensor>& grads) override;
};

class AdaGrad : public Optimizer {
public:
    float eps;
    std::unordered_map<std::string, std::unordered_map<std::string, Tensor>> cache_;

    AdaGrad(float learning_rate = 0.01f, float eps = 1e-8f);
    void update(std::unordered_map<std::string, Tensor>& params,
                const std::unordered_map<std::string, Tensor>& grads) override;
};

class Adam : public Optimizer {
public:
    float beta1, beta2, eps;
    int t;
    std::unordered_map<std::string, std::unordered_map<std::string, Tensor>> m;
    std::unordered_map<std::string, std::unordered_map<std::string, Tensor>> v;

    Adam(float learning_rate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);
    void update(std::unordered_map<std::string, Tensor>& params,
                const std::unordered_map<std::string, Tensor>& grads) override;
};

} // namespace libAI
