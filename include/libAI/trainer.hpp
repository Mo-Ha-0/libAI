#pragma once
#include "network.hpp"
#include "optimizers/optimizers.hpp"
#include <memory>
#include <vector>

namespace libAI {

class Trainer {
public:
    NeuralNetwork* network;
    Optimizer* optimizer;

    std::vector<float> train_loss_history;
    std::vector<float> train_acc_history;
    std::vector<float> val_loss_history;
    std::vector<float> val_acc_history;

    Trainer(NeuralNetwork* net, Optimizer* opt);

    void train_step(const Tensor& x_batch, const Tensor& y_batch,
                    float& out_loss, float& out_acc);

    void fit(const Tensor& x_train, const Tensor& y_train,
             const Tensor& x_val, const Tensor& y_val,
             int epochs = 100, int batch_size = 32,
             bool verbose = true, int print_every = 10);
};

} // namespace libAI
