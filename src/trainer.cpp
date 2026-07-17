#include "../include/libAI/trainer.hpp"
#include <iostream>
#include <algorithm>
#include <random>
#include <vector>

namespace libAI {

Trainer::Trainer(NeuralNetwork* net, Optimizer* opt)
    : network(net), optimizer(opt) {}

void Trainer::train_step(const Tensor& x_batch, const Tensor& y_batch,
                         float& out_loss, float& out_acc) {
    out_loss = network->compute_loss(x_batch, y_batch, true);

    auto params_grads = network->gradient();
    for (auto& [params, grads] : params_grads) {
        optimizer->update(*params, *grads);
    }

    out_acc = network->accuracy(x_batch, y_batch);
}

void Trainer::fit(const Tensor& x_train, const Tensor& y_train,
                  const Tensor& x_val, const Tensor& y_val,
                  int epochs, int batch_size,
                  bool verbose, int print_every) {
    size_t N = x_train.rows();
    size_t D = x_train.cols();
    size_t num_batches = (N + batch_size - 1) / batch_size;

    std::vector<size_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 rng(42);

    for (int epoch = 0; epoch < epochs; epoch++) {
        std::shuffle(indices.begin(), indices.end(), rng);

        float epoch_loss = 0.0f;
        float epoch_acc = 0.0f;

        for (size_t b = 0; b < num_batches; b++) {
            size_t start = b * batch_size;
            size_t end = std::min(start + batch_size, N);
            size_t current_batch_size = end - start;

            Tensor x_batch({current_batch_size, D});
            Tensor y_batch;

            if (y_train.ndim() == 1) {
                y_batch = Tensor({current_batch_size});
            } else {
                y_batch = Tensor({current_batch_size, y_train.cols()});
            }

            for (size_t i = start; i < end; i++) {
                size_t idx = indices[i];
                std::memcpy(&x_batch.data_[(i - start) * D],
                           &x_train.data_[idx * D],
                           D * sizeof(float));
                if (y_train.ndim() == 1) {
                    y_batch.data_[i - start] = y_train.data_[idx];
                } else {
                    size_t y_cols = y_train.cols();
                    std::memcpy(&y_batch.data_[(i - start) * y_cols],
                               &y_train.data_[idx * y_cols],
                               y_cols * sizeof(float));
                }
            }

            float batch_loss, batch_acc;
            train_step(x_batch, y_batch, batch_loss, batch_acc);

            epoch_loss += batch_loss * static_cast<float>(current_batch_size) / static_cast<float>(N);
            epoch_acc += batch_acc * static_cast<float>(current_batch_size) / static_cast<float>(N);
        }

        train_loss_history.push_back(epoch_loss);
        train_acc_history.push_back(epoch_acc);

        if (x_val.size() > 0 && y_val.size() > 0) {
            float val_loss = network->compute_loss(x_val, y_val, false);
            float val_acc = network->accuracy(x_val, y_val);
            val_loss_history.push_back(val_loss);
            val_acc_history.push_back(val_acc);
        }

        if (verbose && (epoch % print_every == 0 || epoch == epochs - 1)) {
            std::cout << "Epoch " << epoch + 1 << "/" << epochs
                      << " - loss: " << epoch_loss
                      << " - acc: " << epoch_acc;
            if (!val_loss_history.empty()) {
                std::cout << " - val_loss: " << val_loss_history.back()
                          << " - val_acc: " << val_acc_history.back();
            }
            std::cout << std::endl;
        }
    }
}

} // namespace libAI
