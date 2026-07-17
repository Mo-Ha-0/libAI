#include "../include/libAI/network.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>

namespace libAI {

Tensor NeuralNetwork::predict(const Tensor& x, bool training) {
    Tensor out = x;
    for (auto& layer : layers) {
        out = layer->forward(out, training);
    }
    return out;
}

float NeuralNetwork::compute_loss(const Tensor& x, const Tensor& y, bool training) {
    Tensor pred = predict(x, training);
    return loss_layer->forward(pred, y, training);
}

float NeuralNetwork::accuracy(const Tensor& x, const Tensor& y) {
    Tensor pred = predict(x, false);
    Tensor pred_classes = pred.argmax(1);

    size_t N = pred.rows();
    size_t correct = 0;

    if (y.ndim() == 1) {
        for (size_t i = 0; i < N; i++) {
            if (std::abs(pred_classes.data_[i] - y.data_[i]) < 1e-6f) {
                correct++;
            }
        }
    } else if (y.ndim() == 2 && y.cols() == 1) {
        for (size_t i = 0; i < N; i++) {
            if (std::abs(pred_classes.data_[i] - y.data_[i]) < 1e-6f) {
                correct++;
            }
        }
    } else {
        Tensor y_classes = y.argmax(1);
        for (size_t i = 0; i < N; i++) {
            if (std::abs(pred_classes.data_[i] - y_classes.data_[i]) < 1e-6f) {
                correct++;
            }
        }
    }

    return static_cast<float>(correct) / static_cast<float>(N);
}

std::vector<std::pair<std::unordered_map<std::string, Tensor>*,
                      const std::unordered_map<std::string, Tensor>*>>
NeuralNetwork::gradient() {
    loss_layer->backward(1.0f);

    Tensor dout = loss_layer->backward(1.0f);

    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; i--) {
        dout = layers[i]->backward(dout);
    }

    std::vector<std::pair<std::unordered_map<std::string, Tensor>*,
                          const std::unordered_map<std::string, Tensor>*>> result;
    for (auto& layer : layers) {
        if (!layer->params.empty()) {
            result.push_back({&layer->params, &layer->grads});
        }
    }
    return result;
}

void NeuralNetwork::save(const std::string& filepath) const {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot save to file: " + filepath);
    }

    auto write_tensor = [&](const Tensor& t) {
        size_t ndim = t.ndim();
        file.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
        for (size_t i = 0; i < ndim; i++) {
            size_t dim = t.shape_[i];
            file.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        }
        size_t total = t.size();
        file.write(reinterpret_cast<const char*>(t.data_.data()), total * sizeof(float));
    };

    size_t num_layers = layers.size();
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));

    for (auto& layer : layers) {
        size_t num_params = layer->params.size();
        file.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));
        for (auto& [key, tensor] : layer->params) {
            size_t key_len = key.size();
            file.write(reinterpret_cast<const char*>(&key_len), sizeof(key_len));
            file.write(key.data(), key_len);
            write_tensor(tensor);
        }
    }
}

void NeuralNetwork::load(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot load from file: " + filepath);
    }

    auto read_tensor = [&]() -> Tensor {
        size_t ndim;
        file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
        std::vector<size_t> shape(ndim);
        for (size_t i = 0; i < ndim; i++) {
            file.read(reinterpret_cast<char*>(&shape[i]), sizeof(shape[i]));
        }
        size_t total = 1;
        for (auto s : shape) total *= s;
        std::vector<float> data(total);
        file.read(reinterpret_cast<char*>(data.data()), total * sizeof(float));
        return Tensor(shape, data);
    };

    size_t num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));

    for (size_t l = 0; l < num_layers; l++) {
        size_t num_params;
        file.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));
        for (size_t p = 0; p < num_params; p++) {
            size_t key_len;
            file.read(reinterpret_cast<char*>(&key_len), sizeof(key_len));
            std::string key(key_len, '\0');
            file.read(&key[0], key_len);
            Tensor t = read_tensor();
            if (l < layers.size()) {
                auto it = layers[l]->params.find(key);
                if (it != layers[l]->params.end()) {
                    it->second = t;
                }
            }
        }
    }
}

} // namespace libAI
