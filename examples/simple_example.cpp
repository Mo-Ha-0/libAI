#include <libAI/libAI.hpp>
#include <iostream>
#include <cmath>

using namespace libAI;

int main() {
    std::cout << "Simple classification example with libAI (C++)" << std::endl;

    size_t num_samples = 300;
    size_t num_features = 2;
    size_t num_classes = 3;

    Tensor x_train({num_samples, num_features});
    Tensor y_train({num_samples});

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < num_samples; i++) {
        float angle = (i % num_classes) * 2.0f * 3.14159f / num_classes;
        float r = 0.5f + dist(gen) * 0.3f;
        x_train(i, 0) = r * std::cos(angle) + dist(gen) * 0.1f;
        x_train(i, 1) = r * std::sin(angle) + dist(gen) * 0.1f;
        y_train(i) = static_cast<float>(i % num_classes);
    }

    NeuralNetwork model;
    model.add_layer<Dense>(num_features, 64);
    model.add_layer<ReLU>();
    model.add_layer<Dense>(64, 64);
    model.add_layer<ReLU>();
    model.add_layer<Dense>(64, num_classes);
    model.set_loss<SoftmaxCrossEntropy>();

    Adam optimizer(0.001f);
    Trainer trainer(&model, &optimizer);

    trainer.fit(x_train, y_train, x_train, y_train, 200, 32, true, 20);

    float acc = model.accuracy(x_train, y_train);
    std::cout << "Final training accuracy: " << acc << std::endl;

    return 0;
}
