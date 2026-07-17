#include <libAI/libAI.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>

using namespace libAI;

struct IrisSample {
    float sepal_length, sepal_width, petal_length, petal_width;
    int label;
};

std::vector<IrisSample> load_iris() {
    std::vector<IrisSample> samples;
    std::ifstream file("examples/iris.data");
    if (!file) {
        file.open("iris.data");
    }
    if (!file) {
        std::cerr << "Warning: iris.data not found, generating synthetic data" << std::endl;
        std::mt19937 gen(123);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (int i = 0; i < 150; i++) {
            IrisSample s;
            int cls = i / 50;
            s.sepal_length = 5.0f + cls * 2.0f + dist(gen) * 0.5f;
            s.sepal_width = 3.0f + dist(gen) * 0.5f;
            s.petal_length = 1.5f + cls * 3.0f + dist(gen) * 0.5f;
            s.petal_width = 0.2f + cls * 1.2f + dist(gen) * 0.3f;
            s.label = cls;
            samples.push_back(s);
        }
        return samples;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        IrisSample s;
        std::string token;
        std::getline(ss, token, ','); s.sepal_length = std::stof(token);
        std::getline(ss, token, ','); s.sepal_width = std::stof(token);
        std::getline(ss, token, ','); s.petal_length = std::stof(token);
        std::getline(ss, token, ','); s.petal_width = std::stof(token);
        std::getline(ss, token, ',');
        if (token.find("setosa") != std::string::npos) s.label = 0;
        else if (token.find("versicolor") != std::string::npos) s.label = 1;
        else s.label = 2;
        samples.push_back(s);
    }
    return samples;
}

int main() {
    std::cout << "Iris classification with libAI (C++)" << std::endl;

    auto data = load_iris();
    std::shuffle(data.begin(), data.end(), std::mt19937(42));

    size_t N = data.size();
    size_t split = N * 2 / 3;

    Tensor x_train({split, 4});
    Tensor y_train({split});
    Tensor x_val({N - split, 4});
    Tensor y_val({N - split});

    for (size_t i = 0; i < split; i++) {
        x_train(i, 0) = data[i].sepal_length;
        x_train(i, 1) = data[i].sepal_width;
        x_train(i, 2) = data[i].petal_length;
        x_train(i, 3) = data[i].petal_width;
        y_train(i) = static_cast<float>(data[i].label);
    }
    for (size_t i = split; i < N; i++) {
        x_val(i - split, 0) = data[i].sepal_length;
        x_val(i - split, 1) = data[i].sepal_width;
        x_val(i - split, 2) = data[i].petal_length;
        x_val(i - split, 3) = data[i].petal_width;
        y_val(i - split) = static_cast<float>(data[i].label);
    }

    NeuralNetwork model;
    model.add_layer<Dense>(4, 16);
    model.add_layer<ReLU>();
    model.add_layer<Dense>(16, 8);
    model.add_layer<ReLU>();
    model.add_layer<Dense>(8, 3);
    model.set_loss<SoftmaxCrossEntropy>();

    Adam optimizer(0.005f);
    Trainer trainer(&model, &optimizer);

    trainer.fit(x_train, y_train, x_val, y_val, 100, 16, true, 10);

    float train_acc = model.accuracy(x_train, y_train);
    float val_acc = model.accuracy(x_val, y_val);
    std::cout << "Train accuracy: " << train_acc << std::endl;
    std::cout << "Val accuracy: " << val_acc << std::endl;

    return 0;
}
