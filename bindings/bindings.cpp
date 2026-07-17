#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <libAI/libAI.hpp>
#include <sstream>
#include <cstddef>
#include <stdexcept>

namespace py = pybind11;
namespace la = libAI;

// Recursively flatten Python nested lists into Tensor data + shape
struct ListTensor {
    std::vector<size_t> shape;
    std::vector<float> data;
};

ListTensor flatten_list(py::handle obj, size_t depth = 0) {
    if (py::isinstance<py::float_>(obj) || py::isinstance<py::int_>(obj)) {
        return {{}, {obj.cast<float>()}};
    }
    if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
        auto iter = py::iter(obj);
        size_t count = 0;
        for (auto _ : iter) { (void)_; count++; }
        if (count == 0) {
            return {{0}, {}};
        }
        ListTensor result;
        result.shape.push_back(count);
        std::vector<size_t> sub_shape;
        iter = py::iter(obj);
        for (auto item : iter) {
            ListTensor sub = flatten_list(item, depth + 1);
            result.data.insert(result.data.end(), sub.data.begin(), sub.data.end());
            if (!sub.shape.empty()) {
                if (sub_shape.empty()) {
                    sub_shape = sub.shape;
                }
            }
        }
        result.shape.insert(result.shape.end(), sub_shape.begin(), sub_shape.end());
        return result;
    }
    throw std::runtime_error("Cannot convert Python object to Tensor");
}

la::Tensor list_to_tensor(py::handle obj) {
    if (py::isinstance<py::float_>(obj) || py::isinstance<py::int_>(obj)) {
        return la::Tensor({1}, obj.cast<float>());
    }
    ListTensor lt = flatten_list(obj);
    if (lt.shape.size() == 1 && lt.shape[0] == 1 && lt.data.size() > 1) {
        lt.shape = {static_cast<size_t>(lt.data.size())};
    }
    return la::Tensor(lt.shape, lt.data);
}

PYBIND11_MODULE(_libAI_core, m) {
    m.doc() = "libAI C++ backend";

    // ──── Tensor ────
    py::class_<la::Tensor>(m, "Tensor", "A multi-dimensional array for storing numerical data.")
        .def(py::init<const std::vector<size_t>&>(), "Create a new Tensor with the given shape")
        .def(py::init<const std::vector<size_t>&, float>(), "Create a new Tensor with the given shape and fill value")
        .def(py::init(&list_to_tensor), "Create a new Tensor from a Python list")
        .def_property_readonly("shape", [](const la::Tensor& t) -> py::list {
            py::list s;
            for (auto d : t.shape_) s.append(d);
            return s;
        }, "Get the shape of the Tensor")
        .def_property_readonly("data", [](const la::Tensor& t) -> py::list {
            std::function<py::list(const std::vector<size_t>&, size_t&, size_t)> build;
            build = [&](const std::vector<size_t>& shape, size_t& offset, size_t dim) -> py::list {
                py::list lst;
                for (size_t i = 0; i < shape[dim]; i++) {
                    if (dim == shape.size() - 1) {
                        lst.append(py::float_(t.data_[offset++]));
                    } else {
                        lst.append(build(shape, offset, dim + 1));
                    }
                }
                return lst;
            };
            if (t.shape_.empty()) return py::list();
            size_t offset = 0;
            return build(t.shape_, offset, 0);
        }, "Get the data of the Tensor as a nested list")
        .def_property_readonly("ndim", &la::Tensor::ndim)
        .def_property_readonly("size", &la::Tensor::size)
        .def("__repr__", &la::Tensor::to_string)
        .def("__len__", [](const la::Tensor& t) { return t.rows(); })
        .def("copy", &la::Tensor::copy)

        // Indexing
        .def("__getitem__", [](const la::Tensor& t, size_t i) { return t.at(i); })
        .def("__getitem__", [](const la::Tensor& t, py::tuple idx) {
            if (py::len(idx) == 1) return t.at(idx[0].cast<size_t>());
            return t.at(idx[0].cast<size_t>(), idx[1].cast<size_t>());
        })
        .def("__setitem__", [](la::Tensor& t, size_t i, float v) { t.at(i) = v; })
        .def("__setitem__", [](la::Tensor& t, py::tuple idx, float v) {
            if (py::len(idx) == 1) t.at(idx[0].cast<size_t>()) = v;
            else t.at(idx[0].cast<size_t>(), idx[1].cast<size_t>()) = v;
        })

        // Arithmetic
        .def("__add__", [](const la::Tensor& a, const la::Tensor& b) { return a.add(b); })
        .def("__sub__", [](const la::Tensor& a, const la::Tensor& b) { return a.sub(b); })
        .def("__mul__", [](const la::Tensor& a, const la::Tensor& b) { return a.mul(b); })
        .def("__truediv__", [](const la::Tensor& a, const la::Tensor& b) { return a.div(b); })
        .def("__add__", [](const la::Tensor& a, float s) { return a.add(s); })
        .def("__sub__", [](const la::Tensor& a, float s) { return a.sub(s); })
        .def("__mul__", [](const la::Tensor& a, float s) { return a.mul(s); })
        .def("__truediv__", [](const la::Tensor& a, float s) { return a.div(s); })
        .def("__radd__", [](const la::Tensor& a, float s) { return a.add(s); })
        .def("__rsub__", [](const la::Tensor& a, float s) { return a.mul(-1.0f).add(s); })
        .def("__rmul__", [](const la::Tensor& a, float s) { return a.mul(s); })
        .def("__neg__", [](const la::Tensor& a) { return a.neg(); })

        // Matrix ops
        .def("matmul", &la::Tensor::matmul, "Matrix multiplication of two tensors with compatible shapes using the @ operator")
        .def("transpose", &la::Tensor::transpose)
        .def_property_readonly("T", &la::Tensor::T)

        // Reductions
        .def("sum", py::overload_cast<>(&la::Tensor::sum, py::const_))
        .def("sum", [](const la::Tensor& t, int axis, bool keepdims) { return t.sum(axis, keepdims); },
             py::arg("axis"), py::arg("keepdims") = false)
        .def("mean", py::overload_cast<>(&la::Tensor::mean, py::const_))
        .def("mean", [](const la::Tensor& t, int axis, bool keepdims) { return t.mean(axis, keepdims); },
             py::arg("axis"), py::arg("keepdims") = false)
        .def("argmax", &la::Tensor::argmax, py::arg("axis") = 1)

        // Element-wise
        .def("exp", &la::Tensor::exp)
        .def("log", &la::Tensor::log)
        .def("sqrt", &la::Tensor::sqrt)
        .def("pow", &la::Tensor::pow)
        .def("clip", &la::Tensor::clip)
        .def("max", py::overload_cast<float>(&la::Tensor::max, py::const_))
        .def("reshape", &la::Tensor::reshape)

        // Factories
        .def_static("zeros", &la::Tensor::zeros)
        .def_static("ones", &la::Tensor::ones)
        .def_static("random_uniform", &la::Tensor::random_uniform,
            py::arg("shape"), py::arg("low") = 0.0f, py::arg("high") = 1.0f)
        .def_static("random_normal", &la::Tensor::random_normal,
            py::arg("shape"), py::arg("mean") = 0.0f, py::arg("stddev") = 1.0f);

    // ──── Layer base ────
    py::class_<la::Layer, std::unique_ptr<la::Layer, py::nodelete>>(m, "_Layer")
        .def("forward", &la::Layer::forward)
        .def("backward", &la::Layer::backward);

    py::class_<la::Dense, la::Layer, std::unique_ptr<la::Dense, py::nodelete>>(m, "Dense")
        .def(py::init<size_t, size_t>(), py::arg("input_size"), py::arg("output_size"))
        .def_readonly("input_size", &la::Dense::input_size)
        .def_readonly("output_size", &la::Dense::output_size);

    py::class_<la::Linear, la::Layer, std::unique_ptr<la::Linear, py::nodelete>>(m, "Linear")
        .def(py::init<>());

    py::class_<la::ReLU, la::Layer, std::unique_ptr<la::ReLU, py::nodelete>>(m, "ReLU")
        .def(py::init<>());

    py::class_<la::Sigmoid, la::Layer, std::unique_ptr<la::Sigmoid, py::nodelete>>(m, "Sigmoid")
        .def(py::init<>());

    py::class_<la::Tanh, la::Layer, std::unique_ptr<la::Tanh, py::nodelete>>(m, "Tanh")
        .def(py::init<>());

    py::class_<la::Dropout, la::Layer, std::unique_ptr<la::Dropout, py::nodelete>>(m, "Dropout")
        .def(py::init<float>(), py::arg("drop_rate") = 0.5f);

    py::class_<la::BatchNormalization, la::Layer, std::unique_ptr<la::BatchNormalization, py::nodelete>>(m, "BatchNormalization")
        .def(py::init<size_t, float, float>(),
             py::arg("num_features"), py::arg("momentum") = 0.9f, py::arg("eps") = 1e-5f);

    // ──── Losses ────
    py::class_<la::LossLayer, std::unique_ptr<la::LossLayer, py::nodelete>>(m, "_LossLayer")
        .def("forward", &la::LossLayer::forward)
        .def("backward", &la::LossLayer::backward);

    py::class_<la::MeanSquaredError, la::LossLayer, std::unique_ptr<la::MeanSquaredError, py::nodelete>>(m, "MeanSquaredError")
        .def(py::init<>());

    py::class_<la::SoftmaxCrossEntropy, la::LossLayer, std::unique_ptr<la::SoftmaxCrossEntropy, py::nodelete>>(m, "SoftmaxCrossEntropy")
        .def(py::init<>());

    // ──── Optimizers ────
    py::class_<la::Optimizer>(m, "_Optimizer")
        .def_readwrite("learning_rate", &la::Optimizer::learning_rate)
        .def("update", &la::Optimizer::update);

    py::class_<la::SGD, la::Optimizer>(m, "SGD")
        .def(py::init<float>(), py::arg("learning_rate") = 0.01f);

    py::class_<la::Momentum, la::Optimizer>(m, "Momentum")
        .def(py::init<float, float>(), py::arg("learning_rate") = 0.01f, py::arg("momentum") = 0.9f);

    py::class_<la::AdaGrad, la::Optimizer>(m, "AdaGrad")
        .def(py::init<float, float>(), py::arg("learning_rate") = 0.01f, py::arg("eps") = 1e-8f);

    py::class_<la::Adam, la::Optimizer>(m, "Adam")
        .def(py::init<float, float, float, float>(),
             py::arg("learning_rate") = 0.001f,
             py::arg("beta1") = 0.9f,
             py::arg("beta2") = 0.999f,
             py::arg("eps") = 1e-8f);

    // ──── NeuralNetwork ────
    py::class_<la::NeuralNetwork>(m, "NeuralNetwork")
        .def(py::init<>())
        .def("add_layer", [](la::NeuralNetwork& net, py::object layer_obj) {
            py::object cls = layer_obj.attr("__class__");
            std::string name = py::str(cls.attr("__name__"));
            if (name == "Dense") {
                auto& d = layer_obj.cast<la::Dense&>();
                net.add_layer<la::Dense>(d.input_size, d.output_size);
            } else if (name == "ReLU")          net.add_layer<la::ReLU>();
            else if (name == "Sigmoid")         net.add_layer<la::Sigmoid>();
            else if (name == "Tanh")            net.add_layer<la::Tanh>();
            else if (name == "Linear")          net.add_layer<la::Linear>();
            else if (name == "Dropout") {
                auto& d = layer_obj.cast<la::Dropout&>();
                net.add_layer<la::Dropout>(d.drop_rate);
            } else if (name == "BatchNormalization") {
                auto& b = layer_obj.cast<la::BatchNormalization&>();
                net.add_layer<la::BatchNormalization>(b.num_features, b.momentum, b.eps);
            } else {
                throw std::runtime_error("Unknown layer: " + name);
            }
        })
        .def("set_loss", [](la::NeuralNetwork& net, py::object loss_obj) {
            std::string name = py::str(loss_obj.attr("__class__").attr("__name__"));
            if (name == "MeanSquaredError")       net.set_loss<la::MeanSquaredError>();
            else if (name == "SoftmaxCrossEntropy") net.set_loss<la::SoftmaxCrossEntropy>();
            else throw std::runtime_error("Unknown loss: " + name);
        })
        .def("predict", &la::NeuralNetwork::predict, py::arg("x"), py::arg("training") = false)
        .def("compute_loss", &la::NeuralNetwork::compute_loss)
        .def("accuracy", &la::NeuralNetwork::accuracy)
        .def("gradient", [](la::NeuralNetwork& net) {
            auto result = net.gradient();
            py::list py_result;
            for (auto& pair : result) {
                py::dict params_dict, grads_dict;
                for (auto& [k, v] : *pair.first) {
                    params_dict[py::str(k)] = v;
                }
                for (auto& [k, v] : *pair.second) {
                    grads_dict[py::str(k)] = v;
                }
                py_result.append(py::make_tuple(params_dict, grads_dict));
            }
            return py_result;
        })
        .def("backward", [](la::NeuralNetwork& net) {
            net.gradient();
        })
        .def("update", [](la::NeuralNetwork& net, la::Optimizer& opt) {
            for (auto& layer_ptr : net.layers) {
                auto& layer = *layer_ptr;
                if (!layer.params.empty()) {
                    opt.update(layer.params, layer.grads);
                }
            }
        })
        .def("step", [](la::NeuralNetwork& net, la::Optimizer& opt,
                        const la::Tensor& x, const la::Tensor& y) {
            float loss = net.compute_loss(x, y, true);
            auto pg = net.gradient();
            for (auto& p : pg) {
                opt.update(*p.first, *p.second);
            }
            float acc = net.accuracy(x, y);
            return py::make_tuple(loss, acc);
        })
        .def("save", &la::NeuralNetwork::save)
        .def("load", &la::NeuralNetwork::load);

    // ──── Trainer ────
    py::class_<la::Trainer>(m, "Trainer")
        .def(py::init<la::NeuralNetwork*, la::Optimizer*>(),
             py::arg("network"), py::arg("optimizer"))
        .def("train_step", [](la::Trainer& t, la::Tensor& x, la::Tensor& y) {
            float loss, acc;
            t.train_step(x, y, loss, acc);
            return py::make_tuple(loss, acc);
        })
        .def("fit", [](la::Trainer& t,
                       la::Tensor& x_train, la::Tensor& y_train,
                       la::Tensor& x_val, la::Tensor& y_val,
                       int epochs, int batch_size, bool verbose, int print_every) {
            t.fit(x_train, y_train, x_val, y_val, epochs, batch_size, verbose, print_every);
            return py::make_tuple(
                t.train_loss_history, t.train_acc_history,
                t.val_loss_history, t.val_acc_history
            );
        },
            py::arg("x_train"), py::arg("y_train"),
            py::arg("x_val"), py::arg("y_val"),
            py::arg("epochs") = 100, py::arg("batch_size") = 32,
            py::arg("verbose") = true, py::arg("print_every") = 10)
        .def_readonly("train_loss_history", &la::Trainer::train_loss_history)
        .def_readonly("train_acc_history", &la::Trainer::train_acc_history)
        .def_readonly("val_loss_history", &la::Trainer::val_loss_history)
        .def_readonly("val_acc_history", &la::Trainer::val_acc_history);

    // ──── Tuning ────
    py::class_<la::HyperparameterTuner>(m, "HyperparameterTuner")
        .def(py::init<>())
        .def("grid_search", [](la::HyperparameterTuner& tuner,
                               la::Tensor& x_train, la::Tensor& y_train,
                               la::Tensor& x_val, la::Tensor& y_val,
                               py::dict param_grid,
                               py::function builder,
                               int epochs, int batch_size) {
            std::map<std::string, std::vector<float>> pg;
            for (auto item : param_grid) {
                std::string key = item.first.cast<std::string>();
                py::list vals = item.second.cast<py::list>();
                std::vector<float> v;
                for (auto val : vals) v.push_back(val.cast<float>());
                pg[key] = v;
            }
            auto result = tuner.grid_search(
                x_train, y_train, x_val, y_val, pg,
                [&](const std::map<std::string, float>& params) {
                    py::dict p;
                    for (auto& [k, v] : params) p[py::str(k)] = v;
                    auto result = builder(p);
                    auto pair = result.cast<py::tuple>();
                    la::NeuralNetwork* net = pair[0].cast<la::NeuralNetwork*>();
                    la::Optimizer* opt = pair[1].cast<la::Optimizer*>();
                    return std::make_pair(net, opt);
                },
                epochs, batch_size
            );
            py::dict best;
            for (auto& [k, v] : result) best[py::str(k)] = v;
            return best;
        },
            py::arg("x_train"), py::arg("y_train"),
            py::arg("x_val"), py::arg("y_val"),
            py::arg("param_grid"), py::arg("network_builder"),
            py::arg("epochs") = 50, py::arg("batch_size") = 32);
}
