#pragma once
#include <vector>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <random>
#include <string>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <iostream>

namespace libAI {

class Tensor {
public:
    std::vector<size_t> shape_;
    std::vector<float> data_;

    Tensor() : shape_{0}, data_(0) {}

    Tensor(const std::vector<size_t>& shape)
        : shape_(shape)
    {
        size_t total = 1;
        for (auto s : shape_) total *= s;
        if (shape_.empty()) shape_ = {0};
        data_.resize(total > 0 ? total : 0, 0.0f);
    }

    Tensor(const std::vector<size_t>& shape, float init_val)
        : shape_(shape)
    {
        size_t total = 1;
        for (auto s : shape_) total *= s;
        if (shape_.empty()) shape_ = {0};
        data_.resize(total > 0 ? total : 0, init_val);
    }

    Tensor(const std::vector<size_t>& shape, const std::vector<float>& data)
        : shape_(shape), data_(data)
    {
        size_t total = 1;
        for (auto s : shape_) total *= s;
        if (total != data_.size()) {
            throw std::runtime_error("Tensor: data size mismatch");
        }
    }
    /**
     * Returns the number of dimensions of the tensor.
     */
    size_t ndim() const { return shape_.size(); }
    /**
     * Returns the total number of elements in the tensor.
     */
    size_t size() const { return data_.size(); }
    const std::vector<size_t>& shape() const { return shape_; }
    /**
     * Returns the number of rows (first dimension) of the tensor. If the tensor is empty, returns 0.
     */
    size_t rows() const {
        if (shape_.empty()) return 0;
        return shape_[0];
    }
    /**
     * Returns the number of columns (second dimension) of the tensor. If the tensor has less than 2 dimensions, returns 1.
     */
    size_t cols() const {
        if (shape_.size() < 2) return 1;
        return shape_[1];
    }
    /**
     * Returns a reference to the element at the given index. Throws an exception if the index is out of bounds.
     */
    float& at(size_t i) {
        return data_[i];
    }
    /**
     * Returns a const reference to the element at the given index. Throws an exception if the index is out of bounds.
     */
    const float& at(size_t i) const {
        return data_[i];
    }
    /**
     * Returns a reference to the element at the given row and column indices. Throws an exception if the indices are out of bounds.
     */
    float& at(size_t i, size_t j) {
        return data_[i * cols() + j];
    }
    /**
     * Returns a const reference to the element at the given row and column indices. Throws an exception if the indices are out of bounds.
     */
    const float& at(size_t i, size_t j) const {
        return data_[i * cols() + j];
    }

    /**
     * Returns a reference to the element at the given index. Throws an exception if the index is out of bounds.
     */
    float& operator()(size_t i) { return data_[i]; }
    const float& operator()(size_t i) const { return data_[i]; }
    /**
     * Returns a reference to the element at the given row and column indices. Throws an exception if the indices are out of bounds.
     */
    float& operator()(size_t i, size_t j) { return data_[i * cols() + j]; }
    const float& operator()(size_t i, size_t j) const { return data_[i * cols() + j]; }

    /**
     * Returns a pointer to the underlying data array.
     */
    float* data_ptr() { return data_.data(); }
    const float* data_ptr() const { return data_.data(); }

    static Tensor zeros(const std::vector<size_t>& shape) {
        return Tensor(shape, 0.0f);
    }

    static Tensor ones(const std::vector<size_t>& shape) {
        return Tensor(shape, 1.0f);
    }

    static Tensor random_uniform(const std::vector<size_t>& shape, float low = 0.0f, float high = 1.0f) {
        Tensor result(shape);
        #pragma omp parallel
        {
            unsigned tid = static_cast<unsigned>(omp_get_thread_num());
            std::mt19937 gen(static_cast<unsigned>(std::random_device{}()) ^ tid);
            std::uniform_real_distribution<float> dist(low, high);
            #pragma omp for
            for (size_t i = 0; i < result.data_.size(); i++) {
                result.data_[i] = dist(gen);
            }
        }
        return result;
    }

    static Tensor random_normal(const std::vector<size_t>& shape, float mean = 0.0f, float stddev = 1.0f) {
        Tensor result(shape);
        #pragma omp parallel
        {
            unsigned tid = static_cast<unsigned>(omp_get_thread_num());
            std::mt19937 gen(static_cast<unsigned>(std::random_device{}()) ^ tid);
            std::normal_distribution<float> dist(mean, stddev);
            #pragma omp for
            for (size_t i = 0; i < result.data_.size(); i++) {
                result.data_[i] = dist(gen);
            }
        }
        return result;
    }

    Tensor copy() const {
        return Tensor(shape_, data_);
    }

    Tensor matmul(const Tensor& other) const {
        size_t M = rows();
        size_t K = cols();
        size_t N = other.cols();

        if (other.rows() != K) {
            throw std::runtime_error("matmul: dimension mismatch");
        }

        Tensor result({M, N}, 0.0f);
        const float* a = data_.data();
        const float* b = other.data_.data();
        float* c = result.data_.data();

        #pragma omp parallel for
        for (size_t i = 0; i < M; i++) {
            for (size_t k = 0; k < K; k++) {
                float aik = a[i * K + k];
                size_t row_offset = i * N;
                size_t b_offset = k * N;
                for (size_t j = 0; j < N; j++) {
                    c[row_offset + j] += aik * b[b_offset + j];
                }
            }
        }
        return result;
    }

    Tensor transpose() const {
        size_t M = rows();
        size_t N = cols();
        Tensor result({N, M});
        const float* src = data_.data();
        float* dst = result.data_.data();

        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                dst[j * M + i] = src[i * N + j];
            }
        }
        return result;
    }

    Tensor T() const { return transpose(); }

    template<typename Func>
    Tensor elementwise(Func op) const {
        Tensor result(shape_);
        const float* src = data_.data();
        float* dst = result.data_.data();

        #pragma omp parallel for
        for (size_t i = 0; i < data_.size(); i++) {
            dst[i] = op(src[i]);
        }
        return result;
    }

    Tensor neg() const {
        return elementwise([](float x) { return -x; });
    }

    Tensor abs() const {
        return elementwise([](float x) { return std::fabs(x); });
    }

    Tensor exp() const {
        return elementwise([](float x) { return std::exp(x); });
    }

    Tensor log() const {
        return elementwise([](float x) { return std::log(x + 1e-10f); });
    }

    Tensor sqrt() const {
        return elementwise([](float x) { return std::sqrt(x + 1e-10f); });
    }

    Tensor clip(float low, float high) const {
        return elementwise([low, high](float x) {
            return x < low ? low : (x > high ? high : x);
        });
    }

    Tensor pow(float exponent) const {
        return elementwise([exponent](float x) { return std::pow(x, exponent); });
    }

    Tensor add(const Tensor& other) const {
        if (shape_ == other.shape_) {
            Tensor result(shape_);
            const float* a = data_.data();
            const float* b = other.data_.data();
            float* dst = result.data_.data();
            #pragma omp parallel for
            for (size_t i = 0; i < data_.size(); i++) {
                dst[i] = a[i] + b[i];
            }
            return result;
        }
        if (ndim() == 2 && other.ndim() == 1 && cols() == other.size()) {
            size_t M = rows(), N = cols();
            Tensor result(shape_);
            const float* a_data = data_.data();
            const float* b_data = other.data_.data();
            float* dst = result.data_.data();
            #pragma omp parallel for
            for (size_t i = 0; i < M; i++) {
                size_t base = i * N;
                for (size_t j = 0; j < N; j++) {
                    dst[base + j] = a_data[base + j] + b_data[j];
                }
            }
            return result;
        }
        throw std::runtime_error("Tensor::add: shape mismatch");
    }

    Tensor sub(const Tensor& other) const {
        if (shape_ == other.shape_) {
            Tensor result(shape_);
            const float* a = data_.data();
            const float* b = other.data_.data();
            float* dst = result.data_.data();
            #pragma omp parallel for
            for (size_t i = 0; i < data_.size(); i++) {
                dst[i] = a[i] - b[i];
            }
            return result;
        }
        if (ndim() == 2 && other.ndim() == 1 && cols() == other.size()) {
            size_t M = rows(), N = cols();
            Tensor result(shape_);
            const float* a_data = data_.data();
            const float* b_data = other.data_.data();
            float* dst = result.data_.data();
            #pragma omp parallel for
            for (size_t i = 0; i < M; i++) {
                size_t base = i * N;
                for (size_t j = 0; j < N; j++) {
                    dst[base + j] = a_data[base + j] - b_data[j];
                }
            }
            return result;
        }
        throw std::runtime_error("Tensor::sub: shape mismatch");
    }

    Tensor mul(const Tensor& other) const {
        if (shape_ == other.shape_) {
            Tensor result(shape_);
            const float* a = data_.data();
            const float* b = other.data_.data();
            float* dst = result.data_.data();
            #pragma omp parallel for
            for (size_t i = 0; i < data_.size(); i++) {
                dst[i] = a[i] * b[i];
            }
            return result;
        }
        if (ndim() == 2 && other.ndim() == 1 && cols() == other.size()) {
            size_t M = rows(), N = cols();
            Tensor result(shape_);
            const float* a_data = data_.data();
            const float* b_data = other.data_.data();
            float* dst = result.data_.data();
            #pragma omp parallel for
            for (size_t i = 0; i < M; i++) {
                size_t base = i * N;
                for (size_t j = 0; j < N; j++) {
                    dst[base + j] = a_data[base + j] * b_data[j];
                }
            }
            return result;
        }
        throw std::runtime_error("Tensor::mul: shape mismatch");
    }

    Tensor div(const Tensor& other) const {
        if (shape_ == other.shape_) {
            Tensor result(shape_);
            const float* a = data_.data();
            const float* b = other.data_.data();
            float* dst = result.data_.data();
            #pragma omp parallel for
            for (size_t i = 0; i < data_.size(); i++) {
                dst[i] = a[i] / (b[i] + 1e-10f);
            }
            return result;
        }
        if (ndim() == 2 && other.ndim() == 1 && cols() == other.size()) {
            size_t M = rows(), N = cols();
            Tensor result(shape_);
            const float* a_data = data_.data();
            const float* b_data = other.data_.data();
            float* dst = result.data_.data();
            #pragma omp parallel for
            for (size_t i = 0; i < M; i++) {
                size_t base = i * N;
                for (size_t j = 0; j < N; j++) {
                    dst[base + j] = a_data[base + j] / (b_data[j] + 1e-10f);
                }
            }
            return result;
        }
        throw std::runtime_error("Tensor::div: shape mismatch");
    }

    Tensor add(float scalar) const {
        return elementwise([scalar](float x) { return x + scalar; });
    }

    Tensor sub(float scalar) const {
        return elementwise([scalar](float x) { return x - scalar; });
    }

    Tensor mul(float scalar) const {
        return elementwise([scalar](float x) { return x * scalar; });
    }

    Tensor div(float scalar) const {
        return elementwise([scalar](float x) { return x / scalar; });
    }

    Tensor max(float scalar) const {
        return elementwise([scalar](float x) { return x > scalar ? x : scalar; });
    }

    Tensor min(float scalar) const {
        return elementwise([scalar](float x) { return x < scalar ? x : scalar; });
    }

    Tensor gt(float scalar) const {
        return elementwise([scalar](float x) { return x > scalar ? 1.0f : 0.0f; });
    }

    Tensor lt(float scalar) const {
        return elementwise([scalar](float x) { return x < scalar ? 1.0f : 0.0f; });
    }

    Tensor eq(const Tensor& other) const {
        if (shape_ != other.shape_) {
            throw std::runtime_error("Tensor::eq: shape mismatch");
        }
        Tensor result(shape_);
        const float* a = data_.data();
        const float* b = other.data_.data();
        float* dst = result.data_.data();
        #pragma omp parallel for
        for (size_t i = 0; i < data_.size(); i++) {
            dst[i] = (std::fabs(a[i] - b[i]) < 1e-6f) ? 1.0f : 0.0f;
        }
        return result;
    }

    Tensor operator+(const Tensor& other) const { return add(other); }
    Tensor operator-(const Tensor& other) const { return sub(other); }
    Tensor operator*(const Tensor& other) const { return mul(other); }
    Tensor operator/(const Tensor& other) const { return div(other); }
    Tensor operator+(float scalar) const { return add(scalar); }
    Tensor operator-(float scalar) const { return sub(scalar); }
    Tensor operator*(float scalar) const { return mul(scalar); }
    Tensor operator/(float scalar) const { return div(scalar); }

    float sum() const {
        float total = 0.0f;
        #pragma omp parallel for reduction(+:total)
        for (size_t i = 0; i < data_.size(); i++) {
            total += data_[i];
        }
        return total;
    }

    Tensor sum(int axis, bool keepdims = false) const {
        if (ndim() == 1) {
            float s = sum();
            if (keepdims) return Tensor({1}, s);
            return Tensor({1}, s);
        }
        if (axis == 0) {
            size_t M = rows(), N = cols();
            std::vector<size_t> out_shape = keepdims ? std::vector<size_t>{1, N} : std::vector<size_t>{N};
            Tensor result(out_shape, 0.0f);
            #pragma omp parallel for
            for (size_t j = 0; j < N; j++) {
                float s = 0.0f;
                for (size_t i = 0; i < M; i++) {
                    s += data_[i * N + j];
                }
                result.data_[j] = s;
            }
            return result;
        } else {
            size_t M = rows(), N = cols();
            std::vector<size_t> out_shape = keepdims ? std::vector<size_t>{M, 1} : std::vector<size_t>{M};
            Tensor result(out_shape, 0.0f);
            #pragma omp parallel for
            for (size_t i = 0; i < M; i++) {
                float s = 0.0f;
                size_t base = i * N;
                for (size_t j = 0; j < N; j++) {
                    s += data_[base + j];
                }
                result.data_[i] = s;
            }
            return result;
        }
    }

    float mean() const {
        return sum() / static_cast<float>(data_.size());
    }

    Tensor mean(int axis, bool keepdims = false) const {
        Tensor s = sum(axis, keepdims);
        float divisor = (axis == 0) ? static_cast<float>(rows()) : static_cast<float>(cols());
        return s / divisor;
    }

    Tensor argmax(int axis) const {
        if (axis == 1) {
            size_t M = rows(), N = cols();
            Tensor result({M});
            #pragma omp parallel for
            for (size_t i = 0; i < M; i++) {
                size_t base = i * N;
                float max_val = data_[base];
                size_t max_idx = 0;
                for (size_t j = 1; j < N; j++) {
                    if (data_[base + j] > max_val) {
                        max_val = data_[base + j];
                        max_idx = j;
                    }
                }
                result.data_[i] = static_cast<float>(max_idx);
            }
            return result;
        }
        throw std::runtime_error("argmax: only axis=1 supported");
    }

    void reshape(const std::vector<size_t>& new_shape) {
        size_t total = 1;
        for (auto s : new_shape) total *= s;
        if (total != data_.size()) {
            throw std::runtime_error("reshape: size mismatch");
        }
        shape_ = new_shape;
    }

private:
    std::string format_data(const std::vector<size_t>& shape, size_t dim, size_t& offset) const {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < shape[dim]; i++) {
            if (i > 0) oss << ", ";
            if (dim == shape.size() - 1) {
                oss << data_[offset++];
            } else {
                oss << format_data(shape, dim + 1, offset);
            }
        }
        oss << "]";
        return oss.str();
    }

public:
    std::string to_string() const {
        std::ostringstream oss;
        oss << "Tensor(shape=[";
        for (size_t i = 0; i < shape_.size(); i++) {
            if (i > 0) oss << ", ";
            oss << shape_[i];
        }
        oss << "], data=";
        if (shape_.empty() || data_.empty()) {
            oss << "[]";
        } else {
            size_t offset = 0;
            oss << format_data(shape_, 0, offset);
        }
        oss << ")";
        return oss.str();
    }
};

} // namespace libAI
