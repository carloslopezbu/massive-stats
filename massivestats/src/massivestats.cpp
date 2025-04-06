#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <stdexcept>
#include <thread>
#include <algorithm>
#include <cstring>

namespace py = pybind11;

py::array_t<double> mean(py::array_t<double, py::array::c_style | py::array::forcecast> input, int axis) {
    auto buf = input.request();
    if (buf.ndim != 2)
        throw std::runtime_error("Input must be a 2D NumPy array");

    const size_t rows = buf.shape[0];
    const size_t cols = buf.shape[1];
    const double* data = static_cast<double*>(buf.ptr);

    std::vector<double> result;

    if (axis == 0) {
        result.resize(cols, 0.0);
        unsigned int num_threads = std::thread::hardware_concurrency();
        num_threads = std::max(1u, num_threads);
        size_t chunk_size = (cols + num_threads - 1) / num_threads;

        std::vector<std::thread> threads;
        for (size_t i = 0; i < num_threads; ++i) {
            size_t start_col = i * chunk_size;
            size_t end_col = std::min(start_col + chunk_size, cols);
            if (start_col >= end_col) break;
            threads.emplace_back([&, start_col, end_col]() {
                std::vector<double> local_sums(end_col - start_col, 0.0);
                for (size_t row = 0; row < rows; ++row) {
                    const double* row_data = data + row * cols;
                    for (size_t col = start_col; col < end_col; ++col) {
                        local_sums[col - start_col] += row_data[col];
                    }
                }
                for (size_t i = 0; i < local_sums.size(); ++i) {
                    result[start_col + i] = local_sums[i] / rows;
                }
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
    } else if (axis == 1) {
        result.resize(rows, 0.0);
        unsigned int num_threads = std::thread::hardware_concurrency();
        num_threads = std::max(1u, num_threads);
        size_t chunk_size = (rows + num_threads - 1) / num_threads;

        std::vector<std::thread> threads;
        for (size_t i = 0; i < num_threads; ++i) {
            size_t start_row = i * chunk_size;
            size_t end_row = std::min(start_row + chunk_size, rows);
            if (start_row >= end_row) break;
            threads.emplace_back([&, start_row, end_row]() {
                for (size_t row = start_row; row < end_row; ++row) {
                    double sum = 0.0;
                    const double* row_data = data + row * cols;
                    for (size_t col = 0; col < cols; ++col) {
                        sum += row_data[col];
                    }
                    result[row] = sum / cols;
                }
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
    } else {
        throw std::runtime_error("Axis must be 0 (columns) or 1 (rows)");
    }

    py::array_t<double> output(result.size());
    auto output_buf = output.request();
    std::memcpy(output_buf.ptr, result.data(), result.size() * sizeof(double));
    return output;
}

PYBIND11_MODULE(massivestats, m) {
    m.doc() = "Extensión C++ optimizada para calcular la media de un array 2D por filas o columnas";
    m.def("mean", &mean, "Calcula la media de un array 2D",
          py::arg("input"), py::arg("axis") = 1);
}