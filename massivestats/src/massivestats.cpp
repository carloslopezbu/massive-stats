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

    unsigned int x_size = (axis == 0) ? cols : rows;
    unsigned int y_size = (axis == 0) ? rows : cols;
    std::vector<double> result(x_size, 0.0); // Cambiado a std::vector
    unsigned int num_threads = std::thread::hardware_concurrency();
    num_threads = std::max(1u, num_threads);
    size_t chunk_size = (x_size + num_threads - 1) / num_threads;

    std::vector<std::thread> threads(num_threads); // Cambiado a std::vector
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * chunk_size;
        size_t end = std::min(start + chunk_size, static_cast<size_t>(x_size)); // Conversión explícita
        if (start >= end) break;
        threads[i] = std::thread([&, start, end]() {
            std::vector<double> local_sums(end - start, 0.0); // Cambiado a std::vector
            for (size_t y = 0; y < y_size; ++y) {
                const double* y_data = data + y * x_size;
                for (size_t x = start; x < end; ++x) {
                    local_sums[x - start] += y_data[x];
                }
            }
            for (size_t i = 0; i < local_sums.size(); ++i) {
                result[start + i] = local_sums[i] / y_size;
            }
        });
    }
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
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