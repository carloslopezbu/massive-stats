cmake_minimum_required(VERSION 3.14)
project(massivestats)

# Establece el estándar de C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Configuración de compilación predeterminada
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Busca pybind11 instalado en el sistema
find_package(pybind11 REQUIRED)
if(NOT pybind11_FOUND)
    message(FATAL_ERROR "pybind11 no encontrado. Instálalo antes de continuar.")
endif()

# Busca Python3 y sus componentes necesarios
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
if(NOT Python3_FOUND)
    message(FATAL_ERROR "Python3 no encontrado. Instálalo antes de continuar.")
endif()

# Define el módulo Python
pybind11_add_module(massivestats src/massivestats.cpp)

# Instala el módulo en el directorio de paquetes de Python
install(TARGETS massivestats DESTINATION ${Python3_SITEARCH})