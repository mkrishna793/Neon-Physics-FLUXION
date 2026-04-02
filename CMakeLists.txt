# FLUXION Build Configuration
cmake_minimum_required(VERSION 3.15)
project(FLUXION VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(FLUXION_BUILD_TESTS "Build tests" ON)
option(FLUXION_USE_OPENCL "Enable OpenCL acceleration" ON)
option(FLUXION_BUILD_DOCS "Build documentation" OFF)

# Find dependencies
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(OpenMP)

if(OpenMP_FOUND)
    message(STATUS "OpenMP found: Parallel physical synthesis enabled")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(WARNING "OpenMP not found, running physics engine in single-threaded mode")
endif()

# OpenCL (optional)
if(FLUXION_USE_OPENCL)
    find_package(OpenCL)
    if(OpenCL_FOUND)
        message(STATUS "OpenCL found: ${OpenCL_VERSION_STRING}")
        add_definitions(-DFLUXION_USE_OPENCL)
    else()
        message(WARNING "OpenCL not found, GPU acceleration disabled")
    endif()
endif()

# FLUXION Verilator Export Pass (standalone library)
add_library(fluxion_export STATIC
    src/cpp/V3FluxionExport.cpp
    src/cpp/FluxionEngine.cpp
    src/cpp/BarnesHutTree.cpp
)

target_include_directories(fluxion_export PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)

target_compile_features(fluxion_export PUBLIC cxx_std_17)
if(OpenMP_CXX_FOUND)
    target_link_libraries(fluxion_export PUBLIC OpenMP::OpenMP_CXX)
endif()

# Python bindings (using pybind11 if available)
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    message(STATUS "pybind11 found, building Python bindings")

    pybind11_add_module(fluxion_python
        src/cpp/python_bindings.cpp
    )

    target_link_libraries(fluxion_python PRIVATE fluxion_export)
    target_include_directories(fluxion_python PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp
    )
else()
    message(STATUS "pybind11 not found, Python bindings not built")
endif()

# Tests
if(FLUXION_BUILD_TESTS)
    enable_testing()

    add_executable(fluxion_test
        tests/cpp/test_fluxion_export.cpp
    )

    target_link_libraries(fluxion_test PRIVATE fluxion_export)

    add_test(NAME FluxionExportTest COMMAND fluxion_test)
endif()

# Documentation
if(FLUXION_BUILD_DOCS)
    find_package(Doxygen QUIET)
    if(Doxygen_FOUND)
        message(STATUS "Doxygen found, documentation can be built")
        add_custom_target(docs
            COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Generating documentation"
        )
    endif()
endif()

# Install
install(TARGETS fluxion_export
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES
    src/cpp/V3FluxionExport.h
    DESTINATION include/fluxion
)

# Print configuration summary
message(STATUS "")
message(STATUS "FLUXION Configuration Summary")
message(STATUS "================================")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Tests: ${FLUXION_BUILD_TESTS}")
message(STATUS "OpenCL: ${OpenCL_FOUND}")
message(STATUS "pybind11: ${pybind11_FOUND}")
message(STATUS "")