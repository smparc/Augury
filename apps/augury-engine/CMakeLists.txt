cmake_minimum_required(VERSION 3.20)
project(augury_engine VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Dependencies come via FetchContent rather than vcpkg or Conan so that a fresh
# checkout builds with nothing installed but CMake and a compiler. Both are
# header-only, which keeps configure time tolerable.
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------

add_library(augury_lmsr
    src/lmsr.cpp
    src/backtest.cpp
)
target_include_directories(augury_lmsr PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(MSVC)
    target_compile_options(augury_lmsr PRIVATE /W4 /permissive-)
else()
    target_compile_options(augury_lmsr PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---------------------------------------------------------------------------
# Engine binary
# ---------------------------------------------------------------------------

add_executable(augury_engine src/main.cpp)
target_link_libraries(augury_engine PRIVATE augury_lmsr nlohmann_json::nlohmann_json)
set_target_properties(augury_engine PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ---------------------------------------------------------------------------
# Tests — correctness only. Benchmarks are a separate target on purpose: a
# timing run has no pass/fail and must never gate `ctest`.
# ---------------------------------------------------------------------------

option(AUGURY_BUILD_TESTS "Build the Catch2 test suite" ON)

if(AUGURY_BUILD_TESTS)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.5.2
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(Catch2)

    enable_testing()

    add_executable(engine_tests tests/test_lmsr.cpp)
    target_link_libraries(engine_tests PRIVATE
        augury_lmsr
        nlohmann_json::nlohmann_json
        Catch2::Catch2WithMain
    )
    set_target_properties(engine_tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
    include(Catch)
    # The golden-vector tests locate schemas/testdata by walking up from the
    # working directory, so run from the repository root.
    catch_discover_tests(engine_tests
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../..
    )
endif()

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

option(AUGURY_BUILD_BENCHMARKS "Build the benchmark target" ON)

if(AUGURY_BUILD_BENCHMARKS)
    add_executable(engine_benchmarks benchmarks/engine_benchmarks.cpp)
    target_link_libraries(engine_benchmarks PRIVATE augury_lmsr)
    set_target_properties(engine_benchmarks PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endif()
