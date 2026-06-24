cmake_minimum_required(VERSION 3.16)
project(MonteCarloSim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Catch2 via FetchContent ---
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.7.1
)
FetchContent_MakeAvailable(Catch2)

# --- Shared simulation library ---
add_library(sim_lib STATIC
    src/csv_io.cpp
    src/simulation.cpp
    src/metrics.cpp
)
target_include_directories(sim_lib PUBLIC src)

# --- Main binary ---
add_executable(monte_carlo src/main.cpp)
target_link_libraries(monte_carlo PRIVATE sim_lib)

# --- Test binary ---
add_executable(run_tests tests/test_main.cpp)
target_link_libraries(run_tests PRIVATE sim_lib Catch2::Catch2WithMain)

# Register tests with CTest
include(CTest)
include(Catch)
catch_discover_tests(run_tests)
