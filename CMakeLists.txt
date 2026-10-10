cmake_minimum_required(VERSION 3.25)
project(lodestone VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(lodestone STATIC
    src/dispatch.cpp
    src/gemm_scalar.cpp
    src/nn.cpp
)
target_include_directories(lodestone PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(lodestone PUBLIC cxx_std_23)
target_compile_options(lodestone PRIVATE -Wall -Wextra -Wpedantic -Werror -O3)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64|i686")
    target_sources(lodestone PRIVATE src/gemm_avx2.cpp src/gemm_avx512.cpp)
    target_compile_definitions(lodestone PUBLIC LODESTONE_X86=1)
    set_source_files_properties(src/gemm_scalar.cpp PROPERTIES
        COMPILE_OPTIONS "-O3;-fno-tree-vectorize;-mno-avx;-mno-avx2")
    set_source_files_properties(src/gemm_avx2.cpp PROPERTIES
        COMPILE_OPTIONS "-O3;-mavx2;-mbmi2")
    set_source_files_properties(src/gemm_avx512.cpp PROPERTIES
        COMPILE_OPTIONS "-O3;-mavx512f;-mavx512bw;-mavx512dq;-mavx512vpopcntdq")
endif()

add_library(lodestone_memorize STATIC examples/memorize.cpp)
target_link_libraries(lodestone_memorize PUBLIC lodestone)
target_include_directories(lodestone_memorize PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/examples)
target_compile_options(lodestone_memorize PRIVATE -Wall -Wextra -Wpedantic -Werror -O3)

add_executable(lodestone_train src/demo.cpp)
target_link_libraries(lodestone_train PRIVATE lodestone_memorize)
target_include_directories(lodestone_train PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples)
target_compile_options(lodestone_train PRIVATE -Wall -Wextra -Wpedantic -Werror -O3)
set_target_properties(lodestone_train PROPERTIES OUTPUT_NAME lodestone)

add_executable(lodestone_bench bench/bench_gemm.cpp)
target_link_libraries(lodestone_bench PRIVATE lodestone)
target_compile_options(lodestone_bench PRIVATE -Wall -Wextra -Wpedantic -Werror -O3)
set_target_properties(lodestone_bench PROPERTIES OUTPUT_NAME lodestone_bench)

enable_testing()

function(lodestone_test name src)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE lodestone)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic -Werror -O2)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

lodestone_test(test_binary tests/test_binary.cpp)
lodestone_test(test_layers tests/test_layers.cpp)

add_executable(test_overfit tests/test_overfit.cpp)
target_link_libraries(test_overfit PRIVATE lodestone_memorize)
target_include_directories(test_overfit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
    ${CMAKE_CURRENT_SOURCE_DIR}/examples)
target_compile_options(test_overfit PRIVATE -Wall -Wextra -Wpedantic -Werror -O2)
add_test(NAME test_overfit COMMAND test_overfit)
set_tests_properties(test_overfit PROPERTIES TIMEOUT 180)
