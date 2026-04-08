cmake_minimum_required (VERSION 3.20)

project (TensorCompiler LANGUAGES C CXX)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

option (SANITIZE "Enable Address and Undefined sanitizers" OFF)
option (BUILD_TESTS "Build tests" ON)
option (BUILD_CODEGEN "Build MLIR/LLVM codegen backend" OFF)

if (MSVC)
    set (COMMON_COMPILE_OPTIONS /W4 /WX)
else ()
    set (COMMON_COMPILE_OPTIONS
        -Wall
        -Wextra
        -Wpedantic
        -Werror
    )
endif ()

if (SANITIZE AND NOT MSVC)
    set (SANITIZE_FLAGS -fsanitize=address,undefined)
endif ()

# protobuf
# CONFIG mode provides transitive abseil deps (Arch, Homebrew, vcpkg)
# MODULE mode (FindProtobuf) works with Ubuntu/Debian apt packages
find_package (Protobuf CONFIG)
if (NOT Protobuf_FOUND)
    find_package (Protobuf REQUIRED)
endif ()

set (PROTO_SRC ${CMAKE_CURRENT_SOURCE_DIR}/proto/onnx.proto3)
set (PROTO_GEN_CC ${CMAKE_CURRENT_BINARY_DIR}/onnx.proto3.pb.cc)
set (PROTO_GEN_H  ${CMAKE_CURRENT_BINARY_DIR}/onnx.proto3.pb.h)

add_custom_command (
    OUTPUT  ${PROTO_GEN_CC} ${PROTO_GEN_H}
    COMMAND protobuf::protoc
            --cpp_out=${CMAKE_CURRENT_BINARY_DIR}
            --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/proto
            ${PROTO_SRC}
    DEPENDS ${PROTO_SRC}
    COMMENT "Generating C++ from onnx.proto3"
)

# suppress all warnings in generated protobuf code
set_source_files_properties (${PROTO_GEN_CC}
    PROPERTIES COMPILE_OPTIONS "-w"
)

#  core library
add_library (tc_core
    src/importer.cpp
    src/graph/verify_graph.cpp
    src/graph/dot_export.cpp
    ${PROTO_GEN_CC}
)

target_include_directories (tc_core PUBLIC
    include/
    ${CMAKE_CURRENT_BINARY_DIR}
    ${Protobuf_INCLUDE_DIRS}
)

target_compile_options (tc_core PRIVATE
    ${COMMON_COMPILE_OPTIONS}
    ${SANITIZE_FLAGS}
)

target_link_options (tc_core PRIVATE
    ${SANITIZE_FLAGS}
)

target_link_libraries (tc_core PUBLIC protobuf::libprotobuf)

# MLIR/LLVM codegen backend
if (BUILD_CODEGEN)
    find_package (MLIR REQUIRED CONFIG)
    find_package (LLVM REQUIRED CONFIG)

    message (STATUS "Found MLIR ${MLIR_PACKAGE_VERSION} in ${MLIR_DIR}")
    message (STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} in ${LLVM_DIR}")

    list (APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
    include (AddLLVM)
    include (AddMLIR)

    add_library (tc_codegen
        src/codegen/mlir_gen.cpp
        src/codegen/lowering_pipeline.cpp
        src/codegen/llvm_emitter.cpp
    )

    target_include_directories (tc_codegen PUBLIC
        include/
    )

    target_include_directories (tc_codegen SYSTEM PUBLIC
        ${MLIR_INCLUDE_DIRS}
        ${LLVM_INCLUDE_DIRS}
    )

    target_compile_definitions (tc_codegen PUBLIC
        ${LLVM_DEFINITIONS}
    )

    target_compile_options (tc_codegen PRIVATE
        ${COMMON_COMPILE_OPTIONS}
        ${SANITIZE_FLAGS}
        -Wno-unused-parameter
    )

    target_link_options (tc_codegen PRIVATE
        ${SANITIZE_FLAGS}
    )

    target_link_libraries (tc_codegen PUBLIC
        tc_core
        MLIR
        LLVM
    )
endif ()

# main
add_executable (tc_main src/main.cpp)

target_compile_options (tc_main PRIVATE
    ${COMMON_COMPILE_OPTIONS}
    ${SANITIZE_FLAGS}
)

target_link_options (tc_main PRIVATE
    ${SANITIZE_FLAGS}
)

if (BUILD_CODEGEN)
    target_link_libraries (tc_main PRIVATE tc_core tc_codegen)
    target_compile_definitions (tc_main PRIVATE TC_WITH_CODEGEN)
else ()
    target_link_libraries (tc_main PRIVATE tc_core)
endif ()

# tests
if (BUILD_TESTS)
    enable_testing ()

    find_package (GTest REQUIRED)

    add_executable (tc_tests
        tests/graph_builder_smoke_test.cpp
        tests/importer_test.cpp
    )

    if (BUILD_CODEGEN)
        target_sources (tc_tests PRIVATE
            tests/lowering_pipeline_smoke_test.cpp
        )
    endif ()

    target_compile_definitions (tc_tests PRIVATE
        TEST_MODELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/models"
    )

    target_compile_options (tc_tests PRIVATE
        ${COMMON_COMPILE_OPTIONS}
        ${SANITIZE_FLAGS}
    )

    target_link_options (tc_tests PRIVATE
        ${SANITIZE_FLAGS}
    )

    target_link_libraries (tc_tests PRIVATE tc_core GTest::gtest GTest::gtest_main)

    if (BUILD_CODEGEN)
        target_link_libraries (tc_tests PRIVATE tc_codegen)
    endif ()

    include (GoogleTest)
    gtest_discover_tests (tc_tests)
endif ()
