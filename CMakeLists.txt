cmake_minimum_required(VERSION 3.13)

project(
    minimal-device-lib
    LANGUAGES CXX
    VERSION 0.0.1
    )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-Wall")

# find_package(Vulkan REQUIRED FATAL_ERROR)

set(SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)

add_custom_command(
  COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_BINARY_DIR}
  OUTPUT ${SHADER_BINARY_DIR}
  COMMENT "Creating ${SHADER_BINARY_DIR}"
)

add_subdirectory(shaders)
add_subdirectory(src)