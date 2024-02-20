find_program(glslc_executable NAMES glslangValidator)

file(GLOB SHADERS
  ${SHADER_SOURCE_DIR}/*.comp
  )

foreach(source IN LISTS SHADERS)
  get_filename_component(FILENAME ${source} NAME_WE)
  add_custom_command(
    COMMAND
      ${glslc_executable}
      -V ${source}
      -o ${SHADER_BINARY_DIR}/${FILENAME}.spv
    OUTPUT ${SHADER_BINARY_DIR}/${FILENAME}.spv
    DEPENDS ${SHADER_BINARY_DIR}
    COMMENT "Compiling shader: ${FILENAME}"
  )
  list(APPEND SPV_SHADERS ${SHADER_BINARY_DIR}/${FILENAME}.spv)
endforeach()

add_custom_target(compile-shaders ALL DEPENDS ${SPV_SHADERS})

