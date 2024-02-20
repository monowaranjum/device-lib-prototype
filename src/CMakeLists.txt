add_executable(main main.cpp)

target_include_directories(main PUBLIC
	"${PROJECT_SOURCE_DIR}/build"
	"${PROJECT_SOURCE_DIR}/src"
	${Vulkan_INCLUDE_DIRS}
	)

target_link_libraries(main PUBLIC
	${Vulkan_LIBRARIES}
	)