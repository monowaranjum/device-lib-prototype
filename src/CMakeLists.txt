add_executable(main main.cpp)

target_include_directories(main PUBLIC
	"${PROJECT_SOURCE_DIR}/build"
	"${PROJECT_SOURCE_DIR}/src"
	"/home/m84327620/HiVoyagerRepo/Repo/vendor/open_source/vulkan/include"
	)

target_link_libraries(main PUBLIC
	"/home/m84327620/HiVoyagerRepo/Repo/out/umd/newarch/out/v300-cmodel/bin/libvulkan.so"
	"/home/m84327620/HiVoyagerRepo/Repo/out/umd/newarch/out/v300-cmodel/bin/libtoplevel.so"
	"/home/m84327620/HiVoyagerRepo/Repo/out/umd/newarch/out/v300-cmodel/bin/libbishenggpucompiler.so"
	)