# CMakeList.txt: проект CMake для EfficientViT 
cmake_minimum_required(VERSION 4.4 FATAL_ERROR)
project(EfficientViT)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(TORCH_CUDA_ARCH_LIST "7.5 8.6")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

include("CMakeLists.Files.txt")

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

source_group("Headers" FILES ${HEADERS})
set(SOURCES ${SOURCES} ${HEADERS})

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} ${LIBS})

if (MSVC) 
	file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj /openmp /MP")
endif (MSVC)