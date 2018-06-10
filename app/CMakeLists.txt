##############################
PROJECT( evoimage )
##############################
cmake_minimum_required (VERSION 3.0.2)

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_FLAGS " -fno-lto")
SET(CMAKE_C_FLAGS "")
ADD_COMPILE_OPTIONS(-Wall -pedantic -O3 -march=ivybridge -fuse-linker-plugin -pthread)

# Packages
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Qt5 COMPONENTS Core Widgets Concurrent Gui Test REQUIRED)

# Includes
INCLUDE_DIRECTORIES(
	${OPENCV_INCLUDE_DIR}
	${PROJECT_BINARY_DIR}
	/usr/include/
)

FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	MESSAGE(STATUS "Found OpenMP")
ENDIF()

SET(SOURCE_FILES
	src/main.cpp
	src/types.hpp
)

# Add executable and target link libraries
ADD_EXECUTABLE( evoimage ${SOURCE_FILES})
TARGET_LINK_LIBRARIES(
	evoimage
	${OpenCV_LIBS}
)
