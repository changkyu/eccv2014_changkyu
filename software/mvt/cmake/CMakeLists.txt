cmake_minimum_required (VERSION 2.8.7)
project (Monocular_Multiview_Object_Tracker)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

####################################################################
# Required Packages OpenCV2, Matlab, CUDA, Boost

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})

# Find OpenCV2
find_package(OpenCV2 REQUIRED)
IF(OpenCV2_FOUND)
  MESSAGE("-- Found OpenCV2")
ELSE()
  MESSAGE(SEND_ERROR "[Error] Failed to find \"OpenCV2\" automatically. If \"OpenCV2\" is correctly installed but this message is shown, please manually edit CMakeLists.txt at this line")
# If FindOpenCV.cmake does not work, please manually set following variables
#
#  set(OpenCV2_FOUND 1)
#  set(OpenCV2_INCLUDE_DIRS "")
#  set(OpenCV2_LIBRARIES "")
#
ENDIF()
include_directories(${OpenCV2_INCLUDE_DIRS})

# Find MATLAB
find_package(MATLAB REQUIRED)
IF(MATLAB_FOUND)
  MESSAGE("-- Found Matlab")
ELSE()
  MESSAGE(SEND_ERROR "[Error] Failed to find \"Matlab\" automatically. If \"Matlab\" is correctly installed but this message is shown, please manually edit CMakeLists.txt at this line")
# If FindMATLAB.cmake does not work, please manually set following variables
#
#  set(MATLAB_INCLUDE_DIR "") #include path for mex.h, engine.h
#  set(MATLAB_LIBRARIES "")   #required libraries: libmex, etc
#  set(MATLAB_MEX_LIBRARY "") #path to libmex.lib
#  set(MATLAB_MX_LIBRARY "") #path to libmx.lib
#  set(MATLAB_MAT_LIBRARY "") #path to libmat.lib # added
#  set(MATLAB_ENG_LIBRARY "") #path to libeng.lib
#  set(MATLAB_ROOT "") # path to Matlab's root directory
#
ENDIF()
include_directories(${MATLAB_INCLUDE_DIR})

# Find CUDA
find_package(CUDA REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find Boost
find_package(Boost 1.36.0 COMPONENTS program_options REQUIRED)

# Find MxArray
set(MXARRAY_FOUND OFF)
IF(NOT MXARRAY_FOUND)
    MESSAGE(SEND_ERROR "[Error] Failed to find \"MxArray\" automatically (Git URL: https://github.com/kyamagu/mexopencv). If \"MxArray\" is correctly installed but this message is shown, please manually edit CMakeLists.txt at this line")
set(MXARRAY_FOUND ON)
set(MXARRAY_LIBRARIES "$ENV{HOME}/lib/matlab_toolbox/mexopencv-master/lib")
set(MXARRAY_INCLUDE_DIRS "$ENV{HOME}/lib/matlab_toolbox/mexopencv-master/include")
ENDIF()
MESSAGE("include: ${MXARRAY_INCLUDE_DIRS}")
include_directories(${MXARRAY_INCLUDE_DIRS})

####################################################################
# Compile flags
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

####################################################################
# Includes
include_directories(include)

####################################################################
# Sources
#add_library(mvt_cuda src/select_gpu.cu src/matrix.cu src/rectify.cu src/hog.cu src/convolve.cu svm_common.c src/svm_struct_api.c src/svm_struct_common.c src/cad.c src/tree.c src/distance_transform.c)
add_executable(mvt ../src/main.cpp)

####################################################################
# Libraries
#target_link_libraries(mvt ${OpenCV2_LIBRARIES})
#target_link_libraries(mvt ${MATLAB_LIBRARIES})
#target_link_libraries(mvt ${CUDA_LIBRARIES})
#target_link_libraries(mvt ${OpenMP_LIBRARIES})
#target_link_libraries(mvt ${Boost_LIBRARIES})


