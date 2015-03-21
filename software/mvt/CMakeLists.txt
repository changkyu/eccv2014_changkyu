cmake_minimum_required (VERSION 2.8.7)
project (Monocular_Multiview_Object_Tracker)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_scripts)

####################################################################
# Required Packages OpenCV2, Matlab, CUDA, Boost, mexopencv

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
#  set(OpenCV2_ROOT_DIR "")
#  set(OpenCV2_INCLUDE_DIRS "")
#  set(OpenCV2_LIBRARIES "")
#
ENDIF()
include_directories(${OpenCV2_ROOT_DIR}/include ${OpenCV2_INCLUDE_DIRS})

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
find_package(Boost COMPONENTS program_options REQUIRED)

# Find MxArray
find_package(MEXOPENCV REQUIRED)
include_directories(${MEXOPENCV_INCLUDE_DIRS})

####################################################################
# Find 3rdparty sources
set(ALM_ROOT_DIR         "${PROJECT_SOURCE_DIR}/3rdparty/ALM")
set(ALM_INCLUDE_DIR      "${ALM_ROOT_DIR}/svm_struct_cuda_mpi")
set(ALM_SOURCE_DIR       "${ALM_ROOT_DIR}/svm_struct_cuda_mpi")

set(MTTUMICH_ROOT_DIR    "${PROJECT_SOURCE_DIR}/3rdparty/mtt-umich/standalone_tracker")
set(MTTUMICH_INCLUDE_DIR "${MTTUMICH_ROOT_DIR}/include")
set(MTTUMICH_SOURCE_DIR  "${MTTUMICH_ROOT_DIR}/src")

set(MIL_ROOT_DIR         "${PROJECT_SOURCE_DIR}/3rdparty/gsoc11_tracking-master")
set(MIL_INCLUDE_DIR      "${MIL_ROOT_DIR}/include")
set(MIL_SOURCE_DIR       "${MIL_ROOT_DIR}/src")

####################################################################
# Multiview Tracker path setting

set(MVT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(MVT_SOURCE_DIR  ${PROJECT_SOURCE_DIR}/src)

####################################################################
# Compile flags
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

####################################################################
# Includes
include_directories($ENV{CPLUS_INCLUDE_PATH} $ENV{C_INCLUDE_PATH})
include_directories(${Boost_INCLUDE_DIR})
include_directories(${MTTUMICH_INCLUDE_DIR})
include_directories(${MIL_INCLUDE_DIR})
include_directories(${MVT_INCLUDE_DIR})

link_directories(${Boost_LIBRARY_DIRS})
link_directories(${MEXOPENCV_LIBRARY_DIRS})

####################################################################
# Sources
#add_library(mvt_cuda src/select_gpu.cu src/matrix.cu src/rectify.cu src/hog.cu src/convolve.cu svm_common.c src/svm_struct_api.c src/svm_struct_common.c src/cad.c src/tree.c src/distance_transform.c)

cuda_include_directories(${CUDA_INCLUDE_DIRS})
cuda_include_directories(${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)

set(ALM_SOURCE_FILES
${ALM_SOURCE_DIR}/select_gpu.cu
${ALM_SOURCE_DIR}/matrix.cu
${ALM_SOURCE_DIR}/rectify.cu
${ALM_SOURCE_DIR}/hog.cu
)

cuda_add_library(mvt_cuda STATIC ${ALM_SOURCE_FILES})

set(MVT_SOURCE_FILES 
${MVT_SOURCE_DIR}/detector_alm.cpp
${MVT_SOURCE_DIR}/detector_dpm.cpp
${MIL_SOURCE_DIR}/cv_onlinemil.cpp
${MIL_SOURCE_DIR}/object_tracker.cpp
${MIL_SOURCE_DIR}/cv_onlineboosting.cpp
${MVT_SOURCE_DIR}/online_model_mil.cpp
${MVT_SOURCE_DIR}/mvt_param.cpp
${MVT_SOURCE_DIR}/mvt_timer.cpp
${MVT_SOURCE_DIR}/mvt_state.cpp
${MVT_SOURCE_DIR}/mvt_sampling.cpp
${MVT_SOURCE_DIR}/mvt_2d_object.cpp
${MVT_SOURCE_DIR}/mvt_3d_object.cpp
${MVT_SOURCE_DIR}/mvt_model.cpp
${MVT_SOURCE_DIR}/motion_pairwise.cpp
${MVT_SOURCE_DIR}/motion_prior.cpp
${MVT_SOURCE_DIR}/mvt_tracker.cpp
${MVT_SOURCE_DIR}/main.cpp
)

add_executable(mvt ${MVT_SOURCE_FILES})

#############################################################nullnull~cs#######
# Libraries
#target_link_libraries(mvt ${OpenCV2_LIBRARIES})
target_link_libraries(mvt opencv_core opencv_imgproc opencv_highgui)
target_link_libraries(mvt ${MATLAB_LIBRARIES})
#target_link_libraries(mvt ${CUDA_LIBRARIES})
#target_link_libraries(mvt ${OpenMP_LIBRARIES})
target_link_libraries(mvt boost_program_options)
target_link_libraries(mvt ${MEXOPENCV_LIBRARIES})
target_link_libraries(mvt mvt_cuda)
