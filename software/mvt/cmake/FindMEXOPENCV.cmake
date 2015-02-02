
FIND_PATH(MEXOPENCV_INCLUDE_DIRS
      NAMES mexopencv.hpp MxArray.hpp
      PATHS ${MEXOPENCV_DIR}/include)
FIND_LIBRARY(MEXOPENCV_MXARRAY_LIBRARY
      NAMES MxArray
      PATHS ${MEXOPENCV_DIR}/lib)

MESSAGE("${MEXOPENCV_DIR} ${MEXOPENCV_INCLUDE_DIRS} ${MEXOPENCV_MXARRAY_LIBRARY}")
IF(MEXOPENCV_INCLUDE_DIRS NOT_FOUND)

ENDIF()
ELSE()
MESSAGE(${MEXOPENCV_DIR})
ENDIF()

#set(MEXOPENCV_FOUND OFF)
IF(NOT MEXOPENCV_FOUND)
    MESSAGE(SEND_ERROR "[Error] Failed to find \"mexopencv\" automatically (Git URL: https://github.com/kyamagu/mexopencv).
If \"mexopencv\" is correctly installed but this message is shown, please manually edit CMakeLists.txt at this line.
Or try again with the option -D MEXOPENCV_DIR=<mexopencv_dir>")

#set(MEXOPENCV_FOUND ON)
#set(MEXOPENCV_DIR "$ENV{HOME}/lib/mexopencv")
#set(MEXOPENCV_LIBRARIES "${MEXOPENCV_DIR}/lib")
#set(MEXOPENCV_INCLUDE_DIRS "${MEXOPENCV_DIR}/include")




# display help message
IF(NOT MEXOPENCV_FOUND)
    MESSAGE(SEND_ERROR "[Error] Failed to find \"Matlab\" automatically. If \"Matlab\" is correctly installed but this message is shown, please manually edit CMakeLists.txt at this line")
ENDIF()

#set(MEXOPENCV_FOUND ON)
#set(MEXOPENCV_DIR "$ENV{HOME}/lib/mexopencv")
#set(MEXOPENCV_LIBRARIES "${MEXOPENCV_DIR}/lib")
#set(MEXOPENCV_INCLUDE_DIRS "${MEXOPENCV_DIR}/include")


