# Find mexopencv library path
#
# Author: Changkyu Song (changkyusong86@gmail.com, http://sites.google.com/site/changkyusong86)

IF(WIN32)
    SET(MEXOPENCV_POSSIBLE_ROOT_DIRS
        ${MEXOPENCV_DIR}
        "$ENV{SystemDrive}/mexopencv"
        "$ENV{SystemDrive}/Program Files/mexopencv"
        "$ENV{SystemDrive}/Program Files (x86)/mexopencv"
    )
ELSE(WIN32)
    SET(MEXOPENCV_POSSIBLE_ROOT_DIRS
        ${MEXOPENCV_DIR}
        /usr/local
        /usr
        /opt/local
    )
ENDIF(WIN32)

FIND_PATH(MEXOPENCV_ROOT_DIR
    NAMES include/mexopencv.hpp
    PATHS ${MEXOPENCV_POSSIBLE_ROOT_DIRS}
)

FIND_PATH(MEXOPENCV_INCLUDE_DIRS
    NAMES mexopencv.hpp MxArray.hpp
    PATHS ${MEXOPENCV_ROOT_DIR}/include
)

FIND_LIBRARY(MEXOPENCV_MXARRAY_LIBRARY
    NAMES MxArray
    PATHS ${MEXOPENCV_ROOT_DIR}/lib
)

MESSAGE("${MEXOPENCV_DIR} ${MEXOPENCV_INCLUDE_DIRS} ${MEXOPENCV_MXARRAY_LIBRARY}")
IF(MEXOPENCV_INCLUDE_DIRS)
MESSAGE("YES")
ELSE()
MESSAGE("NO")
ENDIF()

#set(MEXOPENCV_FOUND OFF)
#IF(NOT MEXOPENCV_FOUND)
#    MESSAGE(SEND_ERROR "[Error] Failed to find \"mexopencv\" automatically (Git URL: https://github.com/kyamagu/mexopencv).
#If \"mexopencv\" is correctly installed but this message is shown, please manually edit CMakeLists.txt at this line.
#Or try again with the option -D MEXOPENCV_DIR=<mexopencv_dir>")

#set(MEXOPENCV_FOUND ON)
#set(MEXOPENCV_DIR "$ENV{HOME}/lib/mexopencv")
#set(MEXOPENCV_LIBRARIES "${MEXOPENCV_DIR}/lib")
#set(MEXOPENCV_INCLUDE_DIRS "${MEXOPENCV_DIR}/include")

