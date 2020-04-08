# Set PROJECT_NAME_UPPERCASE and PROJECT_NAME_LOWERCASE variables
string(TOUPPER ${PROJECT_NAME} PROJECT_NAME_UPPERCASE)
string(TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWERCASE)

# Version variables
set(MAJOR_VERSION 1)
set(MINOR_VERSION 0)
set(PATCH_VERSION 0)
set(PROJECT_VERSION ${MAJOR_VERSION}.${MINOR_VERSION}.${PATCH_VERSION})

# INSTALL_LIB_DIR
set(INSTALL_LIB_DIR lib)

# INSTALL_BIN_DIR
set(INSTALL_BIN_DIR bin)

# INSTALL_INCLUDE_DIR
set(INSTALL_INCLUDE_DIR include)

# INSTALL_CMAKE_DIR
set(INSTALL_CMAKE_DIR lib/cmake/${PROJECT_NAME})

# Convert relative path to absolute path (needed later on)
foreach(substring LIB BIN INCLUDE CMAKE)
  set(var INSTALL_${substring}_DIR)

  set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  if(NOT IS_ABSOLUTE ${CMAKE_INSTALL_PREFIX})
    set(${var} "${CMAKE_BINARY_DIR}/${${var}}")
    get_filename_component(${var} "${${var}}" ABSOLUTE)
  endif()

  message(STATUS "${var}: "  "${${var}}")
endforeach()
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}.")

# Set up include-directories
include_directories(
  "${PROJECT_SOURCE_DIR}"
  "${PROJECT_BINARY_DIR}")

# Library name (by default is the project name in lowercase)
# Example: libfoo.so
if(NOT LIBRARY_NAME)
  set(LIBRARY_NAME ${PROJECT_NAME_LOWERCASE})
endif()

# Library folder name (by default is the project name in lowercase)
# Example: #include <foo/foo.h>
if(NOT LIBRARY_FOLDER)
  set(LIBRARY_FOLDER ${PROJECT_NAME_LOWERCASE})
endif()

# The export set for all the targets
set(PROJECT_EXPORT ${PROJECT_NAME}EXPORT)

# Path of the CMake files generated
set(PROJECT_CMAKE_FILES ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY})

# The RPATH to be used when installing
set(CMAKE_INSTALL_RPATH ${INSTALL_LIB_DIR})

# CMake Registry
include(${CMAKE_SOURCE_DIR}/cmake/CMakeRegistry.cmake)
