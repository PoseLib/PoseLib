# Select library type
set(_PN ${PROJECT_NAME})
option(BUILD_SHARED_LIBS "Build ${_PN} as a shared library." OFF)
if(BUILD_SHARED_LIBS)
  set(LIBRARY_TYPE SHARED)
else()
  set(LIBRARY_TYPE STATIC)
endif()

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release'.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)

  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Target
add_library(${LIBRARY_NAME} ${LIBRARY_TYPE} ${SOURCES} ${HEADERS})

# Install library
install(TARGETS ${LIBRARY_NAME}
  EXPORT ${PROJECT_EXPORT}
  RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT shlib
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT stlib
  COMPONENT dev)

# Create 'version.h'
configure_file(version.h.in
  "${CMAKE_CURRENT_BINARY_DIR}/version.h" @ONLY)
set(HEADERS ${HEADERS} ${CMAKE_CURRENT_BINARY_DIR}/version.h)

# Generate 'poselib.h' automatically
file(GLOB HEADER_FILES ${CMAKE_SOURCE_DIR}/PoseLib/*.h)
foreach(file ${HEADER_FILES})
  # get basename of each header file
  get_filename_component(basename ${file} NAME)

  # append '#include <...>' to the 'poselib.h' file
  # ToDo: set(...) creates a list separated with ';'. Find a different way.
  set(LIB_INCLUDES_STRING ${LIB_INCLUDES_STRING} "#include <PoseLib/${basename}>\n")
endforeach(file)
string(REPLACE ";" "" LIB_INCLUDES_STRING "${LIB_INCLUDES_STRING}")
configure_file(poselib.h.in
  "${CMAKE_CURRENT_BINARY_DIR}/poselib.h" @ONLY)
set(HEADERS ${HEADERS} ${CMAKE_CURRENT_BINARY_DIR}/poselib.h)

# Install headers
install(FILES ${HEADERS}
  DESTINATION "${INSTALL_INCLUDE_DIR}/${LIBRARY_FOLDER}" )
