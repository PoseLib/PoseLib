# Set SOURCES variable
file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

# Set HEADERS variable
file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

# re3q3 (included as part of the library)
include_directories(${CMAKE_SOURCE_DIR}/re3q3)
set($SOURCES
	re3q3/re3q3.cc
	re3q3/sturm8.cc
    $SOURCES
)
set($HEADERS
    re3q3/re3q3.h
    re3q3/sturm8.h
    $HEADERS
)

# library configuration
include(${CMAKE_SOURCE_DIR}/cmake/LibraryConfig.cmake)

# Eigen
target_link_libraries(${LIBRARY_NAME} Eigen3::Eigen)
