# Set SOURCES variable
set(SOURCES
    gp3p.cc
    gp4ps.cc
    p1p2ll.cc
    p2p1ll.cc
    p2p2pl.cc
    p3ll.cc
    p3p.cc
    p4pf.cc
    p5lp_radial.cc
    p6lp.cc
    ugp2p.cc
    ugp3ps.cc
    qep.cc
    univariate.cc
    up1p2pl.cc
    up2p.cc
    up4pl.cc
)

# Set HEADERS variable
set(HEADERS
    gp3p.h
    gp4ps.h
    p1p2ll.h
    p2p1ll.h
    p2p2pl.h
    p3ll.h
    p3p.h
    p4pf.h
    p5lp_radial.h
    p6lp.h
    types.h
    ugp2p.h
    ugp3ps.h
    qep.h
    univariate.h
    up1p2pl.h
    up2p.h
    up4pl.h
)

#
# re3q3 (included as part of the library)
#
include_directories(${PROJECT_SOURCE_DIR}/re3q3)
set(SOURCES
	${PROJECT_SOURCE_DIR}/re3q3/re3q3/re3q3.cc
	${PROJECT_SOURCE_DIR}/re3q3/re3q3/sturm8.cc
    ${SOURCES}
)
set(HEADERS
    ${PROJECT_SOURCE_DIR}/re3q3/re3q3/re3q3.h
	${PROJECT_SOURCE_DIR}/re3q3/re3q3/sturm8.h
    ${HEADERS}
)

# library configuration
include(${PROJECT_SOURCE_DIR}/cmake/LibraryConfig.cmake)

# Eigen
target_link_libraries(${LIBRARY_NAME} Eigen3::Eigen)
