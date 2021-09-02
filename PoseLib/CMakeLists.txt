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
    up1p2pl.cc
    up2p.cc
    up4pl.cc
    ugp4pl.cc
    relpose_upright_3pt.cc
    relpose_upright_planar_2pt.cc
    relpose_upright_planar_3pt.cc
    relpose_8pt.cc
    relpose_5pt.cc
    relpose_7pt.cc
    gen_relpose_upright_4pt.cc
    gen_relpose_5p1pt.cc
    misc/qep.cc
    misc/univariate.cc
    misc/essential.cc
    misc/re3q3.cc
    robust/robust.cc
    robust/ransac.cc
    robust/bundle.cc
    robust/colmap_models.cc
    robust/utils.cc
    robust/estimators/absolute_pose.cc
    robust/estimators/relative_pose.cc
    robust/estimators/hybrid_pose.cc
)

# Set HEADERS_PUBLIC variable
set(HEADERS_PUBLIC
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
    up1p2pl.h
    up2p.h
    up4pl.h
    ugp4pl.h
    relpose_upright_3pt.h
    relpose_upright_planar_2pt.h
    relpose_upright_planar_3pt.h
    relpose_8pt.h
    relpose_5pt.h
    relpose_7pt.h
    gen_relpose_upright_4pt.h
    gen_relpose_5p1pt.h
    robust/robust.h
    robust/types.h
    robust/ransac.h
    robust/bundle.h
    robust/colmap_models.h
    robust/utils.h
    robust/estimators/absolute_pose.h
    robust/estimators/relative_pose.h
    robust/estimators/hybrid_pose.h
)

# Set HEADERS_PRIVATE variable
set(HEADERS_PRIVATE
    misc/qep.h
    misc/univariate.h
    misc/sturm.h
    misc/essential.h
    misc/re3q3.h
)

# library configuration
include(${PROJECT_SOURCE_DIR}/cmake/LibraryConfig.cmake)

# Eigen
target_link_libraries(${LIBRARY_NAME} Eigen3::Eigen)
