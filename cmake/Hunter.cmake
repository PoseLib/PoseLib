# Hunter - package manager
include("cmake/HunterGate.cmake")

# Hunter disabled by default
option(HUNTER_ENABLED "Enable Hunter package manager" OFF)
message(STATUS "HUNTER_ENABLED: ${HUNTER_ENABLED} (package manager)")

# Hunter Gate
# See releases: https://github.com/cpp-pm/hunter/releases
HunterGate(
    URL "https://github.com/cpp-pm/hunter/archive/v0.23.260.tar.gz"
    SHA1 "13775235910a3fa85644568d1c5be8271de72e1c"
)
