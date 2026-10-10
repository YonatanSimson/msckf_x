include(FetchContent)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

# Sophus (MIT): header-only. Populate only; we define the target ourselves so its
# own CMake (tests, fmt requirement) stays out of the build.
FetchContent_Declare(sophus
  URL https://github.com/strasdat/Sophus/archive/refs/tags/1.22.10.tar.gz
  SOURCE_SUBDIR do-not-add
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(sophus)
add_library(sophus INTERFACE)
target_include_directories(sophus SYSTEM INTERFACE ${sophus_SOURCE_DIR})
target_compile_definitions(sophus INTERFACE SOPHUS_USE_BASIC_LOGGING)

if(MSCKF_BUILD_TESTS)
  # GoogleTest (BSD-3).
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()
