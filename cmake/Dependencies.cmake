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

  # basalt-headers (BSD-3), test-only: reference camera models for the Stage 0
  # cross-check. Header-only; populate and expose the include dir.
  FetchContent_Declare(basalt_headers
    URL https://github.com/VladyslavUsenko/basalt-headers-mirror/archive/b7f915b195c87c8373bf93f97a0710bcdaffd718.tar.gz
    URL_HASH SHA256=165cb202c1e499684b231fa33ee984567eeceb3df6f036cd4c3d5d46ad5375d0
    SOURCE_SUBDIR do-not-add
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(basalt_headers)
  add_library(basalt_headers INTERFACE)
  target_include_directories(basalt_headers SYSTEM INTERFACE ${basalt_headers_SOURCE_DIR}/include)
  target_link_libraries(basalt_headers INTERFACE Eigen3::Eigen sophus)
endif()
