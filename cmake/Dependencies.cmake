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

# --- Calibration tools (msckf_calib): OpenCV, apriltag, yaml-cpp -------------

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs calib3d highgui)

# apriltag (BSD-2). Its own CMake forces shared libs, examples and install
# rules, so build a static target from the sources we need.
FetchContent_Declare(apriltag
  URL https://github.com/AprilRobotics/apriltag/archive/refs/tags/v3.4.2.tar.gz
  URL_HASH SHA256=7e021bab89f136aa3cf736f772a635aaa353f93f6f8859495f4bd8c519be4805
  SOURCE_SUBDIR do-not-add
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(apriltag)
enable_language(C)
find_package(Threads REQUIRED)
file(GLOB APRILTAG_COMMON_SRC ${apriltag_SOURCE_DIR}/common/*.c)
add_library(apriltag STATIC
  ${apriltag_SOURCE_DIR}/apriltag.c
  ${apriltag_SOURCE_DIR}/apriltag_quad_thresh.c
  ${apriltag_SOURCE_DIR}/tag36h11.c
  ${APRILTAG_COMMON_SRC})
set_target_properties(apriltag PROPERTIES C_STANDARD 99 POSITION_INDEPENDENT_CODE ON)
target_compile_options(apriltag PRIVATE -w)
target_include_directories(apriltag SYSTEM PUBLIC ${apriltag_SOURCE_DIR})
target_link_libraries(apriltag PUBLIC Threads::Threads m)

# yaml-cpp (MIT). 0.8.0 declares cmake_minimum_required(3.4), which CMake 4
# rejects without a policy floor.
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
set(YAML_CPP_FORMAT_SOURCE OFF CACHE BOOL "" FORCE)
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(yaml-cpp
  URL https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz
  URL_HASH SHA256=fbe74bbdcee21d656715688706da3c8becfd946d92cd44705cc6098bb23b3a16
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
FetchContent_MakeAvailable(yaml-cpp)
unset(CMAKE_POLICY_VERSION_MINIMUM)
