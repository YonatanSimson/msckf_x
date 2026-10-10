cmake_minimum_required(VERSION 3.24)
project(msckf_x LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

option(MSCKF_BUILD_TESTS "Build unit tests" ON)

include(cmake/Dependencies.cmake)

add_library(msckf
  src/camera/camera_model.cpp
  src/camera/pinhole_radtan.cpp
  src/camera/kannala_brandt.cpp
  src/camera/double_sphere.cpp
  src/camera/eucm.cpp
  src/camera/omni_radtan.cpp
  src/eval/alignment.cpp
  src/eval/metrics.cpp
  src/eval/nees.cpp
  src/eval/report.cpp
  src/eval/timing.cpp
  src/eval/trajectory.cpp
  src/imu/propagation.cpp
  src/sim/scenarios.cpp
  src/sim/simulator.cpp
  src/sim/spline.cpp
)
target_include_directories(msckf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(msckf PUBLIC Eigen3::Eigen sophus)
target_compile_options(msckf PRIVATE -Wall -Wextra -Wpedantic)

add_executable(msckf_eval tools/msckf_eval.cpp)
target_link_libraries(msckf_eval PRIVATE msckf)

# Calibration tools: AprilGrid detection, Kalibr I/O, per-image board fits.
# Kept out of msckf so the estimator core does not depend on OpenCV.
add_library(msckf_calib
  src/calib/aprilgrid.cpp
  src/calib/board_pose.cpp
  src/calib/kalibr_io.cpp
)
target_link_libraries(msckf_calib PUBLIC msckf ${OpenCV_LIBS} PRIVATE apriltag yaml-cpp::yaml-cpp)
target_include_directories(msckf_calib SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
target_compile_options(msckf_calib PRIVATE -Wall -Wextra -Wpedantic)

add_executable(calib_reproj tools/calib_reproj.cpp)
target_link_libraries(calib_reproj PRIVATE msckf_calib)

if(MSCKF_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
