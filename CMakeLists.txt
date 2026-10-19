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
  src/geometry/triangulation.cpp
  src/filter/filter_state.cpp
  src/filter/propagator.cpp
  src/filter/feature_update.cpp
  src/filter/msckf.cpp
  src/filter/sim_run.cpp
  src/filter/observability.cpp
  src/imu/preintegration.cpp
  src/imu/propagation.cpp
  src/init/featureless_init.cpp
  src/init/init_refinement.cpp
  src/init/dynamic_init.cpp
  src/init/static_init.cpp
  src/io/euroc.cpp
  src/sim/lissajous.cpp
  src/sim/scenarios.cpp
  src/sim/simulator.cpp
  src/sim/spline.cpp
  src/sim/time_warp.cpp
)
target_include_directories(msckf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(msckf PUBLIC Eigen3::Eigen sophus)
target_compile_options(msckf PRIVATE -Wall -Wextra -Wpedantic)

add_executable(msckf_eval tools/msckf_eval.cpp)
target_link_libraries(msckf_eval PRIVATE msckf)

add_executable(sim_export tools/sim_export.cpp)
target_link_libraries(sim_export PRIVATE msckf)

add_executable(msckf_sim_eval tools/msckf_sim_eval.cpp)
target_link_libraries(msckf_sim_eval PRIVATE msckf)
add_executable(msckf_observability tools/msckf_observability.cpp)
target_link_libraries(msckf_observability PRIVATE msckf)

# Vision: image utilities, the minimal KLT tracker, bearing RANSAC. Grows into
# the Stage 2 front end. Kept out of msckf so the estimator core does not
# depend on OpenCV.
add_library(msckf_vision
  src/vision/basalt_flow.cpp
  src/vision/bearing_ransac.cpp
  src/vision/image.cpp
  src/vision/feature_tracker.cpp
  src/vision/klt_tracker.cpp
)
target_link_libraries(msckf_vision PUBLIC msckf ${OpenCV_LIBS} PRIVATE basalt_headers)
target_include_directories(msckf_vision SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
target_compile_options(msckf_vision PRIVATE -Wall -Wextra -Wpedantic)

# Calibration tools: AprilGrid detection, Kalibr I/O, per-image board fits.
# Kept out of msckf so the estimator core does not depend on OpenCV.
add_library(msckf_calib
  src/calib/aprilgrid.cpp
  src/calib/board_pose.cpp
  src/calib/kalibr_io.cpp
)
target_link_libraries(msckf_calib PUBLIC msckf msckf_vision ${OpenCV_LIBS} PRIVATE apriltag yaml-cpp::yaml-cpp)
target_include_directories(msckf_calib SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
target_compile_options(msckf_calib PRIVATE -Wall -Wextra -Wpedantic)

add_executable(calib_reproj tools/calib_reproj.cpp)
target_link_libraries(calib_reproj PRIVATE msckf_calib)

add_executable(tracker_eval tools/tracker_eval.cpp)
target_link_libraries(tracker_eval PRIVATE msckf_calib msckf_vision)

add_executable(static_init_eval tools/static_init_eval.cpp)
target_link_libraries(static_init_eval PRIVATE msckf_calib)

add_executable(dynamic_init_eval tools/dynamic_init_eval.cpp)
target_link_libraries(dynamic_init_eval PRIVATE msckf_calib)

add_executable(msckf_run tools/msckf_run.cpp)
target_link_libraries(msckf_run PRIVATE msckf_calib)

if(MSCKF_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
