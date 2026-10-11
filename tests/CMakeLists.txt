include(GoogleTest)

function(msckf_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE msckf GTest::gtest_main)
  gtest_discover_tests(${name} DISCOVERY_TIMEOUT 60)
endfunction()

msckf_add_test(test_conventions)
msckf_add_test(test_camera_models)
msckf_add_test(test_simulator)
msckf_add_test(test_eval)
msckf_add_test(test_static_init)
msckf_add_test(test_featureless_init)
msckf_add_test(test_preintegration)
msckf_add_test(test_klt_tracker)
target_link_libraries(test_klt_tracker PRIVATE msckf_vision)
msckf_add_test(test_basalt_flow)
target_link_libraries(test_basalt_flow PRIVATE msckf_vision)

msckf_add_test(test_camera_basalt_crosscheck)
target_link_libraries(test_camera_basalt_crosscheck PRIVATE basalt_headers)

# GTSAM (BSD), test-only: cross-check of the IMU preintegration. Built by
# scripts/deps/build_gtsam.sh; configure with -DGTSAM_DIR=<prefix>/lib/cmake/GTSAM.
find_package(GTSAM CONFIG QUIET)
if(GTSAM_FOUND)
  msckf_add_test(test_preintegration_gtsam)
  target_link_libraries(test_preintegration_gtsam PRIVATE gtsam)
else()
  message(STATUS "GTSAM not found: skipping test_preintegration_gtsam (see scripts/deps/build_gtsam.sh)")
endif()
