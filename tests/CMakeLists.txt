include(GoogleTest)

function(msckf_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE msckf GTest::gtest_main)
  gtest_discover_tests(${name} DISCOVERY_TIMEOUT 60)
endfunction()

msckf_add_test(test_conventions)
msckf_add_test(test_camera_models)
msckf_add_test(test_simulator)

msckf_add_test(test_camera_basalt_crosscheck)
target_link_libraries(test_camera_basalt_crosscheck PRIVATE basalt_headers)
