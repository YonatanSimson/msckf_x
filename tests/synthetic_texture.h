#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdint>

namespace msckf::test {

// Smooth random texture (blurred noise, 0..255 float), large enough to shift.
inline cv::Mat texture(int w, int h, uint64_t seed) {
  cv::Mat noise(h, w, CV_32F);
  cv::RNG rng(seed);
  rng.fill(noise, cv::RNG::UNIFORM, 0.0, 255.0);
  cv::GaussianBlur(noise, noise, cv::Size(0, 0), 2.0);
  cv::normalize(noise, noise, 0, 255, cv::NORM_MINMAX);
  return noise;
}

}  // namespace msckf::test
