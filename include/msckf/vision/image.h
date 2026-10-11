#pragma once

#include <opencv2/core/mat.hpp>

namespace msckf::vision {

// 8-bit single-channel view of an image: 16-bit images keep their high byte,
// colour images are converted to grey; 8-bit grey passes through unchanged.
cv::Mat toGray8(const cv::Mat& image);

}  // namespace msckf::vision
