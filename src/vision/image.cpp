#include "msckf/vision/image.h"

#include <opencv2/imgproc.hpp>

#include <stdexcept>

namespace msckf::vision {

cv::Mat toGray8(const cv::Mat& image) {
  cv::Mat gray = image;
  if (gray.channels() == 3) cv::cvtColor(gray, gray, cv::COLOR_BGR2GRAY);
  if (gray.channels() == 4) cv::cvtColor(gray, gray, cv::COLOR_BGRA2GRAY);
  if (gray.depth() == CV_16U) gray.convertTo(gray, CV_8U, 1.0 / 256.0);
  if (gray.type() != CV_8UC1) throw std::runtime_error("toGray8: unsupported image type");
  return gray;
}

}  // namespace msckf::vision
