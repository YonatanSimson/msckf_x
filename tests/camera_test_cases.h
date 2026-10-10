// Camera configurations shared by the camera-model tests.
#pragma once

#include "msckf/camera/double_sphere.h"
#include "msckf/camera/eucm.h"
#include "msckf/camera/kannala_brandt.h"
#include "msckf/camera/omni_radtan.h"
#include "msckf/camera/pinhole_radtan.h"

#include <functional>
#include <initializer_list>
#include <string>
#include <vector>

namespace msckf::test {

struct CameraCase {
  std::string label;
  std::function<CameraModelPtr()> make;
  bool expect_beyond_90deg;  // the image contains rays with z < 0
};

inline Eigen::VectorXd vec(std::initializer_list<double> v) {
  Eigen::VectorXd out(static_cast<Eigen::Index>(v.size()));
  Eigen::Index i = 0;
  for (double x : v) out[i++] = x;
  return out;
}

// EuRoC cam0 and TUM-VI cam0 calibrations as published (values quoted from
// memory; the gates here only need realistic parameters). The EUCM and the
// alpha < 0.5 cases are synthetic, to exercise the other validity branch.
// TumviOmniRadtan is TUM-VI cam0's published omni-radtan calibration (xi > 1);
// OmniRadtanXiBelowOne is synthetic, for the xi <= 1 branch. The two
// undistorted omni cases are the ones Basalt's UCM can cross-check.
inline std::vector<CameraCase> cameraCases() {
  return {
      {"EurocRadtan",
       [] {
         return std::make_shared<PinholeRadtan>(
             752, 480,
             vec({458.654, 457.296, 367.215, 248.375, -0.28340811, 0.07395907, 0.00019359,
                  1.76187114e-05, 0.0}));
       },
       false},
      {"RadtanWithK3",
       [] {
         return std::make_shared<PinholeRadtan>(
             640, 480, vec({400.0, 401.0, 321.0, 239.0, -0.3, 0.1, 1e-3, -5e-4, -0.02}));
       },
       false},
      {"TumviKb4",
       [] {
         return std::make_shared<KannalaBrandt>(
             512, 512,
             vec({190.97847715128717, 190.9733070521226, 254.93170605935475, 256.8974428996504,
                  0.0034823894022493434, 0.0007150348452162257, -0.0020532361418706202,
                  0.00020293673591811182}));
       },
       true},
      {"TumviDoubleSphere",
       [] {
         return std::make_shared<DoubleSphere>(
             512, 512,
             vec({158.28600034966977, 158.2743455478755, 254.96116578191653, 256.8894218032305,
                  -0.17213086034353243, 0.5931177593944744}));
       },
       true},
      {"DoubleSphereAlphaBelowHalf",
       [] {
         return std::make_shared<DoubleSphere>(640, 480,
                                               vec({250.0, 250.0, 320.0, 240.0, 0.3, 0.4}));
       },
       false},
      {"Eucm",
       [] {
         return std::make_shared<Eucm>(512, 512, vec({157.9, 157.8, 254.9, 256.9, 0.58, 1.08}));
       },
       true},
      {"EucmAlphaBelowHalf",
       [] { return std::make_shared<Eucm>(640, 480, vec({300.0, 300.0, 320.0, 240.0, 0.4, 1.2})); },
       false},
      {"TumviOmniRadtan",
       [] {
         return std::make_shared<OmniRadtan>(
             512, 512,
             vec({533.340727445877, 533.2556495307942, 254.64689387916482, 256.4835490935692,
                  1.792187901303534, -0.05972430882700243, 0.17468739202093328,
                  0.000737218969875311, 0.000574074894976456}));
       },
       true},
      {"OmniRadtanXiBelowOne",
       [] {
         return std::make_shared<OmniRadtan>(
             640, 480, vec({500.0, 501.0, 321.0, 239.0, 0.8, -0.1, 0.05, 1e-3, -5e-4}));
       },
       false},
      {"OmniNoDistortionXiAboveOne",
       [] {
         return std::make_shared<OmniRadtan>(
             512, 512, vec({533.3, 533.2, 254.6, 256.5, 1.79, 0.0, 0.0, 0.0, 0.0}));
       },
       true},
      {"OmniNoDistortionXiBelowOne",
       [] {
         return std::make_shared<OmniRadtan>(
             640, 480, vec({500.0, 501.0, 321.0, 239.0, 0.8, 0.0, 0.0, 0.0, 0.0}));
       },
       false},
  };
}

}  // namespace msckf::test
