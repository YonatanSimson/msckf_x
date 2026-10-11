// Runs the minimal KLT tracker on a dataset and writes, per observation, the
// error against the dataset's ground truth (plan, "Minimal KLT tracker",
// evaluation layers). The gate checker (python/msckf_gates/stage1_tracker.py)
// computes the metrics from these files.
//
//   tracker_eval euroc --euroc DIR --cam camN --gt GT.csv [--camchain YAML] [--time-offset S]
//                      [--max-frames N] --out DIR
//   tracker_eval aprilgrid --euroc DIR --cam camN --gt GT.csv --camchain YAML --target YAML --out DIR
//   tracker_eval tartanair --traj DIR --out DIR
//   tracker_eval kitti-flow --kitti DIR --out DIR
//   tracker_eval tracks (--euroc DIR --cam camN [--camchain YAML] | --traj DIR) [--max-frames N]
//                       [--tracker klt|basalt] --out DIR
//
// euroc (EuRoC and TUM-VI, each camera a mono sequence): every finished track
// with >= 5 observations is triangulated from ground-truth camera poses (IMU
// pose from GT.csv composed with the camera extrinsic, from
// mav0/camN/sensor.yaml or --camchain), refined by Gauss-Newton on pixel
// residuals, and each observation's reprojection residual is written, with the
// first-to-last angular epipolar error of the track (in pixels at the image
// centre).
//
// aprilgrid (TUM-VI calib sequences): as euroc, but the tracks are AprilGrid
// corner detections on the static board instead of KLT features: the floor of
// the triangulation residual from ground-truth poses and calibration alone.
//
// tartanair (one trajectory P00x, left front camera, 640x640 pinhole f=320):
// each feature's 3D point comes from the depth at its detection pixel; its true
// pixel in every later frame is the projection through the poses. Points on a
// depth edge (3x3 depth spread > 5 %) are not evaluated, nor observations whose
// true pixel is occluded (depth there differs by > 3 %) or leaves the image.
//
// kitti-flow (KITTI 2015 training pairs 000000..000199, image_2, frame 10 ->
// 11): error of each feature tracked into frame 11 against the noc
// ground-truth flow at its detection pixel (see sampleSparse).
//
// Tracker settings (all modes): --fb-px, --ransac-px, --no-ransac, --constant-velocity, --window,
// --levels, --max-features, --min-distance, --quality, --block-size (KltTrackerConfig).
//
// tracks (no ground truth): writes every observation of every track
// (tracks.csv: frame, t_ns, track, u, v) and the tracker's valid-region mask
// (valid_mask.png), in the format python/tracker_compare.py also writes for
// Basalt's tracker, so both are scored by the same code. --tracker basalt runs
// our vendored Basalt flow (BasaltFlowTracker; --basalt-type
// frame_to_frame|patch, --grid, --fb-dist2, --pattern, --iterations,
// --levels, --mask-valid) instead of the KLT tracker.
//
// Output: frames.csv (tracker statistics per frame), observations.csv, and in
// tartanair mode tracks.csv (life cycle against ground-truth visibility).

#include "msckf/calib/aprilgrid.h"
#include "msckf/calib/kalibr_io.h"
#include "msckf/camera/pinhole_radtan.h"
#include "msckf/eval/trajectory.h"
#include "msckf/vision/basalt_flow.h"
#include "msckf/vision/bearing_ransac.h"
#include "msckf/vision/klt_tracker.h"

#include <Eigen/Dense>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace msckf;
using vision::KltTracker;
using vision::TrackedFeature;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

void usage() {
  std::cerr << "usage: tracker_eval euroc --euroc DIR --cam camN --gt GT.csv [--camchain YAML] [--time-offset S] "
               "[--max-frames N] --out DIR\n"
               "       tracker_eval aprilgrid --euroc DIR --cam camN --gt GT.csv --camchain YAML --target YAML --out DIR\n"
               "       tracker_eval tartanair --traj DIR --out DIR\n"
               "       tracker_eval kitti-flow --kitti DIR --out DIR\n"
               "       tracker_eval tracks (--euroc DIR --cam camN [--camchain YAML] | --traj DIR) [--max-frames N] "
               "--out DIR\n";
}

struct Args {
  std::string mode, euroc, cam = "cam0", gt, camchain, traj, kitti, target, out = "tracker_out";
  double time_offset = 0.0;  // t_d: ground truth is looked up at t_cam + t_d
  std::string tracker_type = "klt";  // tracks mode: klt or basalt
  vision::BasaltFlowConfig basalt;   // --basalt-type, --grid, --fb-dist2, --pattern, --iterations, --levels,
                                     // --mask-valid
  vision::KltTrackerConfig tracker;  // --fb-px, --ransac-px, --no-ransac, --window, --levels,
                                     // --max-features, --min-distance, --quality
  int max_frames = 0;        // 0: all
};

class FrameLog {
 public:
  explicit FrameLog(const fs::path& path) : f_(path) {
    f_ << "sequence,frame,t,features,tracked,lost_klt,rejected_fb,rejected_ransac,detected,"
          "median_flow_px,ransac_run,time_ms\n"
       << std::setprecision(15);
  }
  void add(const std::string& seq, int frame, double t, const vision::FeatureTracker& tr) {
    const auto& s = tr.stats();
    f_ << seq << ',' << frame << ',' << t << ',' << tr.features().size() << ',' << s.tracked << ','
       << s.lost_klt << ',' << s.rejected_fb << ',' << s.rejected_ransac << ',' << s.detected << ','
       << s.median_flow_px << ',' << s.ransac_run << ',' << s.time_ms << '\n';
  }

 private:
  std::ofstream f_;
};

// Bilinear sample of a single-channel float image; NaN if any of the four
// neighbours is outside or invalid (non-finite or <= 0 when positive_only).
double bilinear(const cv::Mat& img, const Eigen::Vector2d& uv, bool positive_only) {
  const int u0 = static_cast<int>(std::floor(uv.x())), v0 = static_cast<int>(std::floor(uv.y()));
  if (u0 < 0 || v0 < 0 || u0 + 1 >= img.cols || v0 + 1 >= img.rows) return kNaN;
  const double a = uv.x() - u0, b = uv.y() - v0;
  double s = 0.0;
  for (int dv = 0; dv < 2; ++dv) {
    for (int du = 0; du < 2; ++du) {
      const double x = img.at<float>(v0 + dv, u0 + du);
      if (!std::isfinite(x) || (positive_only && x <= 0.0)) return kNaN;
      s += (du ? a : 1.0 - a) * (dv ? b : 1.0 - b) * x;
    }
  }
  return s;
}

// ---------------------------------------------------------------- euroc ----

struct ImageEntry {
  double t;
  std::string path;
};

std::vector<ImageEntry> readEurocImages(const fs::path& cam_dir) {
  std::ifstream f(cam_dir / "data.csv");
  if (!f) throw std::runtime_error("cannot open " + (cam_dir / "data.csv").string());
  std::vector<ImageEntry> images;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    const auto comma = line.find(',');
    std::string name = line.substr(comma + 1);
    name.erase(std::remove_if(name.begin(), name.end(), ::isspace), name.end());
    images.push_back({std::stoll(line.substr(0, comma)) * 1e-9, (cam_dir / "data" / name).string()});
  }
  return images;
}

struct Obs {
  int frame;
  double t;
  Eigen::Vector2d uv;
  Eigen::Vector3d bearing;  // camera frame
};

// Triangulates from world-frame rays: least squares on the perpendicular
// distances, then Gauss-Newton on pixel residuals through the camera model.
// Returns false for poor geometry or a point behind a camera.
bool triangulate(const CameraModel& cam, const std::vector<Sophus::SE3d>& T_GC,
                 const std::vector<Obs>& obs, Eigen::Vector3d* X, double* parallax_deg) {
  Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
  Eigen::Vector3d b = Eigen::Vector3d::Zero();
  std::vector<Eigen::Vector3d> d(obs.size());
  for (size_t i = 0; i < obs.size(); ++i) {
    d[i] = T_GC[i].so3() * obs[i].bearing;
    const Eigen::Matrix3d P = Eigen::Matrix3d::Identity() - d[i] * d[i].transpose();
    A += P;
    b += P * T_GC[i].translation();
  }
  double max_angle = 0.0;
  for (size_t i = 0; i < d.size(); ++i) {
    for (size_t j = i + 1; j < d.size(); ++j) {
      max_angle = std::max(max_angle, std::atan2(d[i].cross(d[j]).norm(), d[i].dot(d[j])));
    }
  }
  *parallax_deg = max_angle * 180.0 / M_PI;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(A);
  if (es.eigenvalues()[0] < 1e-6 * es.eigenvalues()[2]) return false;
  *X = A.ldlt().solve(b);
  for (int it = 0; it < 10; ++it) {
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    Eigen::Vector3d g = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < obs.size(); ++i) {
      const Eigen::Vector3d p_c = T_GC[i].inverse() * *X;
      Eigen::Vector2d uv;
      Matrix23d J;
      if (!cam.project(p_c, uv, &J)) return false;
      const Eigen::Matrix<double, 2, 3> Jx = J * T_GC[i].so3().inverse().matrix();
      H += Jx.transpose() * Jx;
      g += Jx.transpose() * (obs[i].uv - uv);
    }
    const Eigen::Vector3d dx = H.ldlt().solve(g);
    *X += dx;
    if (dx.norm() < 1e-9 * (1.0 + X->norm())) break;
  }
  for (size_t i = 0; i < obs.size(); ++i) {
    if ((T_GC[i].inverse() * *X).dot(obs[i].bearing) <= 0.0) return false;
  }
  return true;
}

calib::KalibrCamera loadCamera(const Args& a) {
  calib::KalibrCamera kc;
  if (!a.camchain.empty()) {
    bool found = false;
    for (const auto& c : calib::loadKalibrCamchain(a.camchain)) {
      if (c.name == a.cam) {
        kc = c;
        found = true;
      }
    }
    if (!found) throw std::runtime_error(a.cam + " not in " + a.camchain);
  } else {
    kc = calib::loadEurocCamera((fs::path(a.euroc) / "mav0" / a.cam / "sensor.yaml").string());
  }
  return kc;
}

int runEuroc(const Args& a) {
  const calib::KalibrCamera kc = loadCamera(a);
  if (!kc.has_T_IC) throw std::runtime_error("camera has no extrinsic (T_cam_imu / T_BS)");
  const auto gt = eval::readEurocGroundTruth(a.gt);
  const auto images = readEurocImages(fs::path(a.euroc) / "mav0" / a.cam);
  fs::create_directories(a.out);
  FrameLog frames(fs::path(a.out) / "frames.csv");
  std::ofstream csv(fs::path(a.out) / "observations.csv");
  csv << "track,frame,age,u,v,residual_px,track_len,parallax_deg,epipolar_px,status\n"
      << std::setprecision(9);

  KltTracker tracker(kc.model, a.tracker);
  const double px_per_rad = 1.0 / tracker.radiansPerPixel();
  std::map<int64_t, std::vector<Obs>> live;
  std::map<std::string, int> counts;

  auto finish = [&](int64_t id, const std::vector<Obs>& o) {
    std::string status = "ok";
    std::vector<Sophus::SE3d> T_GC(o.size());
    Eigen::Vector3d X = Eigen::Vector3d::Zero();
    double parallax = kNaN, epi = kNaN;
    bool ok = o.size() >= 5;
    if (!ok) status = "short";
    for (size_t i = 0; ok && i < o.size(); ++i) {
      Sophus::SE3d T_GI;
      if (!eval::interpolatePose(gt, o[i].t + a.time_offset, 0.05, T_GI)) {
        ok = false;
        status = "no_gt";
      } else {
        T_GC[i] = T_GI * kc.T_IC;
      }
    }
    if (ok) {
      const Sophus::SE3d T_ab = T_GC.front().inverse() * T_GC.back();  // last camera in first
      if (T_ab.translation().norm() > 0.05) {
        // f_b^T E f_a = 0 with E = [t_ba]x R_ba.
        const Sophus::SE3d T_ba = T_ab.inverse();
        const Eigen::Matrix3d E = Sophus::SO3d::hat(T_ba.translation()) * T_ba.so3().matrix();
        epi = vision::epipolarAngle(E, o.front().bearing, o.back().bearing) * px_per_rad;
      }
      if (!triangulate(*kc.model, T_GC, o, &X, &parallax)) {
        ok = false;
        status = "geometry";
      }
    }
    ++counts[status];
    for (size_t i = 0; i < o.size(); ++i) {
      double res = kNaN;
      Eigen::Vector2d uv;
      if (ok && kc.model->project(T_GC[i].inverse() * X, uv)) res = (o[i].uv - uv).norm();
      csv << id << ',' << o[i].frame << ',' << i << ',' << o[i].uv.x() << ',' << o[i].uv.y() << ','
          << res << ',' << o.size() << ',' << parallax << ',' << epi << ',' << status << '\n';
    }
  };

  // aprilgrid mode: AprilGrid corners replace the tracker. The board is
  // static, so each corner is an exact physical point; consecutive detections
  // of a corner form a track, cut every kGridTrackLen observations so track
  // lengths resemble KLT's. Run through the same ground truth, the residuals
  // are the floor that ground-truth poses and calibration alone produce.
  constexpr size_t kGridTrackLen = 20;
  std::unique_ptr<calib::AprilGridDetector> grid;
  if (a.mode == "aprilgrid") {
    grid = std::make_unique<calib::AprilGridDetector>(calib::AprilGrid::fromKalibrYaml(a.target));
  }
  std::map<int, int64_t> segment;  // corner id -> current segment
  const size_t num_frames = a.max_frames > 0 ? std::min<size_t>(images.size(), a.max_frames) : images.size();
  for (size_t k = 0; k < num_frames; ++k) {
    const cv::Mat image = cv::imread(images[k].path, cv::IMREAD_UNCHANGED);
    if (image.empty()) throw std::runtime_error("cannot read " + images[k].path);
    std::vector<TrackedFeature> grid_feats;
    if (grid) {
      const calib::GridObservation g = grid->detect(image);
      for (size_t i = 0; i < g.size(); ++i) {
        TrackedFeature f;
        const int corner = g.ids[i];
        auto it = live.find(static_cast<int64_t>(corner) * 100000 + segment[corner]);
        if (it != live.end() && it->second.size() >= kGridTrackLen) ++segment[corner];
        f.id = static_cast<int64_t>(corner) * 100000 + segment[corner];
        f.uv = g.pixels[i];
        if (!kc.model->unproject(f.uv, f.bearing)) continue;
        grid_feats.push_back(f);
      }
      // A corner missed in this frame starts a new segment when seen again.
      for (auto& [corner, seg] : segment) {
        const bool seen = std::any_of(grid_feats.begin(), grid_feats.end(),
                                      [&](const TrackedFeature& f) { return f.id / 100000 == corner; });
        if (!seen && live.count(static_cast<int64_t>(corner) * 100000 + seg)) ++seg;
      }
    }
    const auto& feats = grid ? grid_feats : tracker.process(image);
    if (!grid) frames.add(a.cam, static_cast<int>(k), images[k].t, tracker);
    std::map<int64_t, std::vector<Obs>> next;
    for (const TrackedFeature& f : feats) {
      auto it = live.find(f.id);
      std::vector<Obs> o = it != live.end() ? std::move(it->second) : std::vector<Obs>{};
      if (it != live.end()) live.erase(it);
      o.push_back({static_cast<int>(k), images[k].t, f.uv, f.bearing});
      next[f.id] = std::move(o);
    }
    for (auto& [id, o] : live) finish(id, o);  // tracks that ended
    live.swap(next);
  }
  for (auto& [id, o] : live) finish(id, o);
  std::cout << a.cam << ": " << num_frames << " frames; tracks:";
  for (const auto& [s, n] : counts) std::cout << ' ' << s << ' ' << n;
  std::cout << '\n';
  return 0;
}

// ------------------------------------------------------------ tartanair ----

cv::Mat readTartanDepth(const std::string& path) {
  const cv::Mat rgba = cv::imread(path, cv::IMREAD_UNCHANGED);
  if (rgba.empty() || rgba.type() != CV_8UC4) throw std::runtime_error("bad depth image " + path);
  // Four 8-bit channels holding one little-endian float32 per pixel.
  return cv::Mat(rgba.rows, rgba.cols, CV_32F, const_cast<uchar*>(rgba.data)).clone();
}

std::vector<Sophus::SE3d> readTartanPoses(const std::string& path) {
  // "tx ty tz qx qy qz qw": camera body pose in a NED world, body axes x
  // forward, y right, z down. The optical frame (x right, y down, z forward)
  // is the body frame rotated by R_body_opt.
  Eigen::Matrix3d R_body_opt;
  R_body_opt << 0, 0, 1,
                1, 0, 0,
                0, 1, 0;
  const Sophus::SE3d T_body_opt(Sophus::SO3d(R_body_opt), Eigen::Vector3d::Zero());
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::vector<Sophus::SE3d> poses;
  double tx, ty, tz, qx, qy, qz, qw;
  while (f >> tx >> ty >> tz >> qx >> qy >> qz >> qw) {
    const Eigen::Quaterniond q(qw, qx, qy, qz);
    poses.push_back(Sophus::SE3d(q.normalized(), Eigen::Vector3d(tx, ty, tz)) * T_body_opt);
  }
  return poses;
}

std::vector<std::string> sortedFiles(const fs::path& dir) {
  std::vector<std::string> files;
  for (const auto& e : fs::directory_iterator(dir)) files.push_back(e.path().string());
  std::sort(files.begin(), files.end());
  return files;
}

// TartanAir v2 left front camera: 640x640 pinhole, f = 320, no distortion.
CameraModelPtr tartanCamera() {
  Eigen::VectorXd k(9);
  k << 320.0, 320.0, 320.0, 320.0, 0, 0, 0, 0, 0;
  return std::make_shared<PinholeRadtan>(640, 640, k);
}

int runTartanair(const Args& a) {
  const fs::path traj(a.traj);
  const std::vector<std::string> images = sortedFiles(traj / "image_lcam_front");
  const std::vector<std::string> depths = sortedFiles(traj / "depth_lcam_front");
  const auto poses = readTartanPoses((traj / "pose_lcam_front.txt").string());
  if (images.empty() || images.size() != depths.size() || images.size() != poses.size()) {
    throw std::runtime_error("inconsistent TartanAir trajectory " + a.traj);
  }
  const CameraModelPtr camera = tartanCamera();
  const std::string seq = traj.parent_path().parent_path().filename().string() + "/" +
                          traj.parent_path().filename().string() + "/" + traj.filename().string();

  fs::create_directories(a.out);
  FrameLog frames(fs::path(a.out) / "frames.csv");
  std::ofstream csv(fs::path(a.out) / "observations.csv");
  csv << "track,frame,age,u,v,gt_u,gt_v,error_px,oracle_px,status\n" << std::setprecision(9);

  KltTracker tracker(camera, a.tracker);
  struct Birth {
    bool valid;
    std::string why;
    Eigen::Vector3d X_W;  // world point
  };
  std::map<int64_t, Birth> births;
  // Ground-truth floor: the detection patch aligned directly into the current
  // frame (one LK solve, started at the true pixel, no frame-to-frame drift).
  // Its distance to the true pixel is the disagreement between the rendered
  // image and the depth/pose geometry, independent of the tracker. Done up to
  // kOracleMaxAge, beyond which appearance change makes direct alignment fail.
  constexpr int kOracleMaxAge = 5;
  std::map<int, cv::Mat> gray_by_frame;
  std::map<int64_t, std::pair<int, Eigen::Vector2d>> birth_px;  // frame, uv
  // Track life cycle against ground-truth visibility (the plan's main tracker
  // goal: tracks as long as physically possible, few drop-and-rediscover).
  // A point is visible in a frame if it projects inside the tracker's valid
  // region and the depth there agrees (not occluded). Visibility is followed
  // after a track dies, until the point first becomes invisible.
  struct Life {
    int birth = 0, last = 0;
    std::string end = "alive";  // drop reason, or "alive" at the end of the sequence
    double end_detail = std::numeric_limits<double>::quiet_NaN();
    int visible_last = -1;      // last frame of the contiguous visible span from birth
    bool visible_open = true;
    int64_t redetection_of = -1;  // dropped, still-visible track whose point this new one re-detects
  };
  std::map<int64_t, Life> lives;
  constexpr double kRedetectPx = 2.0;
  const cv::Mat& valid = tracker.validMask();
  auto projectVisible = [&](const Eigen::Vector3d& X_W, const Sophus::SE3d& T_CW, const cv::Mat& depth,
                            Eigen::Vector2d* uv) {
    const Eigen::Vector3d p_c = T_CW * X_W;
    if (p_c.z() <= 0.0 || !camera->project(p_c, *uv) || !camera->inImage(*uv)) return false;
    const int u = static_cast<int>(std::lround(uv->x())), v = static_cast<int>(std::lround(uv->y()));
    if (u < 0 || v < 0 || u >= valid.cols || v >= valid.rows || !valid.at<uint8_t>(v, u)) return false;
    const double z_img = bilinear(depth, *uv, true);
    return std::isfinite(z_img) && std::abs(z_img - p_c.z()) <= 0.03 * p_c.z();
  };
  auto reasonName = [](vision::DropReason r) {
    switch (r) {
      case vision::DropReason::kKltFailed: return "klt_failed";
      case vision::DropReason::kLeftValidRegion: return "left_valid_region";
      case vision::DropReason::kForwardBackward: return "forward_backward";
      case vision::DropReason::kRansac: return "ransac";
    }
    return "?";
  };
  for (size_t n = 0; n < images.size(); ++n) {
    const cv::Mat image = cv::imread(images[n], cv::IMREAD_UNCHANGED);
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    gray_by_frame[static_cast<int>(n)] = gray;
    gray_by_frame.erase(static_cast<int>(n) - kOracleMaxAge - 1);
    const cv::Mat depth = readTartanDepth(depths[n]);
    const auto& feats = tracker.process(image);
    frames.add(seq, static_cast<int>(n), 0.1 * n, tracker);
    const Sophus::SE3d T_CW = poses[n].inverse();
    for (const auto& d : tracker.dropped()) {
      lives.at(d.last.id).end = reasonName(d.reason);
      lives.at(d.last.id).end_detail = d.detail;
    }
    // Visibility of every point still in its first visible span (alive or not).
    for (auto& [id, life] : lives) {
      if (!life.visible_open) continue;
      const Birth& b = births.at(id);
      Eigen::Vector2d uv;
      if (b.valid && projectVisible(b.X_W, T_CW, depth, &uv)) {
        life.visible_last = static_cast<int>(n);
      } else {
        life.visible_open = false;
      }
    }
    for (const TrackedFeature& f : feats) {
      if (f.age == 0) {
        // Depth (z along the optical axis) at the detection pixel, unless the
        // 3x3 neighbourhood straddles a depth edge.
        Birth b{false, "edge", Eigen::Vector3d::Zero()};
        const int u = static_cast<int>(std::lround(f.uv.x())), v = static_cast<int>(std::lround(f.uv.y()));
        float lo = std::numeric_limits<float>::max(), hi = 0.0f;
        for (int dv = -1; dv <= 1; ++dv) {
          for (int du = -1; du <= 1; ++du) {
            const int uu = std::clamp(u + du, 0, depth.cols - 1), vv = std::clamp(v + dv, 0, depth.rows - 1);
            lo = std::min(lo, depth.at<float>(vv, uu));
            hi = std::max(hi, depth.at<float>(vv, uu));
          }
        }
        const double z = bilinear(depth, f.uv, true);
        if (std::isfinite(z) && lo > 0.0f && (hi - lo) <= 0.05 * lo && z < 1e3) {
          const Eigen::Vector3d p_c = f.bearing / f.bearing.z() * z;
          b = {true, "ok", poses[n] * p_c};
        } else if (!(z < 1e3)) {
          b.why = "far";
        }
        births[f.id] = b;
        birth_px[f.id] = {static_cast<int>(n), f.uv};
        Life life;
        life.birth = life.last = life.visible_last = static_cast<int>(n);
        life.visible_open = b.valid;
        if (b.valid) {
          // Re-detection: a dropped track whose point is still visible and
          // projects within kRedetectPx of this new detection.
          double best = kRedetectPx;
          for (const auto& [oid, o] : lives) {
            if (o.end == "alive" || !o.visible_open) continue;
            Eigen::Vector2d uv;
            if (!projectVisible(births.at(oid).X_W, T_CW, depth, &uv)) continue;
            const double dist = (uv - f.uv).norm();
            if (dist < best) {
              best = dist;
              life.redetection_of = oid;
            }
          }
        }
        lives[f.id] = life;
        csv << f.id << ',' << n << ",0," << f.uv.x() << ',' << f.uv.y() << ',' << f.uv.x() << ','
            << f.uv.y() << ",,," << b.why << '\n';
        continue;
      }
      lives.at(f.id).last = static_cast<int>(n);
      const Birth& b = births.at(f.id);
      std::string status = b.why;
      double err = kNaN, oracle = kNaN;
      Eigen::Vector2d uv(kNaN, kNaN);
      if (b.valid) {
        const Eigen::Vector3d p_c = T_CW * b.X_W;
        if (p_c.z() <= 0.0 || !camera->project(p_c, uv) || !camera->inImage(uv)) {
          status = "out_of_view";
        } else {
          const double z_img = bilinear(depth, uv, true);
          if (!std::isfinite(z_img) || std::abs(z_img - p_c.z()) > 0.03 * p_c.z()) {
            status = "occluded";
          } else {
            err = (f.uv - uv).norm();
            const auto& [bf, buv] = birth_px.at(f.id);
            if (f.age <= kOracleMaxAge) {
              std::vector<cv::Point2f> p0{cv::Point2f(buv.x(), buv.y())}, p1{cv::Point2f(uv.x(), uv.y())};
              std::vector<uint8_t> st;
              std::vector<float> e;
              cv::calcOpticalFlowPyrLK(gray_by_frame.at(bf), gray, p0, p1, st, e, cv::Size(21, 21), 0,
                                       cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 50, 0.001),
                                       cv::OPTFLOW_USE_INITIAL_FLOW);
              if (st[0]) oracle = std::hypot(p1[0].x - uv.x(), p1[0].y - uv.y());
            }
          }
        }
      }
      csv << f.id << ',' << n << ',' << f.age << ',' << f.uv.x() << ',' << f.uv.y() << ',' << uv.x() << ','
          << uv.y() << ',' << err << ',' << oracle << ',' << status << '\n';
    }
  }
  // One row per track. visible_len counts the frames from birth to the end of
  // the point's first visible span; tracked beyond it means the tracker
  // followed a point that was out of view or occluded.
  std::ofstream tracks(fs::path(a.out) / "tracks.csv");
  tracks << "track,birth,last,tracked_len,end,end_detail,gt_valid,visible_last,visible_len,premature,"
            "overtracked,redetection_of\n";
  for (const auto& [id, l] : lives) {
    const bool gt = births.at(id).valid;
    const int tracked_len = l.last - l.birth + 1;
    const int visible_len = gt ? l.visible_last - l.birth + 1 : 0;
    const bool premature = gt && l.end != "alive" && l.visible_last > l.last;
    const bool over = gt && l.last > l.visible_last;
    tracks << id << ',' << l.birth << ',' << l.last << ',' << tracked_len << ',' << l.end << ',' << l.end_detail
           << ',' << gt << ','
           << l.visible_last << ',' << visible_len << ',' << premature << ',' << over << ','
           << l.redetection_of << '\n';
  }
  std::cout << seq << ": " << images.size() << " frames, " << births.size() << " features\n";
  return 0;
}

// ----------------------------------------------------------- kitti-flow ----

// Sparse ground truth at a sub-pixel location: bilinear when all four
// neighbours are valid, otherwise inverse-distance weighting of the valid
// pixels within 1.5 px (the LiDAR-based flow is semi-dense; requiring four
// valid neighbours left only ~5 % of features evaluable).
bool sampleSparse(const cv::Mat& fu, const cv::Mat& fv, const Eigen::Vector2d& uv, double* du, double* dv) {
  *du = bilinear(fu, uv, false);
  *dv = bilinear(fv, uv, false);
  if (std::isfinite(*du) && std::isfinite(*dv)) return true;
  double w = 0.0, su = 0.0, sv = 0.0;
  const int u0 = static_cast<int>(std::floor(uv.x())), v0 = static_cast<int>(std::floor(uv.y()));
  for (int v = v0 - 1; v <= v0 + 2; ++v) {
    for (int u = u0 - 1; u <= u0 + 2; ++u) {
      if (u < 0 || v < 0 || u >= fu.cols || v >= fu.rows) continue;
      const double x = fu.at<float>(v, u), y = fv.at<float>(v, u);
      const double d = std::hypot(u - uv.x(), v - uv.y());
      if (!std::isfinite(x) || !std::isfinite(y) || d > 1.5) continue;
      const double wi = 1.0 / std::max(d, 1e-3);
      w += wi;
      su += wi * x;
      sv += wi * y;
    }
  }
  if (w == 0.0) return false;
  *du = su / w;
  *dv = sv / w;
  return true;
}

int runKittiFlow(const Args& a) {
  const fs::path root = fs::path(a.kitti) / "training";
  fs::create_directories(a.out);
  FrameLog frames(fs::path(a.out) / "frames.csv");
  std::ofstream csv(fs::path(a.out) / "observations.csv");
  csv << "pair,track,u0,v0,u1,v1,flow_px,error_px,status\n" << std::setprecision(9);
  int pairs = 0;
  for (int p = 0; p < 200; ++p) {
    std::ostringstream id;
    id << std::setw(6) << std::setfill('0') << p;
    const std::string i10 = (root / "image_2" / (id.str() + "_10.png")).string();
    const std::string i11 = (root / "image_2" / (id.str() + "_11.png")).string();
    const std::string fl = (root / "flow_noc" / (id.str() + "_10.png")).string();
    const cv::Mat im10 = cv::imread(i10, cv::IMREAD_UNCHANGED);
    const cv::Mat im11 = cv::imread(i11, cv::IMREAD_UNCHANGED);
    const cv::Mat flow16 = cv::imread(fl, cv::IMREAD_UNCHANGED);
    if (im10.empty() || im11.empty() || flow16.empty()) throw std::runtime_error("missing KITTI pair " + id.str());

    // Rectified intrinsics of camera 2 (P_rect_02) for bearings.
    std::ifstream cf(root / "calib_cam_to_cam" / (id.str() + ".txt"));
    std::string line;
    Eigen::Matrix<double, 3, 4> P = Eigen::Matrix<double, 3, 4>::Zero();
    while (std::getline(cf, line)) {
      if (line.rfind("P_rect_02:", 0) == 0) {
        std::istringstream ss(line.substr(10));
        for (int i = 0; i < 12; ++i) ss >> P(i / 4, i % 4);
      }
    }
    if (P(0, 0) <= 0.0) throw std::runtime_error("no P_rect_02 for pair " + id.str());
    Eigen::VectorXd k(9);
    k << P(0, 0), P(1, 1), P(0, 2), P(1, 2), 0, 0, 0, 0, 0;
    auto camera = std::make_shared<PinholeRadtan>(im10.cols, im10.rows, k);

    // Ground truth: 16-bit PNG, channels (BGR as loaded) valid, v, u; flow =
    // (value - 2^15) / 64.
    std::vector<cv::Mat> ch;
    cv::split(flow16, ch);
    cv::Mat fu, fv;
    ch[2].convertTo(fu, CV_32F, 1.0 / 64.0, -32768.0 / 64.0);
    ch[1].convertTo(fv, CV_32F, 1.0 / 64.0, -32768.0 / 64.0);
    fu.setTo(std::numeric_limits<float>::quiet_NaN(), ch[0] == 0);
    fv.setTo(std::numeric_limits<float>::quiet_NaN(), ch[0] == 0);

    KltTracker tracker(camera, a.tracker);
    std::map<int64_t, Eigen::Vector2d> born;
    for (const auto& f : tracker.process(im10)) born[f.id] = f.uv;
    frames.add(id.str(), 10, 0.0, tracker);
    std::map<int64_t, Eigen::Vector2d> tracked;
    for (const auto& f : tracker.process(im11)) {
      if (f.age == 1) tracked[f.id] = f.uv;
    }
    frames.add(id.str(), 11, 0.1, tracker);
    for (const auto& [tid, uv0] : born) {
      double du = kNaN, dv = kNaN;
      const bool has_gt = sampleSparse(fu, fv, uv0, &du, &dv);
      const auto it = tracked.find(tid);
      std::string status = it == tracked.end() ? "lost" : (has_gt ? "ok" : "no_gt");
      csv << id.str() << ',' << tid << ',' << uv0.x() << ',' << uv0.y() << ',';
      if (it != tracked.end()) csv << it->second.x() << ',' << it->second.y();
      else csv << ',';
      csv << ',' << (has_gt ? std::hypot(du, dv) : kNaN) << ',';
      if (it != tracked.end() && has_gt) csv << (it->second - uv0 - Eigen::Vector2d(du, dv)).norm();
      csv << ',' << status << '\n';
    }
    ++pairs;
  }
  std::cout << "KITTI flow 2015: " << pairs << " pairs\n";
  return 0;
}

// --------------------------------------------------------------- tracks ----

int runTracks(const Args& a) {
  CameraModelPtr camera;
  std::vector<ImageEntry> images;
  std::string seq;
  if (!a.traj.empty()) {
    camera = tartanCamera();
    const auto files = sortedFiles(fs::path(a.traj) / "image_lcam_front");
    for (size_t n = 0; n < files.size(); ++n) images.push_back({0.1 * static_cast<double>(n), files[n]});  // 10 Hz
    seq = fs::path(a.traj).filename().string();
  } else {
    camera = loadCamera(a).model;
    images = readEurocImages(fs::path(a.euroc) / "mav0" / a.cam);
    seq = a.cam;
  }
  fs::create_directories(a.out);
  FrameLog frames(fs::path(a.out) / "frames.csv");
  std::ofstream csv(fs::path(a.out) / "tracks.csv");
  csv << "frame,t_ns,track,u,v\n" << std::setprecision(10);
  std::unique_ptr<vision::FeatureTracker> tr;
  if (a.tracker_type == "basalt") {
    vision::BasaltFlowConfig bc = a.basalt;
    bc.levels = a.tracker.pyramid_levels;
    tr = std::make_unique<vision::BasaltFlowTracker>(camera, bc);
  } else if (a.tracker_type == "klt") {
    tr = std::make_unique<KltTracker>(camera, a.tracker);
  } else {
    throw std::invalid_argument("unknown tracker " + a.tracker_type);
  }
  vision::FeatureTracker& tracker = *tr;
  cv::imwrite((fs::path(a.out) / "valid_mask.png").string(), tracker.validMask());
  const size_t num_frames = a.max_frames > 0 ? std::min<size_t>(images.size(), a.max_frames) : images.size();
  for (size_t k = 0; k < num_frames; ++k) {
    const cv::Mat image = cv::imread(images[k].path, cv::IMREAD_UNCHANGED);
    if (image.empty()) throw std::runtime_error("cannot read " + images[k].path);
    const auto t_ns = std::llround(images[k].t * 1e9);
    for (const TrackedFeature& f : tracker.process(image)) {
      csv << k << ',' << t_ns << ',' << f.id << ',' << f.uv.x() << ',' << f.uv.y() << '\n';
    }
    frames.add(seq, static_cast<int>(k), images[k].t, tracker);
  }
  std::cout << seq << ": " << num_frames << " frames\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    usage();
    return 2;
  }
  Args a;
  a.mode = argv[1];
  try {
    for (int i = 2; i < argc; ++i) {
      const std::string arg = argv[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
        return argv[++i];
      };
      if (arg == "--euroc") a.euroc = next();
      else if (arg == "--cam") a.cam = next();
      else if (arg == "--gt") a.gt = next();
      else if (arg == "--camchain") a.camchain = next();
      else if (arg == "--traj") a.traj = next();
      else if (arg == "--kitti") a.kitti = next();
      else if (arg == "--out") a.out = next();
      else if (arg == "--time-offset") a.time_offset = std::stod(next());
      else if (arg == "--target") a.target = next();
      else if (arg == "--fb-px") a.tracker.fb_threshold_px = std::stod(next());
      else if (arg == "--ransac-px") a.tracker.ransac_threshold_px = std::stod(next());
      else if (arg == "--no-ransac") a.tracker.ransac = false;
      else if (arg == "--constant-velocity") a.tracker.constant_velocity = true;
      else if (arg == "--window") a.tracker.window_px = std::stoi(next());
      else if (arg == "--levels") a.tracker.pyramid_levels = std::stoi(next());
      else if (arg == "--max-features") a.tracker.max_features = std::stoi(next());
      else if (arg == "--min-distance") a.tracker.min_distance_px = std::stod(next());
      else if (arg == "--quality") a.tracker.quality_level = std::stod(next());
      else if (arg == "--block-size") a.tracker.block_size = std::stoi(next());
      else if (arg == "--max-frames") a.max_frames = std::stoi(next());
      else if (arg == "--tracker") a.tracker_type = next();
      else if (arg == "--basalt-type") {
        const std::string t = next();
        if (t != "frame_to_frame" && t != "patch") throw std::invalid_argument("--basalt-type " + t);
        a.basalt.type = t == "patch" ? vision::BasaltFlowConfig::Type::kPatch
                                     : vision::BasaltFlowConfig::Type::kFrameToFrame;
      }
      else if (arg == "--grid") a.basalt.grid_size = std::stoi(next());
      else if (arg == "--fb-dist2") a.basalt.max_recovered_dist2 = std::stof(next());
      else if (arg == "--pattern") a.basalt.pattern = std::stoi(next());
      else if (arg == "--iterations") a.basalt.max_iterations = std::stoi(next());
      else if (arg == "--mask-valid") a.basalt.mask_valid_region = true;
      else throw std::invalid_argument("unknown argument " + arg);
    }
    if (a.mode == "euroc" || a.mode == "aprilgrid") return runEuroc(a);
    if (a.mode == "tartanair") return runTartanair(a);
    if (a.mode == "kitti-flow") return runKittiFlow(a);
    if (a.mode == "tracks") return runTracks(a);
    usage();
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "tracker_eval: " << e.what() << '\n';
    return 1;
  }
}
