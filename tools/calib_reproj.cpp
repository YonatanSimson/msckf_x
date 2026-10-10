// Reprojection error of a fixed camera calibration on an AprilGrid sequence:
// detects the grid in every image, fits the board pose per image with the
// intrinsics held fixed, and reports the corner residual statistics.
//
//   calib_reproj --euroc DIR --camchain camchain.yaml --target april.yaml [options]
//
// Options:
//   --cam NAME     camera to evaluate (default: every camera in the camchain
//                  that has a mav0/NAME directory)
//   --step N       use every N-th image (default 1)
//   --out DIR      writes frames.csv and corners.csv (default calib_out)
//   --no-open-junctions  detect without the junction-opening dilation (for
//                  comparison; see AprilGridDetector::detect)
//
// Images come from DIR/mav0/<cam>/data.csv (EuRoC layout; TUM-VI uses it too).

#include "msckf/calib/aprilgrid.h"
#include "msckf/calib/board_pose.h"
#include "msckf/calib/kalibr_io.h"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace msckf::calib;

namespace {

void usage() {
  std::cerr << "usage: calib_reproj --euroc DIR --camchain YAML --target YAML [--cam NAME] "
               "[--step N] [--out DIR] [--no-open-junctions]\n";
}

struct ImageEntry {
  long long stamp_ns;
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
    images.push_back({std::stoll(line.substr(0, comma)), (cam_dir / "data" / name).string()});
  }
  return images;
}

struct Stats {
  size_t count = 0;
  int frames = 0;
  double rmse = 0.0, mean = 0.0, median = 0.0, std = 0.0;  // of |e|
  double std_x = 0.0, std_y = 0.0;                         // per axis
};

double median(std::vector<double> v);

// Residual statistics over the fitted frames; kept_only restricts to corners
// that survived outlier filtering.
template <typename Frames>
Stats collect(const Frames& frames, bool kept_only) {
  std::vector<double> norms;
  double sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0;
  Stats st;
  for (const auto& f : frames) {
    if (!f.fit.ok) continue;
    ++st.frames;
    for (size_t i = 0; i < f.all.size(); ++i) {
      if (kept_only && !f.keep[i]) continue;
      const Eigen::Vector2d& e = f.fit.residuals[i];
      norms.push_back(e.norm());
      sx += e.x();
      sy += e.y();
      sxx += e.x() * e.x();
      syy += e.y() * e.y();
    }
  }
  st.count = norms.size();
  if (st.count < 2) return st;
  const double n = static_cast<double>(st.count);
  double s = 0.0, ss = 0.0;
  for (double x : norms) {
    s += x;
    ss += x * x;
  }
  st.mean = s / n;
  st.rmse = std::sqrt(ss / n);
  st.std = std::sqrt(std::max(0.0, (ss - n * st.mean * st.mean) / (n - 1)));
  st.std_x = std::sqrt(std::max(0.0, (sxx - sx * sx / n) / (n - 1)));
  st.std_y = std::sqrt(std::max(0.0, (syy - sy * sy / n) / (n - 1)));
  st.median = median(norms);
  return st;
}

// Refits a frame's board pose to its kept corners and recomputes the
// residuals of all its corners at that pose.
template <typename Frame>
void refit(const msckf::CameraModel& camera, const AprilGrid& grid, Frame* f) {
  GridObservation kept;
  for (size_t i = 0; i < f->all.size(); ++i) {
    if (!f->keep[i]) continue;
    kept.ids.push_back(f->all.ids[i]);
    kept.pixels.push_back(f->all.pixels[i]);
  }
  f->fit = fitBoardPose(camera, grid, kept);
  if (!f->fit.ok) return;
  const BoardPoseResult kept_fit = f->fit;
  f->fit.residuals.resize(f->all.size());
  for (size_t i = 0; i < f->all.size(); ++i) {
    Eigen::Vector2d uv;
    if (camera.project(kept_fit.T_cam_board * grid.corner(f->all.ids[i]), uv)) {
      f->fit.residuals[i] = f->all.pixels[i] - uv;
    } else {
      f->fit.residuals[i].setConstant(std::numeric_limits<double>::infinity());
    }
  }
}

void print(const char* label, const Stats& s) {
  std::cout << "  " << label << ": " << s.count << " corners, RMSE " << s.rmse << " px; |e| mean "
            << s.mean << " median " << s.median << " std " << s.std << " px\n";
}

double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  const size_t m = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + m, v.end());
  return v[m];
}

}  // namespace

int main(int argc, char** argv) {
  std::string euroc, camchain, target, only_cam, out = "calib_out";
  int step = 1;
  AprilGridDetectorOptions detector_options;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
        return argv[++i];
      };
      if (arg == "--euroc") euroc = next();
      else if (arg == "--camchain") camchain = next();
      else if (arg == "--target") target = next();
      else if (arg == "--cam") only_cam = next();
      else if (arg == "--step") step = std::max(1, std::stoi(next()));
      else if (arg == "--out") out = next();
      else if (arg == "--no-open-junctions") detector_options.open_junctions = false;
      else if (arg == "-h" || arg == "--help") {
        usage();
        return 0;
      } else {
        throw std::invalid_argument("unknown argument " + arg);
      }
    }
    if (euroc.empty() || camchain.empty() || target.empty()) {
      usage();
      return 2;
    }

    const AprilGrid grid = AprilGrid::fromKalibrYaml(target);
    const AprilGridDetector detector(grid, detector_options);
    fs::create_directories(out);
    std::ofstream frames_csv(fs::path(out) / "frames.csv");
    std::ofstream corners_csv(fs::path(out) / "corners.csv");
    frames_csv << "cam,timestamp_ns,tags,corners,inliers,fit_ok,rms_px\n";
    corners_csv << "cam,timestamp_ns,corner_id,u,v,du,dv,inlier\n";
    frames_csv << std::setprecision(6);
    corners_csv << std::setprecision(9);

    bool any = false;
    for (const KalibrCamera& cam : loadKalibrCamchain(camchain)) {
      const fs::path cam_dir = fs::path(euroc) / "mav0" / cam.name;
      if ((!only_cam.empty() && cam.name != only_cam) || !fs::exists(cam_dir)) continue;
      any = true;
      const auto images = readEurocImages(cam_dir);

      // Detect and fit every frame with all corners.
      struct Frame {
        long long stamp_ns;
        GridObservation all;     // every detected corner
        std::vector<bool> keep;  // corner survives outlier filtering
        BoardPoseResult fit;     // fit to the kept corners, residuals for all
      };
      std::vector<Frame> frames;
      int considered = 0;
      for (size_t k = 0; k < images.size(); k += step) {
        ++considered;
        const cv::Mat image = cv::imread(images[k].path, cv::IMREAD_UNCHANGED);
        if (image.empty()) throw std::runtime_error("cannot read " + images[k].path);
        Frame f{images[k].stamp_ns, detector.detect(image), {}, {}};
        f.keep.assign(f.all.size(), true);
        if (f.all.size() > 0) f.fit = fitBoardPose(*cam.model, grid, f.all);
        frames.push_back(std::move(f));
      }

      const Stats raw = collect(frames, false);

      // Kalibr's corner outlier rule (kalibr_calibrate_cameras): a corner is an
      // outlier when either residual component exceeds 4 sigma of that
      // component over all corners. Remove, refit the affected frames, repeat
      // until nothing changes.
      int removed = 0, rounds = 0;
      for (; rounds < 20; ++rounds) {
        const Stats cur = collect(frames, true);
        int removed_now = 0;
        for (Frame& f : frames) {
          if (!f.fit.ok) continue;
          bool changed = false;
          for (size_t i = 0; i < f.all.size(); ++i) {
            const Eigen::Vector2d& e = f.fit.residuals[i];
            if (f.keep[i] && (std::abs(e.x()) > 4.0 * cur.std_x || std::abs(e.y()) > 4.0 * cur.std_y)) {
              f.keep[i] = false;
              changed = true;
              ++removed_now;
            }
          }
          if (changed) refit(*cam.model, grid, &f);
        }
        removed += removed_now;
        if (removed_now == 0) break;
      }
      const Stats filtered = collect(frames, true);

      for (const Frame& f : frames) {
        int inliers = 0;
        for (bool k : f.keep) inliers += k;
        frames_csv << cam.name << ',' << f.stamp_ns << ',' << f.all.num_tags << ',' << f.all.size()
                   << ',' << inliers << ',' << f.fit.ok << ',' << f.fit.rms << '\n';
        if (!f.fit.ok) continue;
        for (size_t i = 0; i < f.all.size(); ++i) {
          const Eigen::Vector2d& e = f.fit.residuals[i];
          corners_csv << cam.name << ',' << f.stamp_ns << ',' << f.all.ids[i] << ','
                      << f.all.pixels[i].x() << ',' << f.all.pixels[i].y() << ',' << e.x() << ','
                      << e.y() << ',' << f.keep[i] << '\n';
        }
      }

      std::cout << std::fixed << std::setprecision(4) << cam.name << " (" << cam.model_name
                << "): " << raw.frames << " frames fitted of " << considered << " read\n";
      print("all corners", raw);
      print("Kalibr 4-sigma filtered", filtered);
      std::cout << "    removed " << removed << " corners (" << std::setprecision(2)
                << 100.0 * removed / std::max<size_t>(raw.count, 1) << " %) in " << rounds
                << " rounds\n";
    }
    if (!any) throw std::runtime_error("no camera of the camchain found under " + euroc + "/mav0");
  } catch (const std::exception& e) {
    std::cerr << "calib_reproj: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
