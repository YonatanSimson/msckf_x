// Readers for recorded sequences: the front end's tracks and EuRoC's state
// ground truth.

#include <gtest/gtest.h>

#include "msckf/io/euroc.h"

#include <filesystem>
#include <fstream>
#include <string>

namespace msckf::io {
namespace {

std::string writeFile(const std::string& name, const std::string& text) {
  const std::string path = (std::filesystem::path(::testing::TempDir()) / name).string();
  std::ofstream(path) << text;
  return path;
}

TEST(Io, ReadTracksGroupsObservationsByFrameInOrder) {
  const std::string path = writeFile("tracks.csv",
                                     "frame,t_ns,track,u,v\n"
                                     "1,2000000000,7,10.5,20.25\n"
                                     "0,1000000000,3,1,2\n"
                                     "1,2000000000,8,30,40\n");

  const std::vector<TrackedFrame> frames = readTracks(path);

  ASSERT_EQ(frames.size(), 2u);
  EXPECT_DOUBLE_EQ(frames[0].t, 1.0);
  ASSERT_EQ(frames[1].uv.size(), 2u);
  EXPECT_EQ(frames[1].uv[0].first, 7);
  EXPECT_EQ(frames[1].uv[0].second, Eigen::Vector2d(10.5, 20.25));
}

TEST(Io, ReadTracksThrowsOnMissingFile) {
  EXPECT_THROW(readTracks("/nonexistent/tracks.csv"), std::runtime_error);
}

TEST(Io, ReadEurocStatesReadsVelocityAndBiases) {
  const std::string path = writeFile("state.csv",
                                     "#timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z,v_x,v_y,v_z,bw_x,bw_y,bw_z,ba_x,ba_y,ba_z\n"
                                     "1500000000,0,0,0,1,0,0,0,1,2,3,0.1,0.2,0.3,-1,-2,-3\n");

  const std::vector<EurocState> states = readEurocStates(path);

  ASSERT_EQ(states.size(), 1u);
  EXPECT_DOUBLE_EQ(states[0].t, 1.5);
  EXPECT_EQ(states[0].v, Eigen::Vector3d(1, 2, 3));
  EXPECT_EQ(states[0].bg, Eigen::Vector3d(0.1, 0.2, 0.3));
  EXPECT_EQ(states[0].ba, Eigen::Vector3d(-1, -2, -3));
}

TEST(Io, ReadEurocStatesIsEmptyForPoseOnlyFiles) {
  const std::string path = writeFile("pose.csv", "#t,px,py,pz,qw,qx,qy,qz\n1000000000,0,0,0,1,0,0,0\n");

  EXPECT_TRUE(readEurocStates(path).empty());
}

}  // namespace
}  // namespace msckf::io
