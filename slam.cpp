#include "SLAM.hpp"
using namespace std;
using namespace cv;

Mat imdisp_debug(0,0, CV_32F);

void SLAM::waitForInit() {
  do {
    mongoose.FetchMongoose();
  } while (!mongoose.isInit);
}

matf SLAM::CameraState::getLocalCoordinatesPoint(const matf & p2d) const {
  matf out(3,3);
  matf a0 = out.col(0), a1 = out.col(1), a2 = out.col(2);
  a2 = p2d(0) * KRinv.col(0) + p2d(1) * KRinv.col(1) + KRinv.col(2);
  a2 /= norm(a2);
  a2.cross(-R.col(1)).copyTo(a0);
  a2.cross(a0).copyTo(a1);
  return out;
}

matf SLAM::CameraState::getLocalCoordinatesPoint(const Point2f & pt2d) const {
  matf out(3,3);
  matf a0 = out.col(0), a1 = out.col(1), a2 = out.col(2);
  a2 = pt2d.x * KRinv.col(0) + pt2d.y * KRinv.col(1) + KRinv.col(2);
  a2 /= norm(a2);
  a2.cross(-R.col(1)).copyTo(a0);
  a2.cross(a0).copyTo(a1);
  return out;
}

matf SLAM::CameraState::getLocalCoordinatesPoint(const Point2i & pt2d) const {
  matf out(3,3);
  matf a0 = out.col(0), a1 = out.col(1), a2 = out.col(2);
  a2 = (float)pt2d.x * KRinv.col(0) + (float)pt2d.y * KRinv.col(1) + KRinv.col(2);
  a2 /= norm(a2);
  a2.cross(-R.col(1)).copyTo(a0);
  a2.cross(a0).copyTo(a1);
  return out;
}
