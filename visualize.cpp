#include "SLAM.hpp"
#include<opencv/highgui.h>
using namespace cv;

void projectElem(Mat dst, const matf & P, const KalmanSLAM & kalman) {
  int h = dst.size().height, w = dst.size().width;
  matf p(4,1);
  for (int i = 0; i < kalman.getNPts(); ++i) {
    matf p3d = kalman.getPt3d(i);
    p(0) = p3d(0);
    p(1) = p3d(1);
    p(2) = p3d(2);
    p(3) = 1.0f;
    matf p2 = P*p;
    p2 /= p2(2);
    if ((p2(0) >= 0) && (p2(1) >= 0) && (p2(0) < w) && (p2(1) < h))
      circle(dst, Point(p2(0),p2(1)), 5, Scalar(0,0,255));
  }
}


void SLAM::visualize() const {
  int h = 200, w = 200;
  matf K2(3,3,0.0f);
  K2(0,0) = K2(1,1) = 50;
  K2(0,2) = w/2; K2(1,2) = h/2; K2(2,2) = 1.0f;
  Mat_<Vec3b> todisp(h, 2*w, Vec3b(0,0,0));
  matf R(matf::eye(3,3));
  matf t(3,1,0.0f);
  t(0) = 5;
  t(2) = 10;
  matf P(3,4);
  R.copyTo(P(Range(0,3),Range(0,3)));
  P(Range(0,3),Range(3,4)) = -R*t;
  P = K2*P;
  projectElem(todisp(Range(0,h),Range(0,w)), P, kalman);
  t(0) = 0;
  t(1) = 5;
  t(2) = 1;
  R.setTo(0);
  R(0,0) = 1;
  R(1,2) = -1;
  R(2,1) = 1;
  R.copyTo(P(Range(0,3),Range(0,3)));
  P(Range(0,3),Range(3,4)) = -R*t;
  P = K2*P;
  projectElem(todisp(Range(0,h),Range(w,2*w)), P, kalman);
  namedWindow("visualize");
  imshow("visualize", todisp);
  cvWaitKey(1);
}
