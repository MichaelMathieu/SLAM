#include "SLAM.hpp"
#include <cmath>
using namespace cv;
using namespace std;

matf SLAM::getCovariancePointAlone(const matf & pt2d) const {
  matf Minv = (K * kalman.getRot().toMat()).inv(); //TODO: once and for all
  matf p (3,1); p(0) = pt2d(0); p(1) = pt2d(1); p(2) = 1.0f;
  matf eigv(3,3);
  matf a = eigv.col(0);
  a = Minv * p;
  a /= norm(a);
  int k = (abs(a(0))<min(abs(a(1)),abs(a(0))))?0:((abs(a(1))<abs(a(2)))?1:2);
  matf v (3, 1, 0.0f); v(k) = 1.0f;
  eigv.col(1) = a.cross(v);
  eigv.col(2) = a.cross(eigv.col(1));
  matf eig(3,3,0.0f);
  eig(0) = 100;
  eig(1) = eig(2) = 1;
  return eigv * eig * eigv.inv();
}

void SLAM::projectGaussian(const matf & sigma, matf & output) const {
  matf x(3,1,1.0f), Mx;
  float alpha, beta, gamma;
  matf Minv = (K * kalman.getRot().toMat()).inv(); //TODO: once and for all
  matf C = kalman.getPos();
  gamma = matf(C.t() * sigma * C)(0);
  float den = 2.*3.1415926*sqrt(determinant(sigma)); //TODO:det once
  cout << "----- " << endl;
  cout << den << endl;
  cout << sigma << endl;
  for (int i = 0; i < output.size().width; ++i)
    for (int j = 0; j < output.size().height; ++j) {
      x(0) = i; x(1) = j; x(2) = 1.0f;
      Mx = Minv * x;
      alpha = matf(Mx.t() * sigma * Mx)(0);
      beta = matf(C.t() * sigma * Mx)(0);
      output(j, i) = exp(0.5*(beta*beta/(alpha*alpha)-gamma))/den;
    }
}

void SLAM::addNewFeatures(const matb & im, const vector<KeyPoint> & keypoints,
			  const vector<Match> & matches,
			  int n, float minDist, int dx, int dy) {
  //TODO that could have linear complexity (with a bit more memory)
  vector<int> newpoints;
  for (size_t i = 0; (i < keypoints.size()) && (n > 0); ++i) {
    const Point2f & pt = keypoints[i].pt;
    for (size_t j = 0; j < matches.size(); ++j)
      if (norm(matches[j].pos - pt) < minDist)
	goto badAddNewFeature;
    for (size_t j = 0; j < newpoints.size(); ++j)
      if (norm(keypoints[newpoints[j]].pt - pt) < minDist)
	goto badAddNewFeature;
    if ((pt.y-dy < 0) || (pt.x-dx < 0) || (pt.y+dy >= im.size().height) ||
	(pt.x+dx >= im.size().width))
      goto badAddNewFeature;
    {
      // add the new point to the kalman filter
      matf x(3,1);
      x(0) = pt.x; x(1) = pt.y; x(2) = 1.0f;
      matf P = kalman.getP();
      matf px = P.t() * (P * P.t()).inv() * x;
      cout << "Adding new point px " << px << endl;
      px = px/px(3);
      px = px(Range(0,3),Range(0,1));
      //px = kalman.getPos() + (px - kalman.getPos()) * 100;
      matf cov = 10*matf::eye(3,3);
      kalman.addNewPoint(px, cov);
      // add the new point to the feature vector
      features.push_back(Feature(kalman, kalman.getNPts()-1, im, pt, px,
				 dx, dy, Feature::Line)); 
      newpoints.push_back(i);
      --n;
    }
  badAddNewFeature:;
  }
}
