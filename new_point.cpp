#include "SLAM.hpp"
#include <cmath>
using namespace cv;
using namespace std;

matf SLAM::getCovariancePointAlone(const CameraState & state, const matf & pt2d,
				   float Lambda, float lambda) const {
  matf eigv = state.getLocalCoordinatesPoint(pt2d);
  //cout << eigv << endl;
  matf eig(3,3,0.0f);
  eig(0,0) = Lambda;
  eig(1,1) = lambda;
  eig(2,2) = lambda;
  matf out = eigv * eig * eigv.inv();
  return out;
}

inline float projectGaussianAux(int i, int j, matf & x, matf & Mx,
				const matf & Minv, const matf & sigmainv,
				const matf & muC, float gamma, float den) {
  x(0) = i;
  x(1) = j;
  x(2) = 1.0f;
  Mx = Minv * x;
  Mx /= norm(Mx);
  const float alpha = matf(Mx.t() * sigmainv * Mx)(0);
  const float beta = matf(muC.t() * sigmainv * Mx)(0);
  return exp(0.5*(beta*beta/alpha-gamma))*sqrt(alpha)/den;
}
  
void SLAM::projectGaussian(const matf & mu, const matf & sigma, matf & output) const {
  matf x(3,1,1.0f), Mx;
  //TODO: M should be uptodate, same above
  matf M = K * kalman.getRot().toMat();
  matf Minv = M.inv(); //TODO: once and for all
  matf C = kalman.getPos();
  float den = 2.*3.1415926*sqrt(determinant(sigma)); //TODO:det once
  matf muC = mu - C;
  matf sigmainv = sigma.inv();
  float gamma = matf(muC.t() * sigmainv * muC)(0);
  
  // project the mean of the 3d gaussian
  matf meanp = M * (mu - C);
  int xstart = round(meanp(0)/meanp(2));
  int ystart = round(meanp(1)/meanp(2));

  // fill the output
  const float val = projectGaussianAux(xstart,ystart,x,Mx,Minv,sigmainv,
				       muC,gamma,den);
  output(ystart,xstart) = val;
  const float thval = val/20;
  float maxval;
  for (int j = ystart+1; j < output.size().height; ++j) {
    const float val = projectGaussianAux(xstart,j,x,Mx,Minv,sigmainv,muC,gamma,den);
    output(j,xstart) = val;
    if (val < thval)
      break;
  }
  for (int j = ystart-1; j >= 0; --j) {
    const float val = projectGaussianAux(xstart,j,x,Mx,Minv,sigmainv,muC,gamma,den);
    output(j,xstart) = val;
    if (val < thval)
      break;
  }
  for (int i = xstart+1; i < output.size().width; ++i) {
    maxval = 0.f;
    for (int j = ystart; j < output.size().height; ++j) {
      const float val = projectGaussianAux(i,j,x,Mx,Minv,sigmainv,muC,gamma,den);
      output(j,i) = val;
      if ((val < thval) && (output(j,i-1)==0.f))
	break;
      if (val > maxval)
	maxval = val;
    }
    for (int j = ystart-1; j >= 0; --j) {
      const float val = projectGaussianAux(i,j,x,Mx,Minv,sigmainv,muC,gamma,den);
      output(j,i) = val;
      if (val < thval)
	break;
      if ((val < thval) && (output(j,i+1)==0.f))
	maxval = val;
    }
    if (maxval < thval)
      break;
  }
  for (int i = xstart-1; i >=0; --i) {
    maxval = 0.f;
    for (int j = ystart; j < output.size().height; ++j) {
      const float val = projectGaussianAux(i,j,x,Mx,Minv,sigmainv,muC,gamma,den);
      output(j,i) = val;
      if ((val < thval) && (output(j,i-1)==0.f))
	break;
      if (val > maxval)
	maxval = val;
    }
    for (int j = ystart-1; j >= 0; --j) {
      const float val = projectGaussianAux(i,j,x,Mx,Minv,sigmainv,muC,gamma,den);
      output(j,i) = val;
      if ((val < thval) && (output(j,i+1)==0.f))
	break;
      if (val > maxval)
	maxval = val;
    }
    if (maxval < thval)
      break;
  }
}

bool compareKeyPoints(const KeyPoint & p1, const KeyPoint & p2) {
  return p1.response > p2.response;
}

void SLAM::getSortedKeyPoints(const matb & im, size_t nMinPts,
			     vector<KeyPoint> & out) const {
  out.clear();
  while (out.size() < nMinPts) {
    out.clear();
    //SURF surf(fastThreshold);
    //surf(im, noArray(), out);
    FAST(im, out, fastThreshold, true);
    fastThreshold *= 0.5f;
  }
  fastThreshold *= 2.f;
  sort(out.begin(), out.end(), compareKeyPoints);
  if (out.size() > 2*nMinPts)
    fastThreshold = out[nMinPts*1.9].response;
}

void SLAM::computeNewLines(const matb & im, const CameraState & state,
			   const vector<Match> & matches,
			   int n, float minDist, int rx, int ry) {
  vector<KeyPoint> keypoints;
  getSortedKeyPoints(im, 10*minTrackedPerImage, keypoints);
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
    if ((pt.y-ry < 0) || (pt.x-rx < 0) || (pt.y+ry >= im.size().height) ||
	(pt.x+rx >= im.size().width))
      goto badAddNewFeature;
    lineFeatures.push_back(LineFeature(im, state, pt, rx, ry));
    newpoints.push_back(i);
    --n;
  badAddNewFeature:;
  }
}

void SLAM::lineToFeature(const Mat_<imtype> & im, const CameraState & state,
			 const Point2i & pt2d, int iLineFeature) {
  matf cov;
  matf p3d = lineFeatures[iLineFeature].cone.getMaxPGlobalCoord(&cov);
  kalman.addNewPoint(p3d, cov);
  int iKalman = kalman.getNPts()-1;
  int dx = lineFeatures[iLineFeature].descriptor.size().width/2;
  int dy = lineFeatures[iLineFeature].descriptor.size().height/2;
  features.push_back(Feature(im, state, iKalman, pt2d, p3d, dx, dy));
  lineFeatures.erase(lineFeatures.begin()+iLineFeature);
}
