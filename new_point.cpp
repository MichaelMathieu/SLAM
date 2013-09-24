#include "SLAM.hpp"
#include <cmath>
using namespace cv;
using namespace std;

matf SLAM::getCovariancePointAlone(const matf & pt2d,
				   float Lambda, float lambda) const {
  matf eigv = getLocalCoordinates(pt2d);
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

void SLAM::computeNewLines(const matb & im, const vector<Match> & matches,
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
    addNewLine(im, pt, rx, ry);
    newpoints.push_back(i);
    --n;
  badAddNewFeature:;
  }
}

void SLAM::addNewLine(const Mat_<imtype> & im, const Point2f & pt2d,
		      int rx, int ry) {
  cout << "Adding new line" << endl;
  matf x(3,1);
  x(0) = pt2d.x; x(1) = pt2d.y; x(2) = 1.0f;
  const Mat_<imtype> desc = \
    im(Range(max(0.f, pt2d.y-ry), min((float)im.size().height, pt2d.y+ry)),
       Range(max(0.f, pt2d.x-rx), min((float)im.size().width , pt2d.x+rx)));
  lineFeatures.push_back(LineFeature(x, *this, desc.clone(), 5));
  //TODO: P is computed once again
}

void SLAM::lineToFeature(const Mat_<imtype> & im, const Point2i & pt2d,
			 int iLineFeature) {
  matf cov;
  matf p3d = lineFeatures[iLineFeature].cone.getMaxPGlobalCoord(&cov);
  kalman.addNewPoint(p3d, cov);
  int iKalman = kalman.getNPts()-1;
  int dx = lineFeatures[iLineFeature].descriptor.size().width/2;
  int dy = lineFeatures[iLineFeature].descriptor.size().height/2;
  features.push_back(Feature(*this, iKalman, im, pt2d, p3d, dx, dy));
  lineFeatures.erase(lineFeatures.begin()+iLineFeature);
}

void SLAM::LineFeature::addNewPoint(const matf & pos, const matf & cov, float w) {
  posPot.push_back(pos);
  covPot.push_back(cov);
  wPot.push_back(w/sqrt(pow(2.f*3.1415926f, 3)*determinant(cov)));
}

float SLAM::LineFeature::getCovOnLineFromDepth(float depth) const {
  float lambda = (dmax-dmin)/nPotAtStart;
  return lambda;
}

matf SLAM::LineFeature::getCovOfPoint(const matf & p2d, float d, float lambda) const {
  float sigmaP = 3; // TODO: sigma in pixels on the 2d frame
  float f = 0.5*(pslam->K(0,0) + pslam->K(1,1)); // focal distance in pixels
  float sigma = d * sigmaP / f;
  return pslam->getCovariancePointAlone(p2d, lambda, sigma);
}

SLAM::LineFeature::LineFeature(const matf & pt2dh, const SLAM & slam,
			       const Mat_<imtype> & desc, int n)
  :pslam(&slam), pCenter(4,1), pInf(4,1), descriptor(desc), p2d0(pt2dh.rowRange(0,2)),
   P0(slam.kalman.getP()), nPotAtStart(n), dmin(5), dmax(100),/*TODO*/
   
   cone(slam.getLocalCoordinates(pt2dh), slam.kalman.getPos(), 3.f,
	0.5*(slam.K(0,0)+slam.K(1,1)), 5, 100, 20,3)
{
  //pCenter, pInf
  const matf & P = P0; // TODO: recomputed
  matf M = P(Range(0,3),Range(0,3));
  matf Minv = M.inv(); // TODO: recomputed
  cvrange(pCenter,0,3) = -Minv*P.col(3);
  pCenter(3) = 1.0f;
  cvrange(pInf,0,3) = Minv * pt2dh;
  pInf(3) = 0.0f;

  for (int i = 0; i < n; ++i) {
    float d = (dmax-dmin)*((float)i)/(n-1)+dmin;
    float lambda = getCovOnLineFromDepth(d);
    matf posPot = pCenter + d * pInf/norm(pInf);
    matf covPot = getCovOfPoint(pt2dh, d, lambda);
    addNewPoint(posPot, covPot, 1.f/n);
  };
}
/*
void SLAM::LineFeature::newViewPot(int iPot, const matf & P, const matf & p2d,
				   const matf & dir) {
  //1) find angle between the rays
  float cosangle = cvrange(pInf,0,3).dot(dir)/(norm(cvrange(pInf,0,3))*norm(dir));
  cout << cosangle << endl;
  if (cosangle < 0.999) {
    //2) triangulate
    matf newPos(4,1);
    triangulatePoints(P0, P, p2d0, p2d, newPos); //should be done before...
    if (abs(newPos(3)) > 1e-5) {
      newPos /= newPos(3);
      newPos = cvrange(newPos,0,3);
      cout <<"3d localization " << newPos << endl;
      float d = norm(pslam->kalman.getPos()-newPos);
      float lambda = getCovOnLineFromDepth(d);
      matf newCov = getCovOfPoint(p2d, d, lambda);
      matf oldPos = posPot[iPot];
      oldPos /= oldPos(3);
      oldPos = cvrange(oldPos, 0, 3);
      matf oldCov = covPot[iPot];
      matf oldCovinv = oldCov.inv(), newCovinv = newCov.inv();
      matf postCov = (oldCovinv + newCovinv).inv();
      matf postPos = postCov * (oldCovinv * oldPos + newCovinv * newPos);
      postPos.copyTo(cvrange(posPot[iPot], 0, 3));
      posPot[iPot](3) = 1.0f;
      covPot[iPot] = postCov;
      cout << trace(postCov)[0] << "  " << postPos << endl;
      wPot *= 
    }
  }
}
*/
void SLAM::LineFeature::newView(const matf & pt2d) {
  //matf P = pslam->kalman.getP();
  float f = 0.5f*(pslam->K(0,0)+pslam->K(1,1)); //TODO
  FCone newCone(pslam->getLocalCoordinates(pt2d), pslam->kalman.getPos(),
		3./*TODO, other too*/, f);
  //  cout << "before  " << cone.getVal(0,0,0) << endl;
  cone.intersect(newCone);
  //cone.print();
  //cout << "after   " << cone.getVal(0,0,0) << endl;
  /*
  matf P = kalmanSLAM.getP(); //TODO again P
  matf pt2dh(3,1); pt2dh(0) = pt2d(0); pt2dh(1) = pt2d(1); pt2dh(2) = 1.0f;
  matf dir = P(Range(0,3),Range(0,3)).inv() * pt2dh; //TODO: Minv once
  dir /= norm(dir);
  for (size_t ipot = 0; ipot < posPot.size(); ++ipot) {
    newViewPot(ipot, P, pt2d, dir);
  }
  */
}
