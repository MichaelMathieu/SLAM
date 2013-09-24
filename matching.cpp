#include "SLAM.hpp"
#include <algorithm>
#include <opencv/highgui.h>
using namespace cv;
using namespace std;

extern Mat imdisp_debug;

SLAM::Feature::Feature(const SLAM & slam, int iKalman,
		       const Mat_<imtype> & im, const Point2i & pos2d,
		       const matf & pos3d, int dx, int dy)
  :pslam(&slam), iKalman(iKalman), descriptor(im(Range(pos2d.y-dy, pos2d.y+dy),
				   Range(pos2d.x-dx, pos2d.x+dx)).clone()),
   B(4,3) {
  computeParams(pos3d);
}

SLAM::Feature::Feature(const SLAM & slam, int iKalman,
		       const Mat_<imtype> & descr,
		       const matf & pos3d)
  :pslam(&slam), iKalman(iKalman), descriptor(descr), B(4,3) {
  computeParams(pos3d);
}

void SLAM::Feature::computeParams(const matf & p3d) {
  matf P = pslam->kalman.getP(); //TODO P
  matf Rm = pslam->kalman.getRot().inv().toMat(); //TODO bufferize
  matf M = P(Range(0,3),Range(0,3));
  matf c = P(Range(0,3),Range(3,4));
  matf Mu = M*Rm.col(0);
  matf Mv = M*Rm.col(1);
  matf Mp = M*p3d;
  float cp3 = c(2) + Mp(2);
  float alpha = ((Mu(0) - Mu(2)) * cp3 - Mu(2)*(c(0)+Mp(0))) / (cp3*cp3);
  float beta  = ((Mv(1) - Mv(2)) * cp3 - Mv(2)*(c(1)+Mp(1))) / (cp3*cp3);
  B(Range(0,3),Range(0,1)) = Rm.col(0) / alpha;
  B(Range(0,3),Range(1,2)) = Rm.col(1) / beta;
  p3d.copyTo(B(Range(0,3),Range(2,3)));
  B(3,0) = B(3,1) = 0.0f;
  B(3,2) = 1.0f;
}

void SLAM::Feature::newDescriptor(const matf & P, const cv::Mat_<imtype> & im,
				  const Point2i & pt2d) {
  int h = im.size().height, w = im.size().width;
  int dx = descriptor.size().width/2, dy = descriptor.size().height/2; //TODO: this can shrink
  Mat_<imtype> newdescr = im(Range(max(0, pt2d.y-dy), min(h, pt2d.y+dy)),
			     Range(max(0, pt2d.x-dx), min(w, pt2d.x+dx)));
  descriptor = newdescr.clone();
  computeParams(pslam->kalman.getPt3d(iKalman));
}

Mat_<SLAM::imtype> SLAM::Feature::project(const matf & P,
					  Mat_<SLAM::imtype> & mask) const {
  matf A = P * B;
  int dx = descriptor.size().width/2, dy = descriptor.size().height/2;
  const float gc[4][2] = {{-dx,-dy},{-dx,dy},{dx,dy},{dx,-dy}}; //TODO: once &for all
  matf corners(4,3);
  matf p (3,1,1.0f);
  for (int i = 0; i < 4; ++i) {
    p(0) = gc[i][0];
    p(1) = gc[i][1];
    corners.row(i) = (A*p).t();
    corners.row(i) /= corners(i,2);
  }
  int xmin = 1e6, xmax = -1e6, ymin = 1e6, ymax = -1e6;
  for (int i = 0; i < 4; ++i) {
    xmin = min(xmin, (int)ceil(corners(i,0)));
    xmax = max(xmax, (int)floor(corners(i,0)));
    ymin = min(ymin, (int)ceil(corners(i,1)));
    ymax = max(ymax, (int)floor(corners(i,1)));
  }
  Mat_<imtype> proj(ymax-ymin, xmax-xmin, 0.0f);

  matf Am = A.inv();
  p(0) = xmin; p(1) = ymin; p(2) = 1.0f;
  Am.col(2) = Am * p;
  Am.row(0) += dx * Am.row(2);
  Am.row(1) += dy * Am.row(2);
  
  warpPerspective(descriptor, proj, Am, proj.size(), WARP_INVERSE_MAP|INTER_LINEAR);
  warpPerspective(Mat_<imtype>(descriptor.size(), 1), mask, Am, proj.size(),
		  WARP_INVERSE_MAP|INTER_NEAREST);
  /*
  //TODO: xmax + 1 ?
  matf Am = A.inv();
  float a, b;
  for (int x = xmin; x < xmax; ++x)
    for (int y = ymin; y < ymax; ++y) {
      p(0) = x; p(1) = y; p(2) = 1.0f;
      p = Am * p;
      a = p(0) / p(2) + dx;
      b = p(1) / p(2) + dy;
      if ((a >= 0) && (b >= 0) && (a < descriptor.size().width) &&
	  (b < descriptor.size().height))
	proj(y-ymin,x-xmin) = descriptor((int)round(b),(int)round(a));
	}
  */
  return proj;
}

matf SLAM::matchInArea(const Mat_<imtype> & im, const Mat_<imtype> & patch,
		       const Mat_<imtype> &patchmask, const Rect & area,
		       const matf & areamask, int stride) const {
  //TODO: this is not optimized enough, and should be!
  const int ph = patch.size().height;
  const int pw = patch.size().width;
  const int dx = area.x - ceil(0.5f*pw);
  const int dy = area.y - ceil(0.5f*ph);
  const int h = im.size().height;
  const int w = im.size().width;
  Mat_<imtype> tmp;
  matf out(area.height, area.width, 0.0f);
  for (int x = 0; x < area.width; x+=stride)
    for (int y = 0; y < area.height; y+=stride) {
      if (areamask(y, x) > 0.5) {
	double cc = 0.f, i2 = 0.f, p2 = 0.f;
	const int dtop    = max(0,-dy-y);
	const int dbottom = max(0,dy+y+ph-h);
	const int dleft   = max(0,-dx-x);
	const int dright  = max(0,dx+x+pw-w);
	if ((dtop+dbottom < ph) && (dright+dleft < pw)) {
	  const Mat_<imtype> imarea = im(Range(dy+y+dtop,dy+y+ph-dbottom),
					 Range(dx+x+dleft,dx+x+pw-dright));
	  const Mat_<imtype> patcharea = patch(Range(dtop,ph-dbottom),
					       Range(dleft,pw-dright));
	  const Mat_<imtype> maskarea = patchmask(Range(dtop, ph-dbottom),
						  Range(dleft,pw-dright));
	  tmp = imarea.mul(maskarea);
	  cc = tmp.dot(patcharea);
	  i2 = tmp.dot(tmp);
	  p2 = patcharea.dot(patcharea);
	  
	  out(y, x) = cc / sqrt(i2*p2);
	}
      }
    }
  return out.mul(areamask);
}

Point2i SLAM::trackFeatureElem(const Mat_<imtype> & im, const Mat_<imtype> & proj,
			       const Mat_<imtype> & projmask,
			       float searchrad, float threshold,
			       const Point2i & c, int stride,
			       float & response, Mat* disp) const {
  matf result; // TODO: reuse storage
  int hx = ceil(searchrad), hy = ceil(searchrad);
  int ih = im.size().height, iw = im.size().width;
  const int dx = floor(proj.size().width*0.5f);
  const int dy = floor(proj.size().height*0.5f);
  
  if ((c.x+hx < 0) || (c.y+hy < 0) || (c.x-hx >= iw) || (c.y-hy >= ih))
    return Point2i(-1,-1);
  
  matf area(2*hy, 2*hx, 0.f); // TODO reuse
  float sqrad = sq(searchrad);
  for (int i = 0; i < 2*hx; ++i)
    for (int j = 0; j < 2*hy; ++j) {
      float d = sq(i-hx)+sq(j-hy);
      if (d < sqrad)
	area(j,i) = 1.;
    }

  if (disp)
    cvCopyToCrop(area, disp[0], Rect(c.x-hx, c.y-hy, hx*2, hy*2));

  result = matchInArea(im, proj, projmask,
		       Rect(max(0,c.x-hx), max(0,c.y-hy), 2*hx, 2*hy),
		       area, stride);
  
  double maxv;
  Point2i maxp;
  minMaxLoc(result, NULL, &maxv, NULL, &maxp);
  maxp.x += c.x-hx;
  maxp.y += c.y-hy;
  response = maxv;
  if (maxv > threshold) {
    if (disp)
      cvCopyToCrop(proj, disp[1], Rect(maxp.x-dx, maxp.y-dy,
				       proj.size().width, proj.size().height));
    return maxp;
  } else
    return Point2i(-1, -1);
}

Point2i SLAM::trackFeature(const Mat_<imtype>* im, int subsample, int ifeature,
			   float threshold, const matf & P,
			   float & response, Mat* disp) const {
  float searchrad = 35; // TODO
  int stride = 3;
  matf p = P * kalman.getPt3dH(features[ifeature].iKalman);
  Point2i c(round(p(0)/p(2)), round(p(1)/p(2)));
  Mat_<imtype> projmask;
  Mat_<imtype> proj = features[ifeature].project(P, projmask);
  Mat_<imtype> projsubs, projmasksubs;
  resize(proj, projsubs, Size(proj.size().width/subsample,
			      proj.size().height/subsample));
  resize(projmask, projmasksubs, projsubs.size());
  Point2i csubs(c.x/subsample, c.y/subsample);

  Point2i pos = trackFeatureElem(im[0], projsubs, projmasksubs,
				 searchrad/subsample,
				 0.66*threshold, csubs, stride, response, NULL);
  response = 0.f;
  if (pos.x != -1) {
    pos = trackFeatureElem(im[1], proj, projmask, stride*subsample, threshold,
			   Point2i(pos.x*subsample, pos.y*subsample),
			   1, response, disp);
  }
  return pos;
}

void SLAM::matchPoints(const Mat_<imtype> & im, std::vector<Match> & matches,
		       float threshold) {
  matf disp[2] = {matf(im.size().height, im.size().width, 0.0f),
		  matf(im.size().height, im.size().width, 128.f)};

  matf P = matf(3,4,0.0f);
  P(Range(0,3),Range(0,3)) = deltaR * kalman.getRot().toMat();
  P(Range(0,3),Range(3,4)) = - P(Range(0,3),Range(0,3)) * kalman.getPos();
  P = K*P;

  Mat_<imtype> imsubsampled;
  int subsample = 3;
  resize(im, imsubsampled, Size(im.size().width/subsample,
				im.size().height/subsample));
  Mat_<imtype> imsubsamples[] = {imsubsampled, im};

  float response;
  for (size_t i = 0; i < features.size(); ++i) {
    const Point2i pos = trackFeature(imsubsamples, subsample, i,
				     threshold, P, response, disp);
    if (pos.x >= 0) {
      matches.push_back(Match(pos, i));
      if (response < 0.5f+0.5f*threshold)
	features[i].newDescriptor(P, im, pos);
    }
  }

  if (imdisp_debug.size().height != 0) {
    Mat imdisp = imdisp_debug;
    int fromto[] = {0,0};
    Mat channels[3];
    split(imdisp, channels);
    disp[0].convertTo(channels[0], CV_8U, 255.);
    disp[1].convertTo(channels[1], CV_8U);
    merge(channels, 3, imdisp);
  }
}

Point2i SLAM::trackLine(const Mat_<imtype>* im, int subsample, int iline,
			   float threshold, const matf & P, Mat* disp) const {
  const LineFeature & line = lineFeatures[iline];
  int stride = 2;
  Mat_<imtype> proj = line.descriptor;
  Mat_<imtype> projmask(proj.size(), 1);
  Mat_<imtype> projsubs, projmasksubs;
  resize(proj, projsubs, Size(proj.size().width/2,
			      proj.size().height/2));
  resize(projmask, projmasksubs, projsubs.size());
  
  Point2i out(-1,-1);
  float bestresponse = threshold, response;
  for (size_t ipot = 0; ipot < line.nPot(); ++ipot) {
    matf pos3d = line.posPot[ipot];
    matf cov = line.covPot[ipot];
    //float searchrad = determinant(cov); // TODO
    float searchrad = 15; //TODO
    matf p = P * pos3d;
    Point2i c(round(p(0)/p(2)), round(p(1)/p(2)));
    Point2i csubs(c.x/subsample, c.y/subsample);
    Point2i pos = trackFeatureElem(im[0], projsubs, projmasksubs,
				   searchrad/subsample, 0.75*threshold,
				   csubs, stride, response, NULL);
    if (pos.x != -1) {
      pos = trackFeatureElem(im[1], proj, projmask, stride*subsample, threshold,
			     Point2i(pos.x*subsample, pos.y*subsample),
			     1, response, disp);
      if (response > bestresponse) {
	bestresponse = response;
	out = pos;
      }
    }
  }
  return out;
}

void SLAM::matchLines(const Mat_<imtype> & im, vector<Match> & matches,
		      float threshold) const {
  Mat_<imtype> imsubsampled;
  int subsample = 2;
  //TODO: this is computed twice
  resize(im, imsubsampled, Size(im.size().width/subsample,
				im.size().height/subsample));
  matf P = kalman.getP(); //TODO multiple times
  Mat_<imtype> imsubsamples[] = {imsubsampled, im};

  for (size_t i = 0; i < lineFeatures.size(); ++i) {
    //TODO: radius
    const Point2i pos = trackLine(imsubsamples, subsample, i,
				  threshold, P, NULL);
    if (pos.x >= 0)
      matches.push_back(Match(pos, i));
  }
}

matf SLAM::getLocalCoordinates(const matf & p2d) const {
  matf R = kalman.getRot().toMat();
  matf Minv = (K * R).inv(); //TODO: once and for all
  matf p (3,1); p(0) = p2d(0); p(1) = p2d(1); p(2) = 1.0f;
  matf out(3,3);
  matf a = out.col(0);
  a = Minv * p;
  a /= norm(a);
  //int k = (abs(a(0))<min(abs(a(1)),abs(a(0))))?0:((abs(a(1))<abs(a(2)))?1:2);
  //matf v (3, 1, 0.0f); v(k) = 1.0f;
  matf v = R.col(0);
  a.cross(v).copyTo(out.col(1));
  a.cross(out.col(1)).copyTo(out.col(2));
  return out;
}
