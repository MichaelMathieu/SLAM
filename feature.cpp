#include "SLAM.hpp"
using namespace std;
using namespace cv;

SLAM::Feature::Feature(const Mat_<imtype> & im, const CameraState & state,
		       int iKalman, const Point2f & pt2d, const matf & p3d,
		       int rx, int ry)
  :iKalman(iKalman), descriptor(2*ry+1, 2*rx+1), B(4,3) {
  newDescriptor(im, state, pt2d, p3d);
}

SLAM::Feature::Feature(const CameraState & state, const Mat_<imtype> & descr,
		       int iKalman, const matf & p3d)
  :iKalman(iKalman), descriptor(descr), B(4,3) {
  computeParams(state, p3d);
}

void SLAM::Feature::computeParams(const CameraState & state, const matf & p3d) {
  const matf & P = state.P;
  const matf localCoord = state.getLocalCoordinates();
  matf M = P(Range(0,3),Range(0,3));
  matf c = P(Range(0,3),Range(3,4));
  matf Mu = M*localCoord.col(0);
  matf Mv = M*localCoord.col(1);
  matf Mp = M*p3d;
  float cp3 = c(2) + Mp(2);
  float alpha = ((Mu(0) - Mu(2)) * cp3 - Mu(2)*(c(0)+Mp(0))) / (cp3*cp3);
  float beta  = ((Mv(1) - Mv(2)) * cp3 - Mv(2)*(c(1)+Mp(1))) / (cp3*cp3);
  B(Range(0,3),Range(0,1)) = localCoord.col(0) / alpha;
  B(Range(0,3),Range(1,2)) = localCoord.col(1) / beta;
  //p3d.copyTo(B(Range(0,3),Range(2,3)));
  B(3,0) = B(3,1) = 0.0f;
  B(3,2) = 1.0f;
}

void SLAM::Feature::newDescriptor(const cv::Mat_<imtype> & im,
				  const CameraState & state,
				  const Point2f & pt2d, const matf & p3d,
				  bool ignoreIfOnBorder) {
  int h = im.size().height, w = im.size().width;
  int rx = descriptor.size().width/2, ry = descriptor.size().height/2;
  Mat_<imtype> newdescr = im(Range(max(0, iround(pt2d.y-ry)),
				   min(h, iround(pt2d.y+ry+1))),
			     Range(max(0, iround(pt2d.x-rx)),
				   min(w, iround(pt2d.x+rx+1))));
  if (!ignoreIfOnBorder || ((newdescr.size().width == descriptor.size().width) &&
			    (newdescr.size().height == descriptor.size().height))) {
    newdescr.copyTo(descriptor);
    computeParams(state, p3d);
  }
}

Mat_<SLAM::imtype> SLAM::Feature::project(const CameraState & state, const matf & p3d,
					  Mat_<SLAM::imtype> & mask,
					  Rect & location) const {
  if ((descriptor.size().width <= 0) || (descriptor.size().height <= 0)) {
    location = Rect(0,0,0,0);
    return Mat_<imtype>(0,0);
  }
  
  p3d.copyTo(B(Range(0,3),Range(2,3)));
  matf A = state.P * B;
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
  location.x = xmin;
  location.y = ymin;
  location.width = max(0,xmax-xmin);
  location.height = max(0,ymax-ymin);
  if ((location.width == 0) || (location.height == 0)
      || (location.width > 500) || (location.height > 500)) // TODO
    return Mat_<imtype>(0,0);
  
  Mat_<imtype> proj(location.height, location.width, 0.0f);

  matf Am = A.inv();
  p(0) = xmin; p(1) = ymin; p(2) = 1.0f;
  Am.col(2) = Am * p;
  Am.row(0) += dx * Am.row(2);
  Am.row(1) += dy * Am.row(2);
  
  warpPerspective(descriptor, proj, Am, proj.size(), WARP_INVERSE_MAP|INTER_LINEAR);
  warpPerspective(Mat_<imtype>(descriptor.size(), 1), mask, Am, proj.size(),
		  WARP_INVERSE_MAP|INTER_NEAREST);
  return proj;
}

Point2i SLAM::Feature::track(const ImagePyramid<imtype> & pyramid,
			     const CameraState & state, const matf & p3d,
			     float threshold, int stride, float & response,
			     Mat* disp) const {
  int nSubs = pyramid.nSubs();
  Rect proj_location; // TODO: use proj_location
  Mat_<imtype> proj_mask;
  Mat_<imtype> proj = project(state, p3d, proj_mask, proj_location);
  int projw = proj.size().width, projh = proj.size().height;

  if ((projw <= 0) || (projh <= 0)) {
    response = -1;
    return Point2i(-1,-1);
  }

  matf p2d = state.project(p3d);

  float fullResSearchRad = 20; //TODO

  Mat_<imtype> totrack, totrack_mask;
  Point2f trackedPoint(p2d(0), p2d(1));
  float sub, searchRad;
  for (int iSub = nSubs-1; iSub >= 0; --iSub) {
    sub = pyramid.subsamples[iSub];
    if (sub == 1) {
      totrack = proj;
      totrack_mask = proj_mask;
    } else {
      resize(proj, totrack, Size(projw/sub, projh/sub));
      resize(proj_mask, totrack_mask, Size(projw/sub, projh/sub));
    }
    if (iSub == nSubs-1) {
      searchRad = fullResSearchRad / sub;
    } else {
      searchRad = 1.3 * pyramid.subsamples[iSub+1]/sub;
      if (iSub == 0)
	searchRad *= stride;
    }
    if (disp)
      for (int idisp = 0 ; idisp < 3; ++idisp)
	circle(disp[idisp], trackedPoint, searchRad * ((iSub == 0) ? 1 : stride),
	       Scalar(0.5 + 0.5*iSub/nSubs), 2);
    Rect areaRect(round(trackedPoint.x/sub - searchRad),
		  round(trackedPoint.y/sub - searchRad),
		  round(2*searchRad+1), round(2*searchRad+1));
    trackedPoint = matchFeatureInArea(pyramid.images[iSub], totrack,
				      &totrack_mask, areaRect, NULL,
				      (iSub == 0) ? 1 : stride, response);
    trackedPoint *= sub;
    if (response < 0.67 * threshold)
      return Point2i(trackedPoint.x, trackedPoint.y);
  }
  
  if (disp) {
    Rect dstrect(trackedPoint.x-floor(0.5*proj.size().width),
		 trackedPoint.y-floor(0.5*proj.size().height),
		 proj.size().width, proj.size().height);
    Rect dstrectC(max(dstrect.x, 0), max(dstrect.y, 0), 0, 0);
    dstrectC.width  = min(dstrect.width , disp[0].size().width  - dstrectC.x);
    dstrectC.height = min(dstrect.height, disp[0].size().height - dstrectC.y);
    disp[0](dstrectC) *= 0.5;
    if (response > threshold) {
      disp[2](dstrectC) *= 0.5;
      cvConvertToCrop(proj, disp[1], dstrect, CV_32F, 1.f/255.f);
    } else {
      disp[1](dstrectC) *= 0.5;
      cvConvertToCrop(proj, disp[2], dstrect, CV_32F, 1.f/255.f);
    }
  }

  return Point2i(trackedPoint.x, trackedPoint.y);
}
