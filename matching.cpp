#include "SLAM.hpp"
#include <algorithm>
#include <opencv/highgui.h>
using namespace cv;
using namespace std;

extern Mat imdisp_debug;

SLAM::Feature::Feature(const SLAM & slam, int iKalman,
		       const Mat_<imtype> & im, const Point2i & pt2d,
		       const matf & pos3d, int dx, int dy)
  :pslam(&slam), iKalman(iKalman), descriptor(2*dy, 2*dx), B(4,3) {
  newDescriptor(slam.kalman.getP(), im, pt2d); // TODO: P
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
  int dx = descriptor.size().width/2, dy = descriptor.size().height/2;
  Mat_<imtype> newdescr = im(Range(max(0, pt2d.y-dy), min(h, pt2d.y+dy)),
			     Range(max(0, pt2d.x-dx), min(w, pt2d.x+dx)));
  newdescr.copyTo(descriptor);
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
  return proj;
}

Point2i SLAM::matchFeatureInArea(const Mat_<imtype> & im,
				 const Mat_<imtype> & patch,
				 const Mat_<imtype> * patchMask,
				 const Rect & areaRect0,
				 const matb * areaMask,
				 int stride, float & response,
				 bool useExactAreaMask) {
  // sizes + cropping of area
  const int h = im.size().height;
  const int w = im.size().width;
  Rect areaRect = areaRect0;
  areaRect.x = max(areaRect.x, 0);
  areaRect.y = max(areaRect.y, 0);
  areaRect.width = min(w-areaRect.x, areaRect.width + areaRect0.x - areaRect.x);
  areaRect.height = min(h-areaRect.y, areaRect.height + areaRect0.y - areaRect.y);
  const int ah = areaRect.height;
  const int aw = areaRect.width;
  if ((areaRect.x >= w) || (areaRect.y >= h) ||
      (areaRect.x+aw <= 0) || (areaRect.y+ah <= 0)) {
    response = -1;
    return Point2i(0,0);
  }
  matb areaMask0;
  if (areaMask)
    areaMask0 = (*areaMask)(Rect(areaRect.x - areaRect0.x,
				 areaRect.y - areaRect0.y,
				 areaRect.width, areaRect.height));
  const int ph = patch.size().height;
  const int pw = patch.size().width;
  const int x0 = areaRect.x - floor(0.5f*pw);
  const int y0 = areaRect.y - floor(0.5f*ph);
  
  // computation of the "uncropped area" (where the patch is fully used)
  // [x0ua, x1ua[ are useful pixels in the image (size <= aw+pw, x0ua ~= x0)
  // [xl0, xl1[ are useful pixels in the area (size <= aw, xl0 ~= 0)
  int x0ua = max(x0, 0), x1ua = min(w, x0 + aw + pw - 1);
  int y0ua = max(y0, 0), y1ua = min(h, y0 + ah + ph - 1);
  Rect rectua(x0ua, y0ua, x1ua-x0ua, y1ua-y0ua);
  int xl0 = x0ua-x0, xl1 = x1ua-x0-pw+1;
  int yl0 = y0ua-y0, yl1 = y1ua-y0-ph+1;
  Rect rectuascore(xl0, yl0, xl1-xl0, yl1-yl0);

  // variables
  Mat_<imtype> imdotmask;
  matf score(ah, aw, -1.f);
  int x, y;
  
  if (patchMask) {
    const Mat_<imtype> patchMask0 = *patchMask;
    const Mat_<imtype> patch0 = patch.mul(patchMask0); //TODO: do not reallocate

    if (areaMask) { // patchMask, areaMask

      {
	const float p2 = norm(patch0);
	for (x = xl0; x < xl1; x += stride)
	  for (y = yl0; y < yl1; y += stride)
	    if (areaMask0(y, x)) {
	      const Mat_<imtype> imarea = im(Range(y0+y, y0+y+ph),
					     Range(x0+x, x0+x+pw));
	      imdotmask = imarea.mul(patchMask0);
	      const float cc = imdotmask.dot(patch0);
	      const float i2 = norm(imdotmask);
	      score(y, x) = cc / (i2*p2);
	    }
      }
     
      for (x = 0; x < aw; x += stride) {
	const int dl = max(0,-x0-x);
	const int dr = max(0,x0+x+pw-w);
	for (y = 0; y < ah; y += stride) {
	  if ((yl0 <= y) && (y < yl1) && (xl0 <= x) && (x < xl1)) {
	    if (yl1 >= ah)
	      break;
	    else
	      y = yl1;
	  }
	  if (areaMask0(y, x)) {
	    const int dt = max(0,-y0-y);
	    const int db = max(0,y0+y+ph-h);
	    if ((dt+db < ph) && (dr+dl < pw)) {
	      const Mat_<imtype> imarea    =         im(Range(y0+y+dt, y0+y+ph-db),
						        Range(x0+x+dl, x0+x+pw-dr));
	      const Mat_<imtype> patcharea =     patch0(Range(dt, ph-db),
						        Range(dl, pw-dr));
	      const Mat_<imtype> maskarea  = patchMask0(Range(dt, ph-db),
							Range(dl, pw-dr));
	      imdotmask = imarea.mul(maskarea);
	      const float cc = imdotmask.dot(patcharea);
	      const float i2 = norm(imdotmask);
	      const float p2 = norm(patcharea);
	      score(y, x) = cc / (i2*p2);
	    }
	  }
	}
      }
      
    } else { // patchMask, no areaMask

      {
	const float p2 = norm(patch0);
	for (x = xl0; x < xl1; x += stride)
	  for (y = yl0; y < yl1; y += stride) {
	    const Mat_<imtype> imarea = im(Range(y0+y, y0+y+ph),
					   Range(x0+x, x0+x+pw));
	    imdotmask = imarea.mul(patchMask0);
	    const float cc = imdotmask.dot(patch0);
	    const float i2 = norm(imdotmask);
	    score(y, x) = cc / (i2*p2);
	  }
      }
     
      for (x = 0; x < aw; x += stride) {
	const int dl = max(0,-x0-x);
	const int dr = max(0,x0+x+pw-w);
	for (y = 0; y < ah; y += stride) {
	  if ((yl0 <= y) && (y < yl1) && (xl0 <= x) && (x < xl1)) {
	    if (yl1 >= ah)
	      break;
	    else
	      y = yl1;
	  }
	  const int dt = max(0,-y0-y);
	  const int db = max(0,y0+y+ph-h);
	  if ((dt+db < ph) && (dr+dl < pw)) {
	    const Mat_<imtype> imarea    =         im(Range(y0+y+dt, y0+y+ph-db),
						      Range(x0+x+dl, x0+x+pw-dr));
	    const Mat_<imtype> patcharea =     patch0(Range(dt, ph-db),
						      Range(dl, pw-dr));
	    const Mat_<imtype> maskarea  = patchMask0(Range(dt, ph-db),
						      Range(dl, pw-dr));
	    imdotmask = imarea.mul(maskarea);
	    const float cc = imdotmask.dot(patcharea);
	    const float i2 = norm(imdotmask);
	    const float p2 = norm(patcharea);
	    score(y, x) = cc / (i2*p2);
	  }
	}
      }
    }
  } else { // no patchMask
        
    if ((rectuascore.width > 0) && (rectuascore.height > 0))
      matchTemplate(im(rectua), patch, score(rectuascore),
		    CV_TM_CCORR_NORMED); //TODO:stride
    
    if (areaMask) { // no patchMask, areaMask

      if (useExactAreaMask) {
	//TODO: optimize
	for (x = xl0; x < xl1; ++x)
	  for (y = yl0; y < yl1; ++y)
	    if (!areaMask0(y, x))
	      score(y, x) = -1;
      }

      for (x = 0; x < aw; x += stride) {
	const int dl = max(0,-x0-x);
	const int dr = max(0,x0+x+pw-w);
	for (y = 0; y < ah; y += stride) {
	  if ((yl0 <= y) && (y < yl1) && (xl0 <= x) && (x < xl1)) {
	    if (yl1 >= ah)
	      break;
	    else
	      y = yl1;
	  }
	  if (areaMask0(y,x)) {
	    const int dt = max(0,-y0-y);
	    const int db = max(0,y0+y+ph-h);
	    if ((dt+db < ph) && (dr+dl < pw)) {
	      const Mat_<imtype> imarea    =         im(Range(y0+y+dt, y0+y+ph-db),
							Range(x0+x+dl, x0+x+pw-dr));
	      const Mat_<imtype> patcharea =      patch(Range(dt, ph-db),
							Range(dl, pw-dr));
	      const float cc = imarea.dot(patcharea);
	      const float i2 = norm(imarea);
	      const float p2 = norm(patcharea);
	      score(y, x) = cc / (i2*p2);
	    }
	  }
	}
      }

    } else { // no patchMask, no areaMask

      for (x = 0; x < aw; x += stride) {
	const int dl = max(0,-x0-x);
	const int dr = max(0,x0+x+pw-w);
	for (y = 0; y < ah; y += stride) {
	  if ((yl0 <= y) && (y < yl1) && (xl0 <= x) && (x < xl1)) {
	    if (yl1 >= ah)
	      break;
	    else
	      y = yl1;
	  }
	  const int dt = max(0,-y0-y);
	  const int db = max(0,y0+y+ph-h);
	  if ((dt+db < ph) && (dr+dl < pw)) {
	    const Mat_<imtype> imarea    =         im(Range(y0+y+dt, y0+y+ph-db),
						      Range(x0+x+dl, x0+x+pw-dr));
	    const Mat_<imtype> patcharea =      patch(Range(dt, ph-db),
						      Range(dl, pw-dr));
	    const float cc = imarea.dot(patcharea);
	    const float i2 = norm(imarea);
	    const float p2 = norm(patcharea);
	    score(y, x) = cc / (i2*p2);
	  }
	}
      }
    }
  }
    
  double response0;
  Point2i out;
  minMaxLoc(score, NULL, &response0, NULL, &out);
  response = response0;
  out.x += areaRect.x;
  out.y += areaRect.y;
  return out;
}

Point2i SLAM::trackFeatureElem(const Mat_<imtype> & im, const Mat_<imtype> & proj,
			       const Mat_<imtype> & projmask,
			       float searchrad,
			       const Point2i & c, int stride,
			       float & response) const {
  matf result; // TODO: reuse storage
  int hx = ceil(searchrad), hy = ceil(searchrad);
  int ih = im.size().height, iw = im.size().width;
  const int dx = floor(proj.size().width*0.5f);
  const int dy = floor(proj.size().height*0.5f);
  
  if ((c.x+hx < 0) || (c.y+hy < 0) || (c.x-hx >= iw) || (c.y-hy >= ih))
    return Point2i(-1,-1);
  
  matb area(2*hy, 2*hx, 0.f); // TODO reuse
  float sqrad = sq(searchrad);
  for (int i = 0; i < 2*hx; ++i)
    for (int j = 0; j < 2*hy; ++j) {
      float d = sq(i-hx)+sq(j-hy);
      if (d < sqrad)
	area(j,i) = 1;
    }

  Point2i maxp = matchFeatureInArea(im, proj, &projmask,
				    Rect(max(0,c.x-hx), max(0,c.y-hy), 2*hx, 2*hy),
				    &area, stride, response);
  
  return maxp;
}

Point2i SLAM::trackFeature(const Mat_<imtype>* im, int subsample, int ifeature,
			   float threshold, const matf & P,
			   float & response, Mat* disp) const {
  float searchrad = 35; // TODO
  int stride = 2;
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
				 csubs, stride, response);
  if (response > 0.66 * threshold) {
    response = -1.f;
    pos = trackFeatureElem(im[1], proj, projmask, stride*subsample,
			   Point2i(pos.x*subsample, pos.y*subsample),
			   1, response);
    if (disp && (response > threshold))
      cvCopyToCrop(proj, disp[1], Rect(pos.x-floor(0.5*proj.size().width),
				       pos.y-floor(0.5*proj.size().height),
				       proj.size().width, proj.size().height));
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
    if (response > threshold) {
      matches.push_back(Match(pos, i));
      //if (response < 0.5f+0.5f*threshold)
      //features[i].newDescriptor(P, im, pos);
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

void SLAM::matchLines(const Mat_<imtype> & im, vector<Match> & matches,
		      float threshold) const {
  int stride = 1; //TODO: stride, subsample
  vector<float> subsamples;
  subsamples.push_back(1);
  //subsamples.push_back(2);
  ImagePyramid<imtype> impyramid(im, subsamples); // TODO: computed twice
  matf P = kalman.getP(); //TODO multiple times, careful about up to date

  matf* pdisp = NULL;
  matf disp[2];
  if (imdisp_debug.size().height != 0) {
    Mat imdisp = imdisp_debug;
    Mat channels[3];
    split(imdisp, channels);
    channels[0].convertTo(disp[0], CV_32F, 1./255.);
    channels[1].convertTo(disp[1], CV_32F, 1./255.);
    pdisp = disp;
  }

  for (size_t i = 0; i < lineFeatures.size(); ++i) {
    //TODO: radius
    //const Point2i pos = trackLine(imsubsamples, subsample, i,
    //				  threshold, P, NULL);
    float response;
    const Point2i pos = lineFeatures[i].track(impyramid, P, threshold, stride,
					      response, pdisp);
    if (response > threshold)
      matches.push_back(Match(pos, i));
  }

  if (imdisp_debug.size().height != 0) {
    Mat imdisp = imdisp_debug;
    Mat channels[3];
    split(imdisp, channels);
    disp[0].convertTo(channels[0], CV_8U, 255.);
    disp[1].convertTo(channels[1], CV_8U, 255.);
    merge(channels, 3, imdisp);
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
