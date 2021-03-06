#include "SLAM.hpp"
#include <algorithm>
#include <opencv/highgui.h>
using namespace cv;
using namespace std;

extern Mat imdisp_debug;

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

void SLAM::matchPoints(const Mat_<imtype> & im, const CameraState & state,
		       std::vector<Match> & matches, float threshold) {
  matf* pdisp = NULL;
  matf disp[3];
  if (imdisp_debug.size().height != 0) {
    Mat imdisp = imdisp_debug;
    Mat channels[3];
    split(imdisp, channels);
    for (int i = 0; i < 3; ++i)
      channels[i].convertTo(disp[i], CV_32F, 1.f/255.f);
    pdisp = disp;
  }

  int stride = 3;
  vector<float> subsamples;
  subsamples.push_back(1.f);
  subsamples.push_back(3.f);
  ImagePyramid<imtype> pyramid(im, subsamples);

  float response;
  for (size_t i = 0; i < features.size(); ++i) {
    const Point2i pos = features[i].track(pyramid, state,
					  kalman.getPt3d(features[i].iKalman),
					  threshold, stride, response, disp);
    if (response > threshold) {
      matches.push_back(Match(pos, i));
      //if (response > 0.5f+0.5f*threshold)
      //features[i].newDescriptor(im, state, pos,
      //			  kalman.getPt3d(features[i].iKalman), true);
    }
  }

  if (imdisp_debug.size().height != 0) {
    Mat imdisp = imdisp_debug;
    Mat channels[3];
    for (int i = 0; i < 3; ++i)
      disp[i].convertTo(channels[i], CV_8U, 255.);
    merge(channels, 3, imdisp);
  }
}

void SLAM::matchLines(const Mat_<imtype> & im, const CameraState & state,
		      vector<Match> & matches, float threshold) const {
  int stride = 2;
  vector<float> subsamples;
  subsamples.push_back(1);
  subsamples.push_back(2);
  ImagePyramid<imtype> impyramid(im, subsamples); // TODO: computed twice
  matf P = kalman.getP(); //TODO multiple times, careful about up to date


  matf* pdisp = NULL;
  matf disp[3];
  if (imdisp_debug.size().height != 0) {
    Mat imdisp = imdisp_debug;
    Mat channels[3];
    split(imdisp, channels);
    for (int i = 0; i < 3; ++i)
      channels[i].convertTo(disp[i], CV_32F, 1.f/255.f);
    pdisp = disp;
  }
  
  for (size_t i = 0; i < lineFeatures.size(); ++i) {
    //TODO: radius
    //const Point2i pos = trackLine(imsubsamples, subsample, i,
    //				  threshold, P, NULL);
    float response;
    const Point2i pos = lineFeatures[i].track(impyramid, state, threshold, stride,
					      response, pdisp);
    if (response > threshold)
      matches.push_back(Match(pos, i));
  }

  if (imdisp_debug.size().height != 0) {
    Mat imdisp = imdisp_debug;
    Mat channels[3];
    for (int i = 0; i < 3; ++i)
      disp[i].convertTo(channels[i], CV_8U, 255.);
    merge(channels, 3, imdisp);
  }
}
