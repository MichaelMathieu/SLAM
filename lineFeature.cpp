#include "SLAM.hpp"
using namespace std;
using namespace cv;

SLAM::LineFeature::LineFeature(const Mat_<imtype> & im, const CameraState & state,
			       const Point2f & pt2d, int rx, int ry)
  :descriptor(im(Range(max(0, iround(pt2d.y-ry)),
		       min(im.size().height, iround(pt2d.y+ry+1))),
		 Range(max(0, iround(pt2d.x-rx)),
		       min(im.size().width, iround(pt2d.x+rx+1)))).clone()),
   cone(state.getLocalCoordinatesPoint(pt2d), state.t, 3.f, state.f, 5,
	100, 20, 3) /*TODO: cone parameters*/ {
}

void SLAM::LineFeature::newView(const CameraState & state, const matf & pt2d) {
  FCone newCone(state.getLocalCoordinatesPoint(pt2d), state.t,
		3./*TODO, other too*/, state.f);
  cone.intersect(newCone);
}

Point2i SLAM::LineFeature::track(const ImagePyramid<imtype> & pyramid,
				 const CameraState & state, float threshold,
				 int stride, float & response, Mat* disp) const {
  int nSubs = pyramid.nSubs();
  float maxWidth = 100, maxHeight = 100;

  // compute tracking area (at lowest resolution)
  float areaRes = 1.f / pyramid.subsamples[nSubs-1];
  vector<int> relevantBins;
  vector<Point2i> binCenters;
  vector<float> binProjRads;
  Point3i nBinsDims = cone.getNBins();
  float nBins = nBinsDims.x*nBinsDims.y*nBinsDims.z;
  float relevantThreshold = 0.2/nBins;
  assert(nBinsDims.x > 1); // TODO: handle this case
  int nBinLayer = nBinsDims.y*nBinsDims.z;
  int xmin=1000000000, xmax=-1, ymin=1000000000, ymax=-1;
  matf p;
  for (int di = 0; di < nBinsDims.x; ++di)
    for (int xi = 0; xi < nBinsDims.y; ++xi)
      for (int yi = 0; yi < nBinsDims.z; ++yi) {
	bool isRelevant = cone.getProba(di, xi, yi) > relevantThreshold;
	if (isRelevant)
	  relevantBins.push_back(binCenters.size());
	// center
	matf binCenter = cone.getBinCenterGlobalCoord(di, xi, yi);
	p = state.project(binCenter);
	Point2i scaledPt(round(p(0)*areaRes), round(p(1)*areaRes));
	binCenters.push_back(scaledPt);
	// radius
	if (di == 0) {
	  binProjRads.push_back(0);
	} else {
	  const Point2i & ptprev = *(binCenters.end()-nBinLayer);
	  float projRad = norm(scaledPt-ptprev);
	  binProjRads.push_back(projRad);
	  if (isRelevant) {
	    xmin = min(xmin, (int)floor((p(0)-projRad)*areaRes));
	    ymin = min(ymin, (int)floor((p(1)-projRad)*areaRes));
	    xmax = max(xmax, (int)ceil ((p(0)+projRad+1)*areaRes));
	    ymax = max(ymax, (int)ceil ((p(1)+projRad+1)*areaRes));
	  }
	}
      }
  for (int xi = 0, k = 0; xi < nBinsDims.y; ++xi)
    for (int yi = 0; yi < nBinsDims.z; ++yi, ++k) {
      const Point2i & c = binCenters[k];
      float projRad = binProjRads[k+nBinLayer];
      binProjRads[k] = projRad;
      if (cone.getProba(0, xi, yi) > relevantThreshold) {
	xmin = min(xmin, (int)floor((c.x-projRad)*areaRes));
	ymin = min(ymin, (int)floor((c.y-projRad)*areaRes));
	xmax = max(xmax, (int)ceil ((c.x+projRad+1)*areaRes));
	ymax = max(ymax, (int)ceil ((c.y+projRad+1)*areaRes));
      }    
    }
  // if the area is too big, don't try to match
  // TODO: if that happens several times, drop the point
  if ((xmax-xmin > maxWidth) || (ymax-ymin > maxHeight)) {
    response = -1;
    return Point2i(-1,-1);
  }

  Rect areaRect = Rect(xmin, ymin, xmax-xmin, ymax-ymin);
  matb areaMask(ymax-ymin, xmax-xmin, (byte)0);
  for (size_t i0 = 0; i0 < relevantBins.size(); ++i0) {
    size_t i = relevantBins[i0];
    Point2i & pt = binCenters[i];
    pt.x -= xmin;
    pt.y -= ymin; //Careful, it is modified in binCenters!
    const int radius = round(binProjRads[i]);
    //line(areaMask, pt, pt, Scalar(255), diameter);
    circle(areaMask, pt, radius, Scalar(255), -1);
  }
  //namedWindow("debug");
  //imshow("debug", areaMask);
  //cvWaitKey(0);
  //exit(0);

  if (disp) {
    matf imdisp = disp[0];
    cvCopyToCrop(matf(areaRect.height/areaRes, areaRect.width/areaRes, 0.25f),
		 imdisp,
		 Rect(areaRect.x/areaRes, areaRect.y/areaRes,
		      areaRect.width/areaRes, areaRect.height/areaRes));
    for (int i = 0; i < areaRect.width; ++i)
      for (int j = 0; j < areaRect.height; ++j) {
	if (areaMask(j,i)) {
	  int y = (areaRect.y+j)/areaRes, x = (areaRect.x+i)/areaRes;
	  if ((x >= 0) && (y >= 0) && (x < imdisp.size().width) &&
	      (y < imdisp.size().height))
	    imdisp(y, x) = 1.f;
	}
      }
  }

  // multi resolution tracking
  Mat_<imtype> totrack;
  float desch = descriptor.size().height, descw = descriptor.size().width;
  // 1) track in the area at lowest resolution
  resize(descriptor, totrack, Size(round(descw*areaRes), round(desch*areaRes)));
  Point2i trackedPoint = matchFeatureInArea(pyramid.images[nSubs-1],
					    totrack, NULL, areaRect,
					    &areaMask, stride, response);
  
  cout << "Tracked " << trackedPoint*(1./areaRes) << " response=" << response<< endl;
  trackedPoint *= 1./areaRes;
  
  if (disp) {
    matf imdisp = disp[1];
    int dh = descriptor.size().height, dw = descriptor.size().width;
    imdisp *= 255;
    cvCopyToCrop(descriptor, imdisp, Rect(trackedPoint.x-dw/2, trackedPoint.y-dh/2, dw, dh));
    imdisp /= 255.;
  }
  //return trackedPoint;

  //TODO: reactivate (and i think the first step is at res 1)
  // 2) refine tracking
  if (response > threshold * 0.67) {
    for (int iSub = nSubs-2; iSub >= 0; --iSub) {
      float lastsub = pyramid.subsamples[iSub+1];
      float sub = pyramid.subsamples[iSub];
      int laststride = stride;
      int newstride = (iSub == 0) ? 1 : stride;
      int searchRad = sub/lastsub*laststride/newstride;
      Rect areaRect(trackedPoint.x/sub-searchRad, trackedPoint.y/sub-searchRad,
		    2*searchRad+1, 2*searchRad+1);
      resize(descriptor, totrack, Size(descw/sub, desch/sub));
      trackedPoint = SLAM::matchFeatureInArea(pyramid.images[iSub], totrack,
					      NULL, areaRect, NULL, newstride,
					      response);
      trackedPoint *= sub;
      if (response < threshold * 0.67)
	break;
    }
  }
  return trackedPoint;
}		 
