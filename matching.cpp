#include "SLAM.hpp"
#include <algorithm>
#include <opencv/highgui.h>
using namespace cv;
using namespace std;

extern Mat imdisp_debug;

SLAM::Feature::Feature(const KalmanSLAM & kalman, int iKalman,
		       const Mat_<imtype> & im, const Point2i & pos2d,
		       const matf & pos3d, int dx, int dy, Type type)
  :iKalman(iKalman), descriptor(im(Range(pos2d.y-dy, pos2d.y+dy),
				   Range(pos2d.x-dx, pos2d.x+dx)).clone()),
   M(4,3,0.0f), dx(dx), dy(dy), type(type) {
  //matf Rm = kalman.getRot().inv().toMat();
  matf Rm = kalman.getRot().toMat().inv(); // TODO
  matf P = kalman.getP();
  matf R = P(Range(0,3),Range(0,3));
  matf t = P(Range(0,3),Range(3,4));
  matf Ru = R*Rm.col(0);
  matf Rv = R*Rm.col(1);
  matf Rp = R*pos3d;
  float tp3 = t(2) + Rp(2);
  float alpha = ((Ru(0) - Ru(2)) * tp3 - Ru(2)*(t(0)+Rp(0))) / (tp3*tp3);
  float beta  = ((Rv(1) - Rv(2)) * tp3 - Rv(2)*(t(1)+Rp(1))) / (tp3*tp3);
  M(Range(0,3),Range(0,1)) = Rm.col(0) / alpha;
  M(Range(0,3),Range(1,2)) = Rm.col(1) / beta;
  pos3d.copyTo(M(Range(0,3),Range(2,3)));
  M(3,2) = 1.0f;
  //debug
  matf Pp = R*pos3d + t;
  Pp /= Pp(2);
  matf Pdebug = R*(pos3d+Rm.col(0)/alpha) + t;
  Pdebug /= Pdebug(2);
}

Mat_<SLAM::imtype> SLAM::Feature::project(const matf & P,
					  Mat_<SLAM::imtype> & mask) const {
  matf A = P * M;
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
			       const Point2i & c, int stride, Mat* disp) const {
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
  if (maxv > threshold) {
    if (disp)
      cvCopyToCrop(proj, disp[1], Rect(maxp.x-dx, maxp.y-dy,
				       proj.size().width, proj.size().height));
    return maxp;
  } else
    return Point2i(-1, -1);
}

Point2i SLAM::trackFeature(const Mat_<imtype>* im, int subsample, int ifeature,
			   float threshold,
			   const matf & P, Mat* disp) const {
  float searchrad = 25; // TODO
  int stride = 2;
  matf p = P * kalman.getPt3dH(ifeature);
  Point2i c(round(p(0)/p(2)), round(p(1)/p(2)));
  Mat_<imtype> projmask;
  Mat_<imtype> proj = features[ifeature].project(P, projmask);
  Mat_<imtype> projsubs, projmasksubs;
  resize(proj, projsubs, Size(proj.size().width/2,
			      proj.size().height/2));
  resize(projmask, projmasksubs, projsubs.size());
  Point2i csubs(c.x/subsample, c.y/subsample);

  Point2i pos = trackFeatureElem(im[0], projsubs, projmasksubs,
				 searchrad/subsample,
				 0.75*threshold, csubs, stride, NULL);
  if (pos.x != -1) {
    pos = trackFeatureElem(im[1], proj, projmask, stride*subsample, threshold,
			   Point2i(pos.x*subsample, pos.y*subsample),
			   1, disp);
  }
  return pos;
}

void SLAM::match(const Mat_<imtype> & im, std::vector<Match> & matches,
		 float threshold) const {
  matf disp[2] = {matf(im.size().height, im.size().width, 0.0f),
		  matf(im.size().height, im.size().width, 128.f)};

  matf P = matf(3,4,0.0f);
  P(Range(0,3),Range(0,3)) = deltaR * kalman.getRot().toMat();
  P(Range(0,3),Range(3,4)) = - P(Range(0,3),Range(0,3)) * kalman.getPos();
  P = K*P;

  Mat_<imtype> imsubsampled;
  int subsample = 2;
  resize(im, imsubsampled, Size(im.size().width/subsample,
				im.size().height/subsample));
  Mat_<imtype> imsubsamples[] = {imsubsampled, im};

  for (size_t i = 0; i < features.size(); ++i) {
    const Point2i pos = trackFeature(imsubsamples, subsample, i, threshold, P, disp);
    if (pos.x >= 0) {
      matches.push_back(Match(pos, i));
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

