#include "SLAM.hpp"
#include <algorithm>
#include <opencv/highgui.h>
using namespace cv;
using namespace std;

extern Mat imdisp_debug;

SLAM::Feature::Feature(const KalmanSLAM & kalman, int iKalman, const Mat_<imtype> & im,
		       const Point2i & pos2d, const matf & pos3d, int dx, int dy)
  :iKalman(iKalman), descriptor(im(Range(pos2d.y-dy, pos2d.y+dy),
				   Range(pos2d.x-dx, pos2d.x+dx)).clone()),
   M(4,3,0.0f), dx(dx), dy(dy) {
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

Mat_<SLAM::imtype> SLAM::Feature::project(const matf & P) const {
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
  return proj;
}

matf SLAM::matchInArea(const Mat_<imtype> & im, const Mat_<imtype> & patch,
		       const Mat_<bool> &patchmask, const Rect & area) const {
  //TODO: this is not optimized enough, and should be!
  const int ph = patch.size().height;
  const int pw = patch.size().width;
  const int dx = area.x - ceil(0.5f*pw);
  const int dy = area.y - ceil(0.5f*ph);
  const int h = im.size().height;
  const int w = im.size().width;
  matf out(area.height, area.width, 0.0f);
  for (int x = 0; x < area.width; ++x)
    for (int y = 0; y < area.height; ++y) {
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
	cc = imarea.dot(patcharea);
	i2 = imarea.dot(imarea);
	p2 = patcharea.dot(patcharea);
	
	out(y, x) = cc / sqrt(i2*p2);
      }
    }
  return out;
}

void SLAM::match(const Mat_<imtype> & im, std::vector<Match> & matches,
		 float threshold) const {
  matf result;
  Point2i maxp;
  double maxv;
  float searchrad = 15; //TODO
  
  matf area(2*searchrad, 2*searchrad);
  matf areaDisp(im.size().height, im.size().width, 0.0f);
  matf P = matf(3,4,0.0f);
  P(Range(0,3),Range(0,3)) = deltaR * kalman.getRot().toMat();
  P(Range(0,3),Range(3,4)) = - P(Range(0,3),Range(0,3)) * kalman.getPos();
  P = K*P;

  for (size_t i = 0; i < features.size(); ++i) {
    // search area
    area.setTo(0);
    matf p = P * kalman.getPt3dH(i);
    int px = floor(p(0)/p(2)+0.5), py = floor(p(1)/p(2)+0.5);
    int hx = searchrad, hy = searchrad;

    if ((px+hx < 0) || (py+hy < 0) || (px-hx >= im.size().width) ||
	(py-hy >= im.size().height))
      continue;

    for (int k = 0; k < 2*hx; ++k)
      for (int j = 0; j < 2*hy; ++j) {
	float d = sq(k-hx)+sq(j-hy);
	if (d < sq(searchrad))
	  area(j,k) = 1.;
      }
    
    //area.copyTo(areaDisp(Range(py-hy,py+hy),Range(px-hx,px+hx)));
    cvCopyToCrop(area, areaDisp, Rect(px-hx, py-hy,hx*2, hy*2));

    Mat_<imtype> proj = features[i].project(P);
    if (i == 4) {
      namedWindow("test2");
      imshow("test2", proj);
      cvWaitKey(1);
    }

    const int dx = floor(features[i].descriptor.size().width*0.5f);
    const int dy = floor(features[i].descriptor.size().height*0.5f);
    //matchTemplate(im, features[i].descriptor, result, CV_TM_CCORR_NORMED);
    /*
    matchTemplate(im, proj, result, CV_TM_CCORR_NORMED);
    matf areaCrop = area(Range(dy,result.size().height+dy),
			 Range(dx,result.size().width+dx));
    result = result.mul(areaCrop);
    minMaxLoc(result, NULL, &maxv, NULL, &maxp);
    */
    result = matchInArea(im, proj, Mat_<bool>(proj.size(), 1),
			 Rect(max(0,px-hx), max(0,py-hy), 2*hx, 2*hy));
    result.mul(area);
    minMaxLoc(result, NULL, &maxv, NULL, &maxp);
    maxp.x += px-hx;
    maxp.y += py-hy;
    if (maxv > threshold) {
      matches.push_back(Match(Point2i(maxp.x, maxp.y), i));
    }
  }

  namedWindow("test");
  imshow("test", areaDisp);
  cvWaitKey(1);
  if (imdisp_debug.size().height != 0) {
    Mat imdisp = imdisp_debug;
    int fromto[] = {0,0};
    Mat channels[3];
    split(imdisp, channels);
    areaDisp.convertTo(channels[0], CV_8U, 255.);
    merge(channels, 3, imdisp);
  }
}

