#include "SLAM.hpp"
#include <opencv/highgui.h>
#include<signal.h>
using namespace std;
using namespace cv;

inline float sq(float a) {
  return a*a;
}
inline float cvval(matf a) {
  return a(0);
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

void SLAM::match(const matb & im, std::vector<Match> & matches,
		 float threshold) const {
  matf result;
  Point2i maxp;
  double maxv;
  
  matf area(im.size().height, im.size().width);
  matf areaDisp(im.size().height, im.size().width, 0.0f);
  matf P = matf(3,4,0.0f);
  P(Range(0,3),Range(0,3)) = kalman.getRot().toMat() * deltaR;
  P(Range(0,3),Range(3,4)) = - P(Range(0,3),Range(0,3)) * kalman.getPos();
  P = K*P;

  for (size_t i = 0; i < features.size(); ++i) {
    // search area
    area.setTo(0);
    matf p = P * kalman.getPt3dH(i);
    int px = floor(p(0)/p(2)+0.5), py = floor(p(1)/p(2)+0.5);
    float sigmagauss = 15; // TODO
    int hx = sigmagauss, hy = sigmagauss;
    //float kgauss = 1./sqrt(2.*3.14159*sigmagauss);
    for (int k = max(0,px-hx); k < min(im.size().width,px+hx); ++k)
      for (int j = max(0,py-hy); j < min(im.size().height,py+hy); ++j) {
	float d = sq(k-px)+sq(j-py);
	//area(j,k) = exp(-0.5*d/sigmagauss);
	if (d < sq(sigmagauss)) {
	  area(j,k) = 1.;
	  areaDisp(j,k) = 1.;
	}
      }

    matchTemplate(im, features[i].descriptor, result, CV_TM_CCORR_NORMED);
    const int dx = floor(features[i].descriptor.size().width*0.5f);
    const int dy = floor(features[i].descriptor.size().height*0.5f);
    matf areaCrop = area(Range(dy,result.size().height+dy),
			 Range(dx,result.size().width+dx));
    result = result.mul(areaCrop);
    minMaxLoc(result, NULL, &maxv, NULL, &maxp);
    if (maxv > threshold) {
      matches.push_back(Match(Point2i(maxp.x+dx, maxp.y+dy), i));
    }
  }
  namedWindow("test");
  imshow("test", areaDisp);
  cvWaitKey(1);
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
    features.push_back(Feature(matf(),
			       im(Range(pt.y-dy, pt.y+dy),
				  Range(pt.x-dx, pt.x+dx)),
			       matf::zeros(3,3)));

    newpoints.push_back(i);
    // add the new point to the kalman filter
    {
      matf x(3,1);
      x(0) = pt.x; x(1) = pt.y; x(2) = 1.0f;
      matf P = kalman.getP();
      matf px = P.t() * (P * P.t()).inv() * x;
      cout << "Adding new point px " << px << endl;
      px = px/px(3);
      px = px(Range(0,3),Range(0,1));
      matf cov = 10*matf::eye(3,3);
      kalman.addNewPoint(px, cov);
    }
    //
    --n;
  badAddNewFeature:;
  }
}

matf rotmat2TaitBryan(const matf & M) {
  matf out(3,1);
  //TODO: this only works if -pi/2 < beta < pi/2
  out(1) = asin(M(2,0));
  out(0) = atan2(M(2,1), M(2,2));
  out(2) = atan2(M(1,0), M(0,0));
  return out;
}

void SLAM::waitForInit() {
  do {
    mongoose.FetchMongoose();
  } while (!mongoose.isInit);
  matf R = mongooseAlign.inv() * mongoose.rotmat.inv() * mongooseAlign;
  R.copyTo(lastR);
}

void SLAM::newImage(const matb & im) {
  mongoose.FetchMongoose();
  if (features.size() == 0) {
    newInitImage(im);
  } else {
    vector<KeyPoint> keypoints;
    getSortedKeyPoints(im, 10*minTrackedPerImage, keypoints);

    matf R = mongooseAlign.inv() * mongoose.rotmat.inv() * mongooseAlign;
    deltaR = lastR.inv() * R;
    //kalman.setRVel(rotmat2TaitBryan(deltaR));
    R.copyTo(lastR);

    cout << kalman.getPos() << endl;
    
    vector<Match> matches;
    match(im, matches, 0.99f);
    addNewFeatures(im, keypoints, matches, minTrackedPerImage-matches.size(), 100.f);
    //cout << features.size() << " " << matches.size() << endl;

    if (matches.size() > 0) {
      //TODO: the next delta is greater
#if 0
      vector<Match> matches0 = matches;
      matches.clear();
      for (int i = 0; i < (signed)matches0.size(); ++i)
	if (matches0[i].iFeature < 4)
	  matches.push_back(matches0[i]);
#endif

      matf y(2*matches.size(),1);
      vector<int> activePts;
      for (int i = 0; i < (signed)matches.size(); ++i) {
	y(2*i) = matches[i].pos.x;
	y(2*i+1) = matches[i].pos.y;
	activePts.push_back(matches[i].iFeature);
      }
      float delta = 0.1f;
      kalman.setActivePoints(activePts);
      kalman.update(matf(0,0), y, &delta);
      kalman.renormalize();
    }
    
    Mat imdisp; 
    cvtColor(im, imdisp, CV_GRAY2RGB);

    //matf Pdebug(3,3,0.0f); Pdebug(2,0) = 1; Pdebug(0,1) = 1; Pdebug(1,2) = -1;
    //matf P = debugmatf*((debugmatf2*Pdebug).inv()*mongoose.rotmat*Pdebug).inv();
    /*
    if (p0debug.size().height == 0)
      p0debug = kalman.getPos();
    for (size_t i = 0; i < features.size(); ++i) {
      matf p = kalman.getPt3d(i).clone();
      p = K * P * (p - p0debug);
      p = p / p(2);
      if ((p(0) >= 5) && (p(0) < imdisp.size().height-5) &&
	  (p(1) >= 5) && (p(1) < imdisp.size().width-5)) {
	cout << p << endl;
	circle(imdisp, Point2f(p(0), p(1)), 4, Scalar(255, 0, 0));
      }
    }
    */

    for (size_t i = 0; i < matches.size(); ++i) {
      circle(imdisp, matches[i].pos, 5, Scalar(0, 0, 255));
      matf p = kalman.getPt3d(matches[i].iFeature).clone();
      p = K * kalman.getRot().toMat() * (p - kalman.getPos());
      p = p / p(2);
      circle(imdisp, Point2f(p(0), p(1)), 4, Scalar(0, 255, 0));
    }
    for (size_t i = 0; i < keypoints.size(); ++i)
      circle(imdisp, keypoints[i].pt, 2, Scalar(255, 0, 0));

    imshow("disp", imdisp);
    cvWaitKey(1);
    visualize();
  }
}

bool SLAM::newInitImage(const matb & im, const Size & pattern) {
  vector<Point2f> corners;
  bool found = findChessboardCorners(im, pattern, corners,
				     CALIB_CB_FAST_CHECK);
  Mat imdisp;
  cvtColor(im, imdisp, CV_GRAY2RGB);
  drawChessboardCorners(imdisp, pattern, corners, found);
  imshow("disp", imdisp);
  cvWaitKey(1);
  
  if (found) {
    
    vector<Point3f> pts3d;
    Mat_<double> rvec, tvec;
    for (int i = 0; i < pattern.height; ++i)
      for (int j = 0; j < pattern.width; ++j)
	pts3d.push_back(Point3f(i,j,0.0f));
    solvePnP(pts3d, corners, K, matf(5,1,0.0f), rvec, tvec);
    Mat_<double> rotmat(3,3,0.0f);
    Rodrigues(rvec, rotmat);
    Quaternion q(rotmat);
    Mat_<double> Kd(3,3);
    for (int i = 0; i <9; ++i)
      Kd(i) = K(i);
    tvec = -rotmat.inv()*tvec;
    matf tvecf(3,1);
    for (int i = 0; i < 3; ++i)
      tvecf(i) = tvec(i);

    kalman.setPos(tvecf);
    kalman.setRot(q);
    //debugmatf = q.toMat();

    int gc[] = {0, pattern.width-1, (pattern.height-1)*pattern.width,
		pattern.width*pattern.height-1};
    matf cornersPos[] = {matf3(0, 0, 0),
			 matf3(0, pattern.width-1, 0),
			 matf3(pattern.height-1, 0, 0),
			 matf3(pattern.height-1, pattern.width-1, 0)};
    vector<KeyPoint> goodCorners;
    for (int i = 0; i < 4; ++i)
      goodCorners.push_back(KeyPoint(corners[gc[i]], 7));
    int xmin = 1000000000, ymin = 1000000000, xmax = -1, ymax = -1;
    for (int i = 0; i < 4; ++i) {
      xmin = min(xmin, (int)goodCorners[i].pt.x);
      ymin = min(ymin, (int)goodCorners[i].pt.y);
      xmax = max(xmin, (int)goodCorners[i].pt.x);
      ymax = max(ymin, (int)goodCorners[i].pt.y);
    }
    const int d = max(xmax-xmin, ymax-ymin) / min(pattern.height,
						  pattern.width);
    for (int i = 0; i < 4; ++i) {
      const Point2f & pt = goodCorners[i].pt;
      if ((pt.y-d>=0) && (pt.x-d>=0) && (pt.y+d<im.size().height) &&
	  (pt.x+d<im.size().width))
	features.push_back(Feature(cornersPos[i],
				   im(Range(pt.y-d,pt.y+d),Range(pt.x-d,pt.x+d)),
				   matf::eye(3,3)));
    }
    if (features.size() != 4) {
      features.clear();
      return false;
    }
      
    for (size_t i = 0; i < goodCorners.size(); ++i)
      line(imdisp, goodCorners[i].pt, goodCorners[i].pt, Scalar(0, 0, 0), 2*d+1);

    for (int i = 0; i < 4; ++i) {
      kalman.setPt3d(i, features[i].pos);
      kalman.setPt3dCov(i, 5e-2);
    }
    // reproject the points
    {
      matf R = kalman.getRot().toMat();
      matf t = kalman.getPos();
      matf K = kalman.getK();
      matf points = kalman.getPts3d();
      for (int i = 0; i < points.size().height; ++i) {
	matf p = K * R * (points.row(i).t() - t);
	p = p/p(2);
	Point pt(p(0), p(1));
	cout << pt << endl;
	line(imdisp, pt, pt, Scalar(255,255,255), 10);
      }
    }


    imshow("disp", imdisp);
    cvWaitKey(0);
    
    cout << kalman << endl;
  }
  return found;
}


matb loadImage(const string & path) {
  Mat im0 = imread(path);
  Mat output;
  if (im0.type() == CV_8UC3) {
    Mat im0b;
    //cvtColor(im0, im0b, CV_RGB2GRAY);
    cvtColor(im0, output, CV_RGB2GRAY);
    //im0b.convertTo(output, CV_32F);
  } else {
    output = im0;
    //im0.convertTo(output, CV_32F);
  }
  //return output/255.0f;
  return output;
}
matb getFrame(VideoCapture & cam) {
  Mat im0;
  cam.grab();
  cam.grab();
  cam.grab();
  cam.grab();
  cam.read(im0);
  Mat output;
  if (im0.type() == CV_8UC3) {
    Mat im0b;
    //cvtColor(im0, im0b, CV_RGB2GRAY);
    cvtColor(im0, output, CV_RGB2GRAY);
    //im0b.convertTo(output, CV_32F);
  } else {
    output = im0;
    //im0.convertTo(output, CV_32F);
  }
  //return output/255.0f;
  return output;
}

SLAM* slamp = NULL;
void handler(int s) {
  printf("Ctrl-C (or Ctrl-\\)\n");
  if (slamp) {
    delete slamp;
    slamp = NULL;
  }
  exit(0);
}

int main () {
  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = handler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, NULL);
  sigaction(SIGQUIT, &sigIntHandler, NULL);

  matf MR(3,3,0.0f);
  //MR(2,0) = 1; MR(0,1) = 1; MR(1,2) = -1;
  MR(0,2) = 1; MR(1,1) = 1; MR(2,0) = 1;

  matf K = matf::eye(3,3);
  //K(0,0) = K(1,1) = 275;
  //K(0,2) = 158;
  //K(1,2) = 89;
  K(0,0) = K(1,1) = 818;
  K(0,2) = 333;
  K(1,2) = 231;
  matf distCoeffs(5,1, 0.0f);
  //distCoeffs(0) = -0.5215f;
  //distCoeffs(1) =  0.4426f;
  //distCoeffs(2) =  0.0012f;
  //distCoeffs(3) =  0.0071f;
  //distCoeffs(4) = -0.0826f;
  distCoeffs(0) = -0.0142f;
  distCoeffs(1) =  0.0045f;
  distCoeffs(2) =  0.0011f;
  distCoeffs(3) =  0.0056f;
  distCoeffs(4) = -0.5707f;
  namedWindow("disp");
  int camidx = 0;
  VideoCapture camera(camidx);
  camera.set(CV_CAP_PROP_FPS, 30);
  if (!camera.isOpened()) {
    cerr << "Could not open camera " << camidx << endl;
    return -1;
  }

  slamp = new SLAM(K, distCoeffs, MR, 10);
  slamp->waitForInit();
  while(true)
    slamp->newImage(getFrame(camera));
  if (slamp)
    delete slamp;

  //Mat lena = loadImage("lena.png");
  //slam.newImage(lena);
  //slam.newImage(lena);
  return 0;
}
