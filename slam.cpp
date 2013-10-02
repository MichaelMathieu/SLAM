#include "SLAM.hpp"
#include <opencv/highgui.h>
#include <signal.h>
using namespace std;
using namespace cv;

Mat imdisp_debug(0,0, CV_32F);

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
}

void SLAM::newImage(const mat3b & imRGB) {
  mongoose.FetchMongoose();
  if (features.size() == 0) {
    newInitImage(imRGB, Size(7,7));
  } else {
    matb im;
    cvtColor(imRGB, im, CV_BGR2GRAY);
    Mat & imdisp = imdisp_debug;
    cvtColor(im, imdisp, CV_GRAY2BGR);

    { // mongoose
      matf R = mongooseAlign.t() * mongoose.rotmat.t() * mongooseAlign;
      deltaR = R * lastR.t();
      kalman.setRVel(rotmat2TaitBryan(deltaR));
      R.copyTo(lastR);
    }

    // CameraState estimated with IMU alone
    matf R = deltaR * kalman.getRot().toMat();
    CameraState statePrior(K, R, kalman.getPos());
    
    // match points
    vector<Match> pointMatches;
    matchPoints(im, statePrior, pointMatches, 0.98f);

    // update the EKF
    if (pointMatches.size() > 0) {
      //TODO: the next delta is greater
      vector<int> activePts;
      for (size_t i = 0; i < pointMatches.size(); ++i) {
	const size_t iFeature = pointMatches[i].iFeature;
	const size_t iKalman = features[iFeature].iKalman;
	activePts.push_back(iKalman);
      }
      matf y(2*activePts.size(),1);
      for (size_t i = 0; i < activePts.size(); ++i) {
	y(2*i) = pointMatches[i].pos.x;
	y(2*i+1) = pointMatches[i].pos.y;
      }
      float delta = 0.3f;
      if (activePts.size() > 0) { // TODO same (delta)
	kalman.setActivePoints(activePts);
	kalman.update(matf(0,0), y, &delta);
	kalman.renormalize();
      }
    }
    
    CameraState statePosterior(kalman);

    // match lines
    vector<Match> lineMatches;
    matchLines(im, lineMatches, 0.9f);
    for (size_t i = 0; i < lineMatches.size(); ++i) {
      Point2f pt = lineMatches[i].pos;
      line(imdisp, pt, pt, Scalar(255,255,255), 20);
      matf pt2d(2,1); pt2d(0) = pt.x; pt2d(1) = pt.y;
      lineFeatures[lineMatches[i].iFeature].newView(pt2d);
      lineFeatures[lineMatches[i].iFeature].cone.display(imdisp, kalman.getP());
    }

    // add new features
    computeNewLines(im, pointMatches,
		    //minTrackedPerImage-pointMatches.size()-lineMatches.size(),
		    3-lineFeatures.size(), 100.f, 40, 40);
    
    // convert good lines to points
    {
      matf p3dline, covline;
      for (int i = lineMatches.size()-1; i >= 0; --i) {
	int iFeature = lineMatches[i].iFeature;
	if (lineFeatures[iFeature].isLocalized()) {
	  lineToFeature(im, statePosterior, lineMatches[i].pos, iFeature);
	}
      }
    }

    // debug display
   
    for (size_t i = 0; i < pointMatches.size(); ++i) {
      circle(imdisp, pointMatches[i].pos, 5, Scalar(0, 0, 255));
      matf p = kalman.getPt3d(pointMatches[i].iFeature).clone();
      p = K * kalman.getRot().toMat() * (p - kalman.getPos());
      p = p / p(2);
      circle(imdisp, Point2f(p(0), p(1)), 4, Scalar(0, 255, 0));
    }
    //for (size_t i = 0; i < keypoints.size(); ++i)
    //circle(imdisp, keypoints[i].pt, 2, Scalar(255, 0, 0));

    imshow("disp", imdisp);
    cvWaitKey(1);
    //visualize();
  }
}

bool SLAM::newInitImage(const mat3b & imRGB, const Size & pattern) {
  matf R = mongooseAlign.t() * mongoose.rotmat.t() * mongooseAlign;
  R.copyTo(lastR);
  matb imGray;
  cvtColor(imRGB, imGray, CV_BGR2GRAY);
  matf imGrayf;
  imGray.convertTo(imGrayf, CV_32F, 1./255);
  matb channels[3];
  split(imRGB, channels);
  float colors[3][5] = {{ 70,  40,  29, 3, -300},
			{ 19,  21, 114, 3, -400},
			{ 52,  68,  37, 3, -350}};
  Mat imdisp = imRGB.clone();
  bool found = true;
  vector<Point2f> corners[3];
  for (int j = 0; j < 3; ++j) {
    matb im;
    matf tmp2(imRGB.size()), tmp3;
    for (int x = 0; x < tmp2.size().width; ++x)
      for (int y = 0; y < tmp2.size().height; ++y) {
	float c = 0, n1 = 0, n2 = 0;
	for (int i = 0; i < 3; ++i) {
	  c += colors[j][i] * channels[i](y, x);
	  n1 += colors[j][i] * colors[j][i];
	  n2 += channels[i](y, x) * channels[i](y, x);
	}
	tmp2(y, x) = c / sqrt(n1*n2);
      }
    double m;
    minMaxLoc(tmp2, NULL, &m, NULL, NULL);
    threshold(tmp2, tmp3, m-5./255, 1, THRESH_BINARY);
    matf kernel(50,50);
    kernel.setTo(1);
    dilate(tmp3, tmp3, kernel);
    tmp2 = tmp3.mul(imGrayf) + (1-tmp3);
    tmp2.convertTo(im, CV_8U, colors[j][3]*255., colors[j][4]);
    
    bool foundElem = findChessboardCorners(im, pattern, corners[j],
					   CALIB_CB_FAST_CHECK|CV_CALIB_CB_ADAPTIVE_THRESH);
    //drawChessboardCorners(imdisp, pattern, corners[j], foundElem);
    drawChessboardCorners(im, pattern, corners[j], foundElem);
    found &= foundElem;

    imshow("disp", im);
    cvWaitKey(0);
  }

  imshow("disp",imdisp);
  cvWaitKey(1);
  
  if (found) {

    // Get chessboard orientations :
    //  we first find the corners close to the origin
    int cornerspos[] = {0, pattern.width-1, (pattern.height-1)*pattern.width,
			pattern.width*pattern.height-1};
    float min_d = 1e10;
    int min_c[3] = {0,0,0};
    for (int i = 0; i < 4; ++i) {
      Point2f & c1 = corners[0][cornerspos[i]];
      for (int j = 0; j < 4; ++j) {
	Point2f & c2 = corners[1][cornerspos[j]];
	for (int k = 0; k < 4; ++k) {
	  Point2f & c3 = corners[2][cornerspos[k]];
	  float d = norm(c1-c2) + norm(c2-c3) + norm(c3-c1);
	  if (d < min_d) {
	    min_d = d;
	    min_c[0] = i;
	    min_c[1] = j;
	    min_c[2] = k;
	  }
	}
      }
    }

    for (int i = 0; i < 3; ++i)
      line(imdisp,corners[i][0],corners[i][0],Scalar(255,0,255),15);

    //  and we reorder the corners so that the first corner is close to the origin
    int w = pattern.width, h = pattern.height;
    for (int i = 0; i < 3; ++i) {
      vector<Point2f> newCorners;
      switch (min_c[i]) {
      case 0:
	break;
      case 1:
	for (int j = 0; j < w; ++j)
	  for (int k = 0; k < h; ++k)
	    newCorners.push_back(corners[i][w*k+(w-j-1)]);
	corners[i] = newCorners;
	break;
      case 2:
	for (int j = 0; j < w; ++j)
	  for (int k = 0; k < h; ++k)
	    newCorners.push_back(corners[i][w*(h-k-1)+j]);
	corners[i] = newCorners;
	break;
      case 3:
	for (int j = 0; j < w; ++j)
	  for (int k = 0; k < h; ++k)
	    newCorners.push_back(corners[i][w*(h-k-1)+(w-j-1)]);
	corners[i] = newCorners;
	break;
      }
    }
    for (int i = 0; i < 3; ++i)
      line(imdisp,corners[i][0],corners[i][0],Scalar(255,255,255),10);

    //  now, on each chessboard, we identify the axis, and reorder the corners
    //  so that the rows of blue go to x, rows of red to y and rows of green to z
    int correspNear[3] = {2, 0, 1};
    int potentialCorners[2] = {w-1, (h-1)*w};
    for (int i = 0; i < 3; ++i) {
      const Point2f & pt = corners[i][1];
      float min_d = 1e20;
      int min_f;
      for (int j = 0; j < 3; ++j)
	if (j != i) {
	  for(int k = 0; k < 2; ++k) {
	    const Point2f & pt2 = corners[j][potentialCorners[k]];
	    float d = norm(pt-pt2);
	    if (d < min_d) {
	      min_d = d;
	      min_f = j;
	    }
	  }
	}
      if (min_f != correspNear[i]) {
	vector<Point2f> newCorners(corners[i].size());
	for (int j = 0; j < w; ++j)
	  for (int k = 0; k < h; ++k) {
	    newCorners[k*w+j] = corners[i][j*h+k];
	  }
	corners[i] = newCorners;
      }
    }
    for (int i = 0; i < 3; ++i)
      drawChessboardCorners(imdisp, pattern, corners[i], true);      
    for (int i = 0; i < 3; ++i)
      line(imdisp,corners[i][w-1],corners[i][w-1],Scalar(0,0,0),10);
    imshow("disp",imdisp);
    cvWaitKey(0);
    
    // Compute and 3d points relative camera position (PnP)
    vector<Point3f> pts3d;
    matf tvecf(3,1);
    Quaternion q;
    {
      Mat_<double> rvec, tvec;
      int planesAxis[3][2] = {{0, 1}, {1, 2}, {2, 0}};
      vector<Point2f> pts2d;
      for (int k = 0; k < 3; ++k) {
	for (int i = 0; i < pattern.height; ++i)
	  for (int j = 0; j < pattern.width; ++j) {
	    matf pt(3, 1, 0.0f);
	    pt(planesAxis[k][1]) = 3+2*i;
	    pt(planesAxis[k][0]) = 3+2*j;
	    pts3d.push_back(Point3f(pt));
	  }
	for (size_t i = 0; i < corners[k].size(); ++i)
	  pts2d.push_back(corners[k][i]);
      }
      solvePnP(pts3d, pts2d, K, matf(5,1,0.0f), rvec, tvec, false, CV_EPNP);
      Mat_<double> rotmat(3,3,0.0f);
      Rodrigues(rvec, rotmat);
      q = Quaternion(rotmat);
      Mat_<double> Kd(3,3);
      for (int i = 0; i <9; ++i)
	Kd(i) = K(i);
      tvec = -rotmat.inv()*tvec;
      for (int i = 0; i < 3; ++i)
	tvecf(i) = tvec(i);
    }

    // Set camera position
    kalman.setPos(tvecf);
    kalman.setRot(q);
    CameraState stateInit(kalman);

    // Initialize features and 3d points
    {
      const int nGC = 9;
      int gc[9][2] = {//{0, 0},
		       {0, pattern.width-1},
		       {0, (pattern.height-1)*pattern.width},
		       {0, pattern.width*pattern.height-1},
		       //{1, 0},
		       {1, pattern.width-1},
		       {1, (pattern.height-1)*pattern.width},
		       {1, pattern.width*pattern.height-1},
		       //{2, 0},
		       {2, pattern.width-1},
		       {2, (pattern.height-1)*pattern.width},
		       {2, pattern.width*pattern.height-1}};
		       
      matf cornersPos[] = {//matf3(3, 3, 0),
			   matf3(3+2*(pattern.height-1), 3, 0),
			   matf3(3, 3+2*(pattern.width-1), 0),
			   matf3(3+2*(pattern.height-1), 3+2*(pattern.width-1), 0),
			   //matf3(0, 3, 3),
			   matf3(0, 3+2*(pattern.height-1), 3),
			   matf3(0, 3, 3+2*(pattern.width-1)),
			   matf3(0, 3+2*(pattern.height-1), 3+2*(pattern.width-1)),
			   //matf3(3, 0, 3),
			   matf3(3, 0, 3+2*(pattern.width-1)),
			   matf3(3+2*(pattern.height-1), 0, 3),
			   matf3(3+2*(pattern.height-1), 0, 3+2*(pattern.width-1))};
      vector<KeyPoint> goodCorners;
      for (int i = 0; i < nGC; ++i)
	goodCorners.push_back(KeyPoint(corners[gc[i][0]][gc[i][1]], 7));
      int xmin = 1000000000, ymin = 1000000000, xmax = -1, ymax = -1;
      // compute chessboard size in pixels
      for (int i = 0; i < nGC; ++i) {
	xmin = min(xmin, (int)goodCorners[i].pt.x);
	ymin = min(ymin, (int)goodCorners[i].pt.y);
	xmax = max(xmax, (int)goodCorners[i].pt.x);
	ymax = max(ymax, (int)goodCorners[i].pt.y);
      }
      // the size of a feature is set so that it sees the chessboard corner
      const int d = 1. * max(xmax-xmin, ymax-ymin) / min(pattern.height,
							 pattern.width);
      // check if the corners are not too close to the edge
      for (int i = 0; i < nGC; ++i) {
	const Point2f & pt = goodCorners[i].pt;
	if ((pt.y-d<0) || (pt.x-d<0) || (pt.y+d>=imRGB.size().height) ||
	    (pt.x+d>=imRGB.size().width))
	  return false;
      }
      // add the feature and the 3d point to the kalman filter
      assert(features.size() == 0);
      for (int i = 0; i < nGC; ++i) {
	const Point2f & pt = goodCorners[i].pt;
	features.push_back(Feature(imGray, stateInit, i, pt, cornersPos[i], d, d));
	kalman.setPt3d(i, cornersPos[i]);
	kalman.setPt3dCov(i, 5e-2);
	line(imdisp, goodCorners[i].pt, goodCorners[i].pt, Scalar(0, 0, 0), 2*d+1);
      }
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
	line(imdisp, pt, pt, Scalar(255,255,255), 10);
      }
    }


    imshow("disp", imdisp);
    cvWaitKey(0);
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
mat3b getFrame(VideoCapture & cam) {
  Mat im0;
  cam.grab();
  cam.grab();
  cam.grab();
  cam.grab();
  cam.read(im0);
  return im0;
  /*
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
  */
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
  //MR(2,0) = -1; MR(0,1) = 1; MR(1,2) = -1;
  MR(0,0) = 1; MR(1,2) = 1; MR(2,1) = 1;
  
  //MR(0,0) = 1; MR(1,1) = 1; MR(2,2) = -1;
  ///MR(0,0) = 1; MR(1,2) = 1; MR(2,1) = 1;
  //   MR(0,1) = -1; MR(1,0) = 1; MR(2,2) = 1;
  ///MR(0,1) = 1; MR(1,2) = 1; MR(2,0) = 1;
  //MR(0,2) = 1; MR(1,0) = 1; MR(2,1) = 1;
  //MR(0,2) = 1; MR(1,1) = 1; MR(2,0) = 1;

  matf K = matf::eye(3,3);
  //K(0,0) = K(1,1) = 275;
  //K(0,2) = 158;
  //K(1,2) = 89;
  K(0,0) = 818.3184;
  K(1,1) = 818.4109;
  K(0,2) = 333.229;
  K(1,2) = 230.9768;
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
  int camidx = 1;
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
