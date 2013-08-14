#include "SLAM.hpp"
#include <opencv/highgui.h>
using namespace std;
using namespace cv;

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
  for (size_t i = 0; i < features.size(); ++i) {
    matchTemplate(im, features[i].descriptor, result, CV_TM_CCORR_NORMED);
    minMaxLoc(result, NULL, &maxv, NULL, &maxp);
    if (maxv > threshold) {
      const int dx = floor(features[i].descriptor.size().width*0.5f);
      const int dy = floor(features[i].descriptor.size().height*0.5f);
      matches.push_back(Match(Point2i(maxp.x+dx, maxp.y+dy), i));
    }
    //imshow("disp", result);
    //cvWaitKey(0);
  }
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
    --n;
  badAddNewFeature:;
  }
}

void SLAM::newImage(const matb & im) {
  if (features.size() == 0) {
    newInitImage(im);
  } else {
    vector<KeyPoint> keypoints;
    getSortedKeyPoints(im, 10*minTrackedPerImage, keypoints);
    
    vector<Match> matches;
    match(im, matches, 0.99f);
    //addNewFeatures(im, keypoints, matches,
    //		   minTrackedPerImage - matches.size(), 15.f);
    cout << features.size() << " " << matches.size() << endl;

    if (matches.size() == 4) {
      matf y(8,1);
      for (int i = 0; i < 4; ++i) {
	y(2*i) = matches[i].pos.x;
	y(2*i+1) = matches[i].pos.y;
      }
      float delta = 0.1f;
      kalman.update(matf(0,0), y, &delta);
      cout << kalman.x << endl;
    }
    
    Mat imdisp;
    cvtColor(im, imdisp, CV_GRAY2RGB);
    for (size_t i = 0; i < matches.size(); ++i) {
      circle(imdisp, matches[i].pos, 5, Scalar(0, 0, 255));
      matf p = kalman.getPt3d(matches[i].iFeature).clone();
      p = K * kalman.getRot().toMat() * (p - kalman.getPos());
      p = p / p(2);
      circle(imdisp, Point2f(p(0), p(1)), 4, Scalar(255, 0, 255));
    }
    for (size_t i = 0; i < keypoints.size(); ++i)
      circle(imdisp, keypoints[i].pt, 2, Scalar(255, 0, 0));
    imshow("disp", imdisp);
    cvWaitKey(1);
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

    for (int i = 0; i < pattern.height; ++i)
      for (int j = 0; j < pattern.width; ++j) {
	Mat_<float> p(3,1,0.0f);
	p(0) = i;
	p(1) = j;
	p = K * q.toMat() * (p - tvecf);
	p = p / p(2);
	Point2f pt(p(0),p(1));
	line(imdisp, pt,pt, Scalar(0, 0, 1), 11);
      }
    kalman.setPos(tvecf);
    kalman.setRot(q);

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
      kalman.setPt3dCov(i, 1e-5);
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

int main () {
  matf K = matf::eye(3,3);
  K(0,0) = K(1,1) = 275;
  K(0,2) = 158;
  K(1,2) = 89;
  matf distCoeffs(5,1, 0.0f);
  distCoeffs(0) = -0.5215f;
  distCoeffs(1) =  0.4426f;
  distCoeffs(2) =  0.0012f;
  distCoeffs(3) =  0.0071f;
  distCoeffs(4) = -0.0826f;
  namedWindow("disp");
  int camidx = 0;
  VideoCapture camera(camidx);
  if (!camera.isOpened()) {
    cerr << "Could not open camera " << camidx << endl;
    return -1;
  }

  SLAM slam(K, distCoeffs);
  while(true)
    slam.newImage(getFrame(camera));

  //Mat lena = loadImage("lena.png");
  //slam.newImage(lena);
  //slam.newImage(lena);
  return 0;
}
