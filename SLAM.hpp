#ifndef  __SLAM_HPP__
#define  __SLAM_HPP__

#include "common.hpp"
#include "kalman.hpp"
#include "mongoose.h"
#include "cone.hpp"
#include "imagePyramid.hpp"
#include <opencv2/nonfree/nonfree.hpp>

class Tester;
extern cv::Mat imdisp_debug;

class SLAM {
  friend class Tester;
private:
  typedef byte imtype;
  const matf mongooseAlign;
  KalmanSLAM kalman;
  Mongoose mongoose;
  matf lastR, deltaR;
  mutable float fastThreshold;
  matf K, distCoeffs;
  int minTrackedPerImage;

  struct CameraState {
    const float f;
    const matf K, R, t;
    matf Rinv, P, KR, KRinv;
    explicit inline CameraState(const KalmanSLAM & kalman);
    inline CameraState(const matf & K, const matf & R, const matf & t);
    inline matf project(const matf & p) const;
    // camera coordinate :
    //  if A = getLocalCoordinates(),
    //  A.col(0) is the x coordinate of the image
    //  A.col(1) is the y coordinate of the image
    //  A.col(2) is the princical ray
    inline matf getLocalCoordinates() const;
    // coordinate system so that Z is held by ([p2d] - t)
    //  where [p2d] is the 3d position of p2d
    //  and X, Y are as close as possible to the x,y of the image.
    //  As getLocalCoordinate, these vectors are the columns of the output
    matf getLocalCoordinatesPoint(const cv::Point2f & pt2d) const;
    matf getLocalCoordinatesPoint(const cv::Point2i & pt2d) const;
    matf getLocalCoordinatesPoint(const matf & p2d) const; 
  };

  struct Feature {
    int iKalman;
    cv::Mat_<imtype> descriptor;
    mutable matf B;
    Feature(const cv::Mat_<imtype> & im, const CameraState & state, int iKalman,
	    const cv::Point2f & pt2d, const matf & p3d, int rx, int ry);
    Feature(const CameraState & state, const cv::Mat_<imtype> & descr, int iKalman,
	    const matf & p3d);
    void computeParams(const CameraState & state, const matf & p3d);
    cv::Mat_<imtype> project(const CameraState & state, const matf & p3d,
			     cv::Mat_<imtype> & mask, cv::Rect & location) const;
    void newDescriptor(const cv::Mat_<imtype> & im, const CameraState & state,
		       const cv::Point2f & pt2d, const matf & p3d,
		       bool ignoreIfOnBorder = false);
    cv::Point2i track(const ImagePyramid<imtype> & pyramid,
		      const CameraState & state, const matf & p3d,
		      float threshold, int stride, float & response,
		      cv::Mat* disp = NULL) const;
  };

  struct LineFeature {
    cv::Mat_<imtype> descriptor;
    BinCone cone;
    mutable int timeSinceLastSeen; //TODO: not elegant
    //TODO: harmonize LineFeature constructor with Feature
    LineFeature(const cv::Mat_<imtype> & im, const CameraState & state,
		const cv::Point2f & pt2d, int rx, int ry);
    void newView(const CameraState & state, const matf & pt2d); //TODO: cov2d
    inline bool isLocalized() const;
    cv::Point2i track(const ImagePyramid<imtype> & pyramid,
		      const CameraState & state, float threshold, int stride,
		      float & response, cv::Mat* disp = NULL) const;
  };

  struct Match {
    int iFeature;
    cv::Point2f pos;
    template<typename T> inline Match(const cv::Point_<T> & pos, const int iFeature);
  };

  std::vector<Feature> features;
  std::vector<LineFeature> lineFeatures;
public:
  // matching
  static cv::Point2i matchFeatureInArea(const cv::Mat_<imtype> & im,
					const cv::Mat_<imtype> & patch,
					const cv::Mat_<imtype> * patchMask,
					const cv::Rect & areaRect,
					const matb * areaMask,
					int stride, float & response,
					bool useExackAreaMask = false);
private:
  void lineToFeature(const cv::Mat_<imtype> & im, const CameraState & state,
		     const cv::Point2i & pt2d, int iLineFeature);
  matf getCovariancePointAlone(const CameraState & state, const matf & pt2d,
			       float Lambda, float lambda) const;
  void projectGaussian(const CameraState & state, const matf & mu,
		       const matf & sigma, matf & output) const;
  void matchPoints(const cv::Mat_<imtype> & im, const CameraState & state,
		   std::vector<Match> & matches, float threshold = .95f);
  void matchLines(const cv::Mat_<imtype> & im, const CameraState & state,
		  std::vector<Match> & matches, float threshold = 0.95f) const;
  void addNewLines(const matb & im, const CameraState & state,
		   const std::vector<Match> & matches,
		   const std::vector<Match> & lineMatches,
		   int n, float minDist, int rx = 15, int ry = 15);
  void removeOldLines();
public:
  inline SLAM(const matf & K, const matf & distCoeffs, const matf & mongooseAlign,
	      int minTrackedPerImage = 10);
  void getSortedKeyPoints(const matb & im, size_t nMinPts,
			 std::vector<cv::KeyPoint> & out) const;
  void newImage(const mat3b & im);
  bool newInitImage(const mat3b & im,
		    const cv::Size & pattern = cv::Size(10, 12));
  void waitForInit();
  void visualize() const;
};

//TODO Rinv = R.t(), change when it is sure it's ok
SLAM::CameraState::CameraState(const KalmanSLAM & kalman)
  :f(0.5f*(kalman.getK()(0,0) + kalman.getK()(1,1))),
   K(kalman.getK()), R(kalman.getRot().toMat()), t(kalman.getPos()),
   Rinv(3,3), P(3,4), KR(), KRinv(3,3) {
  Rinv = R.inv();
  R.copyTo(P(cv::Range(0,3),cv::Range(0,3)));
  P(cv::Range(0,3),cv::Range(3,4)) = - R * t;
  P = K * P;
  KR = P(cv::Range(0,3),cv::Range(0,3));
  KRinv = KR.inv();
}

SLAM::CameraState::CameraState(const matf & K, const matf & R, const matf & t)
  :f(0.5f*(K(0,0)+K(1,1))), K(K), R(R), t(t), Rinv(R.inv()), P(3,4),
   KR(), KRinv(3,3) {
  R.copyTo(P(cv::Range(0,3),cv::Range(0,3)));
  P(cv::Range(0,3),cv::Range(3,4)) = - R * t;
  P = K * P;
  KR = P(cv::Range(0,3),cv::Range(0,3));
  KRinv = KR.inv();
}

matf SLAM::CameraState::project(const matf & p) const {
  matf a;
  if (p.size().height == 3)
    a = KR * (p - t);
  else
    a = P * p;
  a /= a(2);
  return a.rowRange(0,2);
}

matf SLAM::CameraState::getLocalCoordinates() const {
  return Rinv;
}

bool SLAM::LineFeature::isLocalized() const {
  float proba;
  cone.getMaxP(&proba);
  return proba > 0.9; //TODO: hard coded
}

template<typename T>
SLAM::Match::Match(const cv::Point_<T> & pos, const int iFeature)
  :pos(pos.x, pos.y), iFeature(iFeature) {};

SLAM::SLAM(const matf & K, const matf & distCoeffs, const matf & mongooseAlign,
	   int minTrackedPerImage)
  :mongooseAlign(mongooseAlign),
   fastThreshold(10.f), K(K), distCoeffs(distCoeffs),
   minTrackedPerImage(minTrackedPerImage),
   kalman(K, 12, 0.1, .1),
   mongoose("/dev/ttyUSB1"), lastR(3,3,0.0f), deltaR(matf::eye(3,3)) {
}

#endif
