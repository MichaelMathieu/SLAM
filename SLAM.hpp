#ifndef  __SLAM_HPP__
#define  __SLAM_HPP__

#include "common.hpp"
#include "kalman.hpp"
#include "mongoose.h"
#include "cone.hpp"
#include "imagePyramid.hpp"
#include <opencv2/nonfree/nonfree.hpp>

class SLAM {
private:
  typedef byte imtype;
  //public:
  const matf mongooseAlign;
  KalmanSLAM kalman;
  Mongoose mongoose;
  matf lastR, deltaR;
  mutable float fastThreshold;
  matf K, distCoeffs;
  int minTrackedPerImage;

  struct Feature {
    const SLAM* pslam;
    int iKalman;
    cv::Mat_<imtype> descriptor;
    matf B;
    Feature(const SLAM & slam, int iKalman, const cv::Mat_<imtype> & im,
	    const cv::Point2i & pos2d, const matf & pos3d, int dx, int dy);
    Feature(const SLAM & slam, int iKalman, const cv::Mat_<imtype> & descr,
	    const matf & pos3d);
    void computeParams(const matf & pos3d);
    cv::Mat_<imtype> project(const matf & P, cv::Mat_<imtype> & mask) const;
    void newDescriptor(const matf & P, const cv::Mat_<imtype> & im,
		       const cv::Point2i & pt2d);
  };

  struct LineFeature {
    const SLAM* pslam;
    cv::Mat_<imtype> descriptor;
    std::vector<matf> posPot;
    BinCone cone;
    LineFeature(const matf & pt2dh, const SLAM & slam,
		const cv::Mat_<imtype> & desc, int n);
    void newView(const matf & pt2d); //TODO: cov2d
    inline bool isLocalized() const;
    matf getCovOfPoint(const matf & p2d, float d, float lambda) const;
    inline size_t nPot() const;
    cv::Point2i track(const ImagePyramid<imtype> & pyramid, const matf & P,
		      float threshold, int stride, float & response,
		      cv::Mat* disp = NULL) const;
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
  void computeNewLines(const matb & im,
		       const std::vector<Match> & matches,
		       int n, float minDist, int rx = 15, int ry = 15);
  void lineToFeature(const cv::Mat_<imtype> & im, const cv::Point2i & pt2d,
		     int iLineFeature);
  void addNewLine(const cv::Mat_<imtype> & im, const cv::Point2f & pt2d,
		  int rx = 15, int ry = 15);
  cv::Point2i trackFeatureElem(const cv::Mat_<imtype> & im,
			       const cv::Mat_<imtype> & proj,
			       const cv::Mat_<imtype> & projmask,
			       float searchrad,
			       const cv::Point2i & c, int stride,
			       float & response) const;
  cv::Point2i trackFeature(const cv::Mat_<imtype>* im, int subsample, int ifeature,
			   float threshold, const matf & P, float & response,
			   cv::Mat* disp = NULL) const;
  matf getCovariancePointAlone(const matf & pt2d,
				      float Lambda, float lambda) const;
  void projectGaussian(const matf & mu, const matf & sigma, matf & output) const;
  void matchPoints(const cv::Mat_<imtype> & im, std::vector<Match> & matches,
		   float threshold = .95f);
  void matchLines(const cv::Mat_<imtype> & im, std::vector<Match> & matches,
		  float threshold = 0.95f) const;
  matf getLocalCoordinates(const matf & p2d) const;
public:
  inline SLAM(const matf & K, const matf & distCoeffs, const matf & mongooseAlign,
	      int minTrackedPerImage = 10)
    :mongooseAlign(mongooseAlign),
     fastThreshold(10.f), K(K), distCoeffs(distCoeffs),
     minTrackedPerImage(minTrackedPerImage),
     kalman(K, 12, 0.1, .1),
     mongoose("/dev/ttyUSB1"), lastR(3,3,0.0f), deltaR(matf::eye(3,3)) {};
  void getSortedKeyPoints(const matb & im, size_t nMinPts,
			 std::vector<cv::KeyPoint> & out) const;
  void newImage(const mat3b & im);
  bool newInitImage(const mat3b & im,
		    const cv::Size & pattern = cv::Size(10, 12));
  void waitForInit();
  void visualize() const;
};

bool SLAM::LineFeature::isLocalized() const {
  float proba;
  cone.getMaxP(&proba);
  return proba > 0.75;
}

size_t SLAM::LineFeature::nPot() const {
  return posPot.size();
}

template<typename T>
SLAM::Match::Match(const cv::Point_<T> & pos, const int iFeature)
  :pos(pos.x, pos.y), iFeature(iFeature) {};


#endif
