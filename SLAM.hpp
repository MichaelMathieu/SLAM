#ifndef  __SLAM_HPP__
#define  __SLAM_HPP__

#include "common.hpp"
#include "kalman.hpp"
#include "mongoose.h"
#include <opencv2/nonfree/nonfree.hpp>

class SLAM {
private:
  typedef byte imtype;
  //public:
  const matf mongooseAlign;
  matf camera;
  KalmanSLAM kalman;
  Mongoose mongoose;
  matf lastR, deltaR;
  struct Feature {
    enum Type {
      Point, Line,
    };
    int iKalman;
    cv::Mat_<imtype> descriptor;
    matf M;
    int dx, dy;
    Type type;
    // dx, dy are half sizes in x and y (in pixels)
    Feature(const KalmanSLAM & kalman, int iKalman, const cv::Mat_<imtype> & im,
	    const cv::Point2i & pos2d, const matf & pos3d, int dx, int dy,
	    Type type);
    cv::Mat_<imtype> project(const matf & P, cv::Mat_<imtype> & mask) const;
  };
  struct Match {
    int iFeature;
    cv::Point2f pos;
    template<typename T>
    inline Match(const cv::Point_<T> & pos, const int iFeature)
      :pos(pos.x, pos.y), iFeature(iFeature) {};
  };
  std::vector<Feature> features;
  mutable float fastThreshold;
  matf K, distCoeffs;
  int minTrackedPerImage;
private:
  void addNewFeatures(const matb & im,
		      const std::vector<cv::KeyPoint> & keypoints,
		      const std::vector<Match> & matches,
		      int n, float minDist, int dx = 7, int dy = 7);
  matf matchInArea(const cv::Mat_<imtype> & im, const cv::Mat_<imtype> & patch,
		   const cv::Mat_<imtype> & patchmask, const cv::Rect & area,
		   const matf & areamask, int stride = 1) const;
  cv::Point2i trackFeatureElem(const cv::Mat_<imtype> & im,
			       const cv::Mat_<imtype> & proj,
			       const cv::Mat_<imtype> & projmask,
			       float searchrad, float threshold,
			       const cv::Point2i & c, int stride,
			       cv::Mat* disp = NULL) const;
  cv::Point2i trackFeature(const cv::Mat_<imtype>* im, int subsample, int ifeature,
			   float threshold, const matf & P,
			   cv::Mat* disp = NULL) const;
  matf getCovariancePointAlone(const matf & pt2d) const;
  void projectGaussian(const matf & sigma, matf & output) const;
  void match(const matb & im, std::vector<Match> & matches,
	     float threshold = 3.f) const;
public:
  inline SLAM(const matf & K, const matf & distCoeffs, const matf & mongooseAlign,
	      int minTrackedPerImage = 10)
    :mongooseAlign(mongooseAlign),
     fastThreshold(10.f), K(K), distCoeffs(distCoeffs),
     minTrackedPerImage(minTrackedPerImage),
     kalman(K, 12, 0.1, .1),
     mongoose("/dev/ttyUSB1"), lastR(3,3,0.0f), deltaR(matf::eye(3,3)) {};
  inline matf project3DPoint(const matf & pt3d) const;
  void getSortedKeyPoints(const matb & im, size_t nMinPts,
			 std::vector<cv::KeyPoint> & out) const;
  void newImage(const mat3b & im);
  bool newInitImage(const mat3b & im,
		    const cv::Size & pattern = cv::Size(10, 12));
  void waitForInit();
  void visualize() const;
};

matf SLAM::project3DPoint(const matf & pt3d) const {
  matf out(4, 1);
  out(0) = pt3d(0); out(1) = pt3d(1); out(2) = pt3d(2); out(3) = 1.0f;
  out = camera * out;
  return out / out(2);
}  

#endif
