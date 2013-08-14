#ifndef  __SLAM_HPP__
#define  __SLAM_HPP__

#include "common.hpp"
#include "kalman.hpp"
#include <opencv2/nonfree/nonfree.hpp>

class SLAM {
private:
  matf camera;
  KalmanSLAM kalman;
  struct Feature {
    matf pos, sigma;
    cv::Mat descriptor;
    inline Feature(const matf & pos, const cv::Mat & descriptor,
		   const matf & sigma)
      :pos(pos), descriptor(descriptor), sigma(sigma) {};
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
  void match(const matb & im, std::vector<Match> & matches,
	     float threshold = 3.f) const;
public:
  inline SLAM(const matf & K, const matf distCoeffs, int minTrackedPerImage = 10)
    :fastThreshold(10.f), K(K), distCoeffs(distCoeffs),
     minTrackedPerImage(minTrackedPerImage),
     kalman(K, 4) {};
  inline matf project3DPoint(const matf & pt3d) const;
  void getSortedKeyPoints(const matb & im, size_t nMinPts,
			 std::vector<cv::KeyPoint> & out) const;
  void newImage(const matb & im);
  bool newInitImage(const matb & im,
		    const cv::Size & pattern = cv::Size(10, 12));
};

matf SLAM::project3DPoint(const matf & pt3d) const {
  matf out(4, 1);
  out(0) = pt3d(0); out(1) = pt3d(1); out(2) = pt3d(2); out(3) = 1.0f;
  out = camera * out;
  return out / out(2);
}
  

#endif
