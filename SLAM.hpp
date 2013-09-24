#ifndef  __SLAM_HPP__
#define  __SLAM_HPP__

#include "common.hpp"
#include "kalman.hpp"
#include "mongoose.h"
#include "cone.hpp"
#include <opencv2/nonfree/nonfree.hpp>
//#include "new_point_kalman.hpp"

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
    matf P0, p2d0;
    matf pCenter, pInf;
    cv::Mat_<imtype> descriptor;
    //std::vector<KalmanNP> kalmans;
    std::vector<matf> posPot;
    std::vector<matf> covPot;
    std::vector<float> wPot;
    int nPotAtStart;
    float dmin, dmax;

    BinCone cone;

    inline LineFeature(const matf & pt2dh, const SLAM & slam,
		       const cv::Mat_<imtype> & desc, int n);
    void addNewPoint(const matf & pos, const matf & cov, float w);
    //void newViewPot(int iPot, const matf & P, const matf & p2d, const matf & dir);
    void newView(const matf & pt2d); //TODO: cov2d
    inline bool isLocalized() const {
      float proba;
      cone.getMaxP(&proba);
      return proba > 0.75;
    }
    matf getCovOfPoint(const matf & p2d, float d, float lambda) const;
    float getCovOnLineFromDepth(float depth) const;
    inline matf projPCenter(const matf & P) const {
      matf out = P * pCenter;
      out /= out(2);
      return out;
    }
    inline matf projPInf(const matf & P) const {
      matf out = P * pInf;
      out /= out(2);
      return out;
    }
    inline size_t nPot() const {
      return posPot.size();
    }
  };
  struct Match {
    int iFeature;
    cv::Point2f pos;
    template<typename T>
    inline Match(const cv::Point_<T> & pos, const int iFeature)
      :pos(pos.x, pos.y), iFeature(iFeature) {};
  };
  std::vector<Feature> features;
  std::vector<LineFeature> lineFeatures;
  mutable float fastThreshold;
  matf K, distCoeffs;
  int minTrackedPerImage;
private:
  void computeNewLines(const matb & im,
		       const std::vector<Match> & matches,
		       int n, float minDist, int rx = 15, int ry = 15);
  void lineToFeature(const cv::Mat_<imtype> & im, const cv::Point2i & pt2d,
		     int iLineFeature);
  void addNewLine(const cv::Mat_<imtype> & im, const cv::Point2f & pt2d,
		  int rx = 15, int ry = 15);
  matf matchInArea(const cv::Mat_<imtype> & im, const cv::Mat_<imtype> & patch,
		   const cv::Mat_<imtype> & patchmask, const cv::Rect & area,
		   const matf & areamask, int stride = 1) const;
  cv::Point2i trackFeatureElem(const cv::Mat_<imtype> & im,
			       const cv::Mat_<imtype> & proj,
			       const cv::Mat_<imtype> & projmask,
			       float searchrad, float threshold,
			       const cv::Point2i & c, int stride,
			       float & response, cv::Mat* disp = NULL) const;
  cv::Point2i trackFeature(const cv::Mat_<imtype>* im, int subsample, int ifeature,
			   float threshold, const matf & P, float & response,
			   cv::Mat* disp = NULL) const;
  cv::Point2i trackLine(const cv::Mat_<imtype>* im, int subsample, int iline,
			float threshold, const matf & P, cv::Mat* disp = NULL) const;
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
