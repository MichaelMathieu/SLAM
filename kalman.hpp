#ifndef __KALMAN_HPP__
#define __KALMAN_HPP__

#include<ostream>
#include "common.hpp"
#include "quaternion.hpp"
#include "../KalmanFilter/kalman.hpp"

class KalmanSLAM : public KalmanFilter<float> {
  friend std::ostream & operator<<(std::ostream &, const KalmanSLAM &);
private:
  matf K; // camera parameters
  std::vector<int> activePts;
public:
  //SLAM-specific functions
  inline const matr getK () const {return K;};
  inline const matr getPos () const {return cvrange(x,0,3);};
  inline void setPos(const matf & pos) {pos.copyTo(cvrange(x, 0, 3));};
  inline Quaternion getRot () const {
    return Quaternion(x(3), x(4), x(5), x(6));
  };
  inline void setRot(const Quaternion & q) {
    x(3) = q.a; x(4) = q.b; x(5) = q.c; x(6) = q.d;
  }
  inline const matr getVel () const {return cvrange(x,7,10);};
  inline void setVel(const matf & pos) {pos.copyTo(cvrange(x,7,10));};
  inline const matr getRvel() const {return cvrange(x,10,13);};
  inline void setRVel(const matf & rvel) {x.rowRange(10,13) += rvel;};
  inline const matr getPt3d(int i) const {
    return cvrange(x, 13+3*i, 16+3*i);
  }
  inline const matr getPt3dH(int i) const {
    matr out(4,1);
    out(3) = 1.0f;
    getPt3d(i).copyTo(out.rowRange(0,3));
    return out;
  }
  inline const matr getPts3d() const {
    int nrows = (nStateParams() - 13)/3;
    return cvrange(x, 13, nStateParams()).reshape(0, nrows);
  }
  inline const matr getPt3dCov(int i) const {
    return cov(cv::Range(13+3*i, 16+3*i), cv::Range(13+3*i,16+3*i));
  }
  inline void setPt3d(int i, const matr & pos) {
    pos.copyTo(cvrange(x, 13+3*i, 16+3*i));
  }
  inline void setPt3dCov(int i, float c) {
    for (int k = 0; k < 3; ++k) {
      for (int j = 0; j < nStateParams(); ++j) {
	cov(13+3*i+k,j) = 0.f;
	cov(j,13+3*i+k) = 0.f;
      }
      cov(13+3*i+k, 13+3*i+k) = c;
    }
  }
  inline void setXCov(int idx, float c) {
    for (int j = 0; j < nStateParams(); ++j) {
      cov(idx, j) = 0.f;
      cov(j, idx) = 0.f;
    }
    cov(idx, idx) = c;
  }
  inline int getNPts() const {
    return (x.size().height-13)/3;
  }
  void addNewPoint(const matf & pos, matf cov);
  inline void setActivePoints(const std::vector<int> & idx) {
    activePts = idx;
  }
  inline void renormalize() {
    x.rowRange(3,7) /= norm(x.rowRange(3,7));
  }
  matf getP() const { // camera matrix
    matf out(3,4,0.0f);
    getRot().toMat().copyTo(out(cv::Range(0,3),cv::Range(0,3)));
    out(cv::Range(0,3),cv::Range(3,4)) =
      -out(cv::Range(0,3),cv::Range(0,3)) * getPos();
    return K*out;
  }
public:
  // constructor and Kalman class parameters
  inline KalmanSLAM(const matf & K, int nPts, float covw, float covv)
    :KalmanFilter<float>(13 + 3*nPts, covw, covv), K(K), activePts() {
    Quaternion q((matf)(matf::eye(3,3)));
    setRot(q);
  };
  virtual ~KalmanSLAM() {};
  virtual int nCommandParams() const { return 0; };
  virtual int nNoise1Params() const { return 6; };
  virtual int nObsParams() const { return 2*activePts.size(); };
  virtual int nNoise2Params() const {return nObsParams(); };

  // math (derivatives and quaternions)
  static Quaternion TB2Q(const matf & m);
  static matf TB2dQ(const matf & m);
  static matf dQRonQ(const Quaternion & r);
  static matf dQRonR(const Quaternion & q);
  static matf dMronrk(const Quaternion & r, int k);
  
  // kalman filter matrices
  virtual matf getA(const void* p) const;
  virtual matf getW(const void* p) const;
  virtual matf getH(const void* p) const;
  virtual matf getV(const void* p) const;

  virtual matf f(const matf & u, const matf & w, const void* p) const;
  virtual matf h(const matf & v, const void* p) const;
};

inline std::ostream & operator<<(std::ostream & cout_v, const KalmanSLAM & k) {
  cout_v << "KalmanSLAM:\n x    = " << k.x
	 << "\n cov  = " << k.cov
	 << "\n covw = " << k.covw
	 << "\n covv = " << k.covv;
  return cout_v;
}

#endif
