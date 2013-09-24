#ifndef __CONE_HPP__
#define __CONE_HPP__

#include "common.hpp"

class BaseCone {
protected:
  matf base, t;
public:
  inline BaseCone(const matf & localBase, const matf & pos);
  virtual float logEvaluateLocalCoord(const matf & local) const = 0;
  inline matf getGlobalCoordFromLocal(const matf & local) const;
  inline matf getLocalCoordFromGlobal(const matf & global) const;
};

class FCone : public BaseCone {
private:
  float sigmaOnF;
public:
  inline FCone(const matf & P, const matf & p2d, float sigma, float f);
  virtual float logEvaluateLocalCoord(const matf & local) const;
};

class BinCone : public BaseCone {
private:
  int nD, nR;
  float dmin, dmax, width;
  matf bins;
public:
  BinCone(const matf & localBase, const matf & pos, float sigma, float f,
	  float dmin, float dmax, int nbinsD, int nbinsR);
  virtual float logEvaluateLocalCoord(const matf & local) const;
  inline matf getBinCenterLocalCoord(int di, int xi, int yi) const;
  inline matf getBinCenterLocalCoord(const cv::Point3i & coord) const;
  inline matf getBinSize(int di, int xi, int yi) const;
  inline void normalize();
  inline float getVal(int di, int xi, int yi) const;
  inline cv::Point3i getMaxP(float* val = NULL) const;
  inline matf getMaxPGlobalCoord(matf* cov = NULL) const;
  inline matf getBinCovLocalCoord(int di, int xi, int yi) const;
  inline matf getBinCovGlobalCoord(int di, int xi, int yi) const;
  void intersect(const BaseCone & other);
  void print() const;
  void display(cv::Mat & im, const matf & P) const;
};

BaseCone::BaseCone(const matf & localBase, const matf & pos)
  :base(localBase), t(pos) {
}

FCone::FCone(const matf & localBase, const matf & pos, float sigma, float f)
  :BaseCone(localBase, pos), sigmaOnF(sigma/f) {
}

matf BinCone::getBinCenterLocalCoord(int di, int xi, int yi) const {
  //TODO: this is not centered well
  matf out(3,1);
  out(0) = dmin + (dmax-dmin) * (float)di / (float)nD;
  float sigmad = out(0) * width;
  out(1) = sigmad * (-1.f + 2.f * (float)xi / (float)nR);
  out(2) = sigmad * (-1.f + 2.f * (float)yi / (float)nR);
  return out;
}

matf BinCone::getBinCenterLocalCoord(const cv::Point3i & coord) const {
  return getBinCenterLocalCoord(coord.x, coord.y, coord.z);
}

matf BinCone::getBinSize(int di, int xi, int yi) const {
  matf out(3,1);
  out(0) = (dmax-dmin) / (float)nD;
  float sigmad = out(0) * width;
  out(2) = out(1) = sigmad * 2.f / (float)nR;
  return out;
}

matf BaseCone::getGlobalCoordFromLocal(const matf & local) const {
  return base * local + t;
}

matf BaseCone::getLocalCoordFromGlobal(const matf & global) const {
  return base.t() * (global - t);
}

void BinCone::normalize() {
  //bins /= sum(bins)[0];
  double pid;
  minMaxIdx(bins, NULL, &pid, NULL, NULL);
  float pi = (float)pid;
  matf expbins(bins.size()); //TODO: do not reallocate
  exp(bins-pi, expbins);
  float norm = pi + log(sum(expbins)[0]);
  bins -= norm;
}

float BinCone::getVal(int di, int xi, int yi) const {
  return bins(di, xi, yi);
}

cv::Point3i BinCone::getMaxP(float* val) const {
  double vald;
  int maxIdx[3];
  cv::minMaxIdx(bins, NULL, &vald, NULL, maxIdx);
  if (val)
    *val = exp(vald);
  return cv::Point3i(maxIdx[0], maxIdx[1], maxIdx[2]);
}

matf BinCone::getMaxPGlobalCoord(matf* cov) const {
  cv::Point3i bin = getMaxP(NULL);
  if (cov)
    *cov = getBinCovGlobalCoord(bin.x, bin.y, bin.z);
  return getGlobalCoordFromLocal(getBinCenterLocalCoord(bin));
}

matf BinCone::getBinCovLocalCoord(int di, int xi, int yi) const {
  //TODO: this is an approximation (the cov. doesn't have to be parallel to the axis)
  matf out(3,3,0.0f);
  matf binSize = getBinSize(di, xi, yi);
  out(0,0) = binSize(0);
  out(1,1) = binSize(1);
  out(2,2) = binSize(2);
  return out;
}

matf BinCone::getBinCovGlobalCoord(int di, int xi, int yi) const {
  return base.t() * getBinCovLocalCoord(di,xi,yi) * base;
}

#endif
