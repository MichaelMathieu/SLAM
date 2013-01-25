#include<cmath>
#include<iostream>
#include<cassert>
#include<opencv/cv.h>
#include<vector>
extern "C" {
#include<luaT.h>
#include<TH/TH.h>
}
//#include<opencv/highgui.h>
using namespace std;
using namespace cv;

typedef THFloatTensor Tensor;
#define ID_TENSOR_STRING "torch.FloatTensor"
#define Tensor_(a) THFloatTensor_##a
typedef float Real;
typedef double accreal;
typedef unsigned char byte;
typedef unsigned short uint16;

typedef cv::Mat_<float> matf;

#define TWO_BITS_PER_FILTER

#ifdef __ARM__
#define __NEON__
#endif

matf Triangulate(const matf & P1, const matf & P2, const Point2f & p1,
		 const Point2f & p2) {
  matf A(6, 4);
  for (int i = 0; i < 4; ++i) {
    A(0, i) = p1.x * P1(2, i) - P1(0, i);
    A(1, i) = p1.y * P1(2, i) - P1(1, i);
    A(2, i) = p2.x * P2(2, i) - P2(0, i);
    A(3, i) = p2.y * P2(2, i) - P2(1, i);
    A(4, i) = p1.x * P1(1, i) - p1.y * P1(0, i);
    A(5, i) = p2.x * P2(1, i) - p2.y * P2(0, i);
  }
  SVD svd(A, SVD::MODIFY_A);
  return svd.vt.row(3);
}

inline bool extrinsicsPossible(const matf & extr, const Point2f & p1,
			       const Point2f & p2) {
  matf pt4d(4, 1);
  const matf p1m(2, 1, const_cast<float*>(&p1.x));
  const matf p2m(2, 1, const_cast<float*>(&p2.x));
  triangulatePoints(matf::eye(3, 4), extr, p1m, p2m, pt4d);
  pt4d = pt4d / pt4d(3,0);
  return (pt4d(2,0) > 0) &&
    (((matf)(extr*pt4d))(2,0)*determinant(extr(Range(0,3),Range(0,3))) > 0);
}

matf getExtrinsicsFromEssential(const matf & essMat, const Point2f & p1full,
				const Point2f & p2full, const matf & Kinv) {
  matf p1p(3,1,1.f), p2p(3,1,1.f);
  p1p(0) = p1full.x;
  p1p(1) = p1full.y;
  p2p(0) = p2full.x;
  p2p(1) = p2full.y;
  p1p = Kinv * p1p;
  p2p = Kinv * p2p;
  Point2f p1(p1p(0)/p1p(2), p1p(1)/p1p(2));
  Point2f p2(p2p(0)/p2p(2), p2p(1)/p2p(2));
  SVD svd(essMat);
  matf W(3, 3, 0.0f);
  W(0, 1) = -1.f;
  W(1, 0) = W(2, 2) = 1.f;
  matf extr(3, 4), tmp;
  //case 1
  tmp = svd.u*W*svd.vt;
  tmp.copyTo(extr(Range(0, 3), Range(0, 3)));
  svd.u.col(2).copyTo(extr.col(3));
  if (extrinsicsPossible(extr, p1, p2))
    return extr;
  //case 2
  extr.col(3) *= -1.f;
  if (extrinsicsPossible(extr, p1, p2))
    return extr; 
  //case 3
  tmp = svd.u*W.t()*svd.vt;
  tmp.copyTo(extr(Range(0, 3), Range(0, 3)));
  if (extrinsicsPossible(extr, p1, p2))
    return extr;
  //case 4
  extr.col(3) *= -1.f;
  if (extrinsicsPossible(extr, p1, p2))
    return extr;
  //should not happen
  return matf::eye(3, 4);
}

static int Undistort(lua_State *L) {
  const char* idreal = ID_TENSOR_STRING;
  const char* idfloat = "torch.FloatTensor";
  THFloatTensor* input  = (THFloatTensor*)luaT_checkudata(L, 1, idfloat);
  THFloatTensor* output = (THFloatTensor*)luaT_checkudata(L, 2, idfloat);
  THFloatTensor* K      = (THFloatTensor*)luaT_checkudata(L, 3, idfloat);
  THFloatTensor* dist   = (THFloatTensor*)luaT_checkudata(L, 4, idfloat);

  const int h = input->size[0];
  const int w = input->size[1];
  const matf input_cv(h, w, THFloatTensor_data(input));
  matf output_cv(h, w, THFloatTensor_data(output));
  const matf K_cv(3, 3, THFloatTensor_data(K));
  const matf dist_cv(5, 1, THFloatTensor_data(dist));
  
  undistort(input_cv, output_cv, K_cv, dist_cv);
  
  return 0;
}

template<typename T, typename T2>
inline void keepGoodInVector(vector<T> & v,
			     const vector<T2> & goods) {
  size_t i, k = 0;
  for (i = 0; i < v.size(); ++i)
    if (goods[i])
      v[k++] = v[i];
  v.resize(k);
}
template<typename T, typename T2>
inline void keepGoodInVector(vector<T> & v, vector<T> & v2,
			     const vector<T2> & goods) {
  size_t i, k = 0;
  for (i = 0; i < v.size(); ++i)
    if (goods[i]) {
      v[k] = v[i];
      v2[k++] = v2[i];
    }
  v.resize(k);
  v2.resize(k);
}

static int Opticalflow(lua_State *L) {
  const char* idreal = ID_TENSOR_STRING;
  const char* idfloat = "torch.FloatTensor";
  THFloatTensor* input1 = (THFloatTensor*)luaT_checkudata(L, 1, idfloat);
  THFloatTensor* input2 = (THFloatTensor*)luaT_checkudata(L, 2, idfloat);
  THFloatTensor* outputDebug = (THFloatTensor*)luaT_checkudata(L, 3, idfloat);
  THFloatTensor* outputDepth = (THFloatTensor*)luaT_checkudata(L, 4, idfloat);
  THFloatTensor* outputErr   = (THFloatTensor*)luaT_checkudata(L, 5, idfloat);
  THFloatTensor* K0     = (THFloatTensor*)luaT_checkudata(L, 6, idfloat);
  
  assert(input1->nDimension == 2);
  const int h = input1->size[0];
  const int w = input1->size[1];
  const matf K(3, 3, THFloatTensor_data(K0));
  const matf Kinv = K.inv();
  float* ip1 = THFloatTensor_data(input1);
  float* ip2 = THFloatTensor_data(input2);
  float* odbgp = THFloatTensor_data(outputDebug);
  float* odepp = THFloatTensor_data(outputDepth);
  float* oerrp = THFloatTensor_data(outputErr);
  const matf input1_cv(h, w, ip1);
  const matf input2_cv(h, w, ip2);
  cv::Mat input1_cv_8U, input2_cv_8U;
  input1_cv.convertTo(input1_cv_8U, CV_8U, 255.f);
  input2_cv.convertTo(input2_cv_8U, CV_8U, 255.f);
  matf outputR_cv(h, w, odbgp);
  matf outputG_cv(h, w, odbgp+outputDebug->stride[0]);
  matf outputB_cv(h, w, odbgp+2*outputDebug->stride[0]);
  matf output_cv(h, w, odepp);
  matf outputErr_cv(h, w, oerrp);

  vector<cv::Point2f> corners, corners2;
  vector<float> err;
  vector<byte> status, inliers;
  cv::goodFeaturesToTrack(input1_cv_8U, corners, 200, 0.01, 7);
  if (corners.size() < 4)
    cv::goodFeaturesToTrack(input1_cv_8U, corners, 200, 1e-10, 1);
  cv::calcOpticalFlowPyrLK(input1_cv_8U, input2_cv_8U, corners,
			   corners2, status, err);
  keepGoodInVector(corners, corners2, status);
  if (corners.size() < 4) {
    cv::goodFeaturesToTrack(input1_cv_8U, corners, 200, 1e-10, 1);
    cv::calcOpticalFlowPyrLK(input1_cv_8U, input2_cv_8U, corners,
			     corners2, status, err);
    keepGoodInVector(corners, corners2, status);
    if (corners.size() < 4)
      return 0;
  }
  
  matf F = cv::findFundamentalMat(corners, corners2, CV_FM_LMEDS,
  				  1., 0.99, inliers);
  size_t i, k = 0;
  keepGoodInVector(corners, corners2, inliers);

  matf E = K.t() * F * K;
  matf extr = getExtrinsicsFromEssential(E, corners[0], corners2[0], Kinv);
  matf R = K * extr(Range(0, 3), Range(0, 3)) * Kinv;

  vector<cv::Point2f> cornersDense, corners2Dense;
  cv::goodFeaturesToTrack(input1_cv_8U, cornersDense, 1000, 0.0001, 8);
  if (cornersDense.size() < 4)
    cornersDense.insert(cornersDense.end(), corners.begin(), corners.end());
  cv::calcOpticalFlowPyrLK(input1_cv_8U, input2_cv_8U, cornersDense,
			   corners2Dense, status, err);
  keepGoodInVector(cornersDense, corners2Dense, status);
  keepGoodInVector(err, status);

  matf outputGray(input1_cv.size());
  cv::warpPerspective(input1_cv, outputGray, R, input1_cv.size());
  outputGray.copyTo(outputR_cv);
  outputGray.copyTo(outputG_cv);
  outputGray.copyTo(outputB_cv);
  vector<Point2f> cornersWarped(cornersDense.size());
  matf p(3,1), p2(3, 1, 1.f);
  for (i = 0, k = 0; i < cornersDense.size(); ++i) {
    p(0) = cornersDense[i].x;
    p(1) = cornersDense[i].y;
    p(2) = 1.f;
    p2(0) = corners2Dense[i].x;
    p2(1) = corners2Dense[i].y;
    if (((matf)(p.t() * F * p2))(0) > 0.3)
      continue;
    p = R * p;
    cornersWarped[k] = Point2f(p(0)/p(2), p(1)/p(2));
    corners2Dense[k] = corners2Dense[i];
    err[k] = err[i];
    ++k;
  }
  cornersWarped.resize(k);
  corners2Dense.resize(k);
  err.resize(k);
  for (i = 0; i < corners.size(); ++i)
    cv::line(outputB_cv, corners[i], corners2[i], 1, 1);
  for (i = 0; i < cornersWarped.size(); ++i) {
    cv::line(outputG_cv, cornersWarped[i], corners2Dense[i], 1, 1);
    const float x = cornersWarped[i].x-w/2, y = cornersWarped[i].y-h/2;
    const float n = sqrt(x*x + y*y);
    if (n < 25)
      continue;
    const float dx = cornersWarped[i].x - corners2Dense[i].x;
    const float dy = cornersWarped[i].y - corners2Dense[i].y;
    const float D =  n / sqrt(dx*dx + dy*dy);
    const float color = min(20.f/D,1.f);
    cv::line(outputR_cv, cornersWarped[i], cornersWarped[i], color, 10);
    cv::line(outputB_cv, cornersWarped[i], cornersWarped[i], 1-color, 10);
    cv::line(output_cv, cornersWarped[i], cornersWarped[i], color, 10);
    cv::line(outputErr_cv, cornersWarped[i], cornersWarped[i],
	     max(0.f, min(1.f, err[i])), 10);
  }

  return 0;
}


static const struct luaL_reg libmatching[] = {
  {"opticalflow", Opticalflow},
  {"undistort", Undistort},
  {NULL, NULL}
};

LUA_EXTERNC int luaopen_libmatching (lua_State *L) {
  luaL_openlib(L, "libmatching", libmatching, 0);
  return 1;
}
