#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include<cmath>
#include<iostream>
#include<cassert>
#include<opencv/cv.h>
#include<vector>

typedef unsigned char byte;
typedef unsigned short uint16;

typedef cv::Mat_<float> matf;
typedef cv::Mat_<byte> matb;

inline matf matf2(float x, float y) {
  matf out(2,1);
  out(0) = x;
  out(1) = y;
  return out;
}
inline matf matf3(float x, float y, float z) {
  matf out(3,1);
  out(0) = x;
  out(1) = y;
  out(2) = z;
  return out;
}
inline matf matf4(float x, float y, float z, float w) {
  matf out(4,1);
  out(0) = x;
  out(1) = y;
  out(2) = z;
  out(3) = w;
  return out;
}
template<typename T>
inline const cv::Mat_<T> cvrange(const cv::Mat_<T> & M, int a, int b) {
  return M(cv::Range(a,b), cv::Range(0,1));
}
template<typename T>
inline cv::Mat_<T> cvrange(cv::Mat_<T> & M, int a, int b) {
  return M(cv::Range(a,b), cv::Range(0,1));
}

#endif
