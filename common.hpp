#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include<cmath>
#include<algorithm>
#include<iostream>
#include<cassert>
#include<opencv/cv.h>
#include<vector>

typedef unsigned char byte;
typedef unsigned short uint16;

typedef cv::Mat_<float> matf;
typedef cv::Mat_<byte> matb;
typedef cv::Mat_<cv::Vec3b> mat3b;

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

inline float sq(float a) {
  return a*a;
}

inline float cvval(matf a) {
  return a(0);
}

template<typename T>
inline T round(T a) {
  return floor(a+0.5);
}

inline void cvCopyToCrop(const cv::Mat & src, const cv::Mat & dst,
			 const cv::Rect & dstArea) {
  const int dtop    = std::max(-dstArea.y, 0);
  const int dleft   = std::max(-dstArea.x, 0);
  const int dright  = std::max(dstArea.x+dstArea.width -dst.size().width , 0);
  const int dbottom = std::max(dstArea.y+dstArea.height-dst.size().height, 0);
  if ((dtop+dbottom < src.size().height) && (dleft+dright < src.size().width))
    src(cv::Range(dtop, src.size().height-dbottom),
	cv::Range(dleft, src.size().width-dright))
      .copyTo(dst(cv::Range(dstArea.y+dtop , dstArea.y+dstArea.height-dbottom),
		  cv::Range(dstArea.x+dleft, dstArea.x+dstArea.width -dright)));
}


#endif
