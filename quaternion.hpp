#ifndef __QUATERNION_HPP__
#define __QUATERNION_HPP__

#include<stdexcept>
#include<ostream>
#include<cmath>
#include"common.hpp"

struct Quaternion {
  typedef float realq;
  static const realq eps_norm = 1e-20f;
  realq a, b, c, d;
  inline Quaternion()
    :a(0.f), b(0.f), c(0.f), d(0.f) {};
  inline Quaternion(realq a, realq b, realq c, realq d)
    :a(a), b(b), c(c), d(d) {};
  template<typename T>
  inline Quaternion(const cv::Mat_<T> & rotmat)
    :a(0.f) {
    d = 0.5f * sqrt(1.f - rotmat(0,0) - rotmat(1,1) + rotmat(2,2));
    if (abs(d) < 1e-2)
      a = 0.5f * sqrt(1.f + rotmat(0,0) + rotmat(1,1) + rotmat(2,2));
    if (abs(a) < abs(d)) {
      //std::cout << "1" << std::endl;
      realq f = 0.25f / d;
      a = f * (rotmat(1,0) - rotmat(0,1));
      b = f * (rotmat(0,2) + rotmat(2,0));
      c = f * (rotmat(1,2) + rotmat(2,1));
    } else {
      //std::cout << "2" << std::endl;
      realq f = 0.25f / a;
      b = f * (rotmat(2,1) - rotmat(1,2));
      c = f * (rotmat(0,2) - rotmat(2,0));
      d = f * (rotmat(1,0) - rotmat(0,1));
    }

    /*
    d = 0.5f * sqrt(1.f + rotmat(0,0) + rotmat(1,1) + rotmat(2,2));
    if (abs(d) < 1e-2)
      a = 0.5f * sqrt(1.f + rotmat(0,0) - rotmat(1,1) - rotmat(2,2));
    if (abs(a) < abs(d)) {
      std::cout << "1" << std::endl;
      realq f = 0.25f / d;
      a = f * (rotmat(2,1) - rotmat(1,2));
      b = f * (rotmat(0,2) - rotmat(2,0));
      c = f * (rotmat(1,0) - rotmat(0,1));
    } else {
      std::cout << "2" << std::endl;
      realq f = 0.25f / a;
      b = f * (rotmat(0,1) + rotmat(1,0));
      c = f * (rotmat(0,2) + rotmat(2,0));
      d = f * (rotmat(2,1) - rotmat(1,2));
    }
    */
  }
  inline Quaternion & operator=(const Quaternion & q) {
    a = q.a;
    b = q.b;
    c = q.c;
    d = q.d;
    return *this;
  }
  inline static bool approxeq(realq a, realq b) {
    return (a - eps_norm < b) && (b < a + eps_norm);
  }
  inline bool operator==(const Quaternion & q) const {
    return approxeq(a, q.a) && approxeq(b, q.b) && approxeq(c, q.c) && approxeq(d, q.d);
  }
  inline bool operator!=(const Quaternion & q) const {
    return !(operator==(q));
  }
  inline Quaternion operator+(const Quaternion & q) const {
    return Quaternion(a+q.a, b+q.b, c+q.c, d+q.d);
  }
  inline Quaternion & operator+=(const Quaternion & q) {
    a += q.a;
    b += q.b;
    c += q.c;
    d += q.d;
    return *this;
  }
  inline Quaternion operator-(const Quaternion & q) const {
    return Quaternion(a-q.a, b-q.b, c-q.c, d-q.d);
  }
  inline Quaternion & operator-=(const Quaternion & q) {
    a -= q.a;
    b -= q.b;
    c -= q.c;
    d -= q.d;
    return *this;
  }
  inline Quaternion operator-() const {
    return Quaternion(-a, -b, -c, -d);
  }
  inline Quaternion operator*(const Quaternion & q) const {
    return Quaternion(a*q.a - b*q.b - c*q.c - d*q.d,
		      a*q.b + b*q.a + c*q.d - d*q.c,
		      a*q.c - b*q.d + c*q.a + d*q.b,
		      a*q.d + b*q.c - c*q.b + d*q.a);
  }
  inline Quaternion & operator*=(const Quaternion & q) {
    return operator=(operator*(q));
  }
  inline realq norm() const {
    return a*a+b*b+c*c+d*d;
  }
  inline Quaternion conjugate() const {
    return Quaternion(a, -b, -c, -d);
  }
  inline Quaternion inv() const {
    const realq n = norm();
    if (n < eps_norm)
      throw std::overflow_error("Quaternion: division by zero");
    const realq in = 1.f / n;
    return Quaternion(a*in, -b*in, -c*in, -d*in);
  }
  inline Quaternion operator/(const Quaternion & q) const {
    return operator*(q.inv());
  }
  inline Quaternion & operator/=(const Quaternion & q) {
    return operator*=(q.inv());
  }
  inline realq & operator[](int i) {
    return (&a)[i];
  }
  inline realq operator[](int i) const {
    return (&a)[i];
  }
  inline matf toMat() const {
    matf out(3,3);
    /*
    out(0,0) = 1.f - 2.f*b*b - 2.f*c*c;
    out(0,1) = 2.f * (a*b - c*d);
    out(0,2) = 2.f * (a*c + b*d);
    out(1,0) = 2.f * (a*b + c*d);
    out(1,1) = 1.f - 2.f*a*a - 2.f*c*c;
    out(1,2) = 2.f * (b*c - a*d);
    out(2,0) = 2.f * (a*c - b*d);
    out(2,1) = 2.f * (a*d + b*c);
    out(2,2) = 1.f - 2.f*a*a - 2.f*b*b;
    */
    out(0,0) = a*a + b*b - c*c - d*d;
    out(0,1) = 2.f * (b*c - a*d);
    out(0,2) = 2.f * (b*d + a*c);
    out(1,0) = 2.f * (b*c + a*d);
    out(1,1) = a*a - b*b + c*c - d*d;
    out(1,2) = 2.f * (c*d - a*b);
    out(2,0) = 2.f * (b*d - a*c);
    out(2,1) = 2.f * (c*d + a*b);
    out(2,2) = a*a - b*b - c*c + d*d;
    return out;
  }
};

inline std::ostream & operator<<(std::ostream & cout_v, const Quaternion & q) {
  cout_v << "(" << q.a << "," << q.b << "," << q.c << "," << q.d << ")";
  return cout_v;
}

#endif
