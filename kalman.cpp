#include<iostream>
#include<vector>
#include<cmath>
#include<cstdlib>
#include"kalman.hpp"
using namespace std;
using namespace cv;

Quaternion KalmanSLAM::TB2Q(const matf & m) {
  const float a = m(0)/2, b = m(1)/2, c = m(2)/2;
  const float ca = cos(a), cb = cos(b), cc = cos(c);
  const float sa = sin(a), sb = sin(b), sc = sin(c);
  return Quaternion(ca*cb*cc + sa*sb*sc,
		    sa*cb*cc - ca*sb*sc,
		    sa*cb*sc + ca*sb*cc,
		    ca*cb*sc - sa*sb*cc);
}


matf KalmanSLAM::TB2dQ(const matf & m) {
  const float a = m(0)/2, b = m(1)/2, c = m(2)/2;
  const float ca = cos(a), cb = cos(b), cc = cos(c);
  const float sa = sin(a), sb = sin(b), sc = sin(c);
  matf out(4,3);
  out(0,0) =  ca*sb*sc - sa*cb*cc;
  out(1,0) =  ca*cb*cc + sa*sb*sc;
  out(2,0) =  ca*cb*sc - sa*sb*cc;
  out(3,0) = -sa*cb*sc - ca*sb*cc;
  out(0,1) =  sa*cb*sc - ca*sb*cc;
  out(1,1) = -sa*sb*cc - ca*cb*sc;
  out(2,1) =  ca*cb*cc - sa*sb*sc;
  out(3,1) = -ca*sb*sc - sa*cb*cc;
  out(0,2) =  sa*sb*cc - ca*cb*sc;
  out(1,2) = -sa*cb*sc - ca*sb*cc;
  out(2,2) =  sa*cb*cc - ca*sb*sc;
  out(3,2) =  ca*cb*cc + sa*sb*sc;
  return 0.5*out;
}

matf KalmanSLAM::dQRonQ(const Quaternion & r) {
  matf out(4,4);
  out(0,0) = r.a;
  out(1,0) = r.b;
  out(2,0) = r.c;
  out(3,0) = r.d;
  out(0,1) = -r.b;
  out(1,1) = r.a;
  out(2,1) = -r.d;
  out(3,1) = r.c;
  out(0,2) = -r.c;
  out(1,2) = r.d;
  out(2,2) = r.a;
  out(3,2) = -r.b;
  out(0,3) = -r.d;
  out(1,3) = -r.c;
  out(2,3) = r.b;
  out(3,3) = r.a;
  return out;
}

matf KalmanSLAM::dQRonR(const Quaternion & q) {
  matf out(4,4);
  out(0,0) = q.a;
  out(1,0) = q.b;
  out(2,0) = q.c;
  out(3,0) = q.d;
  out(0,1) = -q.b;
  out(1,1) = q.a;
  out(2,1) = q.d;
  out(3,1) = -q.c;
  out(0,2) = -q.c;
  out(1,2) = -q.d;
  out(2,2) = q.a;
  out(3,2) = q.b;
  out(0,3) = -q.d;
  out(1,3) = q.c;
  out(2,3) = -q.b;
  out(3,3) = q.a;
  return out;
}

matf KalmanSLAM::dMronrk(const Quaternion & r, int k) {
  assert((k >= 0) && (k < 4));
  const float a = r.a, b = r.b, c = r.c, d = r.d;
  matf out(3, 3);
  switch(k) {
  case 0:
    out(0,0) =  a; out(0,1) = -d; out(0,2) =  c;
    out(1,0) =  d; out(1,1) =  a; out(1,2) = -b;
    out(2,0) = -c; out(2,1) =  b; out(2,2) =  a;
    break;
  case 1:
    out(0,0) =  b; out(0,1) =  c; out(0,2) =  d;
    out(1,0) =  c; out(1,1) = -b; out(1,2) = -a;
    out(2,0) =  d; out(2,1) =  a; out(2,2) = -b;
    break;
  case 2:
    out(0,0) = -c; out(0,1) =  b; out(0,2) =  a;
    out(1,0) =  b; out(1,1) =  c; out(1,2) =  d;
    out(2,0) = -a; out(2,1) =  d; out(2,2) = -c;
    break;
  case 3:
    out(0,0) = -d; out(0,1) = -a; out(0,2) =  b;
    out(1,0) =  a; out(1,1) = -d; out(1,2) =  c;
    out(2,0) =  b; out(2,1) =  c; out(2,2) =  d;
    break;
  }
  return 2.f*out;
}


matf KalmanSLAM::getA(const void* p) const {
  // tested
  float delta = *((float*)p);
  matf out = matf::eye(nStateParams(), nStateParams());
  dQRonQ(TB2Q(delta*getRvel())).copyTo(out(Range(3,7),Range(3,7)));
  out(0,7) = out(1,8) = out(2,9) = delta;
  ((matf)(delta*dQRonR(getRot())*TB2dQ(delta*getRvel()))).copyTo(out(Range(3,7),Range(10,13)));
  return out;
}

matf KalmanSLAM::getW(const void* p) const {
  // tested
  float delta = *((float*)p);
  matf out(nStateParams(), 6, 0.0f);
  out(0,0) = out(1,1) = out(2,2) = delta*delta;
  ((matf)(delta*delta*dQRonR(getRot())*TB2dQ(delta*getRvel()))).copyTo(out(Range(3,7),Range(3,6)));
  out(7,0) = out(8,1) = out(9,2) = delta;
  out(10,3) = out(11,4) = out(12,5) = delta;
  return out;
}

matf KalmanSLAM::getH(const void* p) const {
  float delta = *((float*)p);
  matf out(nObsParams(), nStateParams(), 0.0f);
  int n = getNPts();
  matf KM = K * getRot().toMat();
  matf KdM[4];
  for (int k = 0; k < 4; ++k)
    KdM[k] = K * dMronrk(getRot(), k);
  matf pos = getPos();
  for (int i = 0; i < n; ++i) {
    // dh/dp
    matf KMXmp = KM * (getPt3d(i) - pos);
    float denominator = KMXmp(2)*KMXmp(2);
    out(Range(2*i,2*i+2),Range(0,3)) =
      (-KMXmp(2)*KM.rowRange(0,2) + KMXmp.rowRange(0,2)*KM.row(2))/denominator;
    // dh/dr
    for (int k = 0; k < 4; ++k) {
      matf KdMXmp = KdM[k] * (getPt3d(i) - pos);
      out(Range(2*i,2*i+2),Range(3+k,3+k+1)) =
	(KMXmp(2)*KdMXmp.rowRange(0,2) - KdMXmp(2)*KMXmp.rowRange(0,2))/denominator;
    }
    // dh/dX
    out(Range(2*i,2*i+2),Range(13+3*i,13+3*i+3)) =
      -out(Range(2*i,2*i+2),Range(0,3));
  }
  return out;
}

matf KalmanSLAM::getV(const void* p) const {
  return matf::eye(nObsParams(), nObsParams());
}



matf KalmanSLAM::f(const matf &, const matf & w, const void* p) const {
  float delta = *((float*)p);
  matf acc = cvrange(w,0,3);
  matf racc = cvrange(w,3,6);
  matf out(nStateParams(), 1);
  cvrange(out,7,10) = getVel() + delta*acc;
  cvrange(out,0,3) = getPos() + delta*cvrange(out,7,10);
  cvrange(out,10,13) = getRvel() + delta*racc;
  Quaternion r = getRot() * TB2Q(delta*cvrange(out, 10,13));
  out(3) = r.a;
  out(4) = r.b;
  out(5) = r.c;
  out(6) = r.d;
  cvrange(x,13,nStateParams()).copyTo(cvrange(out,13,nStateParams()));
  return out;
}

matf KalmanSLAM::h(const matf & v, const void* p) const {
  matf pts3d = getPts3d();
  int nPts = pts3d.size().height;
  matf out = matf(nPts, 2);
  matf tmp;
  matf R = getRot().toMat();
  for (int i = 0; i < nPts; ++i) {
    tmp = K * R * (pts3d.row(i).t() - getPos());
    out(i,0) = tmp(0)/tmp(2);
    out(i,1) = tmp(1)/tmp(2);
  }
  matf out1 = out.reshape(1, nPts*2) + v;
  return out1;
}

/*
int main() {
  matf K = matf::eye(3,3);
  K(0,0) = K(1,1) = 600;
  K(0,2) = 300;
  K(1,2) = 200;
  KalmanSLAM kalman(K, 4);
  cout << kalman << endl;
  return 0;
}
*/
