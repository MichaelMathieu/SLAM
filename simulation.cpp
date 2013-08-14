#include "kalman.hpp"
#include "common.hpp"
#include <cstdlib>
#include<iostream>
using namespace std;

int main() {
  float w = 400.0f, h = 400.0f;
  matf K(3,3,0.0f);
  K(0,0) = K(1,1) = 400.0f;
  K(2,2) = 1.0f;
  K(0,2) = w/2;
  K(1,2) = h/2;

  int nPts = 10;
  matf ptsReal(nPts, 3);
  for (int i = 0; i < nPts; ++i) {
    for (int j = 0; j < 3; ++j) {
      ptsReal(i,j) = (float)(rand())/RAND_MAX;
    }
    ptsReal(i,2) += 10;
  }

  matf R = matf::eye(3,3);
  float alpha = 0.3;
  R(0,0) = cos(alpha);
  R(0,1) =-sin(alpha);
  R(1,0) = sin(alpha);
  R(1,1) = cos(alpha);
  matf t(3,1,0.0f);
  matf v(3,1,0.0f);
  v(1) = 0.4;
  v(2) = 0.1;
  KalmanSLAM SLAM(K, nPts);

  {
    matf p0(3,1);
    p0(0) = 0;
    p0(1) = 0.1;
    p0(2) = -10;
    SLAM.setPos(p0);
  }
  {
    matf p0(3,1);
    p0(0) = 0.2;
    p0(1) = 0.1;
    p0(2) = -2;
    SLAM.setVel(p0);
  }
  for (int i = 0; i < 13; ++i)
    SLAM.setXCov(i, 0);
  for (int i = 0; i < 3; ++i)
    SLAM.setXCov(i, 1);

  int i;
  for (i = 0; i < 7; ++i) {
    SLAM.setPt3d(i,ptsReal.row(i).t());
    SLAM.setPt3dCov(i, 0.0001f);
  }
  for (; i < nPts; ++i) {
    matf eps(3,1);
    float sigma = 0.5;
    for (int j = 0; j < 3; ++j)
      eps(j) = ((float)(rand())/RAND_MAX*2.-1.)*sigma;
    SLAM.setPt3d(i,ptsReal.row(i).t()+eps);
    //SLAM.setPt3d(i,matf(3,1,1.0f));
    SLAM.setPt3dCov(i, sigma);
  }

  float delta = 1;
  cout << (int)SLAM.testDerivatives(&delta) << endl;

  matf Y(nPts, 2);
  cout << SLAM.getPos() << endl;
  for (int i = 0; i < 25; ++i) {
    //cout << SLAM.x << endl << endl;
    for (int j = 0; j < nPts; ++j) {
      matf p = K * R * (ptsReal.row(j).t() - t);
      Y(j,0) = p(0)/p(2);
      Y(j,1) = p(1)/p(2);
    }
    float delta = 1;
    SLAM.update(matf(0,0), Y.reshape(1, 2*nPts), &delta);
    //cout << (matf)(SLAM.getPts3d() - ptsReal)<< endl;
    cout << (matf)(SLAM.getPos() - t) << endl;
    cout << "v " << SLAM.getVel() << endl;
    t += v;
  }
  
  return 0;
}
