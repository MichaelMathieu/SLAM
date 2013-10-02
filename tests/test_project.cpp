#include "tests.hpp"
#include "SLAM.hpp"
#include "random.hpp"
#include <opencv/highgui.h>
using namespace std;
using namespace cv;

typedef Mat_<byte> matim;
bool Tester::test_project() {
  for (int itest = 0; itest < 100; ++itest) {

  startover:

    Size imsize(rand() % 1000 + 1, rand() % 1000 + 1);

    matf K(3,3,0.0f);
    K(0,0) = grand(50) + 500;
    K(1,1) = grand(50) + 500;
    K(2,2) = 1.0f;
    K(0,2) = imsize.width/2 + grand(5);
    K(1,2) = imsize.height/2 + grand(5);

    // initial state : camera is at (0,0,0), R = eye
    SLAM::CameraState state(K, matf::eye(3,3), matf(3,1,0.0f));
    // object is at depth dobj, on axis Z
    float dobj = frand() * 25 + 10;
    // object is on plane XY, has size dobjx in X, same Y
    int dobjx = rand() % 100 + 1;
    int dobjy = rand() % 100 + 1;
    int deltax = grand(25);
    int deltay = grand(25);
    float objincrx = dobj/K(0,0);
    float objincry = dobj/K(1,1);
    matf p3d(3,1);
    p3d(0) = deltax; p3d(1) = deltay; p3d(2) = dobj;
    // generate object
    matim object(dobjy, dobjx);
    for (int i = 0; i < dobjx; ++i)
      for (int j = 0; j < dobjy; ++j)
	object(j,i) = rand() % 256;

    // project object
    matim im(imsize, 0);
    for (int i = 0; i < dobjx; ++i)
      for (int j = 0; j < dobjy; ++j) {
	matf p(4,1,1.0f);
	p(0) = (i - floor(dobjx/2)) * objincrx + deltax;
	p(1) = (j - floor(dobjy/2)) * objincry + deltay;
	p(2) = dobj;
	matf p2d = state.project(p);
	if (!((p2d(0) >= 0) && (p2d(1) >= 0) && (round(p2d(0)) < imsize.width) &&
	      (round(p2d(1)) < imsize.height)))
	  goto startover;
	im(round(p2d(1)), round(p2d(0))) = object(j,i);
      }

    // create patch
    matf p(4,1,1.0f), p1(4,1,1.0f);//, p2(4,1,1.0f);
    p (0) = deltax; p (1) = deltay; p (2) = dobj;
    p1(0) = (- floor(dobjx/2)) * objincrx + deltax;
    p1(1) = (- floor(dobjy/2)) * objincry + deltay;
    p1(2) = dobj;
    //p2(0) = (dobjx-1 - floor(dobjx/2) + deltax) * objincrx;
    //p2(1) = (dobjy-1 - floor(dobjy/2) + deltay) * objincry;
    //p2(2) = dobj;
    matf p2d = state.project(p);
    matf p2d1 = state.project(p1);//, p2d2 = state.project(p2);
    int dx = round(p2d(0)-p2d1(0)), dy = round(p2d(1)-p2d1(1));
    SLAM::Feature feature (im, state, 0, Point2i(round(p2d(0)), round(p2d(1))),
			   cvrange(p,0,3), dx, dy);

    //now we rotate and we compare
    for (int irot = 0; irot < 10; ++irot) {
      im.setTo(0);
      
      // generate a random state
      Quaternion rotQ;
      do {
	rotQ = Quaternion(grand(1), grand(1), grand(1), grand(1));
      } while (rotQ.norm() < 0.1);
      rotQ /= rotQ.norm();
      matf R = rotQ.toMat();
      matf t(3,1,0.0f);
      do {
	t(0) = grand(50);
	t(1) = grand(50);
	t(2) = grand(50);
      } while (norm(t-p3d) > grand(1) * dobj);
      SLAM::CameraState state(K, R, t);

      // project object
      bool isSeen = true;
      matim im(imsize, 0);
      for (int i = 0; i < dobjx; ++i)
	for (int j = 0; j < dobjy; ++j) {
	  matf p(4,1,1.0f);
	  p(0) = (i - floor(dobjx/2)) * objincrx + deltax;
	  p(1) = (j - floor(dobjy/2)) * objincry + deltay;
	  p(2) = dobj;
	  matf p2d = state.project(p);
	  if (!((p2d(0) >= 0) && (p2d(1) >= 0) && (round(p2d(0)) < imsize.width) &&
		(round(p2d(1)) < imsize.height))) {
	    isSeen = false;
	    break;
	  }
	  im(round(p2d(1)), round(p2d(0))) = object(j,i);
	}
      
      if(!isSeen) {
	--irot;
	continue;
      }

      Rect proj_location;
      matim mask;
      matim proj = feature.project(state, p3d, mask, proj_location);
      if (proj.size().height > 0) {
	matb todisp[3];
	todisp[0] = matb(imsize,0);
	todisp[1] = im;
	todisp[2] = matb(imsize,0);
	cvCopyToCrop(proj, todisp[2], proj_location);
	mat3b imdisp;
	merge(todisp, 3, imdisp);
	namedWindow("imdisp");
	imshow("imdisp", imdisp);
	cvWaitKey(0);
      }
      


      
    }


    /*
    // generate a random state
    Quaternion rotQ;
    do {
      rotQ = Quaternion(grand(1), grand(1), grand(1), grand(1));
    } while (rotQ.norm() < 0.1);
    rotQ /= rotQ.norm();
    matf R = rotQ.toMat();
    matf t(3,1);
    t(0) = grand(25);
    t(1) = grand(25);
    t(2) = grand(25);
    SLAM::CameraState state(K, R, t);
    
    // We assume that we see an object that is composed of points
    // x0 + a * u + b * v, where a\in[amin,amax] and b\in[bmin,bmax]
    // at first time we see it, u,v are parallel to the camera plane,
    // so the point is at p(x0) + a * k1*i + b * k2*j

    int w2dobj = rand() % 100 + 1;
    int h2dobj = rand() % 100 + 1;
    int l2dobj = floor(w2dobj/2), t2dobj = floor(h2dobj/2);
    matf p2d(3,1,1.0f);
    p2d(0) = rand() % (imsize.width - w2dobj + 1) + l2dobj;
    p2d(1) = rand() % (imsize.height - h2dobj + 1) + t2dobj;
    matf ray = state.KRinv * p2d;
    matf centerObj = t + (frand() * 25 + 10) * ray;
    float objDepth = ((matf)(P*centerObj))(2);

    int wobj = w2dobj;
    int hobj = h2dobj;
    // the object plane is p31*x + p32*y + p33*z + p34-objDepth = 0

    for (int i = p2d(0) - l2dobj; i < p2d(0)-l2dobj+w2dobj; ++i)
      for (int j = p2d(1) - t2dobj; j < p2d(1)-t2dobj+h2dobj; ++j) {
	matf p2d2(3,1,1.0f);
	p2d(0) = i; p2d(1) = j;
	matf ray2 = state.KRinv * p2d2;
	float lambda = (state.P(2,3)-objDepth - state.P.row(2).dot(t)) / (state.P.row(2).dot(ray2));
	matf p3d2 = t + lambda * ray2;
      }
    */
    //
  }
  return true;
}
