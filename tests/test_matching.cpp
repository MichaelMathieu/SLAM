#include "../SLAM.hpp"
#include "tests.hpp"
#include <cstdlib>
using namespace std;
using namespace cv;

vector<bool (*)()> Tester::testers;
vector<string> Tester::names;

Mat* imdisp_debug = NULL;

typedef Mat_<byte> matim;
bool test_matching0(bool useAreaMask, bool usePatchMask) {
  for (int itest = 0; itest < 100; ++itest) {
    Size imsize(rand() % 1000 + 1, rand() % 1000 + 1);
    Size patchsize(min(rand() % min(200,imsize.width) + 1,imsize.width),
		   min(rand() % min(200,imsize.height) + 1,imsize.height));
    
    // im
    matim im(imsize), patch(patchsize);
    for (int i = 0; i < imsize.width; ++i)
      for (int j = 0; j < imsize.height; ++j)
	im(j,i) = (15485867 * i + 15486883 * j) % 7907;
    
    // patch
    int leftpatch = patchsize.width/2, toppatch = patchsize.height/2;
    int xpatch = rand() % imsize.width;
    int ypatch = rand() % imsize.height;
    int ptop = max(ypatch-toppatch, 0), pleft = max(xpatch-leftpatch, 0);
    int pbottom = min(ypatch-toppatch+patchsize.height, imsize.height);
    int pright = min(xpatch-leftpatch+patchsize.width, imsize.width);
    int dptop = ptop - (ypatch-toppatch), dpleft = pleft - (xpatch-leftpatch);
    int dpbottom = (ypatch-toppatch+patchsize.height) - pbottom;
    int dpright = (xpatch-leftpatch+patchsize.width) - pright;
    for (int i = 0; i < patchsize.width; ++i)
      for (int j = 0; j < patchsize.height; ++j)
	patch(j,i) = rand() % 255;
    im(Range(ptop, pbottom),
       Range(pleft, pright)).copyTo(patch(Range(dptop, patchsize.height-dpbottom),
					  Range(dpleft, patchsize.width-dpright)));
    
    
    // area
    int x0area = xpatch - (rand() % 100)  , y0area = ypatch - (rand() % 100)  ;
    int x1area = xpatch + (rand() % 100)+1, y1area = ypatch + (rand() % 100)+1;
    Rect areaRect(x0area, y0area, x1area-x0area, y1area-y0area);
    
    // areaMatch
    matb areaMask;
    matb* pareaMask = NULL;
    if (useAreaMask) {
      areaMask = matb(areaRect.height, areaRect.width);
      for (int i = 0; i < areaRect.width; ++i)
	for (int j = 0; j < areaRect.height; ++j)
	  areaMask(j,i) = ((rand() % 255) > 128) ? 1 : 0;
      areaMask(ypatch-y0area, xpatch-x0area) = 1;
      pareaMask = &areaMask;
    }

    // patchMask
    matim patchMask(patchsize);
    matim* ppatchMask = NULL;
    if (usePatchMask) {
      patchMask = matim(patchsize);
      for (int i = 0; i < patchsize.width; ++i)
	for (int j = 0; j < patchsize.height; ++j)
	  patchMask(j,i) = ((rand() % 255) > 128) ? 1 : 0;
      ppatchMask = & patchMask;
    }
    
    // match!
    float response;
    Point2i m = SLAM::matchFeatureInArea(im,patch,ppatchMask,areaRect,
					 pareaMask,1,response);
    
    /*
    cout << xpatch << ", " << ypatch << "   " << m << " "
	 << patchsize.width << " " << patchsize.height << "  |  "
	 << x0area << " " << y0area << "   " << x1area << " " << y1area << endl;
    */

    // check
    if ((xpatch != m.x) || (ypatch != m.y)) {
      cout << "mismatch, might still pass (response=" << response << ")" << endl;
      //if the result is wrong, maybe there are two identical places in the image
      if ((0 > m.x) || (0 > m.y) ||
	  (m.x >= imsize.width) || (m.y >= imsize.height) || (response <= 0.99))
	return false;
      for (int i = 0; i < patchsize.width; ++i)
	for (int j = 0; j < patchsize.height; ++j)
	  if ((usePatchMask && patchMask(j,i)) || (!usePatchMask)) {
	    int yim = j+m.y-toppatch, xim = i+m.x-leftpatch;
	    if ((xim >= 0) && (yim >= 0) && (xim < imsize.width) &&
		(yim < imsize.height))
	      if (patch(j,i) != im(yim, xim))
		return false;
	  }
    }
  }
  return true;
}

bool test_matching() {
  return test_matching0(false, false);
}

bool test_matchingArea() {
  return test_matching0(true, false);
}

bool test_matchingPatch() {
  return test_matching0(false, true);
}

bool test_matchingPatchArea() {
  return test_matching0(true, true);
}
