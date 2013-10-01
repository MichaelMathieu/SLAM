#include "cone.hpp"
using namespace std;
using namespace cv;

float FCone::logEvaluateLocalCoord(const matf & local) const {
  const float d = local(0), r2 = sq(local(1)) + sq(local(2));
  float sigma1 = sigmaOnF * d;
  float proba = -0.5*r2/sigma1;
  //proba -= log(2.f * 3.1415926f * sigma1);
  return proba;
}

BinCone::BinCone(const matf & localBase, const matf & pos, float sigma, float f,
		 float dmin, float dmax, int nbinsD, int nbinsR)
  :BaseCone(localBase, pos), nD(nbinsD), nR(nbinsR), dmin(dmin),
   dmax(dmax), width(3*sigma/f),
   bins() {
  cout << "   -> " << width << endl;
  int sizes[3] = {nD, nR, nR};
  bins = matf(3, sizes);
  for (int di = 0; di < nD; ++di) {
    double norm = 0.f;
    for (int xi = 0; xi < nR; ++xi) {
      for (int yi = 0; yi < nR; ++yi) {
	matf localBinCenter = getBinCenterLocalCoord(di, xi, yi);
	float sigma1 = sigma * localBinCenter(0);
	float proba = 1.f;
	proba = -0.5*(sq(localBinCenter(1))+sq(localBinCenter(2)))/sigma1;
	//proba /= 2.f * 3.1415926f * sigma1;
	bins(di,xi,yi) = proba;
	norm += exp(proba); //TODO not good
      }
    }
    bins.row(di) -= log(norm);
  }
  normalize();
}

float BinCone::logEvaluateLocalCoord(const matf & local) const {
  const float d = local(0), x = local(1), y = local(2);
  float wd = width * d;
  // get nearest neighbor
  int di = (int)round(((d - dmin) / (dmax-dmin)) * (float)nD);
  int xi = (int)round(((x / wd + 1.f) / 2.f) * (float)nR);
  int yi = (int)round(((y / wd + 1.f) / 2.f) * (float)nR);
  //std::cout << "  " << di << " " << xi << " " << yi << std::endl;
  //float kmul = 1.f;
  if ((di < 0) or (di >= nD) or (xi < 0) or (xi >= nR) or (yi < 0) or (yi >= nR))
    return -1e20;
  else
    return bins(di, xi, yi);
}

void BinCone::intersect(const BaseCone & other) {
  for (int di = 0; di < nD; ++di)
    for (int xi = 0; xi < nR; ++xi)
      for (int yi = 0; yi < nR; ++yi) {
	matf global1Pt = getBinCenterGlobalCoord(di, xi, yi);
	matf local2Pt = other.getLocalCoordFromGlobal(global1Pt);
	float otherval = other.logEvaluateLocalCoord(local2Pt);
	//	cout << di << " " << xi << " " << yi << " " << otherval << endl;
	//cout << " " << local1Pt << "\n " << globalPt << "\n " << local2Pt << endl;
	bins(di, xi, yi) += otherval;
      }
  normalize();
  //print();
  float maxp;
  cout << getBinCenterGlobalCoord(getMaxP(&maxp)) << endl;
  cout << maxp << endl;
  //if (isnan(sum(bins)[0]))
  //exit(0);
}

void BinCone::print() const {
  for (int di = 0; di < nD; ++di) {
    for (int xi = 0; xi < nR; ++xi) {
      for (int yi = 0; yi < nR; ++yi)
	cout << bins(di,xi,yi) << " ";
      cout << endl;
    }
    cout << "\n--------------------\n" << endl;
  }
  cout << endl;
}

void BinCone::display(Mat & im, const matf & P) const {
  const matf R = P(Range(0,3),Range(0,3));
  const matf p4 = P(Range(0,3),Range(3,4));
  for (int di = 0; di < nD; ++di)
    for (int xi = 0; xi < nR; ++xi)
      for (int yi = 0; yi < nR; ++yi) {
	const matf p3d = getBinCenterGlobalCoord(di,xi,yi);
	const matf p = R * p3d + p4;
	const Point2i pt2d(round(p(0)/p(2)), round(p(1)/p(2)));
	float prob = exp(bins(di, xi, yi));
	if (im.type() == CV_32F)
	  line(im, pt2d, pt2d, Scalar(prob,prob,prob), 5);
	else
	  line(im, pt2d, pt2d, Scalar(255*prob,255*prob,255*prob), 5);
      }
}
