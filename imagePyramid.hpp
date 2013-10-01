#ifndef __IMAGE_PYRAMID_HPP__
#define __IMAGE_PYRAMID_HPP__

#include "common.hpp"
#include <vector>
#include <cassert>

template<typename imtype>
class ImagePyramid {
public:
  typedef cv::Mat_<imtype> mati;
  std::vector<mati> images;
  std::vector<float> subsamples;
  // resolution must be in decreasing order (subsample[0] < subsample[1] < ...)
  inline ImagePyramid(const mati & imFullSize,
		      const std::vector<float> & subsamples)
    :subsamples(subsamples) {
    int h = imFullSize.size().height, w = imFullSize.size().width;
    for (size_t i = 0; i < subsamples.size(); ++i) {
      if (i != 0)
	assert(subsamples[i-1] < subsamples[i]);
      float sub = subsamples[i];
      mati imSub;
      resize(imFullSize, imSub, cv::Size(round(w/sub), round(h/sub)));
      images.push_back(imSub);
    }
  }
  inline size_t nSubs() const {
    return subsamples.size();
  }
};

#endif
