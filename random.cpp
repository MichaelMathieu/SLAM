#include "random.hpp"
#include <cstdlib>
#include <random>

float frand() {
  return ((float)rand()) / RAND_MAX;
}

std::default_random_engine generator;
float grand(float std) {
  std::normal_distribution<double> distribution(0.0,std);
  return distribution(generator);
}
