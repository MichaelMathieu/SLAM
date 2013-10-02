#include "tests.hpp"
#include <iostream>
using namespace std;

bool test_matching();
bool test_matchingArea();
bool test_matchingPatch();
bool test_matchingPatchArea();

int main () {
  NEW_F(test_matching);
  NEW_F(test_matchingArea);
  NEW_F(test_matchingPatch);
  NEW_F(test_matchingPatchArea);
  NEW_F(Tester::test_project);
  
  Tester::run();
  return 0;
}
