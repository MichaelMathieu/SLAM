#ifndef __TESTS_HPP__
#define __TESTS_HPP__

#include <string>
#include <vector>
#include <iostream>

class Tester {
public:
  typedef bool (*pf)(void);
  static void newF(pf f, const std::string & name) {
    testers.push_back(f);
    names.push_back(name);
  }
  static void run() {
    std::cout << "\nTesting:" << std::endl;
    bool result_final = true;
    for (size_t i = 0; i < testers.size(); ++i) {
      bool result = testers[i]();
      std::cout << names[i] << " ";
      if (result)
	std::cout << "passed" << std::endl;
      else
	std::cout << "failed" << std::endl;
      result_final &= result;
    }
    if (result_final)
      std::cout << "\n   [[[ Tests passed ]]]\n" << std::endl;
    else
      std::cout << "\n   [[[ Tests FAILED ]]]\n" << std::endl;
  }
private:
  static std::vector<pf> testers;
  static std::vector<std::string> names;
};

#define NEW_F(f) {Tester::newF(f, "f");}

#endif
