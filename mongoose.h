#ifndef __MONGOOSE_H_240912__
#define __MONGOOSE_H_240912__

#include<string>
#include<fcntl.h>
#include"common.hpp"

const size_t BUFFER_SIZE = 1024;
const int LINE_SIZE = 43;

struct Mongoose {
  unsigned char buffer[BUFFER_SIZE];
  int i_buffer;
  FILE* file;
  float acc[3], gyro[3], mag[3];
  matf rotmat;
  double time;
  bool isInit;

  Mongoose(std::string tty);
  Mongoose(const Mongoose & source);//do not copy
  Mongoose & operator=(const Mongoose & source);//do not copy
  inline void close() {
    if (file)
      fclose(file);
    file = NULL;
  }
  inline ~Mongoose() {
    close();
  }
  bool FetchMongooseElem();
  void FetchMongoose();  
};

#endif
