//======================================================================
// File: mongoose.cpp
//
// Description: Mongoose 9 DoF 
//
// Created: Sept 24th, 2012
//
// Author: Michael Mathieu // mathieu@cs.nyu.edu
//======================================================================

#include<iostream>
#include "mongoose.h"

//#define VERBOSE_DEBUG

using namespace std;

inline int readShort(unsigned char* b) {
  return (((int)b[0] - 1) << 8) + b[1] - 32768;
}

inline long readLong(unsigned char* b) {
  return ((b[0] & 63) << 28) + ((b[1] & 127) << 21) + ((b[2] & 127) << 14) +
    ((b[3] & 127) << 7) + (b[4] & 127);
}

template<typename realT>
inline void readV3D(unsigned char* b, realT* dest) {
  dest[0] = (realT)(readShort(b  )) * (realT)0.2;
  dest[1] = (realT)(readShort(b+2)) * (realT)0.2;
  dest[2] = (realT)(readShort(b+4)) * (realT)0.2;
}
template<typename realT>
inline void readV3Dacc(unsigned char* b, realT* dest) {
  dest[0] += (realT)(readShort(b  )) * (realT)0.2;
  dest[1] += (realT)(readShort(b+2)) * (realT)0.2;
  dest[2] += (realT)(readShort(b+4)) * (realT)0.2;
}

Mongoose::Mongoose(std::string tty)
  :i_buffer(0), file(NULL), rotmat(matf::eye(3,3)), isInit(false) {
  memset(buffer, 0, sizeof(char)*BUFFER_SIZE);
  memset(acc, 0, sizeof(float)*3);
  memset(gyro, 0, sizeof(float)*3);
  memset(mag, 0, sizeof(float)*3);
  FILE* p = popen((std::string("stty -F ") + tty +
		   std::string(" hupcl && stty -F ") +
		   tty + std::string(" cs8 115200 ignbrk -brkint -icanon -icrnl -imaxbel -opost -onlcr -isig -iexten -echo -echoe -echok -echoctl -echoke -noflsh ixon -crtscts")).c_str(), "r");
  if (p) {
    pclose(p);
    file = fopen(tty.c_str(), "rb");
    int fd = fileno(file);
    int status = fcntl(fd, F_GETFL, NULL);
    if (status < 0) {
      file = NULL;
    } else {
      status |= O_NONBLOCK;
      if (fcntl(fd, F_SETFL, status) < 0) {
	file = NULL;
      }
    }
  } else {
    file = NULL;
  }
 }

bool Mongoose::FetchMongooseElem() {
  ssize_t n = fread(buffer+i_buffer,sizeof(unsigned char),LINE_SIZE-i_buffer,file);
  if (n < 0)
    return false;

#ifdef VERBOSE_DEBUG
  int ct = 0;
  cout << i_buffer << " " << n << " | ";
  for (int i = 0; i < i_buffer+n; ++i) {
    cout << (int)buffer[i] << " ";
    if (++ct % 10 == 0) cout << "* ";
  }
  cout << endl;
#endif

  for (int i = i_buffer+n-1; i >= max(i_buffer, 1); --i)
    if (buffer[i] == 0) {
      // beginning of line at the wrong place. drop what was before
      memmove(buffer, buffer+i, (i_buffer+n-i)*sizeof(char));
      i_buffer += n-i;
#ifdef VERBOSE_DEBUG
      cout << "zero" << endl;
#endif
      return true;
    }
  i_buffer += n;
  if (buffer[0] != 0) {
    // corrupted line. drop it.
#ifdef VERBOSE_DEBUG
    cout << "corrupted line" << endl;
#endif
    i_buffer = 0;
    return true;
  }
  if (i_buffer < LINE_SIZE)
    return false;
  i_buffer = 0;
  unsigned char checksum = 0;
  for (int i = 0; i < LINE_SIZE-1; ++i)
    checksum += buffer[i];
  if (checksum == 0)
    checksum = 1;
  if (checksum != buffer[LINE_SIZE-1]) {
    //  checksum error. drop the line.
#ifdef VERBOSE_DEBUG
    cout << "corrupted line (chksum)" << endl;
#endif
    return true;
  }
  
  //data[18] = readLong(buffer+1);
  if (buffer[1] & 64) {
    //readV3D<real>(buffer+6, data);
    //readV3D<real>(buffer+12, data+3);
    //readV3D<real>(buffer+18, data+6);  
    //readV3D<real>(buffer+24, data+9);
  } else {
    time = readLong(buffer+1);
    readV3Dacc<float>(buffer+6, acc);
    readV3D<float>(buffer+12, gyro); //TODO: not sure
    readV3D<float>(buffer+18, mag);  //TODO: not sure
    for (int i = 8; i >= 0; --i)
      rotmat(i) = (float)(readShort(buffer+24+2*i)) * (float)0.0001;
  }
  isInit = true;
  return true;
}

void Mongoose::FetchMongoose() {
  memset(acc, 0, 3*sizeof(float));
  while (FetchMongooseElem());
}
