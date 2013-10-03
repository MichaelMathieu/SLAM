#include "SLAM.hpp"
#include <opencv/highgui.h>
#include <signal.h>
using namespace std;
using namespace cv;

matb loadImage(const string & path) {
  Mat im0 = imread(path);
  Mat output;
  if (im0.type() == CV_8UC3) {
    Mat im0b;
    //cvtColor(im0, im0b, CV_RGB2GRAY);
    cvtColor(im0, output, CV_RGB2GRAY);
    //im0b.convertTo(output, CV_32F);
  } else {
    output = im0;
    //im0.convertTo(output, CV_32F);
  }
  //return output/255.0f;
  return output;
}
mat3b getFrame(VideoCapture & cam) {
  Mat im0;
  cam.grab();
  cam.grab();
  cam.grab();
  cam.grab();
  cam.read(im0);
  return im0;
}

SLAM* slamp = NULL;
void handler(int s) {
  printf("Ctrl-C (or Ctrl-\\)\n");
  if (slamp) {
    delete slamp;
    slamp = NULL;
  }
  exit(0);
}

int main () {
  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = handler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, NULL);
  sigaction(SIGQUIT, &sigIntHandler, NULL);

  matf MR(3,3,0.0f);
  MR(0,0) = 1; MR(1,2) = 1; MR(2,1) = 1;

  matf K = matf::eye(3,3);
  K(0,0) = 818.3184;
  K(1,1) = 818.4109;
  K(0,2) = 333.229;
  K(1,2) = 230.9768;
  matf distCoeffs(5,1, 0.0f);
  distCoeffs(0) = -0.0142f;
  distCoeffs(1) =  0.0045f;
  distCoeffs(2) =  0.0011f;
  distCoeffs(3) =  0.0056f;
  distCoeffs(4) = -0.5707f;
  namedWindow("disp");
  int camidx = 1;
  VideoCapture camera(camidx);
  camera.set(CV_CAP_PROP_FPS, 30);
  if (!camera.isOpened()) {
    cerr << "Could not open camera " << camidx << endl;
    return -1;
  }

  slamp = new SLAM(K, distCoeffs, MR, 10);
  slamp->waitForInit();
  while(true)
    slamp->newImage(getFrame(camera));
  if (slamp)
    delete slamp;

  return 0;
}
