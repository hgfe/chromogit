#pragma once
#include <opencv2/opencv.hpp>

class ConnectedRegion;

using namespace cv;
using namespace std;

#define targetH 1200
#define targetW 1600

#define PI 3.1415926

Mat imgUniform(const Mat imgGray, int& resizeH, int& resizeW);
Mat imFill(const Mat BW);
Mat imrotate(Mat src, double angle, String model);
