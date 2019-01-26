#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

extern "C" __declspec(dllexport) int chooseROI(Mat src, Mat srcCircle, Mat& dstClose, Mat& dstOpen);


