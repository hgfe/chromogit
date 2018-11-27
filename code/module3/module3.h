#pragma once
#include <opencv2/opencv.hpp>
#include "conn.h"
#include "main.h"
using namespace cv;
using namespace std;

typedef  struct __declspec(dllexport) position_ {
	Mat cImgMask;
	int cImgBoundingBox[4];
	float cImgOrientation;
} position;

typedef  struct __declspec(dllexport) chromo_ {
	int index;
	int relatedIndex;
	int chromoId;
	bool cImgType;
	Mat cImg;
	Mat cImgRotated;
	position cImgPosition;
	float chromoCategoryInfo[25];
	int chromoUpright;
} chromo;

extern "C" __declspec(dllexport) int ManualSegment(chromo chromoData, vector<chromo>& newChromoDataList, vector<Mat> allLines, Point clickPosi, int segType, int newCutNum);
