#pragma once
#include "conn.h"
#include "main.h"

extern "C" __declspec(dllexport) void moduleScoring(Mat originPicture, String pictureType,
	float & pictureScore, int & singleNum);
void preSeg(Mat img, bool bIntensityReverse, bool bCutTouching,
	int & preSingleNum, int & preSingleArea);
void roughSegChromoRegion(Mat I, Mat & BWmainbody, Mat & innerCutPoints);
float ChromoScore(float avgLength, int singleNum);