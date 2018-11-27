#pragma once

#include "conn.h"
#include "main.h"

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

extern "C" __declspec(dllexport) int moduleCrossSeg(chromo chromoData, int newCutNum, vector<vector<chromo>>& chromoDataList);
