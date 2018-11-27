#pragma once
#include "conn.h"
#include "main.h"

typedef  struct __declspec(dllexport) position_ {
	Mat cImgMask;
	int cImgBoundingBox[4];		// bounding box 位置,y,x,h,w
	float cImgOrientation;		// 旋转角
} position;

typedef  struct __declspec(dllexport) chromo_ {
	int index;					// 索引
	int relatedIndex;			// 关联索引，single为自身索引，否则为从交叉染色体切割下来时交叉染色体的index
	int chromoId;				// 染色体编号
	bool cImgType;				// 分割类型single or cross, 1 or 0

	Mat cImg;					// 分割的原始图（未旋转的）
	Mat cImgRotated;			// 分割的原始图（旋转后的）
	position cImgPosition;		// 位置信息
	float chromoCategoryInfo[25];	// 0位置为自动识别的染色体编号，后面24位为可能性
	int chromoUpright;			// 染色体长臂向下短臂随体向上信息（正立为1，倒立为2，未知未判断为0）
} chromo;

extern "C" __declspec(dllexport) int moduleSplit(Mat originPicture, String pictureType, vector<chromo> chromoDataArray, int newCutIndex,
	chromo& chromoData);

// 需要更改的地方0920：

// 1.chromoId，染色体编号
// 2.cImgType，分割类型
// 3.chromoCategoryInfo[25]，分类信息
// 4.module0中的打分规则