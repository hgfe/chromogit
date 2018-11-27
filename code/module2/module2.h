#pragma once
#include "conn.h"
#include "main.h"

typedef  struct __declspec(dllexport) position_ {
	Mat cImgMask;
	int cImgBoundingBox[4];		// bounding box λ��,y,x,h,w
	float cImgOrientation;		// ��ת��
} position;

typedef  struct __declspec(dllexport) chromo_ {
	int index;					// ����
	int relatedIndex;			// ����������singleΪ��������������Ϊ�ӽ���Ⱦɫ���и�����ʱ����Ⱦɫ���index
	int chromoId;				// Ⱦɫ����
	bool cImgType;				// �ָ�����single or cross, 1 or 0

	Mat cImg;					// �ָ��ԭʼͼ��δ��ת�ģ�
	Mat cImgRotated;			// �ָ��ԭʼͼ����ת��ģ�
	position cImgPosition;		// λ����Ϣ
	float chromoCategoryInfo[25];	// 0λ��Ϊ�Զ�ʶ���Ⱦɫ���ţ�����24λΪ������
	int chromoUpright;			// Ⱦɫ�峤�����¶̱�����������Ϣ������Ϊ1������Ϊ2��δ֪δ�ж�Ϊ0��
} chromo;

extern "C" __declspec(dllexport) int moduleSplit(Mat originPicture, String pictureType, vector<chromo> chromoDataArray, int newCutIndex,
	chromo& chromoData);

// ��Ҫ���ĵĵط�0920��

// 1.chromoId��Ⱦɫ����
// 2.cImgType���ָ�����
// 3.chromoCategoryInfo[25]��������Ϣ
// 4.module0�еĴ�ֹ���