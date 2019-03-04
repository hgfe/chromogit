#pragma once
#include "stdafx.h"
#include "module2.h"

int moduleSplit(Mat originPicture, String pictureType, vector<chromo> chromoDataArray, int newCutIndex,
	chromo& chromoData) {
	/********************************** split module
	* Input:
	* @param1 originPicture			待处理文件，已经打开的图片文件
	* @param2 pictureType			文件类型，"raw"（黑底） 或者 "tif" （白底）
	* @param3 chromoDataArray		需要拼接的两条染色体
	* @param4 newCutIndex			拼接后的新染色体的 index
	*
	* Output:
	* @param1 chromoData			拼接后的染色体
	* @param2 errorCode				函数成功标志，0-成功，1-输入只有一条染色体，2-无法拼接
	*
	***********************************/
	int errorCode = 0;
	bool bIntensityReverse = 0;
	if (originPicture.cols == 0 || originPicture.rows == 0)
		return errorCode = 1;
	if (pictureType == "raw" || pictureType == "Raw") {
		bIntensityReverse = 1;
	}
	else {
		bIntensityReverse = 0;
	}


	// 反色处理
	Mat imgGray;
	if (bIntensityReverse)
		bitwise_not(originPicture, imgGray);
	else
		imgGray = originPicture.clone();

	int chromoArrayLength = chromoDataArray.size();
	if (chromoArrayLength == 2) {			// 判断输入是否为两条染色体
											// 调整尺度大小
											// imgForExtraction 用于提取染色体图像数据
		int eH = 0, eW = 0;
		Mat imgForExtraction = imgUniform(imgGray, eH, eW);

		Mat splitImg1 = Mat::zeros(Size(targetW, targetH), CV_8UC1);
		Mat splitImg2 = Mat::zeros(Size(targetW, targetH), CV_8UC1);

		// 复制第一个
		int bb[4];
		bb[0] = chromoDataArray[0].cImgPosition.cImgBoundingBox[0];
		bb[1] = chromoDataArray[0].cImgPosition.cImgBoundingBox[1];
		bb[2] = chromoDataArray[0].cImgPosition.cImgBoundingBox[2];
		bb[3] = chromoDataArray[0].cImgPosition.cImgBoundingBox[3];

		Mat splitImg1ROI = splitImg1(Rect(bb[1], bb[0], bb[3], bb[2]));
		Mat imgMask = chromoDataArray[0].cImgPosition.cImgMask;
		imgMask.copyTo(splitImg1ROI, imgMask);

		// 复制第二个
		bb[0] = chromoDataArray[1].cImgPosition.cImgBoundingBox[0];
		bb[1] = chromoDataArray[1].cImgPosition.cImgBoundingBox[1];
		bb[2] = chromoDataArray[1].cImgPosition.cImgBoundingBox[2];
		bb[3] = chromoDataArray[1].cImgPosition.cImgBoundingBox[3];

		Mat splitImg2ROI = splitImg2(Rect(bb[1], bb[0], bb[3], bb[2]));
		imgMask = chromoDataArray[1].cImgPosition.cImgMask;
		imgMask.copyTo(splitImg2ROI, imgMask);

		Mat newSplitImgs = splitImg1 + splitImg2;
		Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
		morphologyEx(newSplitImgs, newSplitImgs, MORPH_CLOSE, kernel);

		ConnectedRegion lbl(newSplitImgs, 8);
		if (lbl.connNum_ == 1) {
			lbl.calculateImage();
			lbl.calculateBoundingBox();
			lbl.calculateOrientation();

			vector<int> bb = lbl.boundingBox_[0];
			double angle = lbl.orientation_[0];
			Mat singleObjMask = lbl.image_[0];

			Mat newChromoObj = imgForExtraction(Rect(bb[1], bb[0], bb[3], bb[2]));

			Mat rotatedObjI, rotatedMask, notRotatedMask;
			if (angle > 0) {
				rotatedObjI = imrotate(newChromoObj, 90 - angle, "bilinear");
				rotatedMask = imrotate(singleObjMask, 90 - angle, "neareast");
			}
			else {
				rotatedObjI = imrotate(newChromoObj, -90 - angle, "bilinear");
				rotatedMask = imrotate(singleObjMask, -90 - angle, "neareast");
			}

			bitwise_not(rotatedMask, notRotatedMask);
			bitwise_or(rotatedObjI, 255, rotatedObjI, notRotatedMask);

			ConnectedRegion CCrot(rotatedMask, 8);
			CCrot.calculateBoundingBox();
			vector<int> bbrot = CCrot.boundingBox_[0];
			rotatedObjI = rotatedObjI(Rect(bbrot[1], bbrot[0], bbrot[3], bbrot[2]));


			chromoData.index = newCutIndex;
			chromoData.relatedIndex = newCutIndex;
			chromoData.chromoId = 0;
			chromoData.cImgType = 1;
			chromoData.cImg = newChromoObj.clone();
			chromoData.cImgRotated = rotatedObjI.clone();
			position pos;
			pos.cImgMask = singleObjMask.clone();
			pos.cImgBoundingBox[0] = bb[0];
			pos.cImgBoundingBox[1] = bb[1];
			pos.cImgBoundingBox[2] = bb[2];
			pos.cImgBoundingBox[3] = bb[3];
			pos.cImgBoundingBoxOffset[0] = bbrot[0];
			pos.cImgBoundingBoxOffset[1] = bbrot[1];
			pos.cImgBoundingBoxOffset[2] = bbrot[2];
			pos.cImgBoundingBoxOffset[3] = bbrot[3];
			pos.cImgOrientation = angle;
			chromoData.cImgPosition = pos;
			for (int idx = 0; idx < 25; idx++)
				chromoData.chromoCategoryInfo[idx] = 0;
			chromoData.chromoUpright = 0;

			errorCode = 0;
		}
		else {
			errorCode = 2;
		}

	}
	else {
		errorCode = 1;
	}

	return errorCode;
}

