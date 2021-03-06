#include "stdafx.h"
#include "moduleAdd.h"

int moduleAdd(Mat originPicture, String pictureType, Mat mask, int newCutIndex, chromo& chromoElement) {
	/********************************** module
	* Input:
	* @param1 originPicture			待处理原图片，典型尺寸为3200*2400像素
	* @param2 pictureType			文件类型，"raw"（黑底） 或者 "tif" （白底）
	* @param3 mask					待处理图片要添加的染色体mask，典型尺寸为1600*1200像素
	* @param4 newCutIndex			新增的染色体的index
	*
	* Output:
	* @param1 chromoElement			新增的染色体结构体
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

	// 获取灰度图
	Mat img = originPicture.clone();
	Mat imgGray;
	if (bIntensityReverse)
		imgGray = 255 - img;
	else
		imgGray = img;

	// 预处理
	int eH = 0, eW = 0;
	threshold(mask, mask, 160, 255, THRESH_BINARY);
	Mat mask2;
	resize(mask, mask2, imgGray.size(), 0, 0, 1);
	threshold(mask2, mask2, 160, 255, THRESH_BINARY);
	bitwise_not(mask2, mask2);
	bitwise_or(imgGray, 255, imgGray, mask2);

	Mat imgForExtraction;
	resize(imgGray, imgForExtraction, Size(1600, 1200), 0.0, 0.0, INTER_CUBIC);

	if ((imgForExtraction.size() != mask.size()))
		return errorCode = 1;

	ConnectedRegion newCC(mask, 4);
	newCC.calculateBoundingBox();
	newCC.calculateOrientation();
	newCC.calculateImage();
	newCC.calculateArea();
	int num_singles = newCC.connNum_;

	for (int id = 1; id <= num_singles; id++) {
		if (newCC.area_[id - 1] < 20)
			return errorCode = 1;

		vector<int> bb = newCC.boundingBox_[id - 1];
		double angle = newCC.orientation_[id - 1];

		Mat single_obj_mask = newCC.image_[id - 1];

		Mat originalObj_I = imgForExtraction(Rect(bb[1], bb[0], bb[3], bb[2]));
		Mat originalObjI = originalObj_I.clone();

		Mat single_obj_mask_not;
		bitwise_not(single_obj_mask, single_obj_mask_not);
		bitwise_or(originalObjI, 255, originalObjI, single_obj_mask_not);


		Mat rotatedObj_I, rotated_Mask, rotated_Mask_not;
		if (angle > 0) {
			rotatedObj_I = imrotate(originalObjI, 90 - angle, "bilinear");
			rotated_Mask = imrotate(single_obj_mask, 90 - angle, "neareast");
		}
		else {
			rotatedObj_I = imrotate(originalObjI, -90 - angle, "bilinear");
			rotated_Mask = imrotate(single_obj_mask, -90 - angle, "neareast");
		}

		bitwise_not(rotated_Mask, rotated_Mask_not);
		bitwise_or(rotatedObj_I, 255, rotatedObj_I, rotated_Mask_not);

		ConnectedRegion CCrot(rotated_Mask, 8);
		CCrot.calculateBoundingBox();

		vector<int> bbrot = CCrot.boundingBox_[0];
		rotatedObj_I = rotatedObj_I(Rect(bbrot[1], bbrot[0], bbrot[3], bbrot[2]));


		Mat rotatedObj_Mask = rotated_Mask(Rect(bbrot[1], bbrot[0], bbrot[3], bbrot[2]));

		chromoElement.index = newCutIndex;
		chromoElement.relatedIndex = newCutIndex;
		chromoElement.chromoId = 0;

		chromoElement.cImg = originalObjI.clone();
		chromoElement.cImgRotated = rotatedObj_I.clone();
		chromoElement.cImgType = 1;
		position posElement;
		posElement.cImgMask = single_obj_mask.clone();
		posElement.cImgBoundingBox[0] = bb[0];
		posElement.cImgBoundingBox[1] = bb[1];
		posElement.cImgBoundingBox[2] = bb[2];
		posElement.cImgBoundingBox[3] = bb[3];
		posElement.cImgBoundingBoxOffset[0] = bbrot[0];
		posElement.cImgBoundingBoxOffset[1] = bbrot[1];
		posElement.cImgBoundingBoxOffset[2] = bbrot[2];
		posElement.cImgBoundingBoxOffset[3] = bbrot[3];
		posElement.cImgOrientation = angle;
		chromoElement.cImgPosition = posElement;
		for (int idx = 0; idx < 25; idx++)
			chromoElement.chromoCategoryInfo[idx] = 0;
		chromoElement.chromoUpright = 0;
	}

	return errorCode;
}