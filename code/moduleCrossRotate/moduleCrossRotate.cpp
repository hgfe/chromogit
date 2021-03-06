#include "stdafx.h"
#include "moduleCrossRotate.h"

int moduleCrossRotate(chromo& chromoElement) {
	/********************************** auto segment module
	* Input/Output:
	* @param chromoElement		待处理/处理后染色体结构体
	*
	***********************************/
	int errorCode = 0;

	if (chromoElement.cImgType != 0)
		return errorCode = 1;
	if (chromoElement.cImgPosition.cImgBoundingBoxOffset[0] != 0)
		return errorCode = 1;
	
	Mat originalObjI = chromoElement.cImg.clone();
	Mat clusterObjMask = chromoElement.cImgPosition.cImgMask.clone();

	ConnectedRegion clusterCC(clusterObjMask, 4);
	clusterCC.calculateOrientation();
	clusterCC.calculateImage();
	int num = clusterCC.connNum_;
	if (num != 1)
		return errorCode = 1;
	
	double angle = clusterCC.orientation_[0];

	Mat rotatedObj_I, rotated_Mask, rotated_Mask_not;
	if (angle > 0) {
		rotatedObj_I = imrotate(originalObjI, 90 - angle, "bilinear");
		rotated_Mask = imrotate(clusterObjMask, 90 - angle, "neareast");
	}
	else {
		rotatedObj_I = imrotate(originalObjI, -90 - angle, "bilinear");
		rotated_Mask = imrotate(clusterObjMask, -90 - angle, "neareast");
	}

	bitwise_not(rotated_Mask, rotated_Mask_not);
	bitwise_or(rotatedObj_I, 255, rotatedObj_I, rotated_Mask_not);

	ConnectedRegion CCrot(rotated_Mask, 8);
	CCrot.calculateBoundingBox();

	vector<int> bbrot = CCrot.boundingBox_[0];
	rotatedObj_I = rotatedObj_I(Rect(bbrot[1], bbrot[0], bbrot[3], bbrot[2]));

	chromoElement.cImgRotated = rotatedObj_I.clone();
	chromoElement.cImgPosition.cImgBoundingBoxOffset[0] = bbrot[0];
	chromoElement.cImgPosition.cImgBoundingBoxOffset[1] = bbrot[1];
	chromoElement.cImgPosition.cImgBoundingBoxOffset[2] = bbrot[2];
	chromoElement.cImgPosition.cImgBoundingBoxOffset[3] = bbrot[3];
	chromoElement.cImgPosition.cImgOrientation = angle;
	
	return errorCode;
}