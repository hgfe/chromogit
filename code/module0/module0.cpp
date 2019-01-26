#include "stdafx.h"
#include "module0.h"

int moduleScoring(Mat originPicture, String pictureType,
	float & pictureScore, int & singleNum) {
	/********************************** pre-process module
	* 打分模块
	* Input:
	* @param1 originPicture			待处理文件，已经打开的图片文件
	* @param2 pictureType			文件类型，"raw"（黑底） 或者 "tif" （白底）
	*
	* Output:
	* @param1 pictureScore			图片分数
	* @param2 singleNum				单条染色体个数
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

	int preSingleNum = 0, preSingleArea = 0, preSingleEnvelope = 0;
	preSeg(originPicture, bIntensityReverse, false, preSingleNum, preSingleArea, preSingleEnvelope);

	float avgLength = (float)preSingleArea / preSingleNum;


	// pictureScore = 0.7 * preSingleNum + 0.3 * avgLength;
	if (preSingleNum == 0)
		pictureScore = -10000;
	else
		// pictureScore = 0.7 * (preSingleNum - 5)/5 + 0.3 * (avgLength / 100 - 20) /10;
		pictureScore = ChromoScore(avgLength, preSingleNum);
	singleNum = preSingleNum;

	return errorCode;
}
