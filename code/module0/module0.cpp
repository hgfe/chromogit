#include "stdafx.h"
#include "module0.h"

int moduleScoring(Mat originPicture, String pictureType,
	float & pictureScore, int & singleNum) {
	/********************************** pre-process module
	* ���ģ��
	* Input:
	* @param1 originPicture			�������ļ����Ѿ��򿪵�ͼƬ�ļ�
	* @param2 pictureType			�ļ����ͣ�"raw"���ڵף� ���� "tif" ���׵ף�
	*
	* Output:
	* @param1 pictureScore			ͼƬ����
	* @param2 singleNum				����Ⱦɫ�����
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
