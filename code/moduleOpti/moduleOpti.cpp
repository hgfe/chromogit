//#include "conn.h"
//#include "main.h"
#include "stdafx.h"
#include "moduleOpti.h"

vector<int> histTpl = { 2,2,3,0,2,2,1,1,0,0,2,2,3,4,3,6,7,8,5,7,3,11,4,9,4,4,13,9,4,11,10,13,8,19,18,17,12,11,16,16,
23,22,21,33,27,25,22,31,30,28,31,21,38,39,32,32,40,26,45,42,51,41,39,49,47,54,60,63,63,43,47,56,62,54,66,57,80,68,82,
71,64,79,79,73,77,85,86,78,95,93,88,80,101,86,80,100,86,91,99,90,90,95,91,118,106,93,84,101,108,98,99,106,102,99,106,
95,89,102,92,111,101,106,87,103,95,111,91,116,102,107,81,109,95,107,89,105,97,97,97,94,100,103,108,110,103,103,85,96,
108,87,91,103,103,83,95,82,79,89,80,91,92,88,89,85,71,88,68,98,80,95,57,78,89,77,73,71,71,61,67,66,75,58,70,67,58,55,
47,67,54,54,81,63,61,47,61,57,44,61,43,45,52,55,59,47,54,48,52,50,50,53,44,59,57,57,38,42,52,49,57,47,52,56,48,56,43,
43,69,75,47,61,56,65,49,54,72,74,59,81,79,77,87,89,117,132,0,0,0,0,0,0,0,0,0,0,0,0 };


Mat histeq(Mat img, vector<int> hist) {
	Mat imgEq = Mat::zeros(img.size(), CV_8UC1);
	vector<int> originHist = imhist(img, 256);
	vector<int> transform(originHist.size(), -1);
	int sumHistTpl = 0;
	for (int idx = 0; idx < hist.size(); idx++) {
		sumHistTpl = sumHistTpl + hist[idx];
	}
	if (hist.size() != 256 || originHist.size() != 256)
		return imgEq;
	for (int i = 0; i < transform.size(); i++) {
		int allSum = 0, sk = 0;
		for (int j = 0; j <= i; j++)
			allSum = allSum + originHist[j];
		sk = (double)(255 * allSum) / (img.size().width * img.size().height);
		transform[i] = sk;
	}
	vector<int> Gtransform(hist.size(), -1);
	for (int i = 0; i < Gtransform.size(); i++) {
		int allSum = 0, gk = 0;
		for (int j = 0; j <= i; j++)
			allSum = allSum + hist[j];
		gk = (double)(255 * allSum) / sumHistTpl;
		Gtransform[i] = gk;
	}
	vector<int> Ftransform(originHist.size(), -1);
	int i = 0, j = 0;
	for (i = 0; i < Ftransform.size(); i++) {
		int sk = transform[i];
		for (j = 0; j < Gtransform.size(); j++) {
			if (sk == Gtransform[j])
				break;
			if (sk < Gtransform[j]) {
				j--;
				break;
			}
		}
		Ftransform[i] = j;
	}
	for (int col = 0; col < img.cols; col++) {
		for (int row = 0; row < img.rows; row++) {
			int index = img.at<uchar>(row, col);
			imgEq.at<uchar>(row, col) = Ftransform[index];
		}
	}
	return imgEq;
}

Mat moduleOpti(Mat originPicture, String pictureType) {
	/********************************** pre-process module
	* 优化图片模块
	* Input:
	* @param1 originPicture			待处理文件，已经打开的图片文件
	* @param2 pictureType			文件类型，"raw"（黑底） 或者 "tif" （白底）
	*
	* Output:
	* @param1						优化后的图片	
	*
	***********************************/

	bool bIntensityReverse = 0;
	if (pictureType == "raw" || pictureType == "Raw") {
		bIntensityReverse = 1;
	}
	else {
		bIntensityReverse = 0;
	}

	// 获取灰度图
	Mat imgGray;
	if (bIntensityReverse)
		bitwise_not(originPicture, imgGray);
	else
		imgGray = originPicture.clone();

	// 调整尺度大小
	// imgForExtraction 用于提取染色体图像数据
	int eH = 0, eW = 0;
	Mat imgForExtraction = imgUniform(imgGray, eH, eW);
	int origH = imgGray.rows;
	int origW = imgGray.cols;
	// float scaling = origH / targetH;
	Mat img2;
	Size dsize = Size(eW, eH);
	resize(imgForExtraction, img2, dsize, 0.0, 0.0, INTER_CUBIC);
	int resizeH = img2.rows, resizeW = img2.cols;

	//namedWindow("after imguniform", CV_WINDOW_FREERATIO);
	//namedWindow("after imguniform and resize", CV_WINDOW_FREERATIO);
	//imshow("after imguniform", img2);
	//imshow("after imguniform and resize", img2);
	//waitKey(0);

	/////////////////////////////////////////////////////////////////
	/////////////// 对比度提升 //////////////////////////////////////
	Mat BW;
	threshold(img2, BW, 0.99 * 255, 255, THRESH_BINARY);

	//namedWindow("after threshold", CV_WINDOW_FREERATIO);
	//imshow("after threshold", BW);
	//waitKey(0);

	//bitwise_not(BW, BW);
	BW = 255 - BW;

	//namedWindow("after inverse", CV_WINDOW_FREERATIO);
	//imshow("after inverse", BW);
	//waitKey(0);

	BW = imFill(BW);
	Mat kernel = Mat::ones(5, 5, CV_16UC1);
	cout << kernel << endl;
	morphologyEx(BW, BW, MORPH_OPEN, kernel);

	//namedWindow("after open", CV_WINDOW_FREERATIO);
	//imshow("after open", BW);
	//waitKey(0);


	Mat BW_2;
	bitwise_not(BW, BW_2);

	ConnectedRegion s(BW, 8);
	s.calculateArea();
	s.calculateConvexArea();
	s.calculatePixelList();
	for (int i = 0; i < s.connNum_; i++) {
		if (s.area_[i] > 10000) {
			if ((double)s.area_[i] / s.convexArea_[i] > 0.8) {
				vector<Point> pixelList = s.pixelList_[i];
				for (int idx = 0; idx < pixelList.size(); idx++) {
					int row = pixelList[idx].x;
					int col = pixelList[idx].y;
					BW.at<uchar>(row, col) = 0;
				}
			}
		}
	}

	//namedWindow("after raising contrast", CV_WINDOW_FREERATIO);
	//imshow("after raising contrast", BW);
	//waitKey(0);

	//////////////////////////////////////////////////////////////
	////////////////// 对比度优化 ////////////////////////////////

	vector<int> histOrigin = imhist(img2, 256);

	//cout << "histOrigin" << endl;
	//for (int idx = 0; idx < histTpl.size(); idx++)
	//	cout << histOrigin[idx] << endl;

	vector<int> histTplCopy(histTpl);
	int sumTpl = 0, sumOrigin = 0, k = 0;
	for (int i = 0; i < histTplCopy.size(); i++) {
		sumTpl = sumTpl + histTplCopy[i];
		sumOrigin = sumOrigin + histOrigin[i];
		if (i >= 244) {
			k = k + histOrigin[i];
		}
	}
	
	double coeff = (double)k / sumOrigin;
	coeff = coeff / (1 - coeff);

	for (int i = 244; i < histTplCopy.size(); i++) {
		histTplCopy[i] = (sumTpl * coeff / k) * histOrigin[i];
	}

	//cout << "histTpl" << endl;
	//int sumHistTplFinal = 0;
	//for (int idx = 0; idx < histTpl.size(); idx++) {
	//	cout << histTpl[idx] << endl;
	//	sumHistTplFinal = sumHistTplFinal + histTpl[idx];
	//}
	//cout << "histTplSum" << endl;
	//cout << sumHistTplFinal << endl;

	// 直方图匹配
	Mat matchedImg = histeq(img2, histTplCopy);
	
	Mat optiPicture = matchedImg.clone();
	return optiPicture;
}
