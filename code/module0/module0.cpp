//#include "conn.h"
//#include "main.h"
#include "stdafx.h"
#include "module0.h"

void moduleScoring(Mat originPicture, String pictureType,
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

	bool bIntensityReverse = 0;
	if (pictureType == "raw" || pictureType == "Raw") {
		bIntensityReverse = 1;
	}
	else {
		bIntensityReverse = 0;
	}

	int preSingleNum = 0, preSingleArea = 0;
	preSeg(originPicture, bIntensityReverse, false, preSingleNum, preSingleArea);

	float avgLength = (float)preSingleArea / preSingleNum;


	// pictureScore = 0.7 * preSingleNum + 0.3 * avgLength;
	if (preSingleNum == 0)
		pictureScore = -10000;
	else
		// pictureScore = 0.7 * (preSingleNum - 5)/5 + 0.3 * (avgLength / 100 - 20) /10;
		pictureScore = ChromoScore(avgLength, preSingleNum);
	singleNum = preSingleNum;

	return;
}

float ChromoScore(float avgLength, int singleNum) {
	float number = (float)singleNum;
	// 配置多项式系数
	float score = 0;
	float a = 1.279;
	float b1 = 0.774, b2 = 0.221;
	float c1 = 3.324, c2 = -2.865;
	//输入数据归一化
	float x = avgLength / 3178.7;
	float y = number / 48;
	//计算打分
	score = a + b1 * x + b2 * x*x + c1 * y + c2 * y*y;
	return score;
}


void preSeg(Mat img, bool bIntensityReverse, bool bCutTouching,
	int & preSingleNum, int & preSingleArea) {
	/**********************************
	* Input:
	* @param1 img					原图
	* @param2 bIntensityReverse		是否需要反置
	* @param3 bCutTouching			是否需切分黏连的染色体
	*
	* Output:
	* @param1 preSingleNum			
	* @param2 preSingleArea			
	*
	***********************************/

	// 获取灰度图
	Mat imgGray;
	if (bIntensityReverse)
		bitwise_not(img, imgGray);
	else
		imgGray = img.clone();

	// 调整尺度大小
	// imgForExtraction 用于提取染色体图像数据
	int eH = 0, eW = 0;
	Mat imgForExtraction = imgUniform(imgGray, eH, eW);

	Mat img2;
	Size dsize = Size(0.5 * eW, 0.5 * eH);
	resize(imgForExtraction, img2, dsize, 0.0, 0.0, INTER_CUBIC);
	int resizeH = img2.rows, resizeW = img2.cols;

	// 图像增强
	Mat lowHigh, lowHighOut;
	stretchlim(img2, lowHigh, 0.01, 0.99);
	imadjust(img2, img2, lowHigh, lowHighOut, 1);
	Mat I = img2;

	Mat BWmainbody, innerCutPoints;
	roughSegChromoRegion(I, BWmainbody, innerCutPoints);


	int choromosomeTotalArea = countNonZero(BWmainbody);
	double avgChoromosomeArea = (double)choromosomeTotalArea / 46;
	double singleMaxArea = avgChoromosomeArea * 3;

	Mat thickness;
	distanceTransform(BWmainbody, thickness, DIST_L2, DIST_MASK_PRECISE);

	ConnectedRegion CCregions(BWmainbody, 4);
	CCregions.calculateImage();
	CCregions.calculateBoundingBox();

	Mat clusters = Mat::zeros(resizeH, resizeW, CV_8UC1);
	Mat globalSkeleton = Mat::zeros(resizeH, resizeW, CV_8UC1);
	Mat SE = getStructuringElement(MORPH_RECT, Size(5, 5));

	for (int id = 1; id <= CCregions.connNum_; id++) {
		Mat objMask = CCregions.image_[id - 1];

		vector<int> bbox = CCregions.boundingBox_[id - 1];

		Mat objMaskSmooth;
		morphologyEx(objMask, objMaskSmooth, MORPH_CLOSE, SE);
		medianBlur(objMaskSmooth, objMaskSmooth, 3);

		Mat skr = skeleton(objMaskSmooth);

		threshold(skr, skr, 25, 255, THRESH_BINARY);

		Mat bwDBIskeleton = ThiningDIBSkeleton(skr);

		//imfilter
		Point anchor(0, 0);
		uchar kernel[3][3] = { { 1,1,1 },{ 1,1,1 },{ 1,1,1 } };
		Mat kernelMat = Mat(3, 3, CV_8UC1, &kernel);
		Mat neighbourCount;
		Mat bwThinInt;
		threshold(bwDBIskeleton, bwThinInt, 1, 1, THRESH_BINARY);
		filter2D(bwThinInt, neighbourCount, -1, kernelMat, anchor, 0.0, BORDER_CONSTANT);

		Mat bwBranches, bwEnds;
		bitwise_and(neighbourCount > 3, bwDBIskeleton, bwBranches);
		bitwise_and(neighbourCount <= 2, bwDBIskeleton, bwEnds);

		//把 bwDBIskeleton 复制到 globalSkeleton
		Mat globalSkeletonROI = globalSkeleton(Rect(bbox[1], bbox[0], bbox[3], bbox[2]));
		bwDBIskeleton.copyTo(globalSkeletonROI, bwDBIskeleton);

		//把 bwDBIskeleton 中值为真的索引在 thickness 中的数值取出来
		//计算其中小于 2 的数值个数
		Mat thicknessROI = thickness(Rect(bbox[1], bbox[0], bbox[3], bbox[2]));
		Mat notBWDBISkeleton;
		bitwise_not(bwDBIskeleton, notBWDBISkeleton);
		bitwise_and(thicknessROI, 0, thicknessROI, notBWDBISkeleton);
		int zerosInThicknessROI = bbox[3] * bbox[2] - countNonZero(thicknessROI);
		int sumThicknessSkelSmallerThan2 = countNonZero(thicknessROI < 2.0) - zerosInThicknessROI;

		///////////////////////
		if ((countNonZero(bwBranches) > 0 && countNonZero(bwEnds) > 2) || countNonZero(objMask) > singleMaxArea) {
			//把符合条件的 obj_mask 复制到 cluster 中
			Mat clusterROI = clusters(Rect(bbox[1], bbox[0], bbox[3], bbox[2]));
			objMask.copyTo(clusterROI, objMask);
		}
		else if (sumThicknessSkelSmallerThan2 > 0) {
			//把符合条件的 obj_mask 复制到 cluster 中
			Mat clusterROI = clusters(Rect(bbox[1], bbox[0], bbox[3], bbox[2]));
			objMask.copyTo(clusterROI, objMask);
		}
	}
	Mat singles;
	subtract(BWmainbody, clusters, singles);

	if (bCutTouching) {

		double globalAvg = (double)sum(I & BWmainbody)[0] / 255.0 / choromosomeTotalArea;
		int minArea = round(avgChoromosomeArea / 4);

		Mat notGlobalSkeleton;
		bitwise_not(globalSkeleton, notGlobalSkeleton);
		bitwise_and(thickness, 0, thickness, notGlobalSkeleton);
		double avgThickness = (double)sum(thickness)[0] / countNonZero(globalSkeleton);
	
		Mat innerCutPointsTmp = innerCutPoints.clone();

		ConnectedRegion CCclusters(clusters, 4);
		int numClusters = CCclusters.connNum_;
		Mat clustersLeft = Mat::zeros(clusters.size(), CV_8UC1);

		while (countNonZero(clusters) != 0) {
			CCclusters.calculateBoundingBox();
			CCclusters.calculateImage();

			for (int id = 1; id <= numClusters; id++) {
				Mat objMask = CCclusters.image_[id - 1];
				vector<int> bbox = CCclusters.boundingBox_[id - 1];

				Mat originalObjI = I(Rect(bbox[1], bbox[0], bbox[3], bbox[2]));
				Mat notObjMask; bitwise_not(objMask, notObjMask);
				bitwise_or(originalObjI, 1, originalObjI, notObjMask);

				Mat obj1;
				Mat innerPoints = innerCutPointsTmp(Rect(bbox[1], bbox[0], bbox[3], bbox[2]));
				if (countNonZero(innerPoints) > 0) {
					obj1 = innerCutting(objMask, originalObjI, innerPoints, globalAvg, minArea);
					ConnectedRegion CCobj1(obj1, 4);
					if (CCobj1.connNum_ > 1) {
						Mat zeroROI = Mat::zeros(bbox[2], bbox[3], CV_8UC1);
						Mat innerCutPointsTmpROI = innerCutPointsTmp(Rect(bbox[1], bbox[0], bbox[3], bbox[2]));
						zeroROI.copyTo(innerCutPointsTmpROI, objMask);
					}
					else if (CCobj1.connNum_ == 1 && countNonZero(objMask - obj1) > 0) {
						Mat zeroROI = Mat::zeros(bbox[2], bbox[3], CV_8UC1);
						Mat innerCutPointsTmpROI = innerCutPointsTmp(Rect(bbox[1], bbox[0], bbox[3], bbox[2]));
						zeroROI.copyTo(innerCutPointsTmpROI, objMask);
					}
					else {
						obj1.release();
					}
				}

				if (obj1.empty()) {
					Mat cutPoints = getCutPoints(objMask, 0.20, 40, "and");

					obj1 = cutTouching(objMask, originalObjI, cutPoints, globalAvg, avgThickness, minArea);
				}

				if (obj1.empty()) {
					Mat zeroROI = Mat::zeros(bbox[2], bbox[3], CV_8UC1);
					Mat mask = Mat::ones(bbox[2], bbox[3], CV_8UC1);
					Mat clusterROI = clusters(Rect(bbox[1], bbox[0], bbox[3], bbox[2]));
					Mat clusterLeftROI = clustersLeft(Rect(bbox[1], bbox[0], bbox[3], bbox[2]));
					zeroROI.copyTo(clusterROI, mask);
					objMask.copyTo(clusterLeftROI, mask);
				}
				else {
					Mat localSingles = Mat::zeros(obj1.size(), CV_8UC1);
					Mat localClusters = Mat::zeros(obj1.size(), CV_8UC1);
					Mat newSkel = Mat::zeros(obj1.size(), CV_8UC1);
					findClusters(obj1, localSingles, localClusters, newSkel);

					Mat onesROI = Mat::ones(bbox[2], bbox[3], CV_8UC1);

					Mat singlesROI = singles(Rect(bbox[1], bbox[0], bbox[3], bbox[2]));
					localSingles.copyTo(singlesROI, onesROI);

					Mat zeroROI = Mat::zeros(bbox[2], bbox[3], CV_8UC1);
					Mat clusterROI = clusters(Rect(bbox[1], bbox[0], bbox[3], bbox[2]));
					Mat globalSkeletonROI = globalSkeleton(Rect(bbox[1], bbox[0], bbox[3], bbox[2]));

					//去掉原先的骨架，重新放置新的骨架
					zeroROI.copyTo(clusterROI, objMask);
					localClusters.copyTo(clusterROI, localClusters);
					localClusters.copyTo(clusterROI, onesROI);
					newSkel.copyTo(globalSkeletonROI, onesROI);
				}
			}

			if (countNonZero(clusters) > 0) {
				clusters = bwareaopen(clusters, 50, 4);
				ConnectedRegion CCclustersTmp(clusters, 4);
				numClusters = CCclustersTmp.connNum_;
				CCclusters = CCclustersTmp;
			}

		}

		clusters = clustersLeft;
		singles = bwareaopen(singles, 50, 4);
	}

	Mat singlesForExtraction;
	dsize = Size(eW, eH);
	resize(singles, singlesForExtraction, dsize, 0.0, 0.0, INTER_NEAREST);


	ConnectedRegion singlesCC(singlesForExtraction, 4);

	preSingleNum = singlesCC.connNum_;
	preSingleArea = countNonZero(singlesForExtraction);

	return;
}

void roughSegChromoRegion(Mat I, Mat & BWmainbody, Mat & innerCutPoints) {
	int resizeH = I.rows, resizeW = I.cols;

	Mat BW;
	//adaptthresh
	int blockSize1 = 2 * floor(resizeH / 16) + 1;
	int blockSize2 = 2 * floor(resizeW / 16) + 1;
	int blockSize = (blockSize1 < blockSize2) ? blockSize2 : blockSize1;
	adaptiveThreshold(I, BW, 255,
		CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV,
		blockSize, 1);

	BW = clearBorder(BW);
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	morphologyEx(BW, BW, MORPH_OPEN, element);

	Mat BW2 = 255 * Mat::ones(I.size(), CV_8UC1);
	BW2 = BW2 - BW;
	BW2 = clearBorder(BW2);
	innerCutPoints = bwareaopen(BW2, 25, 4);			// 这里是另一个输出

	BW = imFill(BW);
	bitwise_and(BW, 0, BW, innerCutPoints);
	
	BWmainbody = bwareaopen(BW, 50, 4);					// 这里是输出
	Mat BW_suiti = BW - BWmainbody;

	Mat BW5 = Mat::zeros(BWmainbody.size(), CV_8UC1);
	if (countNonZero(BW_suiti) > 0) {
		int radius = 10;
		ConnectedRegion cc_suiti(BW_suiti, 4);
		cc_suiti.calculatePixelList();

		Mat Disk, search_bw, Disk2;
		for (int i = 0; i < cc_suiti.connNum_; i++) {
			Disk = Mat::zeros(BWmainbody.size(), CV_8UC1);
			Point center = Point(cc_suiti.centroids_.at<double>(i + 1, 0), cc_suiti.centroids_.at<double>(i + 1, 1));
			circle(Disk, center, radius, Scalar::all(255), -1);

			bitwise_and(BWmainbody, Disk, search_bw);
			Disk2 = Mat::zeros(BWmainbody.size(), CV_8UC1);
			if (countNonZero(search_bw) > 0) {
				ConnectedRegion cc_search_bw(search_bw, 8);
				cc_search_bw.calculatePixelList();

				vector<Point> suitiPixel = cc_suiti.pixelList_[i];
				vector<Point> seaBWPixel = cc_search_bw.pixelList_[0];
				cuttingListStru min = pDist2(suitiPixel, seaBWPixel);
				BW5.at<uchar>(min.point2.x, min.point2.y) = 255;

				BW5 = imreconstruct(BW5, search_bw);
				for (int temp = 0; temp < cc_suiti.pixelList_[i].size(); temp++) {
					BW5.at<uchar>(cc_suiti.pixelList_[i][temp].x, cc_suiti.pixelList_[i][temp].y) = 255;
				}

				int radius2 = SQR(min.dist + 5);
				circle(Disk2, Point(min.point1.y, min.point1.x), radius2, Scalar::all(255), -1);
				bitwise_and(BW5, Disk2, BW5);

				int sz = 2 * ceil(min.dist + 1) + 1;
				Mat elementclose = getStructuringElement(MORPH_ELLIPSE, Size(sz, sz));
				morphologyEx(BW5, BW5, MORPH_CLOSE, elementclose);
				BW5 = imFill(BW5);
				bitwise_or(BWmainbody, 1, BWmainbody, BW5);
			}
		}
	}

	return;
}