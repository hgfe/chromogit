// module1d.cpp: 定义 DLL 应用程序的导出函数。
//

#include "stdafx.h"
#include "module1.h"

void moduleSeg(Mat originPicture, String pictureType, String patientId, String glassId, String karyoId,
	Mat& optiPicture, String& optiPictureType, vector<chromo>& chromoData) {
	/********************************** auto segment module
	* Input:
	* @param1 originPicture			待处理文件，已经打开的图片文件
	* @param2 pictureType			文件类型，"raw"（黑底） 或者 "tif" （白底）
	* @param3 patientId				患者 ID
	* @param4 glassId				玻片 ID
	* @param5 karyoId				核型图 ID
	*
	* Output:
	* @param1 optiPicture			优化后的图片
	* @param2 optiPictureType		优化后图片保存格式，"tif"
	* @param3 chromoData			切割的图片数组
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
	Mat img = originPicture.clone();
	Mat imgGray;
	if (bIntensityReverse)
		imgGray = 255 - img;
	else
		imgGray = img;

	// 预处理
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

	//stretchlim(img2, lowHigh, 0.01, 0.99);
	//imadjust(imgForExtraction, imgForExtraction, lowHigh, lowHighOut, 1);


	optiPicture = I;								// 第一个输出
	optiPictureType = "tif";						// 第二个输出

													//adaptthresh
	Mat BW = Mat::zeros(img2.size(), CV_8UC1);
	int blockSize1 = 2 * floor(resizeH / 16) + 1;
	int blockSize2 = 2 * floor(resizeW / 16) + 1;
	int blockSize = (blockSize1 < blockSize2) ? blockSize2 : blockSize1;
	adaptiveThreshold(img2, BW, 255,
		CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV,
		blockSize, 1);

	BW = clearBorder(BW);
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	morphologyEx(BW, BW, MORPH_OPEN, element);

	Mat BW2 = 255 * Mat::ones(img2.size(), CV_8UC1);
	BW2 = BW2 - BW;
	BW2 = clearBorder(BW2);
	Mat innerCutPoints = bwareaopen(BW2, 25, 4);

	BW = imFill(BW);
	bitwise_and(BW, 0, BW, innerCutPoints);

	Mat BWmainbody = bwareaopen(BW, 50, 4);
	Mat BW_suiti = BW - BWmainbody;


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
			Mat BW5 = Mat::zeros(BWmainbody.size(), CV_8UC1);
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
				bitwise_or(BWmainbody, 255, BWmainbody, BW5);
			}
		}
	}

	int choromosomeTotalArea = countNonZero(BWmainbody);
	double globalAvg = (double)sum(I & BWmainbody)[0] / 255.0 / choromosomeTotalArea;

	double avgChoromosomeArea = (double)choromosomeTotalArea / 46;
	int minArea = round(avgChoromosomeArea / 4);
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

	Mat notGlobalSkeleton;
	bitwise_not(globalSkeleton, notGlobalSkeleton);
	bitwise_and(thickness, 0, thickness, notGlobalSkeleton);

	double avgThickness = (double)sum(thickness)[0] / countNonZero(globalSkeleton);

	Mat innerCutPointsTmp = innerCutPoints.clone();
	ConnectedRegion CCclusters(clusters, 4);
	int numClusters = CCclusters.connNum_;
	Mat clustersLeft = Mat::zeros(clusters.size(), CV_8UC1);

	Mat originalIAndCluster = I & clusters;

	while (countNonZero(clusters) != 0) {
		CCclusters.calculateBoundingBox();
		CCclusters.calculateImage();

		for (int id = 1; id <= numClusters; id++) {
			Mat objMask = CCclusters.image_[id - 1];
			vector<int> bbox = CCclusters.boundingBox_[id - 1];

			Mat originalObjI = originalIAndCluster(Rect(bbox[1], bbox[0], bbox[3], bbox[2]));
			Mat notObjMask; bitwise_not(objMask, notObjMask);
			bitwise_or(originalObjI, 255, originalObjI, notObjMask);

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
				Mat clusterROI = clusters(Rect(bbox[1], bbox[0], bbox[3], bbox[2]));
				Mat clusterLeftROI = clustersLeft(Rect(bbox[1], bbox[0], bbox[3], bbox[2]));
				// Mat mask = Mat::ones(bbox[2], bbox[3], CV_8UC1);

				zeroROI.copyTo(clusterROI, objMask);
				objMask.copyTo(clusterLeftROI, objMask);
			}
			else {
				Mat localSingles = Mat::zeros(obj1.size(), CV_8UC1);
				Mat localClusters = Mat::zeros(obj1.size(), CV_8UC1);
				Mat newSkel = Mat::zeros(obj1.size(), CV_8UC1);
				findClusters(obj1, localSingles, localClusters, newSkel);

				Mat onesROI = Mat::ones(bbox[2], bbox[3], CV_8UC1);
				Mat zeroROI = Mat::zeros(bbox[2], bbox[3], CV_8UC1);

				Mat singlesROI = singles(Rect(bbox[1], bbox[0], bbox[3], bbox[2]));
				localSingles.copyTo(singlesROI, localSingles);

				// Mat zeroROI = Mat::zeros(bbox[2], bbox[3], CV_8UC1);
				Mat clusterROI = clusters(Rect(bbox[1], bbox[0], bbox[3], bbox[2]));
				Mat globalSkeletonROI = globalSkeleton(Rect(bbox[1], bbox[0], bbox[3], bbox[2]));

				//去掉原先的骨架，重新放置新的骨架
				zeroROI.copyTo(clusterROI, objMask);
				localClusters.copyTo(clusterROI, localClusters);
				// localClusters.copyTo(clusterROI, onesROI);
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

#pragma region save singles
	Mat singles_forExtraction;
	resize(singles, singles_forExtraction, Size(eW, eH), 0, 0, INTER_NEAREST);
	ConnectedRegion singles_CC(singles_forExtraction, 4);
	singles_CC.calculateBoundingBox();
	singles_CC.calculateOrientation();
	singles_CC.calculateImage();
	int num_singles = singles_CC.connNum_;

	for (int id = 1; id <= num_singles; id++) {
		vector<int> bb = singles_CC.boundingBox_[id - 1];
		double angle = singles_CC.orientation_[id - 1];

		Mat single_obj_mask = singles_CC.image_[id - 1];
		
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

		chromo chromoElement;
		chromoElement.index = id;
		chromoElement.relatedIndex = id;
		chromoElement.chromoId = 0;
		chromoElement.cImgType = 1;
		chromoElement.cImg = originalObjI.clone();
		chromoElement.cImgRotated = rotatedObj_I.clone();
		position posElement;
		posElement.cImgMask = single_obj_mask.clone();
		posElement.cImgBoundingBox[0] = bb[0];
		posElement.cImgBoundingBox[1] = bb[1];
		posElement.cImgBoundingBox[2] = bb[2];
		posElement.cImgBoundingBox[3] = bb[3];
		posElement.cImgOrientation = angle;
		chromoElement.cImgPosition = posElement;
		for (int idx = 0; idx < 25; idx++)
			chromoElement.chromoCategoryInfo[idx] = 0;
		chromoElement.chromoUpright = 0;

		chromoData.push_back(chromoElement);
	}

#pragma endregion


#pragma region save clusters before cutting
	Mat clusters_forExtraction;
	resize(clusters, clusters_forExtraction, Size(eW, eH), 0, 0, INTER_NEAREST);
	ConnectedRegion clusters_CC(clusters_forExtraction, 4);
	clusters_CC.calculateBoundingBox();
	clusters_CC.calculateImage();
	int num_clusters = clusters_CC.connNum_;

	for (int id = 1; id <= num_clusters; id++) {
		vector<int> bb = clusters_CC.boundingBox_[id - 1];
		Mat cluster_obj_mask = clusters_CC.image_[id - 1];
		Mat originalObj_I = imgForExtraction(Rect(bb[1], bb[0], bb[3], bb[2]));
		Mat cluster_obj_mask_not;
		bitwise_not(cluster_obj_mask, cluster_obj_mask_not);

		Mat originalObjI = originalObj_I.clone();
		bitwise_or(originalObjI, 255, originalObjI, cluster_obj_mask_not);

		chromo chromoElement;
		chromoElement.index = num_singles + id;
		chromoElement.relatedIndex = num_singles + id;
		chromoElement.chromoId = 0;
		chromoElement.cImgType = 0;
		chromoElement.cImg = originalObjI.clone();
		chromoElement.cImgRotated = originalObjI.clone();
		position posElement;
		posElement.cImgMask = cluster_obj_mask.clone();
		posElement.cImgBoundingBox[0] = bb[0];
		posElement.cImgBoundingBox[1] = bb[1];
		posElement.cImgBoundingBox[2] = bb[2];
		posElement.cImgBoundingBox[3] = bb[3];
		posElement.cImgOrientation = 0;
		chromoElement.cImgPosition = posElement;
		for (int idx = 0; idx < 25; idx++)
			chromoElement.chromoCategoryInfo[idx] = 0;
		chromoElement.chromoUpright = 0;

		chromoData.push_back(chromoElement);
	}

#pragma endregion

#pragma region cut clusters & save

	Mat labeledClusters, stats, centroids;
	connectedComponentsWithStats(clusters, labeledClusters, stats, centroids, 4);
	vector<Mat> clusters_images;
	for (int id = 1; id <= num_clusters; id++) {
		Mat clusters_image = Mat::zeros(clusters.size(), CV_8UC1);
		for (int row = 0; row < clusters.rows; row++) {
			int * rowPt = labeledClusters.ptr<int>(row);
			uchar * rowPtclusters = clusters_image.ptr<uchar>(row);
			for (int col = 0; col < clusters.cols; col++) {
				rowPtclusters[col] = rowPt[col] == id ? 255 : 0;
			}
		}
		medianBlur(clusters_image, clusters_image, 3);
		clusters_images.push_back(clusters_image);
	}
	// CutClusterComb

	// 用来存每个 cluster 分割出的 single 数目
	vector<int> singleNumInClusters(num_clusters, 0);
	int finalIndex = 1;
	for (int id = 0; id < num_clusters; id++) {
		Mat selected_ch = clusters_images[id];
		ConnectedRegion selected_ch_pps(selected_ch, 8);
		selected_ch_pps.calculateBoundingBox();
		if (selected_ch_pps.connNum_ == 0)
			continue;

		vector<int> BB = selected_ch_pps.boundingBox_[0];
		Mat cropped_bw = selected_ch(Rect(BB[1], BB[0], BB[3], BB[2]));

		Mat selected_chI = I & selected_ch;
		Mat cropped_gray_I = selected_chI(Rect(BB[1], BB[0], BB[3], BB[2]));
		Mat cropped_bw_not;
		bitwise_not(cropped_bw, cropped_bw_not);
		bitwise_or(cropped_gray_I, 255, cropped_gray_I, cropped_bw_not);

		vector<vector<Mat>> cut_comb_final = separateMultipleOverlapped2new(cropped_bw, cropped_gray_I, globalAvg, minArea);

		////////////// save
		int comb_idx = 0;
		
		if (!cut_comb_final.empty()) {
			for (int j = 0; j < cut_comb_final[comb_idx].size(); j++) {

				Mat new_single_img = Mat::zeros(Size(resizeW, resizeH), CV_8UC1);
				Mat ROI = new_single_img(Rect(BB[1], BB[0], BB[3], BB[2]));
				Mat copycut = cut_comb_final[comb_idx][j];
				copycut.copyTo(ROI, copycut);
				Mat single_forExtraction;
				resize(new_single_img, single_forExtraction, Size(eW, eH));

				ConnectedRegion stats(single_forExtraction, 8);
				stats.calculateBoundingBox();
				stats.calculateOrientation();
				stats.calculateImage();
				vector<int> bb = stats.boundingBox_[0];
				double angle = stats.orientation_[0];
				Mat single_obj_mask = stats.image_[0];

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

				chromo chromoElement;
				chromoElement.index = num_singles + num_clusters + finalIndex;
				chromoElement.relatedIndex = num_singles + id + 1;
				chromoElement.chromoId = 0;
				chromoElement.cImgType = 1;
				chromoElement.cImg = originalObjI.clone();
				chromoElement.cImgRotated = rotatedObj_I.clone();
				position posEle;
				posEle.cImgBoundingBox[0] = bb[0];
				posEle.cImgBoundingBox[1] = bb[1];
				posEle.cImgBoundingBox[2] = bb[2];
				posEle.cImgBoundingBox[3] = bb[3];
				posEle.cImgMask = single_obj_mask.clone();
				posEle.cImgOrientation = angle;
				chromoElement.cImgPosition = posEle;
				for (int idx = 0; idx < 25; idx++)
					chromoElement.chromoCategoryInfo[idx] = 0;
				chromoElement.chromoUpright = 0;

				chromoData.push_back(chromoElement);
				finalIndex++;
			}
		}
	}
#pragma endregion 
	return;
}