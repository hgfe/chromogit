#include "stdafx.h"
#include "moduleCrossSeg.h"

Mat findSkeleton(Mat obj, int thresh, vector<Point> & ep, vector<Point> & bp) {
	Mat skr = skeleton(obj);

	//namedWindow("skr1", CV_WINDOW_FREERATIO);
	//imshow("skr1", skr);
	//waitKey(0);

	//imwrite("skr.tif", skr);

	threshold(skr, skr, thresh, 255, THRESH_BINARY);


	Mat skel = ThiningDIBSkeleton(skr);

	// 这一段是为了消掉上面的细化函数
	// 可能出现的错误
	// 譬如长出来很短的一截小枝
	int times = 0;
	while (times < 100) {
		times++;
		ep.clear();
		bp.clear();
		anaskel(skel, ep, bp);
		cuttingListStru cuttingList = pDist2(ep, bp);
		if (cuttingList.dist < 3 && cuttingList.dist > 0.01) {
			Point tmpep = cuttingList.point1;
			Point tmpbp = cuttingList.point2;
			skel.at<uchar>(tmpep.x, tmpep.y) = 0;
		}
		else
			break;
	}


	//namedWindow("skr2", CV_WINDOW_FREERATIO);
	//imshow("skr2", skel);
	//waitKey(0);

	anaskel(skel, ep, bp);
	int nEP = ep.size(), nBP = bp.size();

	if (nBP > 0) {
		vector<Point> ep2(nEP, Point(0, 0));

		for (int i = 0; i < nEP; i++) {
			vector<int> lenEPBP(nBP, 0);
			for (int j = 0; j < nBP; j++) {
				Mat temp = Mat::zeros(skel.size(), CV_8UC1);
				findPathNLength(skel, ep[i], bp[j], temp, lenEPBP[j]);
			}

			// 找最小值
			vector<int>::iterator minimum = min_element(begin(lenEPBP), end(lenEPBP));
			Mat minPath = Mat::zeros(skel.size(), CV_8UC1);
			int minLength = 0;
			int minIndex = distance(begin(lenEPBP), minimum);
			findPathNLength(skel, ep[i], bp[minIndex], minPath, minLength);

			// 获取 bwtraceboundary 的结果
			// 先用 findcontours 找轮廓
			// 然后对点排序
			vector<vector<Point>> contours;
			vector<Vec4i> hierarchy;
			findContours(minPath, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
			vector<Point> contourToBeSorted = contours[0];
			// 排序
			// 先把从第一个开始到 length 个元素拿出来
			int length = countNonZero(minPath);
			vector<Point> temp(length, Point(0, 0));
			copy(contourToBeSorted.begin(), contourToBeSorted.begin() + length, temp.begin());
			// 然后在 temp 里面排序
			Point compTemp(ep[i].y, ep[i].x);		// ep[i] 反转一下
			vector<Point> contour(length, Point(0, 0));
			for (int idx = 0; idx < length; idx++) {
				if (temp[idx] == compTemp) {
					//if (idx == 0) {
					//	copy(temp.begin(), temp.end(), contour.begin());
					//	break;
					//}
					copy(temp.begin() + idx, temp.end(), contour.begin());
					int leftLength = length - idx;
					copy(temp.begin(), temp.begin() + idx, contour.begin() + leftLength);
					break;
				}
			}

			// contour 反转
			for (int idx = 0; idx < length; idx++) {
				int temp = contour[idx].x;
				contour[idx].x = contour[idx].y;
				contour[idx].y = temp;
			}

			int CL = 0;
			if (minLength > 20)
				CL = 10;
			else if (minLength > 10)
				CL = 5;
			else
				CL = minLength;

			// 向外延伸末端点，使其与obj_mask相交
			int L = 30;
			ep2[i] = ep[i] + (ep[i] - contour[CL - 1]) * (L / norm(Mat(ep[i]), Mat(contour[CL - 1])));
			fitExtention(ep2[i], ep[i], contour[CL - 1], obj.size());

			Mat skelExt = Mat::zeros(skel.size(), CV_8UC1);

			line(skelExt, Point(ep2[i].y, ep2[i].x), Point(ep[i].y, ep[i].x), Scalar::all(255));

			//namedWindow("line", CV_WINDOW_FREERATIO);
			//imshow("line", skelExt);
			//waitKey(0);

			Mat skelExtObjMask;
			bitwise_and(skelExt, obj, skelExtObjMask);
			ConnectedRegion sExtCrossingPoints(skelExtObjMask, 8);
			sExtCrossingPoints.calculateBoundingBox();
			int numSExtCrossingPoints = sExtCrossingPoints.connNum_;

			if (numSExtCrossingPoints > 1) {		// 如果有多个交点，选择最近的
				Mat tmp = skelExt.clone();

				double dist = 1000;
				int chosen = 0;
				for (int n = 1; n <= numSExtCrossingPoints; n++) {
					double centerX = sExtCrossingPoints.centroids_.at<double>(n, 0);
					double centerY = sExtCrossingPoints.centroids_.at<double>(n, 1);
					Point center(centerX, centerY);

					if (norm(Mat(center), Mat(ep[i])) < dist) {
						chosen = n;
						dist = norm(Mat(center), Mat(ep[i]));
					}
				}

				vector<Point> chosenPixelList = sExtCrossingPoints.pixelList_[chosen-1];
				for (int n = 0; n < chosenPixelList.size(); n++) {
					tmp.at<uchar>(chosenPixelList[n]) = 0;
				}
				skelExt = skelExt - tmp;
			}
			skel = (skel + skelExt).mul(obj);

		}
	}

	//namedWindow("skeleton2", CV_WINDOW_FREERATIO);
	//imshow("skeleton2", skel);
	//waitKey(0);
	ep.clear(); bp.clear();
	anaskel(skel, ep, bp);

	return skel;
}

vector<Mat> findSkelLengthOrder(Mat skeleton, vector<Point> ep, vector<Point> bp) {
	int imgIndex = 1, nEP = ep.size(), nBP = bp.size();

	vector<int> length;
	vector<Mat> paths;
	for (int i = 1; i <= nEP - 1; i++) {
		for (int j = i + 1; j <= nEP; j++) {
			Mat path = Mat::zeros(skeleton.size(), CV_8UC1);
			int len = 0;
			findPathNLength(skeleton, ep[i - 1], ep[j - 1], path, len);

			length.push_back(len);
			paths.push_back(path);
			imgIndex = imgIndex + 1;
		}
	}

	// 按照 length 的降序给 path 排序
	vector<Mat> sortedPath(paths);
	for (int i = 0; i < length.size() - 1; i++) {
		for (int j = i + 1; j < length.size(); j++) {
			if (length[i] < length[j]) {
				double templength = length[i];
				length[i] = length[j];
				length[j] = templength;

				Mat tempPath = sortedPath[i].clone();
				sortedPath[i] = sortedPath[j].clone();
				sortedPath[j] = tempPath;
			}
		}
	}

	return sortedPath;
}

int moduleCrossSeg(chromo chromoData,int newCutNum, vector<vector<chromo>>& chromoDataList) {

	Mat img = chromoData.cImg.clone();
	Mat obj = chromoData.cImgPosition.cImgMask.clone();

	vector<chromo> diagonalCutting(2, chromoData);
	vector<chromo> leftRightCutting(2, chromoData);
	vector<chromo> topDownCutting(2, chromoData);

	int errorCode = 0;

	vector<Point> ep, bp;
	Mat skel = findSkeleton(obj, 35, ep, bp);	// 55~60
	//namedWindow("skel", CV_WINDOW_FREERATIO);
	//imshow("skel", skel);
	//waitKey(0);

	vector<Mat> skeletonStructEpEpPath = findSkelLengthOrder(skel, ep, bp);

	//namedWindow("skeleton", CV_WINDOW_FREERATIO);
	//imshow("skeleton", skel);
	//waitKey(0);

	//for (int idx = 0; idx < skeletonStructEpEpPath.size(); idx++) {
	//	char windowName = ' ';
	//	sprintf_s(&windowName, 20, "%d path", idx);
	//	namedWindow(&windowName, CV_WINDOW_FREERATIO);
	//	imshow(&windowName, skeletonStructEpEpPath[idx]);
	//	waitKey(0);
	//}

	Mat cutPoints = getCutPoints(obj, 0.15, 30, "or");

	//namedWindow("cutPoints", CV_WINDOW_FREERATIO);
	//imshow("cutPoints", cutPoints);
	//waitKey(0);

	int nEP = ep.size(), nBP = bp.size();
	if ((nBP == 2 && nEP == 4) || (nBP == 1 && nEP == 4)) {
		// 十字交叉型
		if (nBP == 2) {
			double tempDist = norm(Mat(bp[0]), Mat(bp[1]));
			if (tempDist > 8)
				return errorCode = 1;
		}


		Mat Points;
		vector<cuttingListStru> cuttingList;
		vector<Mat> commonArea;
		findPointMuiltipleCluster(obj, cutPoints, skel, bp, ep, Points, cuttingList, commonArea);

		//namedWindow("pointsMap", CV_WINDOW_FREERATIO);
		//imshow("pointsMap", Points);
		//waitKey(0);

		ConnectedRegion cutPointListX(Points, 8);
		cutPointListX.calculatePixelList();

		for (int i = 0; i < 4; i++) {
			cuttingListStru minElement = pDist2(cutPointListX.pixelList_[i], bp);
			cutPointListX.pixelList_[i].clear();
			cutPointListX.pixelList_[i].push_back(minElement.point1);
		}

		vector<Point> pointVec(4, Point(0, 0));
		for (int i = 0; i < 4; i++)
			pointVec[i] = Point(cutPointListX.pixelList_[i][0].x,
				cutPointListX.pixelList_[i][0].y);

		// 找 pointVec 中最大的距离
		double maxDist = norm(Mat(pointVec[0]), Mat(pointVec[1]));
		int firstMaxIdx = 0, secondMaxIdx = 1;
		for (int i = 0; i < 3; i++) {
			for (int j = i+1; j < 4; j++) {
				double dist = norm(Mat(pointVec[i]), Mat(pointVec[j]));
				if (dist > maxDist) {
					maxDist = dist;
					firstMaxIdx = i; secondMaxIdx = j;
				}

			}
		}

		// 按顺序排列下标
		// 距离最大的两个点的下标分别放在 a 的一号位和四号位
		// 剩下两个点的下标分别放在 a 的二号位和三号位
		vector<int> a(4, 0);
		a[0] = firstMaxIdx; a[3] = secondMaxIdx;
		bool oneOk = false;
		for (int idx = 0; idx < 4; idx++) {
			if (idx != firstMaxIdx && idx != secondMaxIdx && oneOk == false) {
				oneOk = true;
				a[1] = idx;
				continue;
			}
			if (idx != firstMaxIdx && idx != secondMaxIdx) {
				a[2] = idx;
				break;
			}
		}
		
#pragma region diagonal first cutting
		// 按要求取点
		// cuttingLists 分为两个vector
		// 两个 vector 里面分别存了两个点
		vector<Point> cuttingListElement(2, Point(0, 0));
		vector<vector<Point>> cuttingLists11(2, cuttingListElement);
		//cuttingListElement = cuttingLists[0];
		cuttingLists11[0][0] = cutPointListX.pixelList_[a[0]][0];
		cuttingLists11[0][1] = cutPointListX.pixelList_[a[1]][0];
		cuttingLists11[1][0] = cutPointListX.pixelList_[a[2]][0];
		cuttingLists11[1][1] = cutPointListX.pixelList_[a[3]][0];

		Mat skeleton1;
		Mat tmpCutLines = Mat::zeros(obj.size(), CV_8UC1);
		line(tmpCutLines, Point(cuttingLists11[0][0].y,
								cuttingLists11[0][0].x),
							Point(cuttingLists11[0][1].y,
								cuttingLists11[0][1].x),
								Scalar::all(255), 1, 4);
		line(tmpCutLines, Point(cuttingLists11[1][0].y,
								cuttingLists11[1][0].x),
							Point(cuttingLists11[1][1].y,
								cuttingLists11[1][1].x),
								Scalar::all(255), 1, 4);
		for (int skelidx = 0; skelidx < skeletonStructEpEpPath.size(); skelidx++) {
			if (!countNonZero(tmpCutLines & skeletonStructEpEpPath[skelidx])) {
				skeleton1 = skeletonStructEpEpPath[skelidx].clone();
				break;
			}
		}
		Mat cutobj = obj.clone();
		bitwise_and(cutobj, 0, cutobj, tmpCutLines);

		//namedWindow("skeleton1", CV_WINDOW_FREERATIO);
		//imshow("skeleton1", skeleton1);
		//namedWindow("cutobj", CV_WINDOW_FREERATIO);
		//imshow("cutobj", cutobj);
		//namedWindow("tmpCutLines", CV_WINDOW_FREERATIO);
		//imshow("tmpCutLines", tmpCutLines);
		//waitKey(0);


		ConnectedRegion cccutobj11(cutobj, 8);
		cccutobj11.calculatePixelList();
		cccutobj11.calculateBoundingBox();
		cccutobj11.calculateOrientation();
		//cccutobj11.calculateImage();
		for (int i = 0; i < cccutobj11.connNum_; i++) {
			Mat tmp = Mat::zeros(cutobj.size(), CV_8UC1);
			vector<Point> tmpPoints = cccutobj11.pixelList_[i];
			for (int j = 0; j < tmpPoints.size(); j++) {
				tmp.at<uchar>(tmpPoints[j].x, tmpPoints[j].y) = 255;
			}
			if (!countNonZero(tmp & skeleton1)) {
				for (int j = 0; j < tmpPoints.size(); j++) {
					cutobj.at<uchar>(tmpPoints[j].x, tmpPoints[j].y) = 0;
				}
			}
		}
		// 此时的obj是大图mask
		Mat imgMasked;
		bitwise_and(img, cutobj, imgMasked);

		//namedWindow("imgMasked", CV_WINDOW_FREERATIO);
		//imshow("imgMasked", imgMasked);
		//waitKey(0);

		Mat notCutobj;
		bitwise_not(cutobj, notCutobj);
		imgMasked = imgMasked + notCutobj;

		//namedWindow("imgMasked1", CV_WINDOW_FREERATIO);
		//imshow("imgMasked1", imgMasked);
		//waitKey(0);

		//namedWindow("notObj", CV_WINDOW_FREERATIO);
		//imshow("notObj", notCutobj);
		//waitKey(0);

		// 获取单条染色体的原图
		ConnectedRegion ROIprops(cutobj, 8);
		ROIprops.calculateBoundingBox();
		ROIprops.calculateOrientation();
		Rect bbox(ROIprops.boundingBox_[0][1], ROIprops.boundingBox_[0][0], 
			ROIprops.boundingBox_[0][3], ROIprops.boundingBox_[0][2]);
		Mat singleImg = imgMasked(bbox).clone();
		Mat singleMask = cutobj(bbox).clone();

		//namedWindow("singleImg", CV_WINDOW_FREERATIO);
		//imshow("singleImg", singleImg);
		//waitKey(0);



		double angle = ROIprops.orientation_[0];
		// 旋转操作
		Mat rotatedObj_I, rotated_Mask, rotated_Mask_not;
		if (angle > 0) {
			rotatedObj_I = imrotate(singleImg, 90 - angle, "bilinear");
			rotated_Mask = imrotate(singleMask, 90 - angle, "neareast");
		}
		else {
			rotatedObj_I = imrotate(singleImg, -90 - angle, "bilinear");
			rotated_Mask = imrotate(singleMask, -90 - angle, "neareast");
		}
		bitwise_not(rotated_Mask, rotated_Mask_not);
		bitwise_or(rotatedObj_I, 255, rotatedObj_I, rotated_Mask_not);
		ConnectedRegion CCrot(rotated_Mask, 8);
		CCrot.calculateBoundingBox();
		vector<int> bbrot = CCrot.boundingBox_[0];
		rotatedObj_I = rotatedObj_I(Rect(bbrot[1], bbrot[0], bbrot[3], bbrot[2]));
		// 旋转操作 end

		position diagonalCutting1Pos;
		diagonalCutting1Pos.cImgMask = singleMask.clone();
		diagonalCutting1Pos.cImgBoundingBox[0] = chromoData.cImgPosition.cImgBoundingBox[0] + ROIprops.boundingBox_[0][0];
		diagonalCutting1Pos.cImgBoundingBox[1] = chromoData.cImgPosition.cImgBoundingBox[1] + ROIprops.boundingBox_[0][1];
		diagonalCutting1Pos.cImgBoundingBox[2] = ROIprops.boundingBox_[0][2];
		diagonalCutting1Pos.cImgBoundingBox[3] = ROIprops.boundingBox_[0][3];
		diagonalCutting1Pos.cImgOrientation = angle;
		diagonalCutting[0].index = newCutNum + 0;
		diagonalCutting[0].relatedIndex = chromoData.relatedIndex;
		diagonalCutting[0].chromoId = 1;
		diagonalCutting[0].cImgType = true;
		diagonalCutting[0].cImg = singleImg.clone();
		diagonalCutting[0].cImgRotated = rotatedObj_I.clone();
		diagonalCutting[0].cImgPosition = diagonalCutting1Pos;
		for (int idx = 0; idx < 25; idx++)
			diagonalCutting[0].chromoCategoryInfo[idx] = 0;
		diagonalCutting[0].chromoUpright = 0;
		///////////////
		//namedWindow("test1", CV_WINDOW_FREERATIO);
		//imshow("test1", cutobj);
		//namedWindow("test2", CV_WINDOW_FREERATIO);
		//imshow("test2", singleImg);
		//namedWindow("test3", CV_WINDOW_FREERATIO);
		//imshow("test3", rotatedObj_I);
		//waitKey(0);

		
#pragma endregion

#pragma region diagonal second cutting
		vector<vector<Point>> cuttingLists12(2, cuttingListElement);
		//cuttingListElement = cuttingLists[0];
		cuttingLists12[0][0] = cutPointListX.pixelList_[a[0]][0];
		cuttingLists12[0][1] = cutPointListX.pixelList_[a[2]][0];
		cuttingLists12[1][0] = cutPointListX.pixelList_[a[1]][0];
		cuttingLists12[1][1] = cutPointListX.pixelList_[a[3]][0];

		Mat skeleton2;
		tmpCutLines = Mat::zeros(obj.size(), CV_8UC1);
		line(tmpCutLines, Point(cuttingLists12[0][0].y,
								cuttingLists12[0][0].x),
							Point(cuttingLists12[0][1].y,
								cuttingLists12[0][1].x),
								Scalar::all(255), 1, 4);
		line(tmpCutLines, Point(cuttingLists12[1][0].y,
								cuttingLists12[1][0].x),
							Point(cuttingLists12[1][1].y,
								cuttingLists12[1][1].x),
								Scalar::all(255), 1, 4);
		for (int skelidx = 0; skelidx < skeletonStructEpEpPath.size(); skelidx++) {
			if (!countNonZero(tmpCutLines & skeletonStructEpEpPath[skelidx])) {
				skeleton2 = skeletonStructEpEpPath[skelidx].clone();
				break;
			}
		}
		cutobj = obj.clone();
		bitwise_and(cutobj, 0, cutobj, tmpCutLines);
		ConnectedRegion cccutobj12(cutobj, 8);
		cccutobj12.calculatePixelList();
		cccutobj12.calculateBoundingBox();
		cccutobj12.calculateOrientation();
		for (int i = 0; i < cccutobj12.connNum_; i++) {
			Mat tmp = Mat::zeros(cutobj.size(), CV_8UC1);
			vector<Point> tmpPoints = cccutobj12.pixelList_[i];
			for (int j = 0; j < tmpPoints.size(); j++) {
				tmp.at<uchar>(tmpPoints[j].x, tmpPoints[j].y) = 255;
			}
			if (!countNonZero(tmp & skeleton2)) {
				for (int j = 0; j < tmpPoints.size(); j++) {
					cutobj.at<uchar>(tmpPoints[j].x, tmpPoints[j].y) = 0;
				}
			}
		}

		bitwise_and(img, cutobj, imgMasked);
		bitwise_not(cutobj, notCutobj);
		imgMasked = imgMasked + notCutobj;

		// 获取单条染色体的原图
		ConnectedRegion ROIprops1(cutobj, 8);
		ROIprops1.calculateBoundingBox();
		ROIprops1.calculateOrientation();
		Rect bbox1(ROIprops1.boundingBox_[0][1], ROIprops1.boundingBox_[0][0],
			ROIprops1.boundingBox_[0][3], ROIprops1.boundingBox_[0][2]);
		singleImg = imgMasked(bbox1).clone();
		singleMask = cutobj(bbox1).clone();

		angle = ROIprops1.orientation_[0];
		// 旋转操作
		// Mat rotatedObj_I, rotated_Mask, rotated_Mask_not;
		if (angle > 0) {
			rotatedObj_I = imrotate(singleImg, 90 - angle, "bilinear");
			rotated_Mask = imrotate(singleMask, 90 - angle, "neareast");
		}
		else {
			rotatedObj_I = imrotate(singleImg, -90 - angle, "bilinear");
			rotated_Mask = imrotate(singleMask, -90 - angle, "neareast");
		}
		bitwise_not(rotated_Mask, rotated_Mask_not);
		bitwise_or(rotatedObj_I, 255, rotatedObj_I, rotated_Mask_not);

		//namedWindow("test4", CV_WINDOW_FREERATIO);
		//imshow("test4", cutobj);
		//namedWindow("test5", CV_WINDOW_FREERATIO);
		//imshow("test5", singleImg);
		//namedWindow("test6", CV_WINDOW_FREERATIO);
		//imshow("test6", rotatedObj_I);
		//namedWindow("test7", CV_WINDOW_FREERATIO);
		//imshow("test7", rotated_Mask);
		//waitKey(0);


		ConnectedRegion CCrot1(rotated_Mask, 8);
		CCrot1.calculateBoundingBox();
		vector<int> bbrot1 = CCrot1.boundingBox_[0];
		rotatedObj_I = rotatedObj_I(Rect(bbrot1[1], bbrot1[0], bbrot1[3], bbrot1[2]));
		// 旋转操作 end

		position diagonalCutting2Pos;
		diagonalCutting2Pos.cImgMask = singleMask.clone();
		diagonalCutting2Pos.cImgBoundingBox[0] = chromoData.cImgPosition.cImgBoundingBox[0] + cccutobj12.boundingBox_[0][0];
		diagonalCutting2Pos.cImgBoundingBox[1] = chromoData.cImgPosition.cImgBoundingBox[1] + cccutobj12.boundingBox_[0][1];
		diagonalCutting2Pos.cImgBoundingBox[2] = cccutobj12.boundingBox_[0][2];
		diagonalCutting2Pos.cImgBoundingBox[3] = cccutobj12.boundingBox_[0][3];
		diagonalCutting2Pos.cImgOrientation = angle;
		diagonalCutting[1].index = newCutNum + 1;
		diagonalCutting[1].relatedIndex = chromoData.relatedIndex;
		diagonalCutting[1].chromoId = 1;
		diagonalCutting[1].cImg = singleImg.clone();
		diagonalCutting[1].cImgRotated = rotatedObj_I.clone();
		diagonalCutting[1].cImgPosition = diagonalCutting2Pos;
		diagonalCutting[1].cImgType = true;
		for (int idx = 0; idx < 25; idx++)
			diagonalCutting[1].chromoCategoryInfo[idx] = 0;
		diagonalCutting[1].chromoUpright = 0;

		chromoDataList.push_back(diagonalCutting);

		//namedWindow("test4", CV_WINDOW_FREERATIO);
		//imshow("test4", cutobj);
		//namedWindow("test5", CV_WINDOW_FREERATIO);
		//imshow("test5", singleImg);
		//namedWindow("test6", CV_WINDOW_FREERATIO);
		//imshow("test6", rotatedObj_I);
		//waitKey(0);
#pragma endregion

#pragma region )( cutting
		vector<vector<Point>> cuttingLists2(2, cuttingListElement);
		//cuttingListElement = cuttingLists[0];
		cuttingLists2[0][0] = cutPointListX.pixelList_[a[0]][0];
		cuttingLists2[0][1] = cutPointListX.pixelList_[a[3]][0];


		cutobj = obj.clone();

		tmpCutLines = Mat::zeros(obj.size(), CV_8UC1);
		line(tmpCutLines, Point(cuttingLists2[0][0].y,
								cuttingLists2[0][0].x),
							Point(cuttingLists2[0][1].y,
								cuttingLists2[0][1].x),
								Scalar::all(255), 1, 4);

		bitwise_and(cutobj, 0, cutobj, tmpCutLines);

		ConnectedRegion cccutobj2(cutobj, 8);
		cccutobj2.calculatePixelIdxList();
		cccutobj2.calculatePixelList();
		for (int i = 0; i < 2; i++) {
			vector<Point> tmpPoints = cccutobj2.pixelList_[i];
			Mat cutobj2 = cutobj.clone();
			for (int j = 0; j < tmpPoints.size(); j++) {
				cutobj2.at<uchar>(tmpPoints[j].x, tmpPoints[j].y) = 0;
			}

			// 依次存入 chromo 结构体向量
			ConnectedRegion cccutobj2Ele(cutobj2, 8);
			cccutobj2Ele.calculateBoundingBox();
			cccutobj2Ele.calculateOrientation();


			// 此时的obj是大图mask
			Mat imgMasked;
			bitwise_and(img, cutobj2, imgMasked);
			Mat notCutobj2;
			bitwise_not(cutobj2, notCutobj2);
			imgMasked = imgMasked + notCutobj2;

			// 获取单条染色体的原图
			Rect bbox(cccutobj2Ele.boundingBox_[0][1], cccutobj2Ele.boundingBox_[0][0],
				cccutobj2Ele.boundingBox_[0][3], cccutobj2Ele.boundingBox_[0][2]);
			Mat singleImg = imgMasked(bbox).clone();
			Mat singleMask = cutobj2(bbox).clone();
			angle = cccutobj2Ele.orientation_[0];
			// 旋转操作
			// Mat rotatedObj_I, rotated_Mask, rotated_Mask_not;
			if (angle > 0) {
				rotatedObj_I = imrotate(singleImg, 90 - angle, "bilinear");
				rotated_Mask = imrotate(singleMask, 90 - angle, "neareast");
			}
			else {
				rotatedObj_I = imrotate(singleImg, -90 - angle, "bilinear");
				rotated_Mask = imrotate(singleMask, -90 - angle, "neareast");
			}
			bitwise_not(rotated_Mask, rotated_Mask_not);
			bitwise_or(rotatedObj_I, 255, rotatedObj_I, rotated_Mask_not);
			ConnectedRegion CCrot(rotated_Mask, 8);
			CCrot.calculateBoundingBox();
			vector<int> bbrot = CCrot.boundingBox_[0];
			rotatedObj_I = rotatedObj_I(Rect(bbrot[1], bbrot[0], bbrot[3], bbrot[2]));
			// 旋转操作 end

			position leftRightCuttingPos;
			leftRightCuttingPos.cImgMask = singleMask.clone();
			leftRightCuttingPos.cImgBoundingBox[0] = chromoData.cImgPosition.cImgBoundingBox[0] + cccutobj2Ele.boundingBox_[0][0];
			leftRightCuttingPos.cImgBoundingBox[1] = chromoData.cImgPosition.cImgBoundingBox[1] + cccutobj2Ele.boundingBox_[0][1];
			leftRightCuttingPos.cImgBoundingBox[2] = cccutobj2Ele.boundingBox_[0][2];
			leftRightCuttingPos.cImgBoundingBox[3] = cccutobj2Ele.boundingBox_[0][3];
			leftRightCuttingPos.cImgOrientation = angle;
			leftRightCutting[i].index = newCutNum + i;
			leftRightCutting[i].relatedIndex = chromoData.relatedIndex;
			leftRightCutting[i].chromoId = 1;
			leftRightCutting[i].cImg = singleImg.clone();
			leftRightCutting[i].cImgRotated = rotatedObj_I.clone();
			leftRightCutting[i].cImgPosition = leftRightCuttingPos;
			leftRightCutting[i].cImgType = true;
			for (int idx = 0; idx < 25; idx++)
				leftRightCutting[i].chromoCategoryInfo[idx] = 0;
			leftRightCutting[i].chromoUpright = 0;

			//namedWindow("test7", CV_WINDOW_FREERATIO);
			//imshow("test7", cutobj2);
			//namedWindow("test8", CV_WINDOW_FREERATIO);
			//imshow("test8", singleImg);
			//namedWindow("test9", CV_WINDOW_FREERATIO);
			//imshow("test9", rotatedObj_I);
			//waitKey(0);
		}
		chromoDataList.push_back(leftRightCutting);

#pragma endregion

#pragma region v ^ cutting
		vector<vector<Point>> cuttingLists3(2, cuttingListElement);
		//cuttingListElement = cuttingLists[0];
		cuttingLists3[0][0] = cutPointListX.pixelList_[a[1]][0];
		cuttingLists3[0][1] = cutPointListX.pixelList_[a[2]][0];


		cutobj = obj.clone();

		tmpCutLines = Mat::zeros(obj.size(), CV_8UC1);
		line(tmpCutLines, Point(cuttingLists3[0][0].y,
								cuttingLists3[0][0].x),
							Point(cuttingLists3[0][1].y,
								cuttingLists3[0][1].x),
								Scalar::all(255), 1, 4);

		bitwise_and(cutobj, 0, cutobj, tmpCutLines);

		ConnectedRegion cccutobj3(cutobj, 8);
		cccutobj3.calculatePixelIdxList();
		cccutobj3.calculatePixelList();
		for (int i = 0; i < 2; i++) {
			vector<Point> tmpPoints = cccutobj3.pixelList_[i];
			Mat cutobj2 = cutobj.clone();
			for (int j = 0; j < tmpPoints.size(); j++) {
				cutobj2.at<uchar>(tmpPoints[j].x, tmpPoints[j].y) = 0;
			}
			// 依次存入 chromo 结构体向量
			ConnectedRegion cccutobj2Ele(cutobj2, 8);
			cccutobj2Ele.calculateBoundingBox();
			cccutobj2Ele.calculateOrientation();

			// 此时的obj是大图mask
			Mat imgMasked;
			bitwise_and(img, cutobj2, imgMasked);
			Mat notCutobj2;
			bitwise_not(cutobj2, notCutobj2);
			imgMasked = imgMasked + notCutobj2;

			// 获取单条染色体的原图
			Rect bbox(cccutobj2Ele.boundingBox_[0][1], cccutobj2Ele.boundingBox_[0][0],
				cccutobj2Ele.boundingBox_[0][3], cccutobj2Ele.boundingBox_[0][2]);
			Mat singleImg = imgMasked(bbox);
			Mat singleMask = cutobj2(bbox);
			angle = cccutobj2Ele.orientation_[0];
			// 旋转操作
			// Mat rotatedObj_I, rotated_Mask, rotated_Mask_not;
			if (angle > 0) {
				rotatedObj_I = imrotate(singleImg, 90 - angle, "bilinear");
				rotated_Mask = imrotate(singleMask, 90 - angle, "neareast");
			}
			else {
				rotatedObj_I = imrotate(singleImg, -90 - angle, "bilinear");
				rotated_Mask = imrotate(singleMask, -90 - angle, "neareast");
			}
			bitwise_not(rotated_Mask, rotated_Mask_not);
			bitwise_or(rotatedObj_I, 255, rotatedObj_I, rotated_Mask_not);
			ConnectedRegion CCrot(rotated_Mask, 8);
			CCrot.calculateBoundingBox();
			vector<int> bbrot = CCrot.boundingBox_[0];
			rotatedObj_I = rotatedObj_I(Rect(bbrot[1], bbrot[0], bbrot[3], bbrot[2]));
			// 旋转操作 end

			position topDownCuttingPos;
			topDownCuttingPos.cImgMask = singleMask.clone();
			topDownCuttingPos.cImgBoundingBox[0] = chromoData.cImgPosition.cImgBoundingBox[0] + cccutobj2Ele.boundingBox_[0][0];
			topDownCuttingPos.cImgBoundingBox[1] = chromoData.cImgPosition.cImgBoundingBox[1] + cccutobj2Ele.boundingBox_[0][1];
			topDownCuttingPos.cImgBoundingBox[2] = cccutobj2Ele.boundingBox_[0][2];
			topDownCuttingPos.cImgBoundingBox[3] = cccutobj2Ele.boundingBox_[0][3];
			topDownCuttingPos.cImgOrientation = angle;
			topDownCutting[i].index = newCutNum + i;
			topDownCutting[i].relatedIndex = chromoData.relatedIndex;
			topDownCutting[i].chromoId = 1;
			topDownCutting[i].cImg = singleImg.clone();
			topDownCutting[i].cImgRotated = rotatedObj_I.clone();
			topDownCutting[i].cImgPosition = topDownCuttingPos;
			topDownCutting[i].cImgType = true;
			for (int idx = 0; idx < 25; idx++)
				topDownCutting[i].chromoCategoryInfo[idx] = 0;
			topDownCutting[i].chromoUpright = 0;

			//namedWindow("test10", CV_WINDOW_FREERATIO);
			//imshow("test10", cutobj2);
			//namedWindow("test11", CV_WINDOW_FREERATIO);
			//imshow("test11", singleImg);
			//namedWindow("test12", CV_WINDOW_FREERATIO);
			//imshow("test12", rotatedObj_I);
			//waitKey(0);
		}
		chromoDataList.push_back(topDownCutting);
		// 以上是三种切割方式，分别存入了 chromoDataList[0] [1] [2]
#pragma endregion

	}
	else
		return errorCode = 1;

	return errorCode;
}