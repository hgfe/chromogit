#include "stdafx.h"
#include "moduleCrossSeg.h"

int moduleCrossSeg(chromo chromoData, int newCutNum, vector<vector<chromo>>& chromoDataList) {

	Mat img = chromoData.cImg.clone();
	Mat obj = chromoData.cImgPosition.cImgMask.clone();

	int errorCode = 0;

	if (img.rows == 0 || img.cols == 0 || obj.rows == 0 || obj.cols == 0)
		return errorCode = 1;

	vector<chromo> diagonalCutting(2, chromoData);
	vector<chromo> leftRightCutting(2, chromoData);
	vector<chromo> topDownCutting(2, chromoData);

	vector<Point> ep, bp;
	Mat skel = findSkeleton(obj, 35, ep, bp);	// 55~60

	if (ep.empty() || bp.empty())
		return errorCode = 1;

	vector<Mat> skeletonStructEpEpPath = findSkelLengthOrder(skel, ep, bp);

	Mat cutPoints = getCutPoints(obj, 0.15, 30, "or");

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

		ConnectedRegion cutPointListX(Points, 8);
		cutPointListX.calculatePixelList();

		if (cutPointListX.connNum_ != 4)
			return errorCode = 1;

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
			for (int j = i + 1; j < 4; j++) {
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


		ConnectedRegion cccutobj11(cutobj, 8);
		cccutobj11.calculatePixelList();
		cccutobj11.calculateBoundingBox();
		cccutobj11.calculateOrientation();

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

		Mat notCutobj;
		bitwise_not(cutobj, notCutobj);
		imgMasked = imgMasked + notCutobj;

		// 获取单条染色体的原图
		ConnectedRegion ROIprops(cutobj, 8);
		ROIprops.calculateBoundingBox();
		ROIprops.calculateOrientation();
		Rect bbox(ROIprops.boundingBox_[0][1], ROIprops.boundingBox_[0][0],
			ROIprops.boundingBox_[0][3], ROIprops.boundingBox_[0][2]);
		Mat singleImg = imgMasked(bbox).clone();
		Mat singleMask = cutobj(bbox).clone();

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
		diagonalCutting1Pos.cImgBoundingBoxOffset[0] = bbrot[0];
		diagonalCutting1Pos.cImgBoundingBoxOffset[1] = bbrot[1];
		diagonalCutting1Pos.cImgBoundingBoxOffset[2] = bbrot[2];
		diagonalCutting1Pos.cImgBoundingBoxOffset[3] = bbrot[3];
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
		diagonalCutting2Pos.cImgBoundingBoxOffset[0] = bbrot1[0];
		diagonalCutting2Pos.cImgBoundingBoxOffset[1] = bbrot1[1];
		diagonalCutting2Pos.cImgBoundingBoxOffset[2] = bbrot1[2];
		diagonalCutting2Pos.cImgBoundingBoxOffset[3] = bbrot1[3];
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
			leftRightCuttingPos.cImgBoundingBoxOffset[0] = bbrot[0];
			leftRightCuttingPos.cImgBoundingBoxOffset[1] = bbrot[1];
			leftRightCuttingPos.cImgBoundingBoxOffset[2] = bbrot[2];
			leftRightCuttingPos.cImgBoundingBoxOffset[3] = bbrot[3];
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
			topDownCuttingPos.cImgBoundingBoxOffset[0] = bbrot[0];
			topDownCuttingPos.cImgBoundingBoxOffset[1] = bbrot[1];
			topDownCuttingPos.cImgBoundingBoxOffset[2] = bbrot[2];
			topDownCuttingPos.cImgBoundingBoxOffset[3] = bbrot[3];
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

		}
		chromoDataList.push_back(topDownCutting);
		// 以上是三种切割方式，分别存入了 chromoDataList[0] [1] [2]
#pragma endregion

	}
	else
		return errorCode = 1;

	return errorCode;
}