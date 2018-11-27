#include "module1.h"

Mat findSkeleton(Mat obj, int thresh, vector<Point> & ep, vector<Point> & bp) {
	Mat skr = skeleton(obj);
	threshold(skr, skr, thresh, 255, THRESH_BINARY);
	Mat skel = ThiningDIBSkeleton(skr);

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

				vector<Point> chosenPixelList = sExtCrossingPoints.pixelList_[chosen];
				for (int n = 0; n < chosenPixelList.size(); n++) {
					tmp.at<uchar>(chosenPixelList[n]) = 0;
				}
				skelExt = skelExt - tmp;
			}
			skel = (skel + skelExt).mul(obj);

		}
	}

	anaskel(skel, ep, bp);

	return skel;
}

vector<Mat> findSkelLengthOrder(Mat skeleton, vector<Point> ep, vector<Point> bp) {
	int imgIndex = 1, nEP = ep.size(), nBP = bp.size();

	vector<int> length;
	vector<Mat> paths;
	for (int i = 1; i <= nEP - 1; i++) {
		for (int j = i + 1; j <= nEP; j++) {
			Mat path;
			int len;
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

				sortedPath[i] = paths[j].clone();
				sortedPath[j] = paths[i].clone();
			}
		}
	}

	return sortedPath;

	//if (nBP != 0) {
	//	vector<int> numEPBP(nEP, 0);
	//	for (int i = 0; i < nEP; i++) {
	//		vector<int> lenEPBP(nBP, 0);
	//		for (int j = 0; j < nBP; j++) {
	//			Mat tempPath;
	//			findPathNLength(skeleton, ep[i], bp[j], tempPath, lenEPBP[j]);
	//		}

	//		// 找最小值的下标
	//		vector<int>::iterator minimum = min_element(begin(lenEPBP), end(lenEPBP));
	//		int minIndex = distance(begin(lenEPBP), minimum);
	//		numEPBP[i] = minIndex;
	//	}

	//	imgIndex = 1;
	//}


}

int moduleCrossSeg(chromo chromoData, vector<vector<chromo>>& chromoDataList) {

	Mat img = chromoData.cImg;
	Mat obj = chromoData.cImgPosition.cImgMask;

	int errorCode = 0;

	vector<Point> ep, bp;
	Mat skel = findSkeleton(obj, 35, ep, bp);
	vector<Mat> skeletonStructEpEpPath = findSkelLengthOrder(skel, ep, bp);

	Mat cutPoints = getCutPoints(obj, 0.15, 30, "or");

	int nEP = ep.size(), nBP = ep.size();

	if ((nBP == 2 && nEP == 4) || (nBP == 1 && nEP == 4)) {
		// 十字交叉型
		Mat Points;
		vector<cuttingListStru> cuttingList;
		vector<Mat> commonArea;
		findPointMuiltipleCluster(obj, cutPoints, skel, bp, ep, Points, cuttingList, commonArea);
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

		// 按顺序排列
		vector<int> a(4, 0);
		a[0] = firstMaxIdx; a[3] = secondMaxIdx;
		for (int idx = 0; idx < 4; idx++) {
			if (idx != firstMaxIdx && idx != secondMaxIdx) {
				a[1] = idx;
				continue;
			}
			if (idx != firstMaxIdx && idx != secondMaxIdx) {
				a[2] = idx;
				break;
			}
		}
		
		// 取出pixelList
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
		line(tmpCutLines, cuttingLists11[0][0], cuttingLists11[0][1], Scalar::all(255), 1, 4);
		line(tmpCutLines, cuttingLists11[1][0], cuttingLists11[1][1], Scalar::all(255), 1, 4);
		for (int skelidx = 0; skelidx < skeletonStructEpEpPath.size(); skelidx++) {
			if (!countNonZero(tmpCutLines & skeletonStructEpEpPath[skelidx])) {
				skeleton1 = skeletonStructEpEpPath[skelidx].clone();
				break;
			}
		}
		Mat cutobj = obj.clone();
		bitwise_and(cutobj, 0, cutobj, tmpCutLines);
		ConnectedRegion cccutobj11(cutobj, 8);
		cccutobj11.calculatePixelIdxList();
		cccutobj11.calculatePixelList();
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
		cut_comb_final[0].chromosome[0].cImg = cutobj.clone();

		vector<vector<Point>> cuttingLists12(2, cuttingListElement);
		//cuttingListElement = cuttingLists[0];
		cuttingLists12[0][0] = cutPointListX.pixelList_[a[0]][0];
		cuttingLists12[0][1] = cutPointListX.pixelList_[a[2]][0];
		cuttingLists12[1][0] = cutPointListX.pixelList_[a[1]][0];
		cuttingLists12[1][1] = cutPointListX.pixelList_[a[3]][0];

		Mat skeleton2;
		tmpCutLines = Mat::zeros(obj.size(), CV_8UC1);
		line(tmpCutLines, cuttingLists12[0][0], cuttingLists12[0][1], Scalar::all(255), 1, 4);
		line(tmpCutLines, cuttingLists12[1][0], cuttingLists12[1][1], Scalar::all(255), 1, 4);
		for (int skelidx = 0; skelidx < skeletonStructEpEpPath.size(); skelidx++) {
			if (!countNonZero(tmpCutLines & skeletonStructEpEpPath[skelidx])) {
				skeleton2 = skeletonStructEpEpPath[skelidx].clone();
				break;
			}
		}
		cutobj = obj.clone();
		bitwise_and(cutobj, 0, cutobj, tmpCutLines);
		ConnectedRegion cccutobj12(cutobj, 8);
		cccutobj12.calculatePixelIdxList();
		cccutobj12.calculatePixelList();
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

		cut_comb_final[0].chromosome[1].cImg = cutobj.clone();

		vector<vector<Point>> cuttingLists2(2, cuttingListElement);
		//cuttingListElement = cuttingLists[0];
		cuttingLists2[0][0] = cutPointListX.pixelList_[a[0]][0];
		cuttingLists2[0][1] = cutPointListX.pixelList_[a[3]][0];


		cutobj = obj.clone();

		tmpCutLines = Mat::zeros(obj.size(), CV_8UC1);
		line(tmpCutLines, cuttingLists2[0][0], cuttingLists2[0][1], Scalar::all(255), 1, 4);

		bitwise_and(cutobj, 0, cutobj, tmpCutLines);

		ConnectedRegion cccutobj2(cutobj, 8);
		cccutobj2.calculatePixelIdxList();
		cccutobj2.calculatePixelList();
		for (int i = 0; i < cccutobj2.connNum_; i++) {
			vector<Point> tmpPoints = cccutobj2.pixelList_[i];
			Mat cutobj2 = cutobj.clone();
			for (int j = 0; j < tmpPoints.size(); j++) {
				cutobj2.at<uchar>(tmpPoints[j].x, tmpPoints[j].y) = 0;
				cut_comb_final[1].chromosome[j].cImg = cutobj.clone();
			}
		}


		vector<vector<Point>> cuttingLists3(2, cuttingListElement);
		//cuttingListElement = cuttingLists[0];
		cuttingLists3[0][0] = cutPointListX.pixelList_[a[1]][0];
		cuttingLists3[0][1] = cutPointListX.pixelList_[a[2]][0];


		cutobj = obj.clone();

		tmpCutLines = Mat::zeros(obj.size(), CV_8UC1);
		line(tmpCutLines, cuttingLists3[0][0], cuttingLists3[0][1], Scalar::all(255), 1, 4);

		bitwise_and(cutobj, 0, cutobj, tmpCutLines);

		ConnectedRegion cccutobj3(cutobj, 8);
		cccutobj3.calculatePixelIdxList();
		cccutobj3.calculatePixelList();
		for (int i = 0; i < cccutobj3.connNum_; i++) {
			vector<Point> tmpPoints = cccutobj3.pixelList_[i];
			Mat cutobj2 = cutobj.clone();
			for (int j = 0; j < tmpPoints.size(); j++) {
				cutobj2.at<uchar>(tmpPoints[j].x, tmpPoints[j].y) = 0;
				cut_comb_final[2].chromosome[j].cImg = cutobj.clone();
			}
		}
		// 以上是三种切割方式，分别存入了cut_comb_final[0][1][2]
		// 这之间有合成显示相关的代码，需要吗？

	}
	else
		return errorCode = 1;



	return errorCode;
}