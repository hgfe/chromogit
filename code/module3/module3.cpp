#pragma once
#include "stdafx.h"
#include "module3.h"

typedef struct var_ {
	Mat skeleton;
	vector<int> sepskel;
	bool iscluster;
	Mat obj;
} var;

int ManualSegment(chromo chromoData, vector<chromo>& newChromoDataList, vector<Mat> allLines, Point clickPosi, int segType, int newCutNum)
{
	int errorCode = 0;
	if (allLines.size() == 0)
		return errorCode = 1;
	for (int idx = 0; idx < allLines.size(); idx++) {
		allLines[idx] = allLines[idx] > 1;
	}

	if (segType == 0)
	{
		// 边缘分割
		chromo newChromoData;
		if (allLines.size() != 1)
			return errorCode = 1;
		Mat lines = allLines[0].clone();
		Mat m_cacheBinaryMask = chromoData.cImgPosition.cImgMask.clone();
		Mat m_recordMouseTrack_EdgeMask = lines.clone();
		Mat m_cacheImgGray = chromoData.cImg.clone();

		//int errorCode = 0;

		//Mat m_cacheBinaryMask = chromoData.cImgPosition.cImgMask.clone();
		//Mat m_recordMouseTrack_EdgeMask = allLines[0].clone();

		//// 1019修改 start
		//m_recordMouseTrack_EdgeMask = (m_recordMouseTrack_EdgeMask > 1);
		//// 1019修改 end

		//Mat m_cacheImgGray = chromoData.cImg.clone();


		int rows = m_recordMouseTrack_EdgeMask.rows;
		int cols = m_recordMouseTrack_EdgeMask.cols;
		if (rows == m_cacheImgGray.rows && cols == m_cacheImgGray.cols) {
			// 在分割图mask上断开边缘的区域
			Mat tmp;
			bitwise_and(m_cacheBinaryMask, m_recordMouseTrack_EdgeMask, tmp);
			Mat cache_mask = m_cacheBinaryMask - tmp;
			Mat tmp_labels;
			connectedComponents(cache_mask, tmp_labels, 4);
			int location_label = tmp_labels.at<int>(clickPosi.y, clickPosi.x);

			//namedWindow("cache_mask", CV_WINDOW_FREERATIO);
			//imshow("cache_mask", cache_mask);
			//waitKey(0);

			if (location_label > 0) //选中有效的染色体区域
			{
				Mat selected_region_Mask = (tmp_labels == location_label);

				//namedWindow("selected_region_Mask", WINDOW_FREERATIO);
				//imshow("selected_region_Mask", selected_region_Mask);
				//waitKey(0);

				Mat selected_region_Img;
				bitwise_and(m_cacheImgGray, selected_region_Mask, selected_region_Img);
				Mat tmp_reverse;
				bitwise_not(selected_region_Mask, tmp_reverse);

				selected_region_Img = selected_region_Img + tmp_reverse;


				// 计算框选的图片的外接矩形框
				vector<vector<Point> > contours;
				vector<Vec4i> hierarchy;
				findContours(selected_region_Mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
				if (contours.empty())
					return errorCode = 1;
				Rect rec = boundingRect(contours[0]);

				//计算orientation 将分割出的染色体进行旋转，并保证请不越界
				int border = max(selected_region_Mask.rows, selected_region_Mask.cols);
				Mat expansion_region_Mask;
				Mat expansion_region_Img;
				copyMakeBorder(selected_region_Mask, expansion_region_Mask, border, border, border, border, cv::BORDER_CONSTANT, Scalar(0));
				copyMakeBorder(selected_region_Img, expansion_region_Img, border, border, border, border, cv::BORDER_CONSTANT, Scalar(255));

				findContours(expansion_region_Mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

				//Rect boundRect = boundingRect(Mat(contours[0])); //只取第一个 存在危险
				RotatedRect rotatedRect = fitEllipse(Mat(contours[0]));

				Mat rot_mat;
				float angle;
				if (rotatedRect.angle < 90)
				{
					rot_mat = getRotationMatrix2D(rotatedRect.center, rotatedRect.angle, 1.0);
					angle = rotatedRect.angle;
				}
				else {
					rot_mat = getRotationMatrix2D(rotatedRect.center, rotatedRect.angle - 180, 1.0);
					angle = rotatedRect.angle - 180;
				}

				Mat dst_mask;
				warpAffine(expansion_region_Mask, dst_mask, rot_mat, expansion_region_Mask.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT);
				Mat dst_img;
				warpAffine(expansion_region_Img, dst_img, rot_mat, expansion_region_Mask.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, Scalar(255));

				//对mask做一次小膨胀
				Mat element = getStructuringElement(cv::MORPH_ELLIPSE, Size(3, 3), Point(-1, -1));
				dilate(dst_mask, dst_mask, element);

				//再次计算bounding box
				findContours(dst_mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

				Rect boundRect = boundingRect(Mat(contours[0])); //只取第一个 存在危险

				Mat cropped_chromo_Mask = dst_mask(boundRect);
				Mat cropped_chromo_Img = dst_img(boundRect);

				// 再截取
				Mat selected_region_Img_final = selected_region_Img(rec);
				Mat selected_region_Mask_final = selected_region_Mask(rec);

				chromo newChromoData;
				newChromoData.index = newCutNum;
				newChromoData.relatedIndex = chromoData.index;
				newChromoData.chromoId = 0;
				newChromoData.cImgType = 1;
				newChromoData.cImg = selected_region_Img_final;
				newChromoData.cImgRotated = cropped_chromo_Img;
				newChromoData.cImgPosition.cImgMask = selected_region_Mask_final;
				newChromoData.cImgPosition.cImgBoundingBox[0] = rec.y + chromoData.cImgPosition.cImgBoundingBox[0];
				newChromoData.cImgPosition.cImgBoundingBox[1] = rec.x + chromoData.cImgPosition.cImgBoundingBox[1];
				newChromoData.cImgPosition.cImgBoundingBox[2] = rec.height; //chromoData.cImgPosition.cImgBoundingBox[2];
				newChromoData.cImgPosition.cImgBoundingBox[3] = rec.width; //chromoData.cImgPosition.cImgBoundingBox[3];
				newChromoData.cImgPosition.cImgBoundingBoxOffset[0] = boundRect.y;
				newChromoData.cImgPosition.cImgBoundingBoxOffset[1] = boundRect.x;
				newChromoData.cImgPosition.cImgBoundingBoxOffset[2] = boundRect.height;
				newChromoData.cImgPosition.cImgBoundingBoxOffset[3] = boundRect.width;
				newChromoData.cImgPosition.cImgOrientation = angle;

				for (int i = 0; i < 25; i++) {
					newChromoData.chromoCategoryInfo[i] = 0;
				}

				newChromoData.chromoUpright = 0;

				newChromoDataList.push_back(newChromoData);
			}
		}
		else {
			errorCode = 1;
		}
	}
	else if (segType == 1)
	{

		// 基于骨架分割
		Mat origin = chromoData.cImg.clone();
		Mat mask = chromoData.cImgPosition.cImgMask.clone();

		//namedWindow("mask", CV_WINDOW_FREERATIO);
		//imshow("mask", mask);
		//waitKey(0);

		if (allLines.size() < 1)
			return errorCode = 1;
		Mat skelSeparate = Mat::zeros(allLines[0].size(), CV_8UC1);
		for (int idx = 0; idx < allLines.size(); idx++) {
			Mat skelTemp = allLines[idx].clone();
			Mat intersection = skelTemp & skelSeparate;
			skelTemp = skelTemp / 255 * (idx + 1);
			bitwise_and(skelSeparate, 0, skelSeparate, intersection);
			skelSeparate = skelSeparate + skelTemp;
		}
		Mat skelOrg;
		threshold(skelSeparate, skelOrg, 0.5, 255, THRESH_BINARY);

		//namedWindow("skelSeparate", CV_WINDOW_FREERATIO);
		//namedWindow("skelOrg", CV_WINDOW_FREERATIO);
		//imshow("skelSeparate", skelSeparate);
		//imshow("skelOrg", skelOrg);
		//cout << skelSeparate << endl;
		//waitKey(0);


		// 取skelOrg的各个连通域
		ConnectedRegion ccskelOrg(skelOrg, 8);
		ccskelOrg.calculatePixelIdxList();
		ccskelOrg.calculatePixelList();

		Mat thicknessMap;
		distanceTransform(mask, thicknessMap, DIST_L2, DIST_MASK_PRECISE);

		//cout << sum(thicknessMap)[0] << endl;
		bitwise_and(thicknessMap, 0, thicknessMap, ~skelOrg);


		double averageThickness = (double)sum(thicknessMap)[0] / countNonZero(skelOrg) * 2;
		vector<var> vars;

		for (int i = 0; i < ccskelOrg.connNum_; i++) {
			var tempvars;
			tempvars.skeleton = (ccskelOrg.label_ == i + 1);
			tempvars.sepskel.clear();
			tempvars.iscluster = 0;
			tempvars.obj = mask.clone();

			for (int j = 0; j < allLines.size(); j++) {

				if (countNonZero(tempvars.skeleton & (skelSeparate == j + 1)) > 2) {
					tempvars.sepskel.push_back(j + 1);
				}
			}
			if (tempvars.sepskel.size() > 1)
				tempvars.iscluster = 1;
			else
				tempvars.iscluster = 0;

			vars.push_back(tempvars);

		}

		Mat cutPointsMap = getCutPoints(mask, 0.3, 40, "or");
		if (cutPointsMap.empty())
			return errorCode = 1;
		Mat inner;
		bitwise_not(mask, inner);
		inner = clearBorder(inner);
		inner = bwareaopen(inner, 60, 8);
		bitwise_or(cutPointsMap, inner, cutPointsMap);

		if (ccskelOrg.connNum_ > 1) {
			for (int i = 0; i < vars.size(); i++) {
				//Mat skel1 = bwmorphThin(vars[i].skeleton);
				Mat skel1 = vars[i].skeleton.clone();
				thinning(skel1);

				for (int j = 0; j < vars.size(); j++) {
					if (j == i)
						continue;
					Mat skel2 = vars[j].skeleton.clone();
					thinning(skel2);

					//namedWindow("skel2", CV_WINDOW_FREERATIO);
					//imshow("skel2", skel2);
					//waitKey(0);

					int kernelLength = 2 * averageThickness + 1;
					Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(kernelLength, kernelLength));

					Mat skel1Dilate, skel2Dilate;
					Mat skel1Spur = bwmorphSpur(skel1, (int)averageThickness / 2);
					Mat skel2Spur = bwmorphSpur(skel2, (int)averageThickness / 2);
					morphologyEx(skel1Spur, skel1Dilate, MORPH_DILATE, kernel);
					morphologyEx(skel2Spur, skel2Dilate, MORPH_DILATE, kernel);
					Mat skelCommon;
					bitwise_and(skel1Dilate, skel2Dilate, skelCommon);

					//namedWindow("skel1Dilate", CV_WINDOW_FREERATIO);
					//imshow("skel1Dilate", skel1Dilate);
					//namedWindow("skel2Dilate", CV_WINDOW_FREERATIO);
					//imshow("skel2Dilate", skel2Dilate);
					//namedWindow("skelCommon", CV_WINDOW_FREERATIO);
					//imshow("skelCommon", skelCommon);
					//waitKey(0);

					Mat skelPtMap;
					bitwise_and(skelCommon, cutPointsMap, skelPtMap);

					//namedWindow("skelPtMap", CV_WINDOW_FREERATIO);
					//imshow("skelPtMap", skelPtMap);
					//waitKey(0);

					if (countNonZero(skelPtMap) > 0) {

						bool isSeparated = false;

						ConnectedRegion cutPtsRegionProps(skelPtMap, 8);
						cutPtsRegionProps.calculatePixelList();

						// 第一个 if 分支没有问题
						if (countNonZero(skel1 & skel2Dilate) == 0) {
							vector<cuttingListStru> cutPointPairs;
							cutPointPairs = extractCutPointPairs(cutPtsRegionProps);

							Mat obj1 = cut(cutPointPairs, origin, vars[i].obj, 11, 300, "NoPalePath");

							//namedWindow("obj1", CV_WINDOW_FREERATIO);
							//imshow("obj1", obj1);
							//waitKey(0);

							if (!obj1.empty()) {
								ConnectedRegion obj1Label(obj1, 4);
								Mat temp1 = obj1Label.label_.clone();
								Mat temp2 = obj1Label.label_.clone();

								//double maxValueSkel1 = 0, minValueSkel1 = 0, maxValueSkel2 = 0, minValueSkel2 = 0;
								//double * maxPtSkel1 = &maxValueSkel1;
								//double * minPtSkel1 = &minValueSkel1;
								//double * maxPtSkel2 = &maxValueSkel2;
								//double * minPtSkel2 = &minValueSkel2;
								//minMaxIdx(temp1, minPtSkel1, maxPtSkel1);
								//minMaxIdx(temp2, minPtSkel2, maxPtSkel2);

								//namedWindow("temp1", CV_WINDOW_FREERATIO);
								//imshow("temp1", temp1);
								//namedWindow("temp2", CV_WINDOW_FREERATIO);
								//imshow("temp2", temp2);
								//waitKey(0);

								for (int rowIdx = 0; rowIdx < skel1.rows; rowIdx++) {
									uchar * rowPtSkel1 = skel1.ptr<uchar>(rowIdx);
									uchar * rowPtSkel2 = skel2.ptr<uchar>(rowIdx);
									for (int colIdx = 0; colIdx < skel1.cols; colIdx++) {
										if (rowPtSkel1[colIdx] == 0)
											temp1.at<int>(rowIdx, colIdx) = 0;
										if (rowPtSkel2[colIdx] == 0)
											temp2.at<int>(rowIdx, colIdx) = 0;
									}
								}

								double maxValueSkel1 = 0, minValueSkel1 = 0, maxValueSkel2 = 0, minValueSkel2 = 0;
								double * maxPtSkel1 = &maxValueSkel1;
								double * minPtSkel1 = &minValueSkel1;
								double * maxPtSkel2 = &maxValueSkel2;
								double * minPtSkel2 = &minValueSkel2;
								minMaxIdx(temp1, minPtSkel1, maxPtSkel1);
								minMaxIdx(temp2, minPtSkel2, maxPtSkel2);
								if (obj1Label.connNum_ > 1 && (int)maxValueSkel1 != (int)maxValueSkel2) {
									Mat temp = obj1Label.label_ == (int)maxValueSkel1;
									vars[i].obj = temp.clone();
									isSeparated = true;

									//namedWindow("varsIObj", CV_WINDOW_FREERATIO);
									//imshow("varsIObj", vars[i].obj);
									//waitKey(0);
								}
								else {
									isSeparated = false;
								}

								//namedWindow("varsIObj", CV_WINDOW_FREERATIO);
								//imshow("varsIObj", vars[i].obj);
								//waitKey(0);
							}
						}

						// 第二个 if 分支没有测试
						else if ((countNonZero(skel1 & skel2Dilate) > 0) |
							(countNonZero(skel2 & skel1Dilate) > 0)) {
							vector<Point> ep, bp;
							anaskel(skel1, ep, bp);
							for (int checkEp = 0; checkEp < ep.size(); checkEp++) {
								if (skel2Dilate.at<uchar>(ep[checkEp].x, ep[checkEp].y)) {
									vector<Point> refEp;
									refEp.push_back(ep[checkEp]);
									vector<cuttingListStru> cutPointPairs;
									cutPointPairs = findClosestToReference(cutPtsRegionProps, refEp);
									if (!cutPointPairs.empty()) {
										Mat lines = Mat::zeros(skelOrg.size(), CV_8UC1);
										Point checkEpChanges = Point(ep[checkEp].y, ep[checkEp].x);
										Point linePoint1 = Point(cutPointPairs[0].point1.y, cutPointPairs[0].point1.x);
										Point linePoint2 = Point(cutPointPairs[0].point2.y, cutPointPairs[0].point2.x);
										line(lines, checkEpChanges, linePoint1, Scalar::all(255), 1, 4);
										line(lines, checkEpChanges, linePoint2, Scalar::all(255), 1, 4);
										Mat obj1 = vars[i].obj & (~lines);

										// 找trace
										vector<vector<Point>> skel1Contours;
										vector<Vec4i> hierarchy;
										findContours(skel1, skel1Contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
										vector<Point> skel1ContourToBeSorted = skel1Contours[0];
										// 排序
										// 先把从第一个开始到 length 个元素拿出来
										int length = countNonZero(skel1);
										vector<Point> temp(length, Point(0, 0));
										copy(skel1ContourToBeSorted.begin(), skel1ContourToBeSorted.begin() + length, temp.begin());
										// 然后在 temp 里面排序
										vector<Point> skel1Contour(length, Point(0, 0));
										for (int idx = 0; idx < length; idx++) {
											if (temp[idx] == checkEpChanges) {
												copy(temp.begin() + idx, temp.end(), skel1Contour.begin());
												int leftLength = length - idx;
												copy(temp.begin(), temp.begin() + idx, skel1Contour.begin() + leftLength);
												break;
											}
										}

										Mat extension = Mat::zeros(skelOrg.size(), CV_8UC1);
										cuttingListStru tempPointPair = cutPointPairs[0];
										double cutPointPairDists = sqrt(
											(tempPointPair.point1.x - tempPointPair.point2.x)
											* (tempPointPair.point1.x - tempPointPair.point2.x) +
											(tempPointPair.point1.y - tempPointPair.point2.y)
											* (tempPointPair.point1.y - tempPointPair.point2.y));
										Point specialElementInSkel1Contour = skel1Contour[(int)cutPointPairDists / 2];
										extension.at<uchar>(specialElementInSkel1Contour.y, specialElementInSkel1Contour.x) = 255;

										int kernelLength = cutPointPairDists - 1;
										Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(kernelLength, kernelLength));
										morphologyEx(extension, extension, MORPH_DILATE, kernel);

										if (!obj1.empty()) {
											ConnectedRegion obj1Label(obj1, 4);
											Mat temp1 = obj1Label.label_.clone();
											Mat temp2 = obj1Label.label_.clone();
											for (int rowIdx = 0; rowIdx < skel1.rows; rowIdx++) {
												uchar * rowPtSkel1 = skel1.ptr<uchar>(rowIdx);
												uchar * rowPtSkel2 = skel2.ptr<uchar>(rowIdx);
												for (int colIdx = 0; colIdx < skel1.cols; colIdx++) {
													if (rowPtSkel1[colIdx] == 0)
														temp1.at<int>(rowIdx, colIdx) = 0;
													if (rowPtSkel2[colIdx] == 0)
														temp2.at<int>(rowIdx, colIdx) = 0;
												}
											}
											double maxValueSkel1 = 0, minValueSkel1 = 0, maxValueSkel2 = 0, minValueSkel2 = 0;
											double * maxPtSkel1 = &maxValueSkel1;
											double * minPtSkel1 = &minValueSkel1;
											double * maxPtSkel2 = &maxValueSkel2;
											double * minPtSkel2 = &minValueSkel2;
											minMaxIdx(temp1, minPtSkel1, maxPtSkel1);
											minMaxIdx(temp2, minPtSkel2, maxPtSkel2);
											if (obj1Label.connNum_ > 1 && (int)maxValueSkel1 != (int)maxValueSkel2) {
												Mat temp = obj1Label.label_ == (int)maxValueSkel1;
												temp = temp | extension;
												vars[i].obj = temp.clone();
												isSeparated = true;
											}
											else {
												isSeparated = false;
											}
										}
									}

								}
							}
							ep.clear(); bp.clear();
							anaskel(skel2, ep, bp);
							for (int checkEp = 0; checkEp < ep.size(); checkEp++) {
								if (skel1Dilate.at<uchar>(ep[checkEp].x, ep[checkEp].y)) {
									vector<Point> refBp;
									refBp.push_back(ep[checkEp]);
									vector<cuttingListStru> cutPointPairs;
									cutPointPairs = findClosestToReference(cutPtsRegionProps, refBp);

									Mat lines = Mat::zeros(skelOrg.size(), CV_8UC1);
									Point linePoint1 = Point(cutPointPairs[0].point1.y, cutPointPairs[0].point1.x);
									Point linePoint2 = Point(cutPointPairs[0].point2.y, cutPointPairs[0].point2.x);
									line(lines, linePoint1, linePoint2, Scalar::all(255), 1, 4);

									Mat obj1 = vars[i].obj & (~lines);
									if (!obj1.empty()) {
										ConnectedRegion obj1Label(obj1, 4);
										Mat temp1 = obj1Label.label_.clone();
										Mat temp2 = obj1Label.label_.clone();

										for (int rowIdx = 0; rowIdx < skel1.rows; rowIdx++) {
											uchar * rowPtSkel1 = skel1.ptr<uchar>(rowIdx);
											uchar * rowPtSkel2 = skel2.ptr<uchar>(rowIdx);
											for (int colIdx = 0; colIdx < skel1.cols; colIdx++) {
												if (rowPtSkel1[colIdx] == 0)
													temp1.at<int>(rowIdx, colIdx) = 0;
												if (rowPtSkel2[colIdx] == 0)
													temp2.at<int>(rowIdx, colIdx) = 0;
											}
										}
										double maxValueSkel1 = 0, minValueSkel1 = 0, maxValueSkel2 = 0, minValueSkel2 = 0;
										double * maxPtSkel1 = &maxValueSkel1;
										double * minPtSkel1 = &minValueSkel1;
										double * maxPtSkel2 = &maxValueSkel2;
										double * minPtSkel2 = &minValueSkel2;
										minMaxIdx(temp1, minPtSkel1, maxPtSkel1);
										minMaxIdx(temp2, minPtSkel2, maxPtSkel2);
										if (obj1Label.connNum_ > 1 && (int)maxValueSkel1 != (int)maxValueSkel2) {
											Mat temp = obj1Label.label_ == (int)maxValueSkel1;
											vars[i].obj = temp.clone();
											isSeparated = true;
										}
										else {
											isSeparated = false;
										}
									}
								}
							}
						}



						// 新增的
						if (isSeparated == 0) {
							vector<cuttingListStru> cutPointPairs;
							cutPointPairs = extractCutPointPairs(cutPtsRegionProps);
							for (int cutPtNum = 0; cutPtNum < cutPointPairs.size(); cutPtNum++) {
								Mat obj1;
								vector<cuttingListStru> toBeCut;
								toBeCut.push_back(cutPointPairs[cutPtNum]);
								obj1 = cut(toBeCut, origin, vars[i].obj, 11, 300, "NoPalePath");
								if (!obj1.empty()) {
									ConnectedRegion obj1Label(obj1, 4);
									Mat temp1 = obj1Label.label_.clone();
									Mat temp2 = obj1Label.label_.clone();
									for (int rowIdx = 0; rowIdx < skel1.rows; rowIdx++) {
										uchar * rowPtSkel1 = skel1.ptr<uchar>(rowIdx);
										uchar * rowPtSkel2 = skel2.ptr<uchar>(rowIdx);
										for (int colIdx = 0; colIdx < skel1.cols; colIdx++) {
											if (rowPtSkel1[colIdx] == 0)
												temp1.at<int>(rowIdx, colIdx) = 0;
											if (rowPtSkel2[colIdx] == 0)
												temp2.at<int>(rowIdx, colIdx) = 0;
										}
									}

									double maxValueSkel1 = 0, minValueSkel1 = 0, maxValueSkel2 = 0, minValueSkel2 = 0;
									double * maxPtSkel1 = &maxValueSkel1;
									double * minPtSkel1 = &minValueSkel1;
									double * maxPtSkel2 = &maxValueSkel2;
									double * minPtSkel2 = &minValueSkel2;
									minMaxIdx(temp1, minPtSkel1, maxPtSkel1);
									minMaxIdx(temp2, minPtSkel2, maxPtSkel2);
									if (obj1Label.connNum_ > 1 && (int)maxValueSkel1 != (int)maxValueSkel2) {
										Mat temp = obj1Label.label_ == (int)maxValueSkel1;
										vars[i].obj = temp.clone();
										break;
									}
									else {
										isSeparated = false;
									}
								}
							}
						}
					}
				}
			}
		}

		cutPointsMap = getCutPoints(mask, 0.15, 30, "or");
		bitwise_not(mask, inner);
		inner = clearBorder(inner);
		inner = bwareaopen(inner, 60, 8);
		bitwise_or(cutPointsMap, inner, cutPointsMap);

		//imshow("cutPointsMap0", cutPointsMap);
		//waitKey(0);


		Mat obj1;
		for (int idx = 0; idx < vars.size(); idx++) {
			if (vars[idx].iscluster) {
				for (int jdx = 0; jdx < vars[idx].sepskel.size(); jdx++) {
					Mat extension;
					Mat tempSkel1 = skelSeparate == vars[idx].sepskel[jdx];

					//namedWindow("tempSkel1", CV_WINDOW_FREERATIO);
					//imshow("tempSkel1", tempSkel1);
					//waitKey(0);

					Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
					Mat skel1;
					morphologyEx(tempSkel1, skel1, MORPH_DILATE, kernel);

					//namedWindow("skel1Dilate", CV_WINDOW_FREERATIO);
					//imshow("skel1Dilate", skel1);
					//waitKey(0);

					thinning(skel1);
					vector<int> otherSkels(vars[idx].sepskel);
					otherSkels.erase(otherSkels.begin() + jdx);
					var newVar;
					Mat tempSkeleton = skelSeparate == vars[idx].sepskel[jdx] + 1;
					newVar.skeleton = tempSkeleton.clone();
					vector<int> tempSepSkel(1, vars[idx].sepskel[jdx] + 1);
					newVar.sepskel = tempSepSkel;
					newVar.obj = vars[idx].obj.clone();
					newVar.iscluster = 0;
					vars.push_back(newVar);

					Mat cutobj = newVar.obj.clone();

					//namedWindow("skel1", CV_WINDOW_FREERATIO);
					//imshow("skel1", skel1);
					//namedWindow("cutObj", CV_WINDOW_FREERATIO);
					//imshow("cutObj", cutobj);
					//waitKey(0);



					//for (int k = 0; k < otherSkels.size(); k++) {
					for (int k = otherSkels.size() - 1; k >= 0; k--) {
						Mat tempSkel2 = skelSeparate == otherSkels[k];
						Mat tempSkel2Dilate;
						morphologyEx(tempSkel2, tempSkel2Dilate, MORPH_DILATE, kernel);

						Mat skel2 = tempSkel2Dilate.clone();
						thinning(skel2);
						vector<Point> refEp, ref_bp;
						anaskel(skel1 | skel2, refEp, ref_bp);

						//namedWindow("skel2", CV_WINDOW_FREERATIO);
						//imshow("skel2", skel2);
						//waitKey(0);

						vector<Point> refBp;
						if (ref_bp.size() > 1) {
							for (int i = 1; i < ref_bp.size(); i++) {
								ref_bp[0].x = ref_bp[0].x + ref_bp[i].x;
								ref_bp[0].y = ref_bp[0].y + ref_bp[i].y;
							}
							ref_bp[0].x = ref_bp[0].x / ref_bp.size();
							ref_bp[0].y = ref_bp[0].y / ref_bp.size();

							refBp.push_back(ref_bp[0]);
						}

						if (!refBp.empty()) {
							Mat CutPoints_map;
							vector<cuttingListStru> cutPointPairs;
							vector<Mat> commonArea;

							//namedWindow("varsIdxObj", CV_WINDOW_FREERATIO);
							//imshow("varsIdxObj", vars[idx].obj);
							//namedWindow("skel1OrSkel2", CV_WINDOW_FREERATIO);
							//imshow("skel1OrSkel2", skel1 | skel2);
							//namedWindow("allCutPoints", CV_WINDOW_FREERATIO);
							//imshow("allCutPoints", cutPointsMap);
							//waitKey(0);
							findPointMuiltipleCluster(vars[idx].obj, cutPointsMap, (skel1 | skel2), refBp, refEp, CutPoints_map, cutPointPairs, commonArea);
							ConnectedRegion cutPointListX(CutPoints_map, 8);
							cutPointListX.calculatePixelList();
							vector<cuttingListStru> cuttingList1;

							//namedWindow("cutPointsMap1", CV_WINDOW_FREERATIO);
							//imshow("cutPointsMap1", CutPoints_map);
							//waitKey(0);

							if (cutPointPairs.size() == 1) {
								for (int pp = 0; pp <= 1; pp++) {
									cuttingListStru dist = pDist2(cutPointListX.pixelList_[pp], refBp);
									cutPointListX.pixelList_[pp].clear();
									cutPointListX.pixelList_[pp].push_back(dist.point1);
								}
								cuttingListStru tmpCuttingList;
								tmpCuttingList.point1 = cutPointListX.pixelList_[0][0];
								tmpCuttingList.point2 = cutPointListX.pixelList_[1][0];
								cuttingList1.push_back(tmpCuttingList);
								Mat tmpCutLines = Mat::zeros(vars[vars.size() - 1].obj.size(), CV_8UC1);
								line(tmpCutLines, cuttingList1[0].point1, cuttingList1[0].point2, Scalar::all(255), 1, 4);
								if (!countNonZero(tmpCutLines & skel1)) {
									bitwise_and(cutobj, 0, cutobj, tmpCutLines);

									ConnectedRegion cutObjStat(cutobj, 8);
									for (int cutStatNum = 0; cutStatNum < cutObjStat.connNum_; cutStatNum++) {
										Mat tmp = (cutObjStat.label_ == (cutStatNum + 1));
										if (countNonZero(tmp & skel1) == 0) {
											bitwise_and(cutobj, 0, cutobj, tmp);
										}
									}
								}
								else {
									vector<Point> skel1Ep, skel1Bp;
									anaskel(skel1, skel1Ep, skel1Bp);
									vector<Point> cuttingList1Points1;
									cuttingList1Points1.push_back(cuttingList1[0].point1);
									cuttingListStru epCutDist = pDist2(skel1Ep, cuttingList1Points1);
									Point chosenEp = epCutDist.point1;
									Mat tmpCutLines = Mat::zeros(vars[vars.size() - 1].obj.size(), CV_8UC1);
									line(tmpCutLines, cuttingList1[0].point1, chosenEp, Scalar::all(255), 1, 4);
									line(tmpCutLines, cuttingList1[0].point2, chosenEp, Scalar::all(255), 1, 4);
									bitwise_and(cutobj, 0, cutobj, tmpCutLines);
									ConnectedRegion cutObjStat(cutobj, 8);
									for (int cutStatNum = 0; cutStatNum < cutObjStat.connNum_; cutStatNum++) {
										Mat tmp = cutObjStat.label_ == (cutStatNum + 1);
										if (countNonZero(tmp & skel1) == 0) {
											bitwise_and(cutobj, 0, cutobj, tmp);
										}
									}
									// 找trace
									vector<vector<Point>> skel1Contours;
									vector<Vec4i> hierarchy;
									findContours(skel1, skel1Contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
									vector<Point> skel1ContourToBeSorted = skel1Contours[0];
									// 排序
									// 先把从第一个开始到 length 个元素拿出来
									int length = countNonZero(skel1);
									vector<Point> temp(length, Point(0, 0));
									copy(skel1ContourToBeSorted.begin(), skel1ContourToBeSorted.begin() + length, temp.begin());
									// 然后在 temp 里面排序
									vector<Point> skel1Contour(length, Point(0, 0));
									Point checkEpChanges = Point(chosenEp.y, chosenEp.x);
									for (int idx = 0; idx < length; idx++) {
										if (temp[idx] == checkEpChanges) {
											copy(temp.begin() + idx, temp.end(), skel1Contour.begin());
											int leftLength = length - idx;
											copy(temp.begin(), temp.begin() + idx, skel1Contour.begin() + leftLength);
											break;
										}
									}
									extension = Mat::zeros(skelOrg.size(), CV_8UC1);
									cuttingListStru tempPointPair = cutPointPairs[0];
									double cutPointPairDists = sqrt(
										(tempPointPair.point1.x - tempPointPair.point2.x)
										* (tempPointPair.point1.x - tempPointPair.point2.x) +
										(tempPointPair.point1.y - tempPointPair.point2.y)
										* (tempPointPair.point1.y - tempPointPair.point2.y));
									Point specialElementInSkel1Contour = skel1Contour[(int)cutPointPairDists / 2];
									extension.at<uchar>(specialElementInSkel1Contour.y, specialElementInSkel1Contour.x) = 255;

									int kernelLength1 = cutPointPairDists - 1;
									Mat kernel1 = getStructuringElement(MORPH_ELLIPSE, Size(kernelLength1, kernelLength1));
									morphologyEx(extension, extension, MORPH_DILATE, kernel1);
									bitwise_or(cutobj, extension, cutobj);
								}
							}
							else if (cutPointPairs.size() == 3) {
								for (int pp = 1; pp <= 3; pp++) {
									cuttingListStru closest
										= pDist2(cutPointListX.pixelList_[pp - 1], refBp);
									cutPointListX.pixelList_[pp - 1].clear();
									cutPointListX.pixelList_[pp - 1].push_back(closest.point1);
								}

								Mat circle = Mat::zeros(skelOrg.size(), CV_8UC1);
								circle = circle + 255;
								for (int pp = 1; pp <= 3; pp++) {
									Mat circle1 = Mat::zeros(skelOrg.size(), CV_8UC1);
									Point circlePoint(cutPointListX.pixelList_[pp - 1][0].y,
										cutPointListX.pixelList_[pp - 1][0].x);
									circle1.at<uchar>(circlePoint.y, circlePoint.x) = 255;

									int kernelLength = 2 * averageThickness * 3 + 1;
									Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(kernelLength, kernelLength));
									morphologyEx(circle1, circle1, MORPH_DILATE, kernel);
									bitwise_and(circle, circle1, circle);
								}

								vector<Point> skel1Ep, skel1Bp;
								anaskel(skel1, skel1Ep, skel1Bp);

								bool isCut = false;
								for (int checkEp = 0; checkEp < skel1Ep.size(); checkEp++) {
									cuttingList1.clear();
									if (circle.at<uchar>(skel1Ep[checkEp].x, skel1Ep[checkEp].y)) {
										cuttingListStru cuttingListElementFirst;
										cuttingListStru cuttingListElementSecond;
										cuttingListStru cuttingListElementThird;
										cuttingListElementFirst.point1 = cutPointListX.pixelList_[0][0];
										cuttingListElementFirst.point2 = skel1Ep[checkEp];
										cuttingList1.push_back(cuttingListElementFirst);
										cuttingListElementSecond.point1 = cutPointListX.pixelList_[1][0];
										cuttingListElementSecond.point2 = skel1Ep[checkEp];
										cuttingList1.push_back(cuttingListElementSecond);
										cuttingListElementThird.point1 = cutPointListX.pixelList_[2][0];
										cuttingListElementThird.point2 = skel1Ep[checkEp];
										cuttingList1.push_back(cuttingListElementThird);
										for (int lineNum = 0; lineNum < 3; lineNum++) {
											Mat tmpCutLines = Mat::zeros(vars[vars.size() - 1].obj.size(), CV_8UC1);
											Point linePoint1(cuttingList1[lineNum].point1.y,
												cuttingList1[lineNum].point1.x);
											Point linePoint2(cuttingList1[lineNum].point2.y,
												cuttingList1[lineNum].point2.x);
											line(tmpCutLines, linePoint1, linePoint2, Scalar::all(255), 1, 4);
											bitwise_and(cutobj, 0, cutobj, tmpCutLines);
										}
										isCut = true;
										ConnectedRegion cutObjStat(cutobj, 8);
										for (int cutStatNum = 0; cutStatNum < cutObjStat.connNum_; cutStatNum++) {
											Mat tmp = cutObjStat.label_ == (cutStatNum + 1);
											if (countNonZero(tmp & skel1) == 0) {
												bitwise_and(cutobj, 0, cutobj, tmp);
											}
										}

										// 找trace, skelContour
										vector<vector<Point>> skel1Contours;
										vector<Vec4i> hierarchy;
										findContours(skel1, skel1Contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
										vector<Point> skel1ContourToBeSorted = skel1Contours[0];
										// 排序
										// 先把从第一个开始到 length 个元素拿出来
										int length = countNonZero(skel1);
										vector<Point> temp(length, Point(0, 0));
										copy(skel1ContourToBeSorted.begin(), skel1ContourToBeSorted.begin() + length, temp.begin());
										// 然后在 temp 里面排序
										vector<Point> skel1Contour(length, Point(0, 0));
										Point checkEpChanges = Point(skel1Ep[checkEp].y, skel1Ep[checkEp].x);
										for (int idx = 0; idx < length; idx++) {
											if (temp[idx] == checkEpChanges) {
												copy(temp.begin() + idx, temp.end(), skel1Contour.begin());
												int leftLength = length - idx;
												copy(temp.begin(), temp.begin() + idx, skel1Contour.begin() + leftLength);
												break;
											}
										}

										Mat extension = Mat::zeros(skelOrg.size(), CV_8UC1);
										Point specialElementInSkel1Contour = skel1Contour[(int)averageThickness / 2];
										extension.at<uchar>(specialElementInSkel1Contour.y, specialElementInSkel1Contour.x) = 255;

										int kernelLength1 = (int)averageThickness + 1;
										Mat kernel1 = getStructuringElement(MORPH_ELLIPSE, Size(kernelLength1, kernelLength1));
										morphologyEx(extension, extension, MORPH_DILATE, kernel1);

										bitwise_or(cutobj, extension, cutobj);
										break;

									}
									else
										isCut = false;
								}

								if (isCut == false) {
									cuttingListStru cuttingListElementFirst;
									cuttingListStru cuttingListElementSecond;
									cuttingListStru cuttingListElementThird;
									cuttingListElementFirst.point1 = cutPointListX.pixelList_[0][0];
									cuttingListElementFirst.point2 = cutPointListX.pixelList_[1][0];
									cuttingList1.push_back(cuttingListElementFirst);
									cuttingListElementSecond.point1 = cutPointListX.pixelList_[1][0];
									cuttingListElementSecond.point2 = cutPointListX.pixelList_[2][0];
									cuttingList1.push_back(cuttingListElementSecond);
									cuttingListElementThird.point1 = cutPointListX.pixelList_[0][0];
									cuttingListElementThird.point2 = cutPointListX.pixelList_[2][0];
									cuttingList1.push_back(cuttingListElementThird);

									for (int lineNum = 0; lineNum < 3; lineNum++) {
										Mat tmpCutLines = Mat::zeros(vars[vars.size() - 1].obj.size(), CV_8UC1);
										Point linePoint1(cuttingList1[lineNum].point1.y,
											cuttingList1[lineNum].point1.x);
										Point linePoint2(cuttingList1[lineNum].point2.y,
											cuttingList1[lineNum].point2.x);
										line(tmpCutLines, linePoint1, linePoint2, Scalar::all(255), 1, 4);

										if (countNonZero(tmpCutLines & skel1) == 0) {
											bitwise_and(cutobj, 0, cutobj, tmpCutLines);
										}
									}

									ConnectedRegion cutObjStat(cutobj, 8);
									for (int cutStatNum = 0; cutStatNum < cutObjStat.connNum_; cutStatNum++) {
										Mat tmp = cutObjStat.label_ == (cutStatNum + 1);
										if (countNonZero(tmp & skel1) == 0) {
											bitwise_and(cutobj, 0, cutobj, tmp);
										}
									}
								}
							}
							else if (cutPointPairs.size() == 6) {

								for (int pp = 1; pp <= 4; pp++) {
									cuttingListStru dist = pDist2(cutPointListX.pixelList_[pp - 1], refBp);
									cutPointListX.pixelList_[pp - 1].clear();
									cutPointListX.pixelList_[pp - 1].push_back(dist.point1);
								}

								vector<Point> pointVec(4, Point(0, 0));
								for (int pp = 0; pp < cutPointListX.connNum_; pp++) {
									pointVec[pp] = cutPointListX.pixelList_[pp][0];
								}

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

								cuttingListStru cuttingListElement;
								cuttingListElement.point1 = cutPointListX.pixelList_[a[0]][0];
								cuttingListElement.point2 = cutPointListX.pixelList_[a[1]][0];
								cuttingList1.push_back(cuttingListElement);
								cuttingListElement.point1 = cutPointListX.pixelList_[a[2]][0];
								cuttingListElement.point2 = cutPointListX.pixelList_[a[3]][0];
								cuttingList1.push_back(cuttingListElement);
								cuttingListElement.point1 = cutPointListX.pixelList_[a[0]][0];
								cuttingListElement.point2 = cutPointListX.pixelList_[a[2]][0];
								cuttingList1.push_back(cuttingListElement);
								cuttingListElement.point1 = cutPointListX.pixelList_[a[1]][0];
								cuttingListElement.point2 = cutPointListX.pixelList_[a[3]][0];
								cuttingList1.push_back(cuttingListElement);

								Mat square = Mat::zeros(skelOrg.size(), CV_8UC1);
								for (int pp = 1; pp <= 4; pp++) {
									Point linePoint1(cuttingList1[pp - 1].point1.y,
										cuttingList1[pp - 1].point1.x);
									Point linePoint2(cuttingList1[pp - 1].point2.y,
										cuttingList1[pp - 1].point2.x);
									line(square, linePoint1, linePoint2, Scalar::all(255), 1, 4);
								}
								Mat squareFilled = imFill(square);

								vector<Point> skel1Ep, skel2Bp;
								anaskel(skel1, skel1Ep, skel2Bp);

								bool isCut = false;
								vector<cuttingListStru> cuttingEp;
								for (int checkEp = 0; checkEp < skel1Ep.size(); checkEp++) {
									cuttingEp.clear();
									if (square.at<uchar>(skel1Ep[checkEp].x, skel1Ep[checkEp].y)) {
										cuttingListStru cuttingEpElement;
										cuttingEpElement.point1 = cutPointListX.pixelList_[0][0];
										cuttingEpElement.point2 = skel1Ep[checkEp];
										cuttingEp.push_back(cuttingEpElement);
										cuttingEpElement.point1 = cutPointListX.pixelList_[1][0];
										cuttingEp.push_back(cuttingEpElement);
										cuttingEpElement.point1 = cutPointListX.pixelList_[2][0];
										cuttingEp.push_back(cuttingEpElement);
										cuttingEpElement.point1 = cutPointListX.pixelList_[3][0];
										cuttingEp.push_back(cuttingEpElement);

										for (int lineNum = 0; lineNum < 4; lineNum++) {
											Mat tmpCutLines = Mat::zeros(vars[vars.size() - 1].obj.size(), CV_8UC1);
											Point linePoint1(cuttingEp[lineNum].point1.y,
												cuttingEp[lineNum].point1.x);
											Point linePoint2(cuttingEp[lineNum].point2.y,
												cuttingEp[lineNum].point2.x);
											line(tmpCutLines, linePoint1, linePoint2, Scalar::all(255), 1, 4);
											bitwise_and(cutobj, 0, cutobj, tmpCutLines);
										}

										isCut = true;
										ConnectedRegion cutObjStat(cutobj, 8);
										for (int cutStatNum = 0; cutStatNum < cutObjStat.connNum_; cutStatNum++) {
											Mat tmp = cutObjStat.label_ == (cutStatNum + 1);
											if (countNonZero(tmp & skel1) == 0) {
												bitwise_and(cutobj, 0, cutobj, tmp);
											}
										}

										// 找trace, skelContour
										vector<vector<Point>> skel1Contours;
										vector<Vec4i> hierarchy;
										findContours(skel1, skel1Contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
										vector<Point> skel1ContourToBeSorted = skel1Contours[0];
										// 排序
										// 先把从第一个开始到 length 个元素拿出来
										int length = countNonZero(skel1);
										vector<Point> temp(length, Point(0, 0));
										copy(skel1ContourToBeSorted.begin(), skel1ContourToBeSorted.begin() + length, temp.begin());
										// 然后在 temp 里面排序
										vector<Point> skel1Contour(length, Point(0, 0));
										Point checkEpChanges = Point(skel1Ep[checkEp].y, skel1Ep[checkEp].x);
										for (int idx = 0; idx < length; idx++) {
											if (temp[idx] == checkEpChanges) {
												copy(temp.begin() + idx, temp.end(), skel1Contour.begin());
												int leftLength = length - idx;
												copy(temp.begin(), temp.begin() + idx, skel1Contour.begin() + leftLength);
												break;
											}
										}

										Mat extension = Mat::zeros(skelOrg.size(), CV_8UC1);
										Point specialElementInSkel1Contour = skel1Contour[(int)averageThickness / 2];
										extension.at<uchar>(specialElementInSkel1Contour.y, specialElementInSkel1Contour.x) = 255;

										int kernelLength1 = (int)averageThickness + 1;
										Mat kernel1 = getStructuringElement(MORPH_ELLIPSE, Size(kernelLength1, kernelLength1));
										morphologyEx(extension, extension, MORPH_DILATE, kernel1);

										bitwise_or(cutobj, extension, cutobj);
									}
									else
										isCut = false;
								}

								if (isCut == false) {
									for (int lineNum = 0; lineNum < 4; lineNum++) {
										Mat tmpCutLines = Mat::zeros(vars[vars.size() - 1].obj.size(), CV_8UC1);
										Point linePoint1(cuttingList1[lineNum].point1.y,
											cuttingList1[lineNum].point1.x);
										Point linePoint2(cuttingList1[lineNum].point2.y,
											cuttingList1[lineNum].point2.x);
										line(tmpCutLines, linePoint1, linePoint2, Scalar::all(255), 1, 4);

										//namedWindow("cutLines", CV_WINDOW_FREERATIO);
										//imshow("cutLines", tmpCutLines);
										//waitKey(0);
										//namedWindow("cutLines&skel1", CV_WINDOW_FREERATIO);
										//imshow("cutLines&skel1", tmpCutLines & skel1);
										//waitKey(0);

										if (countNonZero(tmpCutLines & skel1) == 0) {
											bitwise_and(cutobj, 0, cutobj, tmpCutLines);
											//namedWindow("cutObjInLoop", CV_WINDOW_FREERATIO);
											//imshow("cutObjInLoop", cutobj);
											//waitKey(0);
										}
									}

									//namedWindow("cutFalseCutObj", CV_WINDOW_FREERATIO);
									//imshow("cutFalseCutObj", cutobj);
									//waitKey(0);

									ConnectedRegion cutObjStat(cutobj, 8);
									for (int cutStatNum = 0; cutStatNum < cutObjStat.connNum_; cutStatNum++) {
										Mat tmp = cutObjStat.label_ == (cutStatNum + 1);
										if (countNonZero(tmp & skel1) == 0) {
											bitwise_and(cutobj, 0, cutobj, tmp);
										}
									}
								}
							}
						}
					}
					//namedWindow("cutObjFinal", CV_WINDOW_FREERATIO);
					//imshow("cutObjFinal", cutobj);
					//waitKey(0);
					vars[vars.size() - 1].obj = cutobj.clone();
				}
			}
		}
		int single_index = newCutNum;
		for (int idx = 0; idx < vars.size(); idx++) {
			if (!vars[idx].iscluster) {

				ConnectedRegion ccobj(vars[idx].obj, 8);
				ccobj.calculateBoundingBox();
				ccobj.calculateOrientation();
				ccobj.calculateArea();
				ccobj.calculateImage();
				int maxIdx = 0;
				if (ccobj.connNum_ > 1) {
					int maxArea = ccobj.area_[0];
					for (int i = 0; i < ccobj.connNum_; i++) {
						if (ccobj.area_[i] > maxArea) {
							maxArea = ccobj.area_[i];
							maxIdx = i;
						}
					}
				}
				vector<int> bb = ccobj.boundingBox_[maxIdx];
				double angle = ccobj.orientation_[maxIdx];
				Mat single_obj_mask = ccobj.image_[maxIdx].clone();
				Mat originalObj_I = origin(Rect(bb[1], bb[0], bb[3], bb[2]));
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
				chromoElement.index = single_index;
				single_index++;
				chromoElement.relatedIndex = chromoData.index;
				chromoElement.chromoId = 0;
				chromoElement.cImgType = 1;
				chromoElement.cImg = originalObjI.clone();
				chromoElement.cImgRotated = rotatedObj_I.clone();
				position posElement;
				posElement.cImgMask = single_obj_mask.clone();
				posElement.cImgBoundingBox[0] = bb[0] + chromoData.cImgPosition.cImgBoundingBox[0];
				posElement.cImgBoundingBox[1] = bb[1] + chromoData.cImgPosition.cImgBoundingBox[1];
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

				newChromoDataList.push_back(chromoElement);


			}
		}
	}
	else {
		errorCode = 1;
	}

	return errorCode;
}
