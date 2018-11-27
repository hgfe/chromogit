#include "stdafx.h"
#include "main.h"
#include "conn.h"

// 预处理
Mat roughSegChromoRegion(Mat I) {
	int resizeH = I.rows;
	int resizeW = I.cols;

	//adaptthresh
	Mat BW;
	int blockSize1 = 2 * floor(resizeH / 16) + 1;
	int blockSize2 = 2 * floor(resizeW / 16) + 1;
	int blockSize = (blockSize1 < blockSize2) ? blockSize2 : blockSize1;
	adaptiveThreshold(I, BW, 255,
		CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV,
		blockSize, 1);

	BW = clearBorder(BW);
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	morphologyEx(BW, BW, MORPH_OPEN, kernel);

	Mat BW2; bitwise_not(BW, BW2);
	BW2 = clearBorder(BW2);
	Mat innerCutPoints = bwareaopen(BW2, 25, 4);

	BW = imFill(BW);
	bitwise_and(BW, 0, BW, innerCutPoints);
	Mat BWmainbody = bwareaopen(BW, 50, 4);
	Mat BWsuiti = BW - BWmainbody;

	if (countNonZero(BWsuiti) > 0) {
		int radius = 10;
		ConnectedRegion cc_suiti(BWsuiti, 8);
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

				Mat BW5 = Mat::zeros(BWmainbody.size(), CV_8UC1);
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

	return BWmainbody;
}

// 预处理
Mat imgUniform(const Mat imgGray, int& resizeH, int& resizeW) {

	int origH = imgGray.rows;
	int origW = imgGray.cols;

	/////////////////////////////////////////////////////////////
	///////////////////// resize 图像 ///////////////////////////
	Size dsize = Size(targetW, targetH);
	Mat scalingImgGray;
	resize(imgGray, scalingImgGray, dsize, 0.0, 0.0, INTER_CUBIC);
	Mat img2 = scalingImgGray.clone();

	/////////////////////////////////////////////////////////////
	//////////////////// adjust the bg to 255 ///////////////////
	resizeH = scalingImgGray.rows;
	resizeW = scalingImgGray.cols;

	Mat BW = scalingImgGray < 255;
	double darkForeRatio = (double)countNonZero(BW) / (resizeH * resizeW);
	if (darkForeRatio > 0.2) {
		vector<int> histo = imhist(scalingImgGray, 256);
		int maxValue = *(max_element(histo.begin(), histo.end()));
		if (maxValue == histo[255])
			histo[255] = 0;
		maxValue = *(max_element(histo.begin(), histo.end()));
		vector<int>::iterator maxIter = max_element(histo.begin(), histo.end());
		int maxIndex = distance(histo.begin(), maxIter);				// maxIndex 即为 peak

		vector<int> tmp(histo.begin() + maxIndex + 1, histo.end());

		int neighborhood = -1;
		for (int tmpIdx = 0; tmpIdx < tmp.size(); tmpIdx++) {
			if (tmp[tmpIdx] < (double)maxValue / 1000) {
				neighborhood = tmpIdx;
				break;
			}
		}
		if (neighborhood == -1)
			neighborhood = 255 - maxIndex;

		int thresh = maxIndex - 2 * neighborhood;

		BW = scalingImgGray < thresh;
		BW = bwareaopen(BW, 200, 4);

		Mat BW2;
		bitwise_not(BW, BW2);
		BW2 = clearBorder(BW2);
		BW2 = bwareaopen(BW2, 100, 8);
		BW = imFill(BW);

		bitwise_and(BW, 0, BW, BW2);
	}

	Mat BWNot; bitwise_not(BW, BWNot);
	bitwise_or(img2, 255, img2, BWNot);

	/////////////////////////////////////////////////////////////////////////
	//////////////// remove small impurity & reserve the satellite //////////
	Mat BW3; bitwise_not(BW, BW3);
	BW3 = roughSegChromoRegion(BW3);
	Mat darkForeGround = BW3.clone();

	////////////////////////////////////////////////////////////////////////
	////////////// remove the necleus with a round shape ///////////////////
	int limitedSize = 0.005 * resizeH * resizeW;
	Mat darkForeGroundFilled = imFill(darkForeGround);

	ConnectedRegion s(darkForeGroundFilled, 8);
	s.calculateArea();
	s.calculateConvexArea();
	s.calculatePixelList();
	for (int idx = 0; idx < s.connNum_; idx++) {
		if (s.area_[idx] > limitedSize) {
			if ((double)s.area_[idx] / s.convexArea_[idx] > 0.9) {
				vector<Point> pixelList = s.pixelList_[idx];
				for (int idx = 0; idx < pixelList.size(); idx++) {
					int row = pixelList[idx].x;
					int col = pixelList[idx].y;
					darkForeGroundFilled.at<uchar>(row, col) = 0;
				}
			}
		}

		if (s.area_[idx] < resizeH * resizeW / 6400) {
			vector<Point> pixelList = s.pixelList_[idx];
			for (int idx = 0; idx < pixelList.size(); idx++) {
				int row = pixelList[idx].x;
				int col = pixelList[idx].y;
				darkForeGroundFilled.at<uchar>(row, col) = 0;
			}
		}
	}
	Mat BW4; bitwise_not(darkForeGroundFilled, BW4);

	bitwise_or(img2, 255, img2, BW4);

	return img2;

	/////////////// 以下为旧版本 ////////////////////////////////
	//#pragma region 旧版本
	//	//////////////////////////////////////////////////////////////
	//	////////////// resize the image //////////////////////////////
	//	int origH = imgGray.rows;
	//	int origW = imgGray.cols;
	//
	//	// resize 图像
	//	Size dsize = Size(targetW, targetH);
	//	Mat scalingImgGray;
	//	resize(imgGray, scalingImgGray, dsize, 0.0, 0.0, INTER_CUBIC);
	//	resizeH = scalingImgGray.rows;
	//	resizeW = scalingImgGray.cols;
	//
	//	/////////////////////////////////////////////////////////////
	//	////////////// adjust the background to 255 /////////////////
	//	// median filter 中值滤波
	//	Mat imgGrayMf;
	//	medianBlur(scalingImgGray, imgGrayMf, 5);
	//
	//	Mat img2 = scalingImgGray.clone();
	//	Mat darkForeGround = imgGrayMf < 255; // 得到的 darkForeGround 所有元素非0即255
	//
	//	double darkForeRatio = (double)countNonZero(darkForeGround) / resizeH / resizeW;
	//	if (darkForeRatio > 0.2) { // 背景亮度不够
	//		darkForeGround = imgGrayMf < 204;	// 降低分割阈值
	//		// 以下是针对模块0出现-nan问题的阈值修改
	//		darkForeRatio = (double)countNonZero(darkForeGround) / resizeH / resizeW;
	//		if (darkForeRatio > 0.3) {
	//			darkForeGround = imgGrayMf < 170;	// 再降低分割阈值
	//		}
	//		// 以上是针对模块0出现-nan问题的阈值修改
	//		medianBlur(darkForeGround, darkForeGround, 3);
	//		Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	//		morphologyEx(darkForeGround, darkForeGround, MORPH_OPEN, element);
	//	}
	//
	//	Mat BW3;
	//	bitwise_not(darkForeGround, BW3);
	//	bitwise_or(img2, BW3, img2); // 即 img2(BW3)=255。BW3为真的部分，则置为255，即或运算；BW3为假的部分，保留img2的值，仍然是或运算
	//
	//	/////////////////////////////////////////////////////////////////////////////////
	//	////////////// remove the nucleus with a round shape ////////////////////////////
	//	double limitedSize = 0.005 * resizeH * resizeW;
	//	Mat darkForeGroundFilled = imFill(darkForeGround);
	//
	//	ConnectedRegion s(darkForeGroundFilled, 8);
	//	s.calculateConvexArea();
	//	s.calculateArea();
	//
	//	for (int index = 1; index <= s.connNum_; index++) {
	//		int area = s.area_[index - 1];
	//		if (area > limitedSize) {
	//			int convexArea = s.convexArea_[index - 1];
	//			if ((double)area / convexArea > 0.9) {
	//				vector<Point> pixelList = s.pixelList_[index - 1];
	//				for (int idx = 0; idx < pixelList.size(); idx++) {
	//					int row = pixelList[idx].x;
	//					int col = pixelList[idx].y;
	//					darkForeGroundFilled.at<uchar>(row, col) = 0;
	//				}
	//			}
	//		}
	//	}
	//
	//	bitwise_not(darkForeGroundFilled, BW3);
	//	bitwise_or(img2, BW3, img2);
	//	return img2;
	//#pragma endregion
}

 //imFill 'holes'
 //空洞填充
 //应该是按照 matlab 原函数原理实现的
Mat imFill(const Mat BW) {
	Size originSize = BW.size();
	Mat temp = Mat::zeros(originSize.height + 2, originSize.width + 2, BW.type());
	BW.copyTo(temp(Range(1, originSize.height + 1), Range(1, originSize.width + 1)));
	floodFill(temp, Point(0, 0), Scalar(255));
	Mat cutImg, cutImg_not;
	temp(Range(1, originSize.height + 1), Range(1, originSize.width + 1)).copyTo(cutImg);
	bitwise_not(cutImg, cutImg_not);
	return BW | (cutImg_not);

}

// 清除与边界相连的连通域
// 没有按照 matlab 原函数原理实现
Mat clearBorder(const Mat BW) {
	Mat BW1 = BW.clone();

	rectangle(BW1, Rect(0, 0, BW1.cols, BW1.rows), Scalar(255));
	floodFill(BW1, Point(0, 0), Scalar(0));

	return BW1;
}

vector<int> imhist(Mat &srcImage, unsigned int n) {
	CV_Assert(srcImage.channels() == 1);
	vector<int> hist(n, 0);
	double a = n / 256.0;
	int index = 0;
	int rows = srcImage.rows;
	int cols = srcImage.cols;
	for (int i = 0; i < rows; i++)
	{
		uchar* pdata = srcImage.ptr<uchar>(i);
		for (int j = 0; j < cols; j++)
		{
			index = a * pdata[j];
			++hist[index];
		}
	}
	return hist;
}

void stretchlim(Mat& src, Mat& lowHigh, double tol_low = 0.01, double tol_high = 0.99)
{
	CV_Assert(tol_low <= tol_high);

	int channelNum = src.channels();
	lowHigh.create(channelNum, 2, CV_64F);
	int nbins;
	if (src.depth() == CV_8U)
	{
		nbins = 256;
	}
	else
	{
		nbins = 65536;
	}
	//通道分离
	vector<Mat> channels;
	split(src, channels);


	for (int i = 0; i < channelNum; i++)
	{
		//获取灰度统计信息
		double low = 0, high = 1;
		auto hist = imhist(channels[i]);
		auto toltalSize = std::accumulate(hist.begin(), hist.end(), 0);
		//得到 >tol_low的分布概率的灰度等级
		for (int j = 0; j < hist.size(); ++j)
		{
			auto sum = std::accumulate(hist.begin(), hist.begin() + j, 0.0);
			if ((sum / toltalSize) > tol_low)  // > tol_low
			{
				low = j / (double)nbins;
				break;
			}
		}
		//得到 >tol_high的分布概率的灰度等级
		for (int k = 0; k < hist.size(); ++k)
		{
			auto sum = std::accumulate(hist.begin(), hist.begin() + k, 0.0);
			if ((sum / toltalSize) >= tol_high) // > tol_high
			{
				high = k / double(nbins);
				break;
			}
		}
		if (low == high)
		{
			lowHigh.ptr<double>(i)[0] = 0;
			lowHigh.ptr<double>(i)[1] = 1;
		}
		else
		{
			lowHigh.ptr<double>(i)[0] = low;
			lowHigh.ptr<double>(i)[1] = high;
		}
	}
	return;
}

void imadjust(Mat& src, Mat& dst, Mat& lowHighIn, Mat&lowHighOut, double gamma = 1)
{
	CV_Assert(src.data != NULL);

	int chl = src.channels();
	int rowNum = src.rows;
	int colNum = src.cols;

	//通道分离
	vector<Mat> channels;
	split(src, channels);


	//设置默认值
	if (lowHighIn.data == NULL)
	{
		lowHighIn = Mat::zeros(chl, 2, CV_64F);
		for (int i = 0; i < chl; i++)
		{
			lowHighIn.at<double>(i, 1) = 1;
		}
	}

	if (lowHighOut.data == NULL)
	{
		lowHighOut = Mat::zeros(chl, 2, CV_64F);
		for (int i = 0; i < chl; i++)
		{
			lowHighOut.at<double>(i, 1) = 1;
		}
	}
	for (int m = 0; m < chl; m++)
	{
		//gamma校正查表
		vector<double> lookuptable(256, 0);
		vector<uchar> img(256, 0);
		for (int i = 0; i < 256; i++)
		{
			lookuptable[i] = i / 255.0;
			if (lookuptable[i] <= lowHighIn.at<double>(m, 0))
			{
				lookuptable[i] = lowHighIn.at<double>(m, 0);
			}
			if (lookuptable[i] >= lowHighIn.at<double>(m, 1))
			{
				lookuptable[i] = lowHighIn.at<double>(m, 1);
			}
			lookuptable[i] = (lookuptable[i] - lowHighIn.at<double>(m, 0)) / (lowHighIn.at<double>(m, 1) - lowHighIn.at<double>(m, 0));
			lookuptable[i] = pow(lookuptable[i], gamma);
			lookuptable[i] = lookuptable[i] * (lowHighOut.at<double>(m, 1) - lowHighOut.at<double>(m, 0)) + lowHighOut.at<double>(m, 0);
			img[i] = lookuptable[i] * 255;
		}
		for (int j = 0; j < rowNum; j++)
		{
			for (int k = 0; k < colNum; k++)
			{
				channels[m].at<uchar>(j, k) = img[channels[m].at<uchar>(j, k)];
			}
		}
	}
	merge(channels, dst);
	return;
}

// 提取二值图像的骨骼
Mat skeleton(const Mat BW) {
	int nrow = BW.rows, ncol = BW.cols;
	
	// count junctions
	int jnrow = nrow + 1, jncol = ncol + 1;
	int njunc = 0, jhood = 0;
	for (int rowIdx = 0; rowIdx < jnrow; rowIdx++) {
		for (int colIdx = 0; colIdx < jncol; colIdx++) {
			jhood = jointNeighborhood(BW, rowIdx, colIdx);
			if (jhood != 0 && jhood != 15)
				njunc = njunc + 1;
		}
	}

	// register junctions
	Mat seenj = Mat::zeros(jnrow, jncol, CV_8UC1);
	vector<int>	jx(njunc, 0);
	vector<int> jy(njunc, 0);
	vector<int> seqj(njunc, 0);
	vector<int> edgej(njunc, 0);

	int ijunc = 0, nedge = 0;
	int iseq = 0, ei = 0, ej = 0, lastdir = 0;
	for (int rowIdx = 0; rowIdx < jnrow; rowIdx++) {
		for (int colIdx = 0; colIdx < jncol; colIdx++) {
			jhood = jointNeighborhood(BW, rowIdx, colIdx);

			if (jhood != 0 && jhood != 15 && jhood != 5 && jhood != 10
				&& seenj.at<uchar>(rowIdx, colIdx) == 0) {
				// find new edge, traverse it
				iseq = 0;
				ei = rowIdx; ej = colIdx;
				lastdir = North;

				while (seenj.at<uchar>(ei, ej) == 0
					|| jhood == 5 || jhood == 10) {
					if (seenj.at<uchar>(ei, ej) == 0) {
						jx[ijunc] = ej;
						jy[ijunc] = ei;
						edgej[ijunc] = nedge;
						seqj[ijunc] = iseq;
						iseq = iseq + 1;
						ijunc = ijunc + 1;
						seenj.at<uchar>(ei, ej) = 255;
					}

					// traverse clockwise
					switch (dircode[jhood]) {
					case North:
						ei--;
						lastdir = North;
						break;
					case South:
						ei++;
						lastdir = South;
						break;
					case East:
						ej++;
						lastdir = East;
						break;
					case West:
						ej--;
						lastdir = West;
						break;
					case None:
						switch (lastdir) {
						case East: // go North
							ei--;
							lastdir = North;
							break;
						case West:	// go South
							ei++;
							lastdir = South;
							break;
						case South:
							ej++;	// go East
							lastdir = East;
							break;
						case North: // go West
							ej--;
							lastdir = West;
							break;
						}
						break;
					}
					jhood = jointNeighborhood(BW, ei, ej);
				}
				nedge = nedge + 1;
			}
		}
	}

	// count perimeter along each edge
	vector<int> edgelen(nedge, 0);
	for (ijunc = 0; ijunc < njunc; ijunc++) {
		edgelen[edgej[ijunc]]++;
	}

	// create output
	int mind = 0, mindNE = 0, mindNW = 0, mindSE = 0, mindSW = 0, minjunc = 0;
	int nnear = 0, pspan = 0;
	Mat skr = Mat::zeros(BW.size(), CV_8UC1);
	vector<int>dNE(njunc, 0);
	vector<int>dNW(njunc, 0);
	vector<int>dSE(njunc, 0);
	vector<int>dSW(njunc, 0);
	vector<int>nearj(njunc, 0);
	for (int rowIdx = 0; rowIdx < nrow; rowIdx++) {
		for (int colIdx = 0; colIdx < ncol; colIdx++) {
			//if (rowIdx == 34) {
			//	int kkkkkkk = 100;
			//}
			if (BW.at<uchar>(rowIdx, colIdx)) {
				mind = mindNE = mindNW = mindSE = mindSW = SQR(jnrow + jncol);
				minjunc = -1;
				for (ijunc = 0; ijunc < njunc; ijunc++) {
					dNE[ijunc] = SQR(rowIdx - jy[ijunc]) + SQR(colIdx - jx[ijunc]);
					dNW[ijunc] = SQR(rowIdx - jy[ijunc]) + SQR(colIdx + 1 - jx[ijunc]);
					dSE[ijunc] = SQR(rowIdx + 1 - jy[ijunc]) + SQR(colIdx - jx[ijunc]);
					dSW[ijunc] = SQR(rowIdx + 1 - jy[ijunc]) + SQR(colIdx + 1 - jx[ijunc]);

					if (dNE[ijunc] < mindNE) {
						mindNE = dNE[ijunc];
						if (dNE[ijunc] < mind) {
							mind = dNE[ijunc];
							minjunc = ijunc;
						}
					}
					if (dNW[ijunc] < mindNW) {
						mindNW = dNW[ijunc];
						if (dNW[ijunc] < mind) {
							mind = dNW[ijunc];
							minjunc = ijunc;
						}
					}
					if (dSE[ijunc] < mindSE) {
						mindSE = dSE[ijunc];
						if (dSE[ijunc] < mind) {
							mind = dSE[ijunc];
							minjunc = ijunc;
						}
					}
					if (dSW[ijunc] < mindSW) {
						mindSW = dSW[ijunc];
						if (dSW[ijunc] < mind) {
							mind = dSW[ijunc];
							minjunc = ijunc;
						}
					}
				}

				// find all other junction points at minimal distance
				nnear = pspan = 0;
				for (ijunc = 0; ijunc < njunc; ijunc++) {
					if ((dNE[ijunc] <= MIN(mindNE, dNE[minjunc]))
						|| (dNW[ijunc] <= MIN(mindNW, dNW[minjunc]))
						|| (dSE[ijunc] <= MIN(mindSE, dSE[minjunc]))
						|| (dSW[ijunc] <= MIN(mindSW, dSW[minjunc]))) {
						// we have a candidate junction
						if (edgej[ijunc] != edgej[minjunc]) {
							pspan = -1;
							break;
						}
						else {
							nearj[nnear] = seqj[ijunc];
							nnear++;
						}
					}
				}

				if (pspan >= 0) {
					// compute perimeter span -- find largest gap and take remainder
					quickSort(nearj, nnear);
					pspan = nearj[0] - nearj[nnear - 1] + edgelen[edgej[minjunc]];
					for (int inear = 1; inear < nnear; inear++) {
						if (pspan < nearj[inear] - nearj[inear - 1]) {
							pspan = nearj[inear] - nearj[inear - 1];
						}
					}
					pspan = edgelen[edgej[minjunc]] - pspan;
					if (pspan > 255)
						pspan = 255;
					skr.at<uchar>(rowIdx, colIdx) = pspan;
				}
				else {
					skr.at<uchar>(rowIdx, colIdx) = 255;
				}
			}
			else {
				skr.at<uchar>(rowIdx, colIdx) = 0;
			}
		}
	}


	return skr;
}

// 输入 img 图像中指定 rowIdx 和 colIdx 位置的像素点
// 返回这个像素点的 8 连通区域的信息
int jointNeighborhood(const Mat img, const int rowIdx, const int colIdx) {
	int nrow = img.rows, ncol = img.cols;
	int condition = 8 * (rowIdx <= 0) + 4 * (colIdx <= 0) + 2 * (rowIdx >= nrow) + (colIdx >= ncol);

	switch (condition) {
	case 0:		// all points valid
		return(img.at<uchar>(rowIdx - 1, colIdx - 1) ? 1 : 0)
			+ (img.at<uchar>(rowIdx - 1, colIdx) ? 2 : 0)
			+ (img.at<uchar>(rowIdx, colIdx) ? 4 : 0)
			+ (img.at<uchar>(rowIdx, colIdx - 1) ? 8 : 0);
	case 1:		// right side not valid
		return(img.at<uchar>(rowIdx - 1, colIdx - 1) ? 1 : 0)
			+ (img.at<uchar>(rowIdx, colIdx - 1) ? 8 : 0);
	case 2:		// bottom not valid
		return(img.at<uchar>(rowIdx - 1, colIdx - 1) ? 1 : 0)
			+ (img.at<uchar>(rowIdx - 1, colIdx) ? 2 : 0);
	case 3:		// bottom and right not valid
		return(img.at<uchar>(rowIdx - 1, colIdx - 1) ? 1 : 0);
	case 4:		// left side not valide
		return(img.at<uchar>(rowIdx - 1, colIdx) ? 2 : 0)
			+ (img.at<uchar>(rowIdx, colIdx) ? 4 : 0);
	case 5:		// left and right sides not valid
		return 0;
	case 6:		// left and bottom not valid
		return(img.at<uchar>(rowIdx - 1, colIdx) ? 2 : 0);
	case 7:		// left, bottom, right sides not valid
		return 0;
	case 8:		// top side not valid
		return(img.at<uchar>(rowIdx, colIdx) ? 4 : 0)
			+ (img.at<uchar>(rowIdx, colIdx - 1) ? 8 : 0);
	case 9:		// top and right not valid
		return(img.at<uchar>(rowIdx, colIdx - 1) ? 8 : 0);
	case 10:	// top and bottom not valid
	case 11:	// top, bottom and right not valid
		return 0;
	case 12:	// top and left not valid
		return(img.at<uchar>(rowIdx, colIdx) ? 4 : 0);
	case 13:	// top, left and right sides not valid
	case 14:	// top, left and bottom sides not valid
	case 15:	// no sides valid
		return 0;
	default:
		return -1;
	}
}

// 将一个 vector<int> 快速排序
void quickSort(vector<int>& arr, int n) {
	if (n > 8) {
		int pivot;
		int pivid;
		int lo = 1;    // everything to the left of lo is smaller than pivot
		int hi = n - 1;  // everything to the right of hi is larger than the pivot
		int tmp;

		pivid = rand() % n;
		tmp = arr[0];
		arr[0] = arr[pivid];
		arr[pivid] = tmp;
		pivot = arr[0];  // store value for convenience
		while (hi != lo - 1) {  // keep going until everything has been categorized.
			if (arr[lo] < pivot) {
				// smaller than pivot, so stays here
				lo++;
			}
			else {
				// larger than pivot, so moves to the other end
				tmp = arr[hi];
				arr[hi] = arr[lo];
				arr[lo] = tmp;
				hi--;
			}
		}
		tmp = arr[hi];
		arr[hi] = arr[0];
		arr[0] = tmp;
		quickSort(arr, hi);          // sort smaller side of array

		

		vector<int> arrBehind(n - lo, 0);
		copy(arr.begin() + lo, arr.begin() + n, arrBehind.begin());
		quickSort(arrBehind, n - lo);  // sort bigger side of array
		copy(arrBehind.begin(), arrBehind.end(), arr.begin() + lo);
	}
	else {
		// use insertion sort
		int i, j;
		int tmp;

		for (i = 0; i < n; i++) {
			for (j = i; (j > 0) && (arr[j] < arr[j - 1]); j--) {
				tmp = arr[j];
				arr[j] = arr[j - 1];
				arr[j - 1] = tmp;
			}
		}
	}
}

// 基于索引表的细化细化算法
// 功能：对图象进行细化，即 MATLAB 中 bwdist(BW, 'skel', inf)
Mat ThiningDIBSkeleton(Mat BW)
{
	int lWidth = BW.cols;
	int lHeight = BW.rows;
	uchar * lpDIBBits = new uchar[sizeof(char) * BW.rows * BW.cols];
	// 提取图像imageData数组
	for (int row = 0; row < lHeight; row++)
	{
		uchar* ptr = BW.ptr<uchar>(row);
		for (int col = 0; col < lWidth; col++)
		{
			lpDIBBits[row * lWidth + col] = ptr[col] > 0 ? 1 : 0;
			//imagedata[y*src->width + x] = ptr[x] > 0 ? 1 : 0;
		}
	}

	//循环变量
	long i;
	long j;
	long lLength;

	unsigned char deletemark[256] = {      // 这个即为前人据8领域总结的是否可以被删除的256种情况
		0,0,0,0,0,0,0,1,	0,0,1,1,0,0,1,1,
		0,0,0,0,0,0,0,0,	0,0,1,1,1,0,1,1,
		0,0,0,0,0,0,0,0,	1,0,0,0,1,0,1,1,
		0,0,0,0,0,0,0,0,	1,0,1,1,1,0,1,1,
		0,0,0,0,0,0,0,0,	0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,	0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,	1,0,0,0,1,0,1,1,
		1,0,0,0,0,0,0,0,	1,0,1,1,1,0,1,1,
		0,0,1,1,0,0,1,1,	0,0,0,1,0,0,1,1,
		0,0,0,0,0,0,0,0,	0,0,0,1,0,0,1,1,
		1,1,0,1,0,0,0,1,	0,0,0,0,0,0,0,0,
		1,1,0,1,0,0,0,1,	1,1,0,0,1,0,0,0,
		0,1,1,1,0,0,1,1,	0,0,0,1,0,0,1,1,
		0,0,0,0,0,0,0,0,	0,0,0,0,0,1,1,1,
		1,1,1,1,0,0,1,1,	1,1,0,0,1,1,0,0,
		1,1,1,1,0,0,1,1,	1,1,0,0,1,1,0,0
	};//索引表

	unsigned char p0, p1, p2, p3, p4, p5, p6, p7;
	unsigned char *pmid, *pmidtemp;    // pmid 用来指向二值图像  pmidtemp用来指向存放是否为边缘
	unsigned char sum;
	bool bStart = true;
	lLength = lWidth * lHeight;
	unsigned char *pTemp = new uchar[sizeof(unsigned char) * lWidth * lHeight]();  //动态创建数组 并且初始化

																				   //    P0 P1 P2
																				   //    P7    P3
																				   //    P6 P5 P4

	while (bStart)
	{
		bStart = false;

		//首先求边缘点
		pmid = (unsigned char *)lpDIBBits + lWidth + 1;
		memset(pTemp, 0, lLength);
		pmidtemp = (unsigned char *)pTemp + lWidth + 1; //  如果是边缘点 则将其设为1
		for (i = 1; i < lHeight - 1; i++)
		{
			for (j = 1; j < lWidth - 1; j++)
			{
				if (*pmid == 0)                   //是0 不是我们需要考虑的点
				{
					pmid++;
					pmidtemp++;
					continue;
				}
				p3 = *(pmid + 1);
				p2 = *(pmid + 1 - lWidth);
				p1 = *(pmid - lWidth);
				p0 = *(pmid - lWidth - 1);
				p7 = *(pmid - 1);
				p6 = *(pmid + lWidth - 1);
				p5 = *(pmid + lWidth);
				p4 = *(pmid + lWidth + 1);
				sum = p0 & p1 & p2 & p3 & p4 & p5 & p6 & p7;
				if (sum == 0)
				{
					*pmidtemp = 1;       // 这样周围8个都是1的时候  pmidtemp==1 表明是边缘     					
				}

				pmid++;
				pmidtemp++;
			}
			pmid++;
			pmid++;
			pmidtemp++;
			pmidtemp++;
		}

		//现在开始删除
		pmid = (unsigned char *)lpDIBBits + lWidth + 1;
		pmidtemp = (unsigned char *)pTemp + lWidth + 1;

		for (i = 1; i < lHeight - 1; i++)   // 不考虑图像第一行 第一列 最后一行 最后一列
		{
			for (j = 1; j < lWidth - 1; j++)
			{
				if (*pmidtemp == 0)     //1表明是边缘 0--周围8个都是1 即为中间点暂不予考虑
				{
					pmid++;
					pmidtemp++;
					continue;
				}

				p3 = *(pmid + 1);
				p2 = *(pmid + 1 - lWidth);
				p1 = *(pmid - lWidth);
				p0 = *(pmid - lWidth - 1);
				p7 = *(pmid - 1);
				p6 = *(pmid + lWidth - 1);
				p5 = *(pmid + lWidth);
				p4 = *(pmid + lWidth + 1);

				p1 *= 2;
				p2 *= 4;
				p3 *= 8;
				p4 *= 16;
				p5 *= 32;
				p6 *= 64;
				p7 *= 128;

				sum = p0 | p1 | p2 | p3 | p4 | p5 | p6 | p7;
				//	sum = p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7;
				if (deletemark[sum] == 1)
				{
					*pmid = 0;
					bStart = true;      //  表明本次扫描进行了细化
				}
				pmid++;
				pmidtemp++;
			}

			pmid++;
			pmid++;
			pmidtemp++;
			pmidtemp++;
		}
	}
	delete[]pTemp;

	Mat skeleton = Mat::zeros(BW.size(), CV_8UC1);
	for (int row = 0; row < lHeight; row++) {
		uchar * rowPt = skeleton.ptr<uchar>(row);
		for (int col = 0; col < lWidth; col++) {
			rowPt[col] = lpDIBBits[row * lWidth + col] > 0 ? 255: 0;
		}
	}
	delete(lpDIBBits);
	return skeleton;
}

// 
Mat innerCutting(Mat objMask, Mat originalObjI, Mat innerPointsMap, double globalAvg, double minArea) {
	Mat obj1 = Mat::zeros(objMask.size(), CV_8UC1);

	Mat cutPoints = getCutPoints(objMask, 0.20, 40, "and");
	ConnectedRegion cutPointList(cutPoints, 8);
	cutPointList.calculatePixelList();

	vector<Point> points1;
	for (int row = 0; row < innerPointsMap.rows; row++) {
		uchar * rowPt = innerPointsMap.ptr<uchar>(row);
		for (int col = 0; col < innerPointsMap.cols; col++) {
			if (rowPt[col]) {
				Point coor = Point(row, col);
				points1.push_back(coor);
			}
		}
	}

	if (cutPointList.connNum_ != 0) {
		int nLenCutPointList = cutPointList.connNum_;

		vector<cuttingListStru> cuttingList;
		for (int idx = 1; idx <= nLenCutPointList; idx++) {
			vector<Point> points2 = cutPointList.pixelList_[idx - 1];
			cuttingListStru cuttingListElement = pDist2(points1, points2);

			cuttingList.push_back(cuttingListElement);
		}

		sort(cuttingList.begin(), cuttingList.end(), compDistAscend);

		obj1 = objMask.clone();

		for (int i = 1; i <= nLenCutPointList; i++) {
			cuttingListStru cuttingListEle = cuttingList[0];
			Mat cutLine = findCutLine(cuttingListEle, originalObjI, objMask);

			Mat notObjMask; bitwise_not(objMask, notObjMask);
			bitwise_and(cutLine, 0, cutLine, notObjMask);

			Mat notCutLine; bitwise_not(cutLine, notCutLine);

			int sumObjMask = countNonZero(objMask);
			int sumCutLine = countNonZero(cutLine);
			double objAvg = (double)sum(originalObjI & objMask)[0] / sumObjMask;
			double avg = (double)sum(originalObjI & cutLine)[0] / sumCutLine;

			if (avg > 2 * globalAvg || avg > 2 * objAvg) {
				bitwise_and(obj1, 0, obj1, cutLine);
			}
			// bwareaopen
			obj1 = bwareaopen(obj1, (int)minArea, 8);

			ConnectedRegion CCobj1(obj1, 4);
			if (CCobj1.connNum_ > 1)
				return obj1;
		}
	}

	return obj1;
}

//
Mat getCutPoints(Mat objMask, double paramCurv, double paramAngle, String logic) {
	Mat cutPointsMap;

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(objMask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0,0));

	int jumpCurve = 4, jumpAngle = 8;
	int nLen = contours[0].size();
	if (nLen <= jumpCurve) {
		cutPointsMap = Mat::zeros(objMask.size(), CV_8UC1);
		return cutPointsMap;
	}
	if (nLen <= jumpAngle) {
		cutPointsMap = Mat::zeros(objMask.size(), CV_8UC1);
		return cutPointsMap;
	}


	vector<Point> boundaryPoints = contours[0];

	////////////////////////////////////////////////////////////////////////
	///////////////////curvature detection//////////////////////////////////
	vector<Point> boundaryPointsCurve(boundaryPoints);
	// 在末尾插入最前面 jumpCurve 个元素
	// 在最前面插入末尾 jumpCuve 个元素
	for (int idx = 1; idx <= jumpCurve; idx++) {
		Point forward = boundaryPoints[idx - 1];
		Point behind = boundaryPoints[nLen - jumpCurve + idx - 1];
		boundaryPointsCurve.push_back(forward);
		boundaryPointsCurve.insert(boundaryPointsCurve.begin() + idx - 1, behind);
	}

	Mat curvaturePoints = Mat::zeros(objMask.size(), CV_8UC1);
	for (int idx = 1 + jumpCurve; idx <= nLen + jumpCurve; idx++) {

		// boundaryPointsCurve 中每个 Point 的 x 是列下标， y 是行下标
		// 这里反过来是为了和 MATLAB 保持一致
		// 加一是因为 MATLAB 下标索引从 1 开始
		// 数值上保持一致
		int x1 = boundaryPointsCurve[idx - jumpCurve - 1].x + 1;
		int y1 = boundaryPointsCurve[idx - jumpCurve - 1].y + 1;
		int x2 = boundaryPointsCurve[idx - 1].x + 1;
		int y2 = boundaryPointsCurve[idx - 1].y + 1;
		int x3 = boundaryPointsCurve[idx + jumpCurve - 1].x + 1;
		int y3 = boundaryPointsCurve[idx + jumpCurve - 1].y + 1;

		// 注意这里要 减二，坐标索引不能错
		if (objMask.at<uchar>((y1 + y3 - 2) / 2, (x1 + x3 - 2) / 2) == 0) {
			double curvature = 
				(double)(2 * ABS((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))) /
				sqrt((SQR(x2 - x1) + SQR(y2 - y1))	 *	(SQR(x3 - x1) + SQR(y3 - y1))	*	(SQR(x3 - x2) + SQR(y3 - y2)));
			if (curvature > paramCurv) {
				Point coor = boundaryPoints[idx - 1 - jumpCurve];
				curvaturePoints.at<uchar>(coor.y, coor.x) = 255;
			}
		}
	}

	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
	morphologyEx(curvaturePoints, curvaturePoints, MORPH_DILATE, element);

	//namedWindow("curvaturePoints", CV_WINDOW_FREERATIO);
	//imshow("curvaturePoints", curvaturePoints);
	//waitKey(0);

	///////////////////////////////////////////////////////////////////////
	/////////////calculate angular changes along boundary//////////////////
	vector<Point> boundaryPointsAngle(boundaryPoints);
	// 在末尾插入最前面 jumpAngle 个元素
	// 在最前面插入末尾 jumpAngle 个元素
	for (int idx = 1; idx <= jumpAngle; idx++) {
		Point forward = boundaryPoints[idx - 1];
		Point behind = boundaryPoints[nLen - jumpAngle + idx - 1];
		boundaryPointsAngle.push_back(forward);
		boundaryPointsAngle.insert(boundaryPointsAngle.begin() + idx - 1, behind);
	}

	Mat angleChanges = Mat::zeros(objMask.size(), CV_8UC1);
	//vector<double> testAngles;
	for (int idx = 1 + jumpAngle; idx <= nLen + jumpAngle; idx++) {

		// boundaryPointsAngle 中每个 Point 的 x 是列下标， y 是行下标
		// 这里反过来是为了和 MATLAB 保持一致
		// 加一是因为 MATLAB 下标索引从 1 开始
		// 数值上保持一致
		int x1 = boundaryPointsAngle[idx - jumpAngle - 1].x + 1;
		int y1 = boundaryPointsAngle[idx - jumpAngle - 1].y + 1;
		int x2 = boundaryPointsAngle[idx - 1].x + 1;
		int y2 = boundaryPointsAngle[idx - 1].y + 1;
		int x3 = boundaryPointsAngle[idx + jumpAngle - 1].x + 1;
		int y3 = boundaryPointsAngle[idx + jumpAngle - 1].y + 1;

		if (objMask.at<uchar>(round((y1 + y3)/2) - 1, round((x1 + x3) / 2) - 1) == 0) {
			// p1 = [x1 y1]
			// p2 = [x2 y2]
			// p3 = [x3 y3]
			// v1 = p2 - p1 = [x2 - x1 y2 - y1]
			// v2 = p3 - p2 = [x3 - x2 y3 - y2]
			double detV1V2 = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2);
			double dotV1V2 = (x2 - x1) * (x3 - x2) + (y2 - y1) * (y3 - y2);
			double atanDetDot = atan2(detV1V2, dotV1V2);
			double a1 = atanDetDot - 2 * PI * floor(atanDetDot / (2 * PI));
			double angleOut = (double)(ABS((a1 > (PI / 2) ? 1 : 0) * PI - a1)) * 180.0 / PI;

			//testAngles.push_back(angleOut);

			if ((angleOut > paramAngle) && (angleOut <= 180 - 1.5 * paramAngle)) {
				Point coor = boundaryPoints[idx - 1 - jumpAngle];
				angleChanges.at<uchar>(coor.y, coor.x) = 255;
			}
		}
	}

	//namedWindow("angleChangesBeforeDilate", CV_WINDOW_FREERATIO);
	//imshow("angleChangesBeforeDilate", angleChanges);
	//waitKey(0);

	morphologyEx(angleChanges, angleChanges, MORPH_DILATE, element);

	//namedWindow("angleChanges", CV_WINDOW_FREERATIO);
	//imshow("angleChanges", angleChanges);
	//waitKey(0);

	/////////////////////////////////////////////////////////////////////
	//////////////////specify cut points/////////////////////////////////
	if (logic == "and")
		bitwise_and(curvaturePoints, angleChanges, cutPointsMap);
	else if (logic == "or")
		bitwise_or(curvaturePoints, angleChanges, cutPointsMap);
	
	//namedWindow("cutPointsMapAnd", CV_WINDOW_FREERATIO);
	//imshow("cutPointsMapAnd", cutPointsMap);
	//waitKey(0);
	
	element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	Mat objMaskTmp, objMaskExteriorBoundary, notObjMaskExteriorBoundary;
	morphologyEx(objMask, objMaskTmp, MORPH_DILATE, element);
	subtract(objMaskTmp, objMask, objMaskExteriorBoundary);
	
	//namedWindow("objMaskExt", CV_WINDOW_FREERATIO);
	//imshow("objMaskExt", objMaskExteriorBoundary);
	//waitKey(0);

	bitwise_and(cutPointsMap, objMaskExteriorBoundary, cutPointsMap);

	return cutPointsMap;
}

// 找两个vector<Point>之间最近的两个点 pdist2(points1, points2)
// 返回结构体，保存两个点和其距离
cuttingListStru pDist2(const vector<Point>pointList1, const vector<Point>pointList2) {
	double minDist = norm(Mat(pointList1[0]), Mat(pointList2[0]));
	cuttingListStru nearest;
	nearest.point1 = pointList1[0];
	nearest.point2 = pointList2[0];
	nearest.dist = minDist;

	for (int i = 0; i < pointList1.size(); i++) {
		for (int j = 0; j < pointList2.size(); j++) {
			Point temp1 = pointList1[i];
			Point temp2 = pointList2[j];
			double currDist = norm(Mat(temp1), Mat(temp2));

			if (currDist < minDist) {
				minDist = currDist;
				nearest.point1 = temp1;
				nearest.point2 = temp2;
				nearest.dist = minDist;
			}
		}
	}

	return nearest;
}

// 方便结构体排序用的比较函数
bool compDistAscend(const cuttingListStru & a, const cuttingListStru & b) {
	return a.dist < b.dist;
}

//
Mat findCutLine(cuttingListStru cuttingListEle, Mat originalObjI, Mat objMask) {
	Mat cutLine = Mat::zeros(objMask.size(), CV_8UC1);
	
	Point point1 = cuttingListEle.point1;
	Point point2 = cuttingListEle.point2;
	double dist = cuttingListEle.dist;
 	
	int radius = 0;
	while (dist > 1.0) {
		if (dist > 15.0) {
			radius = 3;
		}
		else {
			radius = ceil(dist / 5);
		}

		// draw circle
		Mat circleContour = Mat::zeros(objMask.size(), CV_8UC1);
		circle(circleContour, Point(point1.y, point1.x), radius, Scalar::all(255));

		//namedWindow("circleContour", CV_WINDOW_FREERATIO);
		//imshow("circleContour", circleContour);
		//waitKey(0);

		vector<cuttingListStru> candidatesPointDist; // 每个 struct 中 point1 存的是目标点
		for (int row = 0; row < circleContour.rows; row++) {
			uchar * rowPt = circleContour.ptr<uchar>(row);
			for (int col = 0; col < circleContour.cols; col++) {
				if (rowPt[col]) {
					Point candidatePoint = Point(row, col);
					double candidateDist = norm(Mat(candidatePoint), Mat(point2));
					if (candidateDist < dist) {
						cuttingListStru candidate;
						candidate.point1 = candidatePoint;
						candidate.point2 = point2;
						candidate.dist = candidateDist;
						candidatesPointDist.push_back(candidate);
					}
				}
			}
		}

		if (candidatesPointDist.empty()) {
			break;
		}

		sort(candidatesPointDist.begin(), candidatesPointDist.end(), compDistAscend);

		double pathIntensity = -1;
		Point chosenPoint;
		Mat chosenLine;
		for (int idx = 0; idx < candidatesPointDist.size(); idx++) {
			Mat line = drawThinLine(point1, candidatesPointDist[idx].point1, objMask.size());

			//namedWindow("line", CV_WINDOW_FREERATIO);
			//imshow("line", line);
			//waitKey(0);

			int sumLine = countNonZero(line);

			Mat originalObjILine;
			bitwise_or(originalObjI, 255, originalObjILine, line);

			//namedWindow("originalObjILine", CV_WINDOW_FREERATIO);
			//imshow("originalObjILine", originalObjILine);
			//waitKey(0);

			double newIntensity = (double)sum(originalObjILine)[0] / 255.0 / sumLine;
			if (newIntensity > pathIntensity) {
				pathIntensity = newIntensity;
				chosenLine = line.clone();
				chosenPoint = candidatesPointDist[idx].point1;
			}
		}
		//namedWindow("chosenLine", CV_WINDOW_FREERATIO);
		//imshow("chosenLine", chosenLine);
		//waitKey(0);

		bitwise_or(cutLine, chosenLine, cutLine);

		point1 = point2;
		point2 = chosenPoint;
		dist = norm(Mat(point1), Mat(point2));
	}
	return cutLine;
}

// 画线
// 注意 dilate 和 hitmiss 操作的 kernel
// 这里很奇怪地，OPENCV 和 MATLAB 的结果不一样
// 因此做了一些修改
// 还需要进一步测试确保任何情况下都相同
Mat drawThinLine(Point point1, Point point2, Size imgSize) {
	Mat thinLine = Mat::zeros(imgSize, CV_8UC1);

	line(thinLine, Point(point1.y, point1.x), Point(point2.y, point2.x), Scalar::all(255));

	//namedWindow("line0", CV_WINDOW_FREERATIO);
	//imshow("line0", thinLine);
	//waitKey(0);
	//cout << "opencvLineFunc" << endl;
	//cout << thinLine << endl << endl;

	int row1 = point1.x, row2 = point2.x;
	int col1 = point1.y, col2 = point2.y;

	if ((col1 < col2 && row1 > row2) || (col1 > col2 && row1 < row2)) {
		Mat interval = (Mat_<int>(3, 3) << 0, -1, 1,
											 0, 1, -1,
											 0, 0, 0);
		Mat element = (Mat_<uchar>(3, 3) << 0, 0, 0,
											0, 0, 0,
											0, 1, 0);
		Mat mask;
		morphologyEx(thinLine, mask, MORPH_HITMISS, interval);
		morphologyEx(mask, mask, MORPH_DILATE, element);
		bitwise_or(mask, thinLine, thinLine);
	}
	else if ((col1 < col2 && row1 < row2) || (col1 > col2 && row1 > row2)) {
		Mat interval = (Mat_<int>(3, 3) << 0, 0, 0,
											 0, 1, -1,
											 0, -1, 1);
		Mat element = (Mat_<uchar>(3, 3) << 0, 1, 0,
											0, 0, 0,
											0, 0, 0);
		Mat mask;
		morphologyEx(thinLine, mask, MORPH_HITMISS, interval);

		//cout << "interval" << endl;
		//cout << interval << endl << endl;
		//cout << "mask" << endl;
		//cout << mask << endl << endl;

		//Mat elementTest = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
		//elementTest.at<uchar>(0, 1) = 0;
		//elementTest.at<uchar>(1, 0) = 0; elementTest.at<uchar>(1, 1) = 0;
		//elementTest.at<uchar>(1, 2) = 0;
		//cout << "elementTest" << endl;
		//cout << elementTest << endl << endl;

		Mat maskDilated;
		morphologyEx(mask, maskDilated, MORPH_DILATE, element);

		//cout << "maskDilated" << endl;
		//cout << maskDilated << endl << endl;
		bitwise_or(maskDilated, thinLine, thinLine);
	}

	//cout << "thinLine" << endl;
	//cout << thinLine << endl << endl;

	return thinLine;
}

// 小连通区域删除
// 这里实现的原理是找轮廓，轮廓包络面积小于 threshold 的则擦除
// MATLAB 的原理，找连通域，连通域面积小于 threshold 的则擦除
// 修改了
Mat bwareaopen(const Mat BW, const int threshold, const int conn) {

	Mat BW1 = Mat::zeros(Size(BW.cols, BW.rows), CV_8UC1);

	//vector<vector<Point>> contours;
	//findContours(BW, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	//for (int idx = 0; idx < contours.size(); idx++) {
	//	double area = contourArea(contours[idx]);

	//	if (area >= threshold) {
	//		drawContours(BW1, contours, idx, Scalar::all(255), -1);
	//	}
	//}
	//return BW1;

	ConnectedRegion CCbw(BW, conn);
	CCbw.calculateArea();
	CCbw.calculateBoundingBox();
	CCbw.calculateImage();
	vector<int> area = CCbw.area_;
	vector<Mat> image = CCbw.image_;
	vector<vector<int>> bbox = CCbw.boundingBox_;
	
	for (int idx = 0; idx < area.size(); idx++) {
		if (area[idx] >= threshold) {
			Mat imageElement = image[idx];
			vector<int> bboxElement = bbox[idx];
			Mat bw1ROI = BW1(Rect(bboxElement[1], bboxElement[0], bboxElement[3], bboxElement[2]));
			imageElement.copyTo(bw1ROI, imageElement);
		}
	}

	return BW1;
}

//
void findClusters(Mat BW, Mat & singles, Mat & clusters, Mat & bwThin) {
	Mat skr = skeleton(BW);

	threshold(skr, skr, 25, 255, THRESH_BINARY);
	bwThin = ThiningDIBSkeleton(skr);

	//namedWindow("bwThin", CV_WINDOW_FREERATIO);
	//imshow("bwThin", bwThin);
	//waitKey(0);

	// imfilter
	Point anchor(0, 0);
	uchar kernel[3][3] = { { 1,1,1 },{ 1,1,1 },{ 1,1,1 } };
	Mat kernelMat = Mat(3, 3, CV_8UC1, &kernel);
	Mat neighbourCount;
	Mat bwThinInt;
	threshold(bwThin, bwThinInt, 1, 1, THRESH_BINARY);
	filter2D(bwThinInt, neighbourCount, -1, kernelMat, anchor, 0.0, BORDER_CONSTANT);

	Mat bwBranches;
	bitwise_and(neighbourCount > 4, bwThin, bwBranches);

	//namedWindow("bwBranches", CV_WINDOW_FREERATIO);
	//imshow("bwBranches", bwBranches);
	//waitKey(0);

	clusters = imreconstruct(bwBranches, BW);
	singles = BW - clusters;

	return;
}

//
Mat imreconstruct(Mat marker, Mat mask) {
	Mat dst;
	cv::min(marker, mask, dst);
	dilate(dst, dst, Mat());
	cv::min(dst, mask, dst);
	Mat temp1 = Mat(marker.size(), CV_8UC1);
	Mat temp2 = Mat(marker.size(), CV_8UC1);
	do
	{
		dst.copyTo(temp1);
		dilate(dst, dst, Mat());
		cv::min(dst, mask, dst);
		compare(temp1, dst, temp2, CV_CMP_NE);
	} while (sum(temp2).val[0] != 0);
	return dst;
}

Mat cutTouching(Mat objMask, Mat originalObjI, Mat cutPointsMap, double globalAvg, double avgThicknes, double minArea) {
	
	//namedWindow("objMask", CV_WINDOW_FREERATIO);
	//imshow("objMask", objMask);
	//waitKey(0);

	//namedWindow("originalObjI", CV_WINDOW_FREERATIO);
	//imshow("originalObjI", originalObjI);
	//waitKey(0);

	//namedWindow("cutPointsMap", CV_WINDOW_FREERATIO);
	//imshow("cutPointsMap", cutPointsMap);
	//waitKey(0);
	
	Mat skr = skeleton(objMask);

	threshold(skr, skr, 25, 255, THRESH_BINARY);
	Mat bwThin = ThiningDIBSkeleton(skr);

	Mat skel = bwThin;
	vector<Point> ep, bp;
	extendSkeleton(objMask, skel, ep, bp);
	int nEP = ep.size();

	ConnectedRegion s(skel, 8);
	s.calculatePixelList();
	ConnectedRegion cutPointRegionProps(cutPointsMap, 8);
	cutPointRegionProps.calculatePixelList();

	Mat obj1;

	int numCutPointRegions = cutPointRegionProps.connNum_;

	if (numCutPointRegions > 1) {
		vector<cuttingListStru> cutPointPairs = extractCutPointPairs(cutPointRegionProps);
		vector<cuttingListStru> validCutPointPairs = reduceCuttingList(cutPointPairs, skel);

		// 保留切割点对的距离在一定范围内的切割点对
		int numValidPairs = validCutPointPairs.size();
		vector<cuttingListStru> distValidCutPointPairs(validCutPointPairs);
		for (int i = 0; i < numValidPairs; i++) {
			int distThresh = round(avgThicknes / 3 * 2) * 5 / 2;	// 距离阈值
			if (validCutPointPairs[numValidPairs - 1 - i].dist > distThresh) {
				distValidCutPointPairs.pop_back();
			}
			else
				break;
		}

		if (!distValidCutPointPairs.empty()) {
			obj1 = cut(distValidCutPointPairs, originalObjI, objMask, globalAvg, minArea, "PalePath");
		}

	}

	/////
	/////

	ConnectedRegion objNum(obj1, 4);
	if (objNum.connNum_ < 2) {		// 只能是空的情况？ 上述情况没切成

		// 对有两个及以上切割点
		if (numCutPointRegions > 1) {
			// 只有一个骨架
			if (s.connNum_ == 1) {
				if (nEP < 3) {
					// 找到弯曲点
					Mat angleChanges = findAngleChanges(skel, ep[0]);
					ConnectedRegion angleStats(angleChanges, 8);

					if (angleStats.connNum_ > 1) {
						Point sumCenter = Point(0, 0);
						for (int idx = 0; idx < angleStats.connNum_; idx++) {
							Point center = Point(
								angleStats.centroids_.at<double>(idx + 1, 1),
								angleStats.centroids_.at<double>(idx + 1, 0));
							sumCenter = sumCenter + center;
						}
						Point ap;
						ap.x = round((double)sumCenter.x / angleStats.connNum_);
						ap.y = round((double)sumCenter.y / angleStats.connNum_);

						vector<cuttingListStru> cutPointPairs = extractCutPointPairs(cutPointRegionProps);
						vector<cuttingListStru> validCutPointPairs = reduceCuttingList(cutPointPairs, skel);

						// 保留切割点对的距离在一定范围内的切割点对
						int numValidPairs = validCutPointPairs.size();
						vector<cuttingListStru> distValidCutPointPairs(validCutPointPairs);
						for (int i = 0; i < numValidPairs; i++) {
							int distThresh = 20;	// 距离阈值
							if (validCutPointPairs[numValidPairs - 1 - i].dist > distThresh) {
								distValidCutPointPairs.pop_back();
							}
							else
								break;
						}

						if (!distValidCutPointPairs.empty()) {
							obj1 = cut(distValidCutPointPairs, originalObjI, objMask, globalAvg, minArea, "PalePath");
						}
						else {
							obj1.release();
						}
					}
				}
				else if (nEP == 3 && bp.size() == 1) {		// 大概指 T 型
					vector<cuttingListStru> cutPointPairs = findClosestToReference(cutPointRegionProps, bp); //从近到远排序
					vector<cuttingListStru> validCutPointPairs = reduceCuttingList(cutPointPairs, skel);

					// 保留切割点对的距离在一定范围内的切割点对
					int numValidPairs = validCutPointPairs.size();
					vector<cuttingListStru> distValidCutPointPairs(validCutPointPairs);
					for (int i = 0; i < numValidPairs; i++) {
						int distThresh = 22;	// 距离阈值，MATLAB 中为 20
												// 但是这里由于之前计算点位存在误差
												// 导致距离判断出错
						if (validCutPointPairs[numValidPairs - 1 - i].dist > distThresh) {
							distValidCutPointPairs.pop_back();
						}
						else
							break;
					}

					if (distValidCutPointPairs.empty()) {
						obj1.release();
					}
					else {
						vector<double> brightness(distValidCutPointPairs.size(), 0.0);
						for (int i = 0; i < brightness.size(); i++) {
							Mat cutLine = findCutLine(distValidCutPointPairs[i], originalObjI, objMask);

							Mat notObjMask; bitwise_not(objMask, notObjMask);
							bitwise_and(cutLine, 0, cutLine, notObjMask);

							brightness[i] = (double)sum(originalObjI & cutLine)[0] / 255.0 / countNonZero(cutLine);
						}
						// 按照 brightness 的降序给 distValidCutPointPairs 排序
						for (int i = 0; i < brightness.size() - 1; i++) {
							for (int j = i + 1; j < brightness.size(); j++) {
								if (brightness[i] < brightness[j]) {
									double tempBrightness = brightness[i];
									brightness[i] = brightness[j];
									brightness[j] = tempBrightness;
									cuttingListStru tempDVCPP = distValidCutPointPairs[i];
									distValidCutPointPairs[i] = distValidCutPointPairs[j];
									distValidCutPointPairs[j] = tempDVCPP;
								}
							}
						}

						obj1 = cut(distValidCutPointPairs, originalObjI, objMask, globalAvg, minArea, "PalePath");
					}
				}
				else if (nEP == 4) {	// 大概指背靠背的形式
					vector<cuttingListStru> cutPointPairs = extractCutPointPairs(cutPointRegionProps);
					vector<cuttingListStru> reducedCutPointPairs = reduceCuttingList(cutPointPairs, skel);

					for (int i = 0; i < reducedCutPointPairs.size(); i++) {
						vector<cuttingListStru> reducedCutPointPairsI;
						reducedCutPointPairsI.push_back(reducedCutPointPairs[i]);
						obj1 = cut(reducedCutPointPairsI, originalObjI, objMask, globalAvg, minArea, "PalePath");
						if (!obj1.empty()) {
							Mat localSingles = Mat::zeros(obj1.size(), CV_8UC1);
							Mat localClusters = Mat::zeros(obj1.size(), CV_8UC1);
							Mat newSkel = Mat::zeros(obj1.size(), CV_8UC1);
							findClusters(obj1, localSingles, localClusters, newSkel);

							ConnectedRegion singleCC(localSingles, 4);
							if (countNonZero(localClusters) == 0
								&& singleCC.connNum_ == 2) {		// 刚好切成两个染色体
								break;
							}
							else {
								obj1.release();
							}
						}
					}
					
					if (obj1.empty()) {
						// 4个端点，2个分叉点的情况
						if (bp.size() == 2) {
							vector<Point> branch1, branch2;
							for (int i = 1; i <= 4; i++) {
								if (norm(Mat(ep[i - 1]), Mat(bp[0])) < norm(Mat(ep[i - 1]), Mat(bp[1]))) {
									branch1.push_back(ep[i - 1]);								
								}
								else {
									branch2.push_back(ep[i - 1]);
								}
							}

							// 每支两个端点
							if (branch1.size() == 2) {
								double angle1 = angle3points(branch1[0], bp[0], branch1[1]);
								double angle2 = angle3points(branch2[0], bp[1], branch2[1]);

								if (angle1 < 30 || angle1 > 120) {
									cutPointPairs.clear();
									reducedCutPointPairs.clear();
									vector<Point> bpI;
									bpI.push_back(bp[0]);
									cutPointPairs = findClosestToReference(cutPointRegionProps, bpI);
									reducedCutPointPairs = reduceCuttingList(cutPointPairs, skel);
									if (reducedCutPointPairs.empty()) {
										obj1.release();
									}
									else {
										vector<cuttingListStru> reducedCutPointPairsI;
										reducedCutPointPairsI.push_back(reducedCutPointPairs[0]);
										obj1 = cut(reducedCutPointPairsI, originalObjI, objMask, globalAvg, minArea, "PalePath");
									}

									if (obj1.empty() && (angle2 < 30 || angle2 > 120)) {
										cutPointPairs.clear();
										reducedCutPointPairs.clear();
										vector<Point> bpI;
										bpI.push_back(bp[1]);
										cutPointPairs = findClosestToReference(cutPointRegionProps, bpI);
										reducedCutPointPairs = reduceCuttingList(cutPointPairs, skel);

										if (reducedCutPointPairs.empty()) {
											obj1.release();
										}
										else {
											vector<cuttingListStru> reducedCutPointPairsI;
											reducedCutPointPairsI.push_back(reducedCutPointPairs[0]);
											obj1 = cut(reducedCutPointPairsI, originalObjI, objMask, globalAvg, minArea, "PalePath");
										}
									}
								}
								else if (angle2 < 30 || angle2 > 120) {
									cutPointPairs.clear();
									reducedCutPointPairs.clear();
									vector<Point> bpI;
									bpI.push_back(bp[1]);
									cutPointPairs = findClosestToReference(cutPointRegionProps, bpI);
									reducedCutPointPairs = reduceCuttingList(cutPointPairs, skel);
									if (reducedCutPointPairs.empty()) {
										obj1.release();
									}
									else {
										vector<cuttingListStru> reducedCutPointPairsI;
										reducedCutPointPairsI.push_back(reducedCutPointPairs[0]);
										obj1 = cut(reducedCutPointPairsI, originalObjI, objMask, globalAvg, minArea, "PalePath");
									}
								}
								else {
									obj1.release();
								}
							}
						}
						else {
							obj1.release();
						}
					}
				}
				else {
					for (int i = 0; i < bp.size(); i++) {
						vector<Point> bpLeft(bp);
						bpLeft.erase(bpLeft.begin()+i);
						Point point1 = bp[i];

						int count = 0;
						for (int idx = 0; idx < bpLeft.size(); idx++) {
							Point point2 = bpLeft[idx];
							if (norm(Mat(point1), Mat(point2)) > 10)
								count++;
						}

						if (count > 1) {
							vector<Point> bpI;
							bpI.push_back(point1);
							vector<cuttingListStru> cutPointPairs = findClosestToReference(cutPointRegionProps, bpI);
							vector<cuttingListStru> reducedCutPointPairs = reduceCuttingList(cutPointPairs, skel);
							if (!reducedCutPointPairs.empty()) {
								vector<cuttingListStru> reducedCutPointPairsI;
								reducedCutPointPairsI.push_back(reducedCutPointPairs[i]);
								obj1 = cut(reducedCutPointPairsI, originalObjI, objMask, globalAvg, minArea, "PalePath");
							}
						}
					}
				}
			}
			else if (s.connNum_ > 1) {
				obj1.release();
			}
		}
		else {
			obj1.release();
		}
	}

	return obj1;
}

void extendSkeleton(Mat objMask, Mat& skel, vector<Point>& ep, vector<Point>& bp) {
	// ep: endPoints
	// bp: junctions
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
			if (contours.empty())
				return;
			// 排序
			// 先把从第一个开始到 length 个元素拿出来
			int length = countNonZero(minPath);
			vector<Point> temp(length, Point(0,0));
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
					copy(temp.begin(), temp.begin() + idx, contour.begin()+leftLength);
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
			fitExtention(ep2[i], ep[i], contour[CL - 1], objMask.size());

			Mat skelExt = Mat::zeros(skel.size(), CV_8UC1);

			line(skelExt, Point(ep2[i].y, ep2[i].x), Point(ep[i].y, ep[i].x), Scalar::all(255));

			//namedWindow("line", CV_WINDOW_FREERATIO);
			//imshow("line", skelExt);
			//waitKey(0);

			Mat skelExtObjMask;
			bitwise_and(skelExt, objMask, skelExtObjMask);
			ConnectedRegion sExtCrossingPoints(skelExtObjMask, 8);
			sExtCrossingPoints.calculateBoundingBox();
			int numSExtCrossingPoints = sExtCrossingPoints.connNum_;

			if (numSExtCrossingPoints > 1) {		// 如果有多个交点，选择最近的
				Mat tmpSkel = Mat::zeros(skelExt.size(), CV_8UC1);

				// 找最小值
				Mat centroids = sExtCrossingPoints.centroids_;
				int minIdx = 0;
				Point center = Point(centroids.at<double>(1, 1), centroids.at<double>(1, 0));
				double minDist = norm(Mat(ep[i]), Mat(center));
				for (int idx = 1; idx < numSExtCrossingPoints; idx++) {
					Point center = Point(centroids.at<double>(idx + 1, 1), centroids.at<double>(idx + 1, 0));
					double curDist = norm(Mat(ep[i]), Mat(center));
					if (curDist < minDist) {
						minIdx = idx;
						minDist = curDist;
					}
				}

				vector<int> bbox = sExtCrossingPoints.boundingBox_[minIdx];
				Mat onesROI = Mat::ones(bbox[2], bbox[3], CV_8UC1) * 255;
				Mat tmpSkelROI = tmpSkel(Rect(bbox[1], bbox[0], bbox[3], bbox[2]));
				onesROI.copyTo(tmpSkelROI, onesROI);
				skelExt = tmpSkel;
			}
			skel = (skel + skelExt).mul(objMask);

			//namedWindow("skelInLoop", CV_WINDOW_FREERATIO);
			//imshow("skelInLoop", skel);
			//waitKey(0);
		}


		//namedWindow("skelAfterForLoop", CV_WINDOW_FREERATIO);
		//imshow("skelAfterForLoop", skel);
		//waitKey(0);
		//anaskel(skel, ep, bp);

	}
	return;
}

// 分析骨架
void anaskel(Mat skel, vector<Point>& endPoints, vector<Point>& junctions) {
	// trim skeleton
	Mat skelTrim = doctrim(skel);

	// count junctions and endpoints
	// obtain junctions and endpoints
	for (int row = 0; row < skelTrim.rows; row++) {
		uchar * rowPt = skelTrim.ptr<uchar>(row);
		for (int col = 0; col < skelTrim.cols; col++) {
			if (rowPt[col]) {
				int hood = neighborhood(skelTrim, row, col);
				
				// endpoints
				if (nbr_branches[hood] < 2) {
					endPoints.push_back(Point(row, col));			// 这里原来的cpp中加一，是因为MATLAB下标索引从1开始
				}

				// junctions
				if (nbr_branches[hood] > 2) {
					junctions.push_back(Point(row, col));
				}
			}
		}
	}
	return;
}

// 对输入的骨架进行剪枝
// anaskel 中的子函数
Mat doctrim(Mat skel) {
	Mat doctrimSkel = skel.clone();

	for (int row = 0; row < doctrimSkel.rows; row++) {
		uchar * rowPt = doctrimSkel.ptr<uchar>(row);
		for (int col = 0; col < doctrimSkel.cols; col++) {
			if (rowPt[col]) {
				int hood = neighborhood(doctrimSkel, row, col);
				rowPt[col] = (connected_nbrs[hood] > 1) || (nbr_branches[hood] == 1);
			}
			else {
				rowPt[col] = 0;
			}
		}
	}
	return doctrimSkel;
}

// 返回 img 中指定位置的像素的邻域信息
// 12点方向起始顺时针
// anaskel 和 doctrim 的子函数
int neighborhood(const Mat img, const int rowIdx, const int colIdx) {
	int nrow = img.rows, ncol = img.cols;

	int condition = 8 * (rowIdx <= 0) + 4 * (colIdx <= 0) + 2 * (rowIdx >= nrow - 1) + (colIdx >= ncol - 1);

	switch (condition) {
	case 0:	// all sides valid
		return (img.at<uchar>(rowIdx - 1, colIdx) ? 1 : 0) + (img.at<uchar>(rowIdx - 1, colIdx + 1) ? 2 : 0) +
			(img.at<uchar>(rowIdx, colIdx + 1) ? 4 : 0) + (img.at<uchar>(rowIdx + 1, colIdx + 1) ? 8 : 0) +
			(img.at<uchar>(rowIdx + 1, colIdx) ? 16 : 0) + (img.at<uchar>(rowIdx + 1, colIdx - 1) ? 32 : 0) +
			(img.at<uchar>(rowIdx, colIdx - 1) ? 64 : 0) + (img.at<uchar>(rowIdx - 1, colIdx - 1) ? 128 : 0);
	case 1:	// right side not valid
		return (img.at<uchar>(rowIdx - 1, colIdx) ? 1 : 0) + (img.at<uchar>(rowIdx + 1, colIdx) ? 16 : 0) +
			(img.at<uchar>(rowIdx + 1, colIdx - 1) ? 32 : 0) + (img.at<uchar>(rowIdx, colIdx - 1) ? 64 : 0) +
			(img.at<uchar>(rowIdx - 1, colIdx - 1) ? 128 : 0);
	case 2:	// bottom not valid
		return (img.at<uchar>(rowIdx - 1, colIdx) ? 1 : 0) + (img.at<uchar>(rowIdx - 1, colIdx + 1) ? 2 : 0) +
			(img.at<uchar>(rowIdx, colIdx + 1) ? 4 : 0) + (img.at<uchar>(rowIdx, colIdx - 1) ? 64 : 0) +
			(img.at<uchar>(rowIdx - 1, colIdx - 1) ? 128 : 0);
	case 3:	// bottom and right not valid
		return (img.at<uchar>(rowIdx - 1, colIdx) ? 1 : 0) + (img.at<uchar>(rowIdx, colIdx - 1) ? 64 : 0) +
			(img.at<uchar>(rowIdx - 1, colIdx - 1) ? 128 : 0);
	case 4:	// left side not valid
		return (img.at<uchar>(rowIdx - 1, colIdx) ? 1 : 0) + (img.at<uchar>(rowIdx - 1, colIdx + 1) ? 2 : 0) +
			(img.at<uchar>(rowIdx, colIdx + 1) ? 4 : 0) + (img.at<uchar>(rowIdx + 1, colIdx + 1) ? 8 : 0) +
			(img.at<uchar>(rowIdx + 1, colIdx) ? 16 : 0);
	case 5:	// left and right sides not valid
		return (img.at<uchar>(rowIdx -1, colIdx) ? 1 : 0) + (img.at<uchar>(rowIdx + 1, colIdx) ? 16 : 0);
	case 6:	// left and bottom sides not valid
		return (img.at<uchar>(rowIdx - 1, colIdx) ? 1 : 0) + (img.at<uchar>(rowIdx - 1, colIdx + 1) ? 2 : 0) +
			(img.at<uchar>(rowIdx, colIdx + 1) ? 4 : 0);
	case 7:	// left, bottom and right sides not valid
		return (img.at<uchar>(rowIdx - 1, colIdx) ? 1 : 0);
	case 8:	// top side not valid
		return (img.at<uchar>(rowIdx, colIdx + 1) ? 4 : 0) + (img.at<uchar>(rowIdx + 1, colIdx + 1) ? 8 : 0) +
			(img.at<uchar>(rowIdx + 1, colIdx) ? 16 : 0) + (img.at<uchar>(rowIdx + 1, colIdx - 1) ? 32 : 0) +
			(img.at<uchar>(rowIdx, colIdx - 1) ? 64 : 0);
	case 9:	// top and right sides not valid
		return (img.at<uchar>(rowIdx + 1, colIdx) ? 16 : 0) + (img.at<uchar>(rowIdx + 1, colIdx - 1) ? 32 : 0) +
			(img.at<uchar>(rowIdx, colIdx - 1) ? 64 : 0);
	case 10:	// top and bottom sides not valid
		return (img.at<uchar>(rowIdx, colIdx + 1) ? 4 : 0) + (img.at<uchar>(rowIdx, colIdx - 1) ? 64 : 0);
	case 11:	// top, bottom and right sides not valid
		return (img.at<uchar>(rowIdx, colIdx - 1) ? 64 : 0);
	case 12:	// top and left not valid
		return (img.at<uchar>(rowIdx, colIdx + 1) ? 4 : 0) + (img.at<uchar>(rowIdx + 1, colIdx + 1) ? 8 : 0) +
			(img.at<uchar>(rowIdx + 1, colIdx) ? 16 : 0);
	case 13:	// top, left and right sides not valid
		return (img.at<uchar>(rowIdx + 1, colIdx) ? 16 : 0);
	case 14:	// top, left and bottom sides not valid
		return (img.at<uchar>(rowIdx, colIdx + 1) ? 4 : 0);
	case 15:	// no sides valid
		return 0;
	}
}

// 完成
void findPathNLength(Mat skeletonPath, Point point1, Point point2, Mat& path, int& len) {
	Mat D1 = bwdistgeodesic(skeletonPath, point1.y, point1.x);
	Mat D2 = bwdistgeodesic(skeletonPath, point2.y, point2.x);

	Mat D = D1 + D2;

	Mat Dround = Mat::zeros(D.size(), CV_64FC1);
	for (int row = 0; row < D.rows; row++) {
		double * rowPt = D.ptr<double>(row);
		double * rowPtDround = Dround.ptr<double>(row);
		for (int col = 0; col < D.cols; col++) {
			rowPtDround[col] = round(rowPt[col] * 8) / 8;
		}
	}

	imregionalmin(Dround, path, len);
	return;
}

// 获取测地距离
Mat bwdistgeodesic(const Mat src, double cols, double rows)
{
	Mat distance = Mat::ones(src.size(), CV_64FC1) * (INT_MAX);
	Point point(cols, rows);
	queue<Point> next;
	next.push(point);
	distance.at<double>(point) = 0;

	Point curPos;
	int i = 0, j = 0;
	while (!next.empty())//BFS
	{
		curPos = next.front();
		next.pop();
		//for a point  : col-->x  row-->y				for mat  :  at function visit by  (  row , col )
		//处理当前点的八领域  
		const uchar* mask_data = NULL;
		double* dist_data = NULL;
		for (i = curPos.y - 1; i <= curPos.y + 1; ++i)
		{
			if (i == -1 || i == src.rows) continue;
			j = 0;
			mask_data = src.ptr<uchar>(i);
			dist_data = distance.ptr<double>(i);
			for (j = curPos.x - 1; j <= curPos.x + 1; ++j)
			{
				if (j == -1 || j == src.cols) continue;
				if ((int(mask_data[j]) != 0) && (int(dist_data[j] == INT_MAX)))//当前点属于地且未处理
				{
					const uchar* mask_data2 = NULL;
					double* dist_data2 = NULL;
					double temp = 10000;
					int ii, jj, mini = -2, minj = -2;

					for (ii = -1; ii <= 1; ++ii) {
						if (i + ii == -1 || i + ii == src.rows) continue;

						mask_data2 = src.ptr<uchar>(i + ii);
						dist_data2 = distance.ptr<double>(i + ii);
						for (jj = -1; jj <= 1; ++jj) {
							if (j + jj == -1 || j + jj == src.cols) continue;
							if (dist_data2[j + jj] != -1 && dist_data2[j + jj] <= temp) {
								temp = dist_data2[j + jj];
								mini = ii;
								minj = jj;
							}
						}
					}
					if (temp != 10000)
						dist_data[j] = temp + sqrt(mini*mini + minj * minj);
					else
						dist_data[j] = temp;
					next.push(Point(j, i));
				}
			}
		}
	}

	return distance;
}

// 获取局部最小值
void imregionalmin(Mat src, Mat& path, int& len) {
	double min = 10000;
	for (int row = 0; row < src.rows; row++) {
		double * rowPt = src.ptr<double>(row);
		for (int col = 0; col < src.cols; col++) {
			if (rowPt[col] < min) min = rowPt[col];
		}
	}

	for (int row = 0; row < src.rows; row++) {
		double * rowPt = src.ptr<double>(row);
		uchar * rowPtpath = path.ptr<uchar>(row);
		for (int col = 0; col < src.cols; col++) {
			if (rowPt[col] == min) rowPtpath[col] = 255;
			else
				rowPtpath[col] = 0;
		}
	}
	len = countNonZero(path);
	return;
}

vector<cuttingListStru> extractCutPointPairs(ConnectedRegion cutPointRegionProps) {
	int numRegions = cutPointRegionProps.connNum_;
	int numPairs = numRegions * (numRegions - 1) / 2;

	vector<cuttingListStru> cutPointPairs(numPairs, { Point(0,0), Point(0,0), 0.0 });

	int count = 1;
	for (int i = 1; i <= numRegions - 1; i++) {
		for (int j = i + 1; j <= numRegions; j++) {
			vector<Point> temp1 = cutPointRegionProps.pixelList_[i - 1];
			vector<Point> temp2 = cutPointRegionProps.pixelList_[j - 1];
			cuttingListStru temp = pDist2(temp1, temp2);

			cutPointPairs[count - 1] = temp;
			count = count + 1;
		}
	}

	sort(cutPointPairs.begin(), cutPointPairs.end(), compDistAscend);
	return cutPointPairs;
}

// 获得能够切割骨架的切割点组合
vector<cuttingListStru> reduceCuttingList(vector<cuttingListStru> cutPointPairs, Mat skel) {
	
	vector<cuttingListStru> validCutPointPairs;
	
	int numPairs = cutPointPairs.size();

	for (int i = 0; i < numPairs; i++) {
		Mat cutLine = Mat::zeros(skel.size(), CV_8UC1);
		Point tempPoint1 = cutPointPairs[i].point1;
		Point tempPoint2 = cutPointPairs[i].point2;
		
		// MATLAB 中 drawLine 函数，是两次操作，一次画线，一次膨胀
		line(cutLine, Point(tempPoint1.y, tempPoint1.x), Point(tempPoint2.y, tempPoint2.x), Scalar::all(255));
		Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
		morphologyEx(cutLine, cutLine, MORPH_DILATE, element);

		Mat skelCut; bitwise_and(cutLine, skel, skelCut);
		if (countNonZero(skelCut) > 0) {
			validCutPointPairs.push_back(cutPointPairs[i]);
		}
	}
	return validCutPointPairs;
}

Mat cut(vector<cuttingListStru> cutPointPairs, Mat originalObjI, Mat objMask, 
	double globalAvg, double minArea, String logic) {
	Mat obj1;

	if (logic == "NoPalePath") {
		for (int i = 0; i < cutPointPairs.size(); i++) {
			cuttingListStru cp = cutPointPairs[i];
			Mat cutLine = findCutLine(cp, originalObjI, objMask);
			obj1 = objMask.clone();

			bitwise_and(obj1, 0, obj1, cutLine);

			Mat obj1AreaOpen = bwareaopen(obj1, minArea, 4);
			obj1 = obj1AreaOpen;

			ConnectedRegion cc(obj1, 4);
			if (cc.connNum_ > 1) {
				break;
			}
		}
	}
	else {
		for (int i = 0; i < cutPointPairs.size(); i++) {
			cuttingListStru cp = cutPointPairs[i];
			Mat cutLine = findCutLine(cp, originalObjI, objMask);

			//namedWindow("cutLine", CV_WINDOW_FREERATIO);
			//imshow("cutLine", cutLine);
			//waitKey(0);

			Mat notObjMask; bitwise_not(objMask, notObjMask);
			bitwise_and(cutLine, 0, cutLine, notObjMask);

			//namedWindow("cutLineNot", CV_WINDOW_FREERATIO);
			//imshow("cutLineNot", cutLine);
			//waitKey(0);

			int sumObjMask = countNonZero(objMask);
			int sumCutLine = countNonZero(cutLine);
			double objAvg = (double)sum(originalObjI & objMask)[0] / 255.0 / sumObjMask;
			double avg = (double)sum(originalObjI & cutLine)[0] / 255.0 / sumCutLine;

			if (avg > 2 * globalAvg || avg > 2 * objAvg) {
				obj1 = objMask.clone();
				bitwise_and(obj1, 0, obj1, cutLine);

				obj1 = bwareaopen(obj1, minArea, 4);
				ConnectedRegion cc(obj1, 4);

				if (cc.connNum_ > 1)
					break;
			}
			else {
				continue;
			}
		}
	}
	return obj1;
}

void fitExtention(Point& pointEx, Point point1, Point ref, Size imgSize) {
	if (pointEx.x < 1) {
		pointEx.x = 1;
		pointEx.y = round((double)(1 - point1.x) / (point1.x - ref.x) * (point1.y - ref.y) + point1.y);
	}

	if (pointEx.x > imgSize.height) {
		pointEx.x = imgSize.height;
		pointEx.y = round((double)(imgSize.height - point1.x) / (point1.x - ref.x) * (point1.y - ref.y) + point1.y);
	}

	if (pointEx.y < 1) {
		pointEx.y = 1;
		pointEx.x = round((double)(1 - point1.y) / (point1.y - ref.y) * (point1.x - ref.x) + point1.x);
	}

	if (pointEx.y > imgSize.width) {
		pointEx.y = imgSize.width;
		pointEx.x = round((double)(imgSize.width - point1.y) / (point1.y - ref.y) * (point1.x - ref.x) + point1.x);
	}
	return;
}

vector<cuttingListStru> findClosestToReference(ConnectedRegion cutPointProps, vector<Point> refBp) {
	int numCutPoints = cutPointProps.connNum_;
	int numCutPointPair = numCutPoints * (numCutPoints - 1) / 2;

	vector<cuttingListStru> cutPointPairs;

	if (numCutPoints >= 2) {
		for (int i = 1; i <= numCutPoints - 1; i++) {
			for (int j = i + 1; j <= numCutPoints; j++) {
				vector<Point> temp1 = cutPointProps.pixelList_[i - 1];
				cuttingListStru pointDist1 = pDist2(temp1, refBp);

				vector<Point> temp2 = cutPointProps.pixelList_[j - 1];
				cuttingListStru pointDist2 = pDist2(temp2, refBp);

				cuttingListStru element;
				element.point1 = pointDist1.point1;
				element.point2 = pointDist2.point1;
				element.dist = pointDist1.dist + pointDist2.dist;

				cutPointPairs.push_back(element);
			}
		}

		sort(cutPointPairs.begin(), cutPointPairs.end(), compDistAscend);
	}

	return cutPointPairs;
}

double angle3points(Point point1, Point point2, Point point3) {
	Point v1 = point1 - point2;
	Point v2 = point2 - point3;

	double detV1V2 = v1.x * v2.y - v1.y * v2.x;
	double dotV1V2 = v1.x * v2.x + v1.y * v2.y;

	double atanDetDot = atan2(detV1V2, dotV1V2);
	double angle = atanDetDot - 2 * PI * floor(atanDetDot / (2 * PI));
	angle = (double)(ABS((angle > (PI / 2) ? 1 : 0) * PI - angle)) * 180.0 / PI;

	return angle;
}

Mat findAngleChanges(Mat line, Point startPoint) {
	Mat angleChanges = Mat::zeros(line.size(), CV_8UC1);

	int numLinePoints = countNonZero(line);

	int jumpAngle = 5;
	if (numLinePoints <= jumpAngle) {
		return angleChanges;
	}

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(line, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
	vector<Point> linePoints = contours[0];

	if (numLinePoints > linePoints.size()) {		// 这种情况已经不正常了
		return angleChanges;
	}

	vector<Point> linePointsCut(linePoints.begin(), linePoints.begin() + numLinePoints);
	// 排序
	//int idx = 0;
	//for (; idx < numLinePoints; idx++) {
	//	if (linePointsCut[idx] == Point(startPoint.y, startPoint.x)) {
	//		break;
	//	}
	//}
	//vector<Point> linePointsCutSorted(linePointsCut);
	//copy(linePointsCut.begin() + idx, linePointsCut.end(), linePointsCutSorted.begin());
	//copy(linePointsCut.begin(), linePointsCut.begin() + idx, linePointsCutSorted.begin() + numLinePoints - idx);
	for (int i = 1 + jumpAngle; i <= numLinePoints - jumpAngle; i++) {
		//Point p1 = linePointsCutSorted[i - jumpAngle - 1];
		//Point p2 = linePointsCutSorted[i - 1];
		//Point p3 = linePointsCutSorted[i + jumpAngle - 1];
		Point p1 = linePointsCut[i - jumpAngle - 1];
		Point p2 = linePointsCut[i - 1];
		Point p3 = linePointsCut[i + jumpAngle - 1];

		double angleOut = angle3points(p1, p2, p3);

		if (angleOut > 50 && angleOut < 130) {
			Point p = linePointsCut[i - 1];
			angleChanges.at<uchar>(p.y, p.x) = 255;
		}
	}

	//namedWindow("insideAngleChanges", CV_WINDOW_FREERATIO);
	//imshow("insideAngleChanges", angleChanges);
	//waitKey(0);

	Mat element = (Mat_<int>(3, 3) << 1, 1, 1, 
										1, 1, 1, 
										1, 1, 1);
	morphologyEx(angleChanges, angleChanges, MORPH_DILATE, element);

	Mat notLine; bitwise_not(line, notLine);
	bitwise_and(angleChanges, 0, angleChanges, notLine);

	return angleChanges;
}