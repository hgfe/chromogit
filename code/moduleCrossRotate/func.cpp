// all the functions in project Chromosome
// 20190112
#include "stdafx.h"
#include "main.h"
#include "conn.h"

// 1.直方图匹配
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

// 2.粗分割
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

// 3.预处理，标准化
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
		vector<int> histo = imhist(scalingImgGray);
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
	Mat t;
	roughSegChromoRegion(BW3, BW3, t);
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
			if ((double)s.area_[idx] / s.convexArea_[idx] > 0.85) {
				vector<Point> pixelList = s.pixelList_[idx];
				for (int i = 0; i < pixelList.size(); i++) {
					darkForeGroundFilled.at<uchar>(pixelList[i].x, pixelList[i].y) = 0;
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

	//1223新增

	Mat BW5 = Mat::zeros(img2.size(), CV_8UC1);
	threshold(img2, BW5, 160, 255, THRESH_BINARY);
	bitwise_not(BW5, BW5);
	Mat BW5en = Mat::zeros(BW5.size(), CV_8UC1);
	calcEnvelope(BW5, BW5en);

	Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));
	morphologyEx(BW5en, BW5en, MORPH_DILATE, element);

	bitwise_not(BW5en, BW5en);
	bitwise_or(img2, 255, img2, BW5en);

	return img2;
}

// 4.空洞填充
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

// 5.清除与边界相连的连通域，没有按照 matlab 原函数原理实现
Mat clearBorder(const Mat BW) {
	Mat BW1 = BW.clone();

	rectangle(BW1, Rect(0, 0, BW1.cols, BW1.rows), Scalar(255));
	floodFill(BW1, Point(0, 0), Scalar(0));

	return BW1;
}

// 6.直方图统计
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

// 7.对比度拉伸
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

// 8.对比度调整
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

// 9.二值图像提取骨架
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

// 10.提取连通域信息
int jointNeighborhood(const Mat img, const int rowIdx, const int colIdx) {
	// 输入 img 图像中指定 rowIdx 和 colIdx 位置的像素点
	// 返回这个像素点的 8 连通区域的信息
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

// 11.整型容器快速排序
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

// 12.细化算法
Mat ThiningDIBSkeleton(Mat BW) {
	// 基于索引表的细化算法
	// 功能：对图象进行细化，即 MATLAB 中 bwdist(BW, 'skel', inf)
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
			rowPt[col] = lpDIBBits[row * lWidth + col] > 0 ? 255 : 0;
		}
	}
	delete(lpDIBBits);
	return skeleton;
}

// 13.内部裁剪
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

// 14.获取切割点
Mat getCutPoints(Mat objMask, double paramCurv, double paramAngle, String logic) {
	Mat cutPointsMap;

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(objMask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
	if (contours.empty())
		return cutPointsMap;

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

		if (objMask.at<uchar>(round((y1 + y3) / 2) - 1, round((x1 + x3) / 2) - 1) == 0) {
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

			if ((angleOut > paramAngle) && (angleOut <= 180 - 1.5 * paramAngle)) {
				Point coor = boundaryPoints[idx - 1 - jumpAngle];
				angleChanges.at<uchar>(coor.y, coor.x) = 255;
			}
		}
	}
	morphologyEx(angleChanges, angleChanges, MORPH_DILATE, element);

	/////////////////////////////////////////////////////////////////////
	//////////////////specify cut points/////////////////////////////////
	if (logic == "and")
		bitwise_and(curvaturePoints, angleChanges, cutPointsMap);
	else if (logic == "or")
		bitwise_or(curvaturePoints, angleChanges, cutPointsMap);

	element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	Mat objMaskTmp, objMaskExteriorBoundary, notObjMaskExteriorBoundary;
	morphologyEx(objMask, objMaskTmp, MORPH_DILATE, element);
	subtract(objMaskTmp, objMask, objMaskExteriorBoundary);

	bitwise_and(cutPointsMap, objMaskExteriorBoundary, cutPointsMap);

	return cutPointsMap;
}

// 15.获取最短距离
cuttingListStru pDist2(const vector<Point>pointList1, const vector<Point>pointList2) {
	// 找两个vector<Point>之间最近的两个点 pdist2(points1, points2)
	// 返回结构体，保存两个点和其距离
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

// 16.结构体比较
bool compDistAscend(const cuttingListStru & a, const cuttingListStru & b) {
	return a.dist < b.dist;
}

// 17.寻找切割线
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

			int sumLine = countNonZero(line);

			Mat originalObjILine;
			bitwise_or(originalObjI, 255, originalObjILine, line);

			double newIntensity = (double)sum(originalObjILine)[0] / 255.0 / sumLine;
			if (newIntensity > pathIntensity) {
				pathIntensity = newIntensity;
				chosenLine = line.clone();
				chosenPoint = candidatesPointDist[idx].point1;
			}
		}

		bitwise_or(cutLine, chosenLine, cutLine);

		point1 = point2;
		point2 = chosenPoint;
		dist = norm(Mat(point1), Mat(point2));
	}
	return cutLine;
}

// 18.画细线
Mat drawThinLine(Point point1, Point point2, Size imgSize) {
	// 画线
	// 注意 dilate 和 hitmiss 操作的 kernel
	// 这里很奇怪地，OPENCV 和 MATLAB 的结果不一样
	// 因此做了一些修改
	// 还需要进一步测试确保任何情况下都相同
	Mat thinLine = Mat::zeros(imgSize, CV_8UC1);

	line(thinLine, Point(point1.y, point1.x), Point(point2.y, point2.x), Scalar::all(255));

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

		Mat maskDilated;
		morphologyEx(mask, maskDilated, MORPH_DILATE, element);
		bitwise_or(maskDilated, thinLine, thinLine);
	}
	return thinLine;
}

// 19.小连通区域删除
Mat bwareaopen(const Mat BW, const int threshold, const int conn) {
	// 这里实现的原理是找轮廓，轮廓包络面积小于 threshold 的则擦除
	// MATLAB 的原理，找连通域，连通域面积小于 threshold 的则擦除
	// 修改了
	Mat BW1 = Mat::zeros(Size(BW.cols, BW.rows), CV_8UC1);

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

// 20.寻找cluster
void findClusters(Mat BW, Mat & singles, Mat & clusters, Mat & bwThin) {
	Mat skr = skeleton(BW);

	threshold(skr, skr, 25, 255, THRESH_BINARY);
	bwThin = ThiningDIBSkeleton(skr);

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

	clusters = imreconstruct(bwBranches, BW);
	singles = BW - clusters;

	return;
}

// 21.重建
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

// 22.接触染色体切割
Mat cutTouching(Mat objMask, Mat originalObjI, Mat cutPointsMap, double globalAvg, double avgThicknes, double minArea) {

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

	ConnectedRegion objNum(obj1, 4);
	if (objNum.connNum_ < 2) {		// 只能是空的情况？ 上述情况没切成

									// 对有两个及以上切割点
		if (numCutPointRegions > 1) {
			// 只有一个骨架
			if (s.connNum_ == 1) {
				if (nEP < 3) {
					// 找到弯曲点

					Mat angleChanges;
					if (!ep.empty()) {
						angleChanges = findAngleChanges(skel, ep[0]);
					}
					else
						return obj1;
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
						bpLeft.erase(bpLeft.begin() + i);
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

// 23.延伸骨架
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
			if (contours.empty())
				return;
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
				CL = contour.size();

			// 向外延伸末端点，使其与obj_mask相交
			int L = 30;
			ep2[i] = ep[i] + (ep[i] - contour[CL - 1]) * (L / norm(Mat(ep[i]), Mat(contour[CL - 1])));
			fitExtention(ep2[i], ep[i], contour[CL - 1], objMask.size());

			Mat skelExt = Mat::zeros(skel.size(), CV_8UC1);

			line(skelExt, Point(ep2[i].y, ep2[i].x), Point(ep[i].y, ep[i].x), Scalar::all(255));

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
		}
	}
	return;
}

// 24.分析骨架
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

// 25.对输入的骨架进行剪枝
Mat doctrim(Mat skel) {
	// anaskel 中的子函数
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

// 26.邻域信息
int neighborhood(const Mat img, const int rowIdx, const int colIdx) {
	// 返回 img 中指定位置的像素的邻域信息
	// 12点方向起始顺时针
	// anaskel 和 doctrim 的子函数
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
		return (img.at<uchar>(rowIdx - 1, colIdx) ? 1 : 0) + (img.at<uchar>(rowIdx + 1, colIdx) ? 16 : 0);
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

// 27.获取路径长度
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

// 28.获取测地距离
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

// 29.获取局部最小值
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

// 30.寻找切割点对
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

// 31.获得能够切割骨架的切割点组合
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

// 32.切割
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

			Mat notObjMask; bitwise_not(objMask, notObjMask);
			bitwise_and(cutLine, 0, cutLine, notObjMask);

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

// 33.扩展
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

// 34.寻找最近点
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

// 35.计算三点角度
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

// 36.寻找角度变化
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
	for (int i = 1 + jumpAngle; i <= numLinePoints - jumpAngle; i++) {
		Point p1 = linePointsCut[i - jumpAngle - 1];
		Point p2 = linePointsCut[i - 1];
		Point p3 = linePointsCut[i + jumpAngle - 1];

		double angleOut = angle3points(p1, p2, p3);

		if (angleOut > 50 && angleOut < 130) {
			Point p = linePointsCut[i - 1];
			angleChanges.at<uchar>(p.y, p.x) = 255;
		}
	}

	Mat element = (Mat_<int>(3, 3) << 1, 1, 1,
		1, 1, 1,
		1, 1, 1);
	morphologyEx(angleChanges, angleChanges, MORPH_DILATE, element);

	Mat notLine; bitwise_not(line, notLine);
	bitwise_and(angleChanges, 0, angleChanges, notLine);

	return angleChanges;
}

// 37.旋转
Mat imrotate(Mat src, double angle, String model) {
	Point center(round(src.cols / 2), round(src.rows / 2));
	Mat rotate_model = getRotationMatrix2D(center, angle, 1);
	Mat dst;

	Rect bbox;
	bbox = RotatedRect(center, src.size(), angle).boundingRect();
	rotate_model.at<double>(0, 2) += bbox.width / 2 - center.x;
	rotate_model.at<double>(1, 2) += bbox.height / 2 - center.y;

	if (model == "bilinear") {
		warpAffine(src, dst, rotate_model, bbox.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar::all(255));
	}
	else {
		warpAffine(src, dst, rotate_model, bbox.size(), INTER_NEAREST, BORDER_CONSTANT, Scalar::all(0));
	}

	return dst;
}

// 38.分离重叠染色体
vector<vector<Mat>> separateMultipleOverlapped2new(Mat obj_mask, Mat obj_img, double globalAvg, double minArea) {
	Mat skr = skeleton(obj_mask);
	threshold(skr, skr, 25, 255, THRESH_BINARY);
	Mat skel = ThiningDIBSkeleton(skr);

	vector<Point> ep, bp;
	extendSkeleton(obj_mask, skel, ep, bp);

	vector<vector<Mat>> cut_comb_final;
	if (ep.empty())
		return cut_comb_final;

	vector<Mat> Ep2Ep_Path = findEnd2EndPathOnSkeleton2(skel, ep);

	Mat cutPoints_map = getCutPoints(obj_mask, 0.15, 30, "or");

	if (ep.size() == 2) {
		bool cutComplete = true;
		bool loop = true;
		int erodeVal = 3;

		Mat erodedCopy;
		while (loop) {
			Mat eroded;
			Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(erodeVal, erodeVal));
			morphologyEx(obj_mask, eroded, MORPH_ERODE, kernel);
			erodedCopy = eroded.clone();
			erodeVal = erodeVal + 1;

			ConnectedRegion objLabel(eroded, 8);

			if (objLabel.connNum_ > 1) {
				loop = false;
				cutComplete = true;
			}
			else if (erodeVal > 10) {
				loop = false;
				cutComplete = false;
			}
		}

		if (cutComplete) {
			vector<Mat> tempvector;
			for (int objLabelNum = 1; objLabelNum <= 2; objLabelNum++) {
				Mat dilated;
				Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(erodeVal, erodeVal));
				ConnectedRegion objLabel(erodedCopy, 8);
				Mat BW = objLabel.label_ == objLabelNum;
				morphologyEx(BW, dilated, MORPH_DILATE, kernel);
				tempvector.push_back(dilated);
			}
			cut_comb_final.push_back(tempvector);
		}
		else {
			ConnectedRegion cutPoint_RegionProps(cutPoints_map, 8);
			cutPoint_RegionProps.calculatePixelList();
			cutPoint_RegionProps.calculatePixelIdxList();
			if (cutPoint_RegionProps.connNum_ < 2)
				return cut_comb_final;
			else {
				vector<cuttingListStru> cutPointPairs = extractCutPointPairs(cutPoint_RegionProps);
				vector<cuttingListStru> validCutPointPairs = reduceCuttingList(cutPointPairs, skel);
				cutPointPairs = validCutPointPairs;
				if (!cutPointPairs.empty()) {
					for (int i = 0; i < cutPointPairs.size(); i++) {
						Mat cutobj = obj_mask.clone();
						Mat cutLine = findCutLine(cutPointPairs[i], obj_img, cutobj);
						bitwise_and(cutobj, 0, cutobj, cutLine);

						ConnectedRegion cutobjstat(cutobj, 8);
						cutobjstat.calculatePixelIdxList();
						cutobjstat.calculatePixelList();
						int num_chromos = cutobjstat.connNum_;

						// cut_comb_final[i]初始化
						vector<Mat> tempvector;
						for (int idx = 0; idx < num_chromos; idx++) {
							Mat tempmask = Mat::zeros(obj_mask.size(), CV_8UC1);
							tempvector.push_back(tempmask);
						}
						cut_comb_final.push_back(tempvector);

						for (int j = 0; j < num_chromos; j++) {
							Mat cutobj2 = Mat::zeros(obj_mask.size(), CV_8UC1);
							vector<Point> tempvectorpoint = cutobjstat.pixelList_[j];
							for (int k = 0; k < cutobjstat.pixelList_[j].size(); k++) {
								Point temppoint = tempvectorpoint[k];
								cutobj2.at<uchar>(temppoint.x, temppoint.y) = 255;
							}
							cut_comb_final[i][j] = cutobj2;
						}
					}
				}
			}
		}
	}
	/*
	else if (ep.size() == 4 && bp.size() <= 2) { // 交叉染色体的处理
	// 修改的地方：
	// 1.删除了 moduleCrossSeg.cpp 中 findskeleton 和 findskellengthorder
	// 两个函数的定义
	// 2.把上面两个函数的定义放在了 func.cpp 中
	// 3.在 main.h 中添加了上面两个函数的声明
	vector<Mat> skeletonStructEpEpPath = findSkelLengthOrder(skel, ep, bp);
	Mat cutPoints = getCutPoints(obj_mask, 0.15, 30, "or");
	if (bp.size() == 2) {
	double tempDist = norm(Mat(bp[0]), Mat(bp[1]));
	if (tempDist > 8) {
	return cut_comb_final;
	}
	}
	Mat Points;
	vector<cuttingListStru> cuttingList;
	vector<Mat> commonArea;
	findPointMuiltipleCluster(obj_mask, cutPoints, skel, bp, ep, Points, cuttingList, commonArea);
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

	vector<Mat> singleChroms;
	// 两个 vector 里面分别存了两个点
	vector<Point> cuttingListElement(2, Point(0, 0));
	vector<vector<Point>> cuttingLists11(2, cuttingListElement);
	//cuttingListElement = cuttingLists[0];
	cuttingLists11[0][0] = cutPointListX.pixelList_[a[0]][0];
	cuttingLists11[0][1] = cutPointListX.pixelList_[a[1]][0];
	cuttingLists11[1][0] = cutPointListX.pixelList_[a[2]][0];
	cuttingLists11[1][1] = cutPointListX.pixelList_[a[3]][0];

	Mat skeleton1;
	Mat tmpCutLines = Mat::zeros(obj_img.size(), CV_8UC1);
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
	Mat cutobj = obj_mask.clone();
	bitwise_and(cutobj, 0, cutobj, tmpCutLines);
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

	Mat singleMask = cutobj.clone();
	singleChroms.push_back(singleMask);

	vector<vector<Point>> cuttingLists12(2, cuttingListElement);
	//cuttingListElement = cuttingLists[0];
	cuttingLists12[0][0] = cutPointListX.pixelList_[a[0]][0];
	cuttingLists12[0][1] = cutPointListX.pixelList_[a[2]][0];
	cuttingLists12[1][0] = cutPointListX.pixelList_[a[1]][0];
	cuttingLists12[1][1] = cutPointListX.pixelList_[a[3]][0];

	Mat skeleton2;
	tmpCutLines = Mat::zeros(obj_img.size(), CV_8UC1);
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
	cutobj = obj_mask.clone();
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
	singleMask = cutobj.clone();
	singleChroms.push_back(singleMask);

	cut_comb_final.push_back(singleChroms);
	}
	*/
	else if (ep.size() < 5) {

		Mat CutPoints_map;
		vector<cuttingListStru> CutPoint_pairs;
		vector<Mat> commonArea;

		findPointMuiltipleCluster(obj_mask, cutPoints_map, skel, bp, ep,
			CutPoints_map, CutPoint_pairs, commonArea);

		if (CutPoints_map.empty() && CutPoint_pairs.empty())
			return cut_comb_final;
		else {
			vector<Mat> cut_comb;

			for (int skelidx = 0; skelidx < Ep2Ep_Path.size(); skelidx++) {
				Mat sub_skeleton = Ep2Ep_Path[skelidx];
				vector<cuttingListStru> cutPoints_valid = reduceCuttingList(CutPoint_pairs, sub_skeleton);

				// 剔除 valid
				vector<cuttingListStru> cutPoint_pairs1;
				for (int idx = 0; idx < CutPoint_pairs.size(); idx++) {
					cuttingListStru toBeCompared = CutPoint_pairs[idx];
					int flagSame = 0;
					for (int jdx = 0; jdx < cutPoints_valid.size(); jdx++) {
						cuttingListStru validElement = cutPoints_valid[jdx];
						if (validElement.point1 == toBeCompared.point1 && validElement.point2 == toBeCompared.point2) {
							flagSame = 1;
							break;
						}
					}
					if (flagSame == 0)
						cutPoint_pairs1.push_back(toBeCompared);
				}

				int num_left_cutPoint_pairs = cutPoint_pairs1.size();
				Mat cutobj = obj_mask.clone();
				for (int i = 0; i < num_left_cutPoint_pairs; i++) {
					Mat cutLine = Mat::zeros(cutobj.size(), CV_8UC1);
					Point pt1 = cutPoint_pairs1[i].point1;
					Point pt2 = cutPoint_pairs1[i].point2;
					line(cutLine, Point(pt1.y, pt1.x), Point(pt2.y, pt2.x), Scalar::all(255), 1, 4);
					bitwise_and(cutobj, 0, cutobj, cutLine);
				}

				Mat recutobj = imreconstruct(sub_skeleton, cutobj);
				cutobj = recutobj;
				medianBlur(cutobj, cutobj, 3);

				Mat cutobj_skr = skeleton(cutobj);
				Mat cutobj_skel;
				threshold(cutobj_skr, cutobj_skr, 25, 255, THRESH_BINARY);
				cutobj_skel = ThiningDIBSkeleton(cutobj_skr);

				vector<Point> cutobj_ep, cutobj_bp;
				anaskel(cutobj_skel, cutobj_ep, cutobj_bp);

				if (cutobj_ep.size() > 2 || cutobj_ep.empty()) {
					continue;
				}
				else {
					Mat angleChanges_map = findAngleChanges(cutobj_skel, cutobj_ep[0]);
					if (countNonZero(angleChanges_map) > 0)
						continue;

					cut_comb.push_back(cutobj.clone());
				}
			}

			if (!commonArea.empty()) {

				int num_comb = cut_comb.size();
				vector<int> intersection_index(num_comb, 0);
				for (int i = 0; i < num_comb; i++) {
					for (int j = 0; j < commonArea.size(); j++) {
						Mat intersection;
						bitwise_and(cut_comb[i], commonArea[j], intersection);
						if (countNonZero(intersection) > 0)
							intersection_index[i] += 1;
					}
				}
				for (int i = 0; i < num_comb; i++) {
					if (intersection_index[num_comb - i - 1] < (*max_element(intersection_index.begin(), intersection_index.end()))) {
						// cut_comb[num_comb - i - 1].release();
						vector<Mat>::iterator iter = cut_comb.begin() + num_comb - i - 1;
						cut_comb.erase(iter);
					}
				}

				int d = cut_comb.size();
				int k = commonArea.size();

				// dec to bin
				int pWidth = k, pHeight = pow(2, k);
				Mat p = Mat::zeros(Size(pWidth, pHeight), CV_8UC1);
				for (int idx = 0; idx < pHeight; idx++) {
					dec2bin(idx, k - 1, p);
				}

				int combos_accum_num = 0;
				for (int i = 0; i < d; i++) {
					Mat obj1 = obj_mask.clone();
					bitwise_and(obj1, 0, obj1, cut_comb[i]);

					vector<Mat> commonArea2_Masks(k, Mat::zeros(obj_mask.size(), CV_8UC1));
					for (int n = 0; n < k; n++) {
						bitwise_and(cut_comb[i], commonArea[n], commonArea2_Masks[n]);
					}
					int combo_j_start = 0;
					if (i == 0)
						combo_j_start = 0;
					else
						combo_j_start = 1;
					for (int j = combo_j_start; j < pow(2, k); j++) {
						Mat obj2 = obj1.clone();

						for (int m = 0; m < k; m++) {
							if (p.at<uchar>(j, m)) {
								bitwise_or(obj2, commonArea2_Masks[m], obj2);
							}
						}
						obj2 = imFill(obj2);
						ConnectedRegion ccobj2(obj2, 4);
						if (ccobj2.connNum_ == 1) {
							Mat skr2 = skeleton(obj2);
							Mat skel2;
							threshold(skr2, skr2, 25, 255, THRESH_BINARY);
							skel2 = ThiningDIBSkeleton(skr2);
							vector<Point> ep2, bp2;
							extendSkeleton(obj2, skel2, ep2, bp2);

							if ((ep.size() == 5 && bp.size() == 3) && (ep2.size() == 3)) {
								Mat cmignored;
								vector<Mat> cmaignored;
								findPointMuiltipleCluster(obj2, cutPoints_map, skel2, bp2, ep2, cmignored, CutPoint_pairs, cmaignored);

								if (!CutPoint_pairs.empty()) {
									obj2 = cut(CutPoint_pairs, obj_img, obj2, globalAvg, minArea, "NoPalePath");
								}
							}
						}

						ConnectedRegion s_obj2(obj2, 8);
						s_obj2.calculatePixelIdxList();
						s_obj2.calculatePixelList();
						int num_chromos = s_obj2.connNum_ + 1;
						combos_accum_num = combos_accum_num + 1;
						vector<Mat> chromosome_Masks(num_chromos, Mat::zeros(obj_mask.size(), CV_8UC1));
						chromosome_Masks[0] = cut_comb[i].clone();

						for (int c = 1; c < num_chromos; c++) {
							Mat tmp_img = Mat::zeros(obj_mask.size(), CV_8UC1);
							vector<Point> tempPts = s_obj2.pixelList_[c - 1];
							for (int idx = 0; idx < tempPts.size(); idx++) {
								Point tempPt = tempPts[idx];
								tmp_img.at<uchar>(tempPt.x, tempPt.y) = 255;
							}
							chromosome_Masks[c] = tmp_img.clone();
						}
						cut_comb_final.push_back(chromosome_Masks);
					}
				}
			}
			else {
				for (int i = 0; i < cut_comb.size(); i++) {
					Mat obj1 = obj_mask.clone();
					bitwise_and(obj1, 0, obj1, cut_comb[i]);
					Mat obj1open = bwareaopen(obj1, 50, 8);
					obj1 = obj1open;

					ConnectedRegion obj1_s(obj1, 8);
					obj1_s.calculatePixelIdxList();
					obj1_s.calculatePixelList();
					int num_chromos = obj1_s.connNum_ + 1;
					vector<Mat> chromosome_Masks(num_chromos, Mat::zeros(obj_mask.size(), CV_8UC1));
					chromosome_Masks[0] = cut_comb[i].clone();

					for (int j = 1; j < num_chromos; j++) {
						Mat obj2 = Mat::zeros(obj_mask.size(), CV_8UC1);
						for (int idx = 0; idx < obj1_s.pixelList_[j - 1].size(); idx++) {
							obj2.at<uchar>(obj1_s.pixelList_[j - 1][idx].x, obj1_s.pixelList_[j - 1][idx].y) = 255;
						}
						chromosome_Masks[j] = obj2.clone();
					}
					cut_comb_final.push_back(chromosome_Masks);
				}
			}
		}
	}

	int num_combos = cut_comb_final.size();
	vector<int> cut_comb_final_NumChromos(num_combos, 0);
	vector<int> cut_comb_final_NumBends(num_combos, 0);
	vector<double> cut_comb_final_AvgLens(num_combos, 0);
	for (int i = 0; i < num_combos; i++) {
		cut_comb_final_NumChromos[i] = cut_comb_final[i].size();
		double accumlength = 0.0;
		int bendnumber = 0;
		for (int j = 0; j < cut_comb_final_NumChromos[i]; j++) {
			Mat skr_tmp = skeleton(cut_comb_final[i][j]);
			threshold(skr_tmp, skr_tmp, 25, 255, THRESH_BINARY);
			Mat skel2_tmp = ThiningDIBSkeleton(skr_tmp);

			vector<Point> ep, bp;
			extendSkeleton(cut_comb_final[i][j], skel2_tmp, ep, bp);
			accumlength = accumlength + countNonZero(skel2_tmp);// easy
			if (ep.size() == 2) {
				Mat angleChanges = findAngleChanges(skel2_tmp, ep[0]);
				ConnectedRegion ccangch(angleChanges, 8);
				bendnumber = bendnumber + ccangch.connNum_;
			}
			else
				bendnumber = INT_MAX;

		}

		cut_comb_final_AvgLens[i] = accumlength / cut_comb_final_NumChromos[i];
		cut_comb_final_NumBends[i] = bendnumber;
	}

	// sortrows
	// change index
	if (!cut_comb_final.empty()) {
		for (int i = 0; i < cut_comb_final.size() - 1; i++) {
			for (int j = i + 1; j < cut_comb_final.size(); j++) {
				if (cut_comb_final_NumChromos[i] > cut_comb_final_NumChromos[j])
				{
					swap(cut_comb_final[i], cut_comb_final[j]);
				}
				else if (cut_comb_final_NumChromos[i] == cut_comb_final_NumChromos[j]) {
					if (cut_comb_final_NumBends[i] > cut_comb_final_NumBends[j]) {
						swap(cut_comb_final[i], cut_comb_final[j]);
					}
					else if (cut_comb_final_NumBends[i] == cut_comb_final_NumBends[j]) {
						if (cut_comb_final_AvgLens[i] < cut_comb_final_AvgLens[j]) {
							swap(cut_comb_final[i], cut_comb_final[j]);
						}
						else
							continue;
					}
					else
						continue;
				}
				else
					continue;
			}

		}
	}
	return cut_comb_final;
}
// 39.寻找骨架路径
vector<Mat> findEnd2EndPathOnSkeleton2(Mat skel, vector<Point> ep) {
	int num_pairs = int(ep.size()*(ep.size() - 1) / 2);
	vector<Mat> Ep2Ep_Path;
	vector<int> Ep2Ep_Length;
	for (int i = 0; i < ep.size() - 1; i++) {
		for (int j = i + 1; j < ep.size(); j++) {
			Mat temppath = Mat::zeros(skel.size(), CV_8UC1);
			int templen = 0;
			findPathNLength(skel, ep[i], ep[j], temppath, templen);
			Ep2Ep_Path.push_back(temppath);
			Ep2Ep_Length.push_back(templen);
		}
	}

	for (int idxi = 0; idxi < Ep2Ep_Path.size() - 1; idxi++) {
		for (int idxj = idxi + 1; idxj < Ep2Ep_Path.size(); idxj++) {
			if (Ep2Ep_Length[idxi] < Ep2Ep_Length[idxj]) {
				int templen = Ep2Ep_Length[idxj];
				Ep2Ep_Length[idxj] = Ep2Ep_Length[idxi];
				Ep2Ep_Length[idxi] = templen;

				Mat temppath = Ep2Ep_Path[idxj];
				Ep2Ep_Path[idxj] = Ep2Ep_Path[idxi];
				Ep2Ep_Path[idxi] = temppath;
				// exchange
			}
		}
	}
	return Ep2Ep_Path;
}

// 40.寻找multiple cluster的点
void findPointMuiltipleCluster(Mat obj_mask, Mat cutPoints_map, Mat skel, vector<Point> bp, vector<Point> ep, Mat& CutPoints_map, vector<cuttingListStru>& cutPoint_Pairs, vector<Mat>& commonArea) {
	// commonArea是个只有一个属性的struct

	Mat skeleton2;
	if (bp.size() == 4 && ep.size() == 6) {
		int k = 2;
		skeleton2 = splitSkeleton(skel, k, bp);

		ConnectedRegion skelprops(skeleton2, 8);
		skelprops.calculateImage();

		vector<Point> ep1, bp1, ep2, bp2;
		anaskel(skelprops.image_[0], ep1, bp1);
		anaskel(skelprops.image_[1], ep2, bp2);
		if (ep1.size() != 4 || ep2.size() != 4) {
			cutPoint_Pairs.clear();
			CutPoints_map.release();
			commonArea.clear();
			return;
		}
	}

	else if (bp.size() == 3 && ep.size() == 5) {
		int k = 1;
		skeleton2 = splitSkeleton(skel, k, bp);
		ConnectedRegion skelprops(skeleton2, 8);
		skelprops.calculateImage();
		vector<Point> ep1, bp1, ep2, bp2;
		anaskel(skelprops.image_[0], ep1, bp1);
		anaskel(skelprops.image_[1], ep2, bp2);
		if ((ep1.size() == 3 && ep2.size() == 4) || (ep1.size() == 4 && ep2.size() == 3)) {}
		else {
			cutPoint_Pairs.clear();
			CutPoints_map.release();
			commonArea.clear();
			return;
		}
	}

	else if ((bp.size() <= 2 && ep.size() == 4) || (bp.size() == 1 && ep.size() == 3)) {
		skeleton2 = skel.clone();
	}
	else {
		cutPoint_Pairs.clear();
		CutPoints_map.release();
	}

	ConnectedRegion skel_props(skeleton2, 8);
	skel_props.calculatePixelIdxList();
	skel_props.calculatePixelList();
	cutPoint_Pairs.clear();
	CutPoints_map = Mat::zeros(cutPoints_map.size(), CV_8UC1);
	int commonareanum = 0;
	for (int skel_num = 1; skel_num <= skel_props.connNum_; skel_num++) {
		Mat sub_skel = Mat::zeros(skeleton2.size(), CV_8UC1);
		vector<Point> tempPoints = skel_props.pixelList_[skel_num - 1];
		for (int idx = 0; idx < tempPoints.size(); idx++) {
			Point temp = tempPoints[idx];
			sub_skel.at<uchar>(temp.x, temp.y) = 255;
		}
		vector<Point> sub_skel_ep, sub_skel_bp;
		anaskel(sub_skel, sub_skel_ep, sub_skel_bp);

		Mat CutPoints_map_iter = Mat::zeros(cutPoints_map.size(), CV_8UC1);
		vector<cuttingListStru> cutPoint_Pairs_iter;

		findCutPoints(cutPoints_map, obj_mask, sub_skel, sub_skel_ep, sub_skel_bp,
			CutPoints_map_iter, cutPoint_Pairs_iter);

		// CutPoint_Pairs的连接
		cutPoint_Pairs.insert(cutPoint_Pairs.end(), cutPoint_Pairs_iter.begin(), cutPoint_Pairs_iter.end());


		bitwise_or(CutPoints_map, CutPoints_map_iter, CutPoints_map);

		ConnectedRegion ccPoints(CutPoints_map_iter, 8);
		int numPoints = ccPoints.connNum_;
		if (numPoints == 4) {
			Mat commarea = Mat::zeros(obj_mask.size(), CV_8UC1);
			if (cutPoint_Pairs_iter.size() == 6) {
				for (int i = 0; i < 6; i++) {
					Mat thinLine = Mat::zeros(obj_mask.size(), CV_8UC1);
					cuttingListStru temp = cutPoint_Pairs_iter[i];
					line(thinLine, Point(temp.point1.y, temp.point1.x), Point(temp.point2.y, temp.point2.x), Scalar::all(255));
					bitwise_or(thinLine, commarea, commarea);
				}
				Mat commareafilled = imFill(commarea);

				// 去轮廓
				Mat boundary = Mat::zeros(commareafilled.size(), CV_8UC1);
				vector<vector<Point>> contours;
				findContours(commareafilled, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
				for (int idx = 0; idx < contours.size(); idx++) {
					drawContours(boundary, contours, idx, Scalar::all(255), 1);
				}

				commarea = commareafilled - boundary;
				commonArea.push_back(commarea);
				commonareanum = commonareanum + 1;

			}
		}
	}

	return;
}

// 41.切割骨架
Mat splitSkeleton(Mat skel, int k, vector<Point> bp) {

	vector<double> bpDist;
	vector<Point> bpNum;
	int nBP = bp.size();
	for (int i = 0; i < nBP - 1; i++) {
		for (int j = i + 1; j < nBP; j++) {
			bpDist.push_back(norm(Mat(bp[i]), Mat(bp[j])));
			bpNum.push_back(Point(i, j));
		}
	}

	vector<double> bpDist2(bpDist);
	for (int i = 0; i < k; i++) {
		vector<double>::iterator minimum = min_element(begin(bpDist2), end(bpDist2));
		bpDist2.erase(minimum);
	}

	vector<double>::iterator minimum = min_element(begin(bpDist2), end(bpDist2));
	int index = distance(begin(bpDist), minimum);

	Point bp1 = bp[bpNum[index].x], bp2 = bp[bpNum[index].y];
	Mat path; int len;
	findPathNLength(skel, bp1, bp2, path, len);

	// 找trace
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(skel, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
	vector<Point> contourToBeSorted = contours[0];
	// 排序
	// 先把从第一个开始到 length 个元素拿出来
	int length = countNonZero(path);
	vector<Point> temp(length, Point(0, 0));
	copy(contourToBeSorted.begin(), contourToBeSorted.begin() + length, temp.begin());
	// 然后在 temp 里面排序
	Point compTemp(bp1.y, bp1.x);		// bp1 反转一下
	vector<Point> contour(length, Point(0, 0));
	for (int idx = 0; idx < length; idx++) {
		if (temp[idx] == compTemp) {
			copy(temp.begin() + idx, temp.end(), contour.begin());
			int leftLength = length - idx;
			copy(temp.begin(), temp.begin() + idx, contour.begin() + leftLength);
			break;
		}
	}

	// 取 contour 中间附近的点，并反转
	Point midPoint = contour[round(length / 2)];
	int mid = midPoint.y; midPoint.y = midPoint.x; midPoint.x = mid;

	Mat skeleton2 = skel.clone();
	Mat skeleton3 = Mat::zeros(skel.size(), CV_8UC1);
	skeleton3.at<uchar>(midPoint.x, midPoint.y) = 255;

	Mat SE = getStructuringElement(MORPH_RECT, Size(3, 3));
	morphologyEx(skeleton3, skeleton3, MORPH_DILATE, SE);

	bitwise_and(skeleton2, 0, skeleton2, skeleton3);

	return skeleton2;
}

// 42.寻找切割点
void findCutPoints(Mat cutPoints_map, Mat obj_mask, Mat skel, vector<Point> ep, vector<Point>bp,
	Mat& CutPoints_map, vector<cuttingListStru>& cutPoint_pairs) {
	if (bp.size() > 1) {
		double sumx = 0, sumy = 0;
		for (int idx = 0; idx < bp.size(); idx++) {
			sumx = sumx + bp[idx].x;
			sumy = sumy + bp[idx].y;
		}
		sumx = sumx / bp.size();
		sumy = sumy / bp.size();
		bp.clear();
		bp.push_back(Point(round(sumx), round(sumy)));
	}
	double dist = meanpdist2(bp, ep);
	vector<Point> ep2 = extentdPoints(ep, bp[0], dist, skel.size());

	Mat mask2 = Mat::zeros(obj_mask.size(), CV_8UC1);
	Mat skel_extend = skel.clone();

	for (int i = 0; i < ep.size(); i++) {
		line(skel_extend, Point(ep[i].y, ep[i].x), Point(ep2[i].y, ep2[i].x), Scalar::all(255));
	}
	double radius;
	if (dist > 40)
		radius = 20;
	else if (dist > 20)
		radius = 10;
	else
		radius = dist / 2;
	vector<Point> points;
	findPointsOnSegments_forAngleCalc(skel_extend, radius, points, ep2, bp);

	if (bp.size() > 1) {
		Point sum = Point(0, 0);
		for (int idx = 0; idx < bp.size(); idx++) {
			sum = sum + bp[idx];
		}
		double sumx = sum.x / bp.size();
		double sumy = sum.y / bp.size();
		bp.clear();
		bp.push_back(Point(round(sumx), round(sumy)));
	}
	Mat valid_CutPoints_map = Mat::zeros(obj_mask.size(), CV_8UC1);
	for (int i = 0; i < ep2.size() - 1; i++) {
		for (int j = i + 1; j < ep2.size(); j++) {
			Point p1 = ep2[i];
			Point p2 = bp[0];
			Point p3 = ep2[j];

			double ang = angle3points(points[i], p2, points[j]);

			if (ang > 45 && ang < 135) {
				vector<Point> points2;
				points2 = extentdPoints(points, bp[0], dist, skel.size());

				Mat mask = skel_extend.clone();
				line(mask, Point(p1.y, p1.x), Point(points2[i].y, points2[i].x), Scalar::all(255), 1, 4);
				line(mask, Point(p3.y, p3.x), Point(points2[j].y, points2[j].x), Scalar::all(255), 1, 4);

				Point cor;
				cor.x = points2[i].x + points2[j].x - p2.x;
				cor.y = points2[i].y + points2[j].y - p2.y;
				if (cor.x < 0)
					cor.x = 0;
				else if (cor.x >= obj_mask.rows)
					cor.x = obj_mask.rows - 1;

				if (cor.y < 0)
					cor.y = 0;
				else if (cor.y >= obj_mask.cols)
					cor.y = obj_mask.cols - 1;

				line(mask, Point(points2[i].y, points2[i].x), Point(cor.y, cor.x), Scalar::all(255));
				line(mask, Point(points2[j].y, points2[j].x), Point(cor.y, cor.x), Scalar::all(255));

				Mat mask_filled = imFill(mask);
				mask = mask_filled;
				mask2 = mask + mask2;

				Mat cutPoints_masked;
				bitwise_and(cutPoints_map, mask, cutPoints_masked);

				ConnectedRegion cutPoints_cc(cutPoints_masked, 8);
				if (cutPoints_cc.connNum_ == 1)
					bitwise_or(valid_CutPoints_map, cutPoints_masked, valid_CutPoints_map);
				else if (cutPoints_cc.connNum_ > 1) {
					cutPoints_cc.calculatePixelIdxList();
					cutPoints_cc.calculatePixelList();
					int num_props = cutPoints_cc.connNum_;

					vector<Point> center_points;
					for (int k = 1; k <= cutPoints_cc.connNum_; k++) {
						Point center = Point(
							cutPoints_cc.centroids_.at<double>(k, 1),
							cutPoints_cc.centroids_.at<double>(k, 0));
						center_points.push_back(center);
					}
					cuttingListStru dists = pDist2(center_points, bp);
					vector<Point>::iterator it;
					it = find(center_points.begin(), center_points.end(), Point(dists.point1));
					int min_Idx = distance(begin(center_points), it);
					for (int t = 0; t < cutPoints_cc.pixelList_[min_Idx].size(); t++) {
						valid_CutPoints_map.at<uchar>(cutPoints_cc.pixelList_[min_Idx][t].x, cutPoints_cc.pixelList_[min_Idx][t].y) = 255;
					}

				}
			}
		}
	}

	ConnectedRegion cutPoint_props(valid_CutPoints_map, 8);
	cutPoint_props.calculatePixelIdxList();
	cutPoint_props.calculatePixelList();
	cutPoint_pairs = findClosestToReference(cutPoint_props, bp);
	CutPoints_map = valid_CutPoints_map;

	return;
}

// 43.
double meanpdist2(vector<Point> pointList1, vector<Point> pointList2) {
	double sumdist = 0, meandist = 0, num = 0;

	for (int i = 0; i < pointList1.size(); i++) {
		for (int j = 0; j < pointList2.size(); j++) {
			Point temp1 = pointList1[i];
			Point temp2 = pointList2[j];
			sumdist = sumdist + norm(Mat(temp1), Mat(temp2));
			num = num + 1;
		}
	}
	meandist = sumdist / num;
	return meandist;
}

// 44.
vector<Point> extentdPoints(vector<Point>points1, Point ref, double dist, Size imgSize) {
	vector<Point> pointsEx;
	for (int i = 0; i < points1.size(); i++) {
		Point temppoint;
		if (norm(Mat(points1[i]), Mat(ref)) < dist) {
			temppoint.x = points1[i].x + round((points1[i].x - ref.x)*(dist / norm(Mat(points1[i]), Mat(ref))));
			temppoint.y = points1[i].y + round((points1[i].y - ref.y)*(dist / norm(Mat(points1[i]), Mat(ref))));
			fitExtention(temppoint, points1[i], ref, imgSize);
		}

		else
			temppoint = points1[i];

		if (temppoint.x < 0) {
			temppoint.x = 0;
		}
		if (temppoint.y < 0) {
			temppoint.y = 0;
		}
		if (temppoint.y >= imgSize.width) {
			temppoint.y = imgSize.width - 1;
		}
		if (temppoint.x >= imgSize.height) {
			temppoint.x = imgSize.height - 1;
		}
		pointsEx.push_back(temppoint);
	}

	return pointsEx;
}

// 45.
void dec2bin(int num, int size, Mat & str) {


	for (int i = size; i >= 0; i--)
	{
		if (num & (1 << i)) {
			str.at<uchar>(num, size - i) = 0;
		}
		else {
			str.at<uchar>(num, size - i) = 255;
		}
	}
	return;
}

// 46
void findPointsOnSegments_forAngleCalc(Mat skel, double radius, vector<Point>& points_forAngleCalc, vector<Point>& ep, vector<Point>& bp) {

	ep.clear(); bp.clear();
	anaskel(skel, ep, bp);
	int num_endpoints = ep.size();
	int num_branchpoints = bp.size();
	vector<int> Ep2Ep_lens(num_endpoints, 0);
	vector<int> corresponding_bp(num_endpoints, 0);
	vector<Mat> branch_img(num_endpoints, Mat::zeros(skel.size(), CV_8UC1));
	for (int i = 0; i < num_endpoints; i++) {
		vector<int> lengths(num_branchpoints, 0);
		vector<Mat> paths(num_branchpoints, Mat::zeros(skel.size(), CV_8UC1));
		for (int j = 0; j < num_branchpoints; j++) {
			Mat skeleton_path = Mat::zeros(skel.size(), CV_8UC1);
			int path_len = 0;
			findPathNLength(skel, ep[i], bp[j], skeleton_path, path_len);
			paths[j] = skeleton_path.clone();
			lengths[j] = path_len;
		}
		int min_len = *min_element(lengths.begin(), lengths.end());
		int index = min_element(lengths.begin(), lengths.end()) - lengths.begin();
		Ep2Ep_lens[i] = min_len;
		branch_img[i] = paths[index].clone();
		corresponding_bp[i] = index;
	}
	vector<Point> points(num_endpoints, Point(0, 0));
	points_forAngleCalc = points;
	for (int i = 0; i < num_endpoints; i++) {
		Size sz = skel.size();
		Point circle_center = bp[corresponding_bp[i]];
		double dist = norm(Mat(ep[i]), Mat(circle_center));
		double radius_s = radius;
		if (dist < radius_s)
			radius_s = dist;
		Mat circle_img = Mat::zeros(skel.size(), CV_8UC1);
		circle(circle_img, Point(circle_center.y, circle_center.x), radius_s, Scalar::all(255), 1, 4);
		Mat a;
		bitwise_and(circle_img, branch_img[i], a);

		ConnectedRegion a_props(a, 8);
		a_props.calculatePixelIdxList();
		a_props.calculatePixelList();
		if (a_props.connNum_ > 0) {
			vector<vector<Point>> pointa = a_props.pixelList_;
			Point point;
			if (pointa[0].size() > 1) {
				Point maxPoint = pointa[0][0];
				for (int k = 0; k < pointa[0].size(); k++) {
					if (norm(Mat(pointa[0][k]), Mat(circle_center)) > norm(Mat(maxPoint), Mat(circle_center)))
						maxPoint = pointa[0][k];
				}
				point = maxPoint;
			}
			else
				point = pointa[0][0];
			points_forAngleCalc[i] = point;
		}
		else
			points_forAngleCalc[i] = ep[i];
	}

	return;
}

// 47
void calcEnvelope(Mat src, Mat& dst) {
	if (src.size() != dst.size()) {
		dst = Mat::zeros(src.size(), CV_8UC1);
		return;
	}
	int rows = src.rows, cols = src.cols;
	int i = 0, j = 0;
	for (i = 0; i < rows; i++) {
		// 按行扫描
		int left = -1, right = -1;
		for (j = 0; j < cols; j++) {
			if (src.at<uchar>(i, j) > 0) {
				left = j;
				break;
			}
		}
		for (j = cols - 1; j > left + 1; j--) {
			if (src.at<uchar>(i, j) > 0) {
				right = j;
				break;
			}
		}
		if (left > 0 && right > 0 && left <= right) {
			for (j = left; j <= right; j++) {
				dst.at<uchar>(i, j) = 255;
			}
		}
	}
	for (j = 0; j < cols; j++) {
		// 按列扫描
		int up = -1, down = -1;
		for (i = 0; i < rows; i++) {
			if (src.at<uchar>(i, j) > 0) {
				up = i;
				break;
			}
		}
		for (i = rows - 1; i > up + 1; i--) {
			if (src.at<uchar>(i, j) > 0) {
				down = i;
				break;
			}
		}
		if (up > 0 && down > 0 && up <= down) {
			for (i = up; i <= down; i++) {
				dst.at<uchar>(i, j) = 255;
			}
		}
	}
	dst = imFill(dst);

}

// 48
Mat bwmorphSpur(Mat src, int n) {
	Mat dst = Mat::zeros(src.size(), CV_8UC1);
	if (n < 0)return dst;
	Mat lastdst = src.clone();
	for (int idx = 0; idx < n; idx++) {
		for (int i = 0; i < lastdst.rows; i++) {
			for (int j = 0; j < lastdst.cols; j++) {
				if ((i - 1 < 0) || (j - 1 < 0) || (i == lastdst.rows - 1) || (j == lastdst.cols - 1)) {
					dst.at<uchar>(i, j) = lastdst.at<uchar>(i, j);
					continue;
				}
				if (lastdst.at<uchar>(i, j) > 0) {
					int conn4 = 0, conn8 = 0;
					for (int vi = -1; vi <= 1; vi++) {
						for (int vj = -1; vj <= 1; vj++) {
							if (vi == 0 && vj == 0)continue;
							if ((vi == 0 || vj == 0) && lastdst.at<uchar>(i + vi, j + vj) > 0) conn4++;
							if (vi != 0 && vj != 0 && lastdst.at<uchar>(i + vi, j + vj) > 0) conn8++;
						}
					}
					if (conn4 >= 2 || conn8 >= 2) dst.at<uchar>(i, j) = 255;
					else if (conn4 == 0 && conn8 == 0) dst.at<uchar>(i, j) = 255;
					else if ((conn4 == 0 && conn8 == 1) || (conn4 == 1 && conn8 == 0)) dst.at<uchar>(i, j) = 0;
					else if (conn4 == 1 && conn8 == 1) {
						int conn4i = -2, conn4j = -2;
						int conn8i = -2, conn8j = -3;
						for (int vi = -1; vi <= 1; vi++) {
							for (int vj = -1; vj <= 1; vj++) {
								if (vi == 0 && vj == 0)continue;
								if ((vi == 0 || vj == 0) && lastdst.at<uchar>(i + vi, j + vj) > 0) {
									conn4i = vi;
									conn4j = vj;
								}

								if (vi != 0 && vj != 0 && lastdst.at<uchar>(i + vi, j + vj) > 0) {
									conn8i = vi;
									conn8j = vj;
								}
							}
						}
						if (conn4i == conn8i || conn4j == conn8j)
							dst.at<uchar>(i, j) = 0;
						else
							dst.at<uchar>(i, j) = 255;
					}
					else
						dst.at<uchar>(i, j) = 0;
				}
			}
		}
		lastdst = dst.clone();
	}
	return dst;
}

// 49
void thinningIteration(Mat& im, int iter) {
	/**
	* Perform one thinning iteration.
	* Normally you wouldn't call this function directly from your code.
	*
	* @param  im    Binary image with range = 0-1
	* @param  iter  0=even, 1=odd
	*/
	Mat marker = Mat::zeros(im.size(), CV_8UC1);

	for (int i = 1; i < im.rows - 1; i++)
	{
		for (int j = 1; j < im.cols - 1; j++)
		{
			uchar p2 = im.at<uchar>(i - 1, j);
			uchar p3 = im.at<uchar>(i - 1, j + 1);
			uchar p4 = im.at<uchar>(i, j + 1);
			uchar p5 = im.at<uchar>(i + 1, j + 1);
			uchar p6 = im.at<uchar>(i + 1, j);
			uchar p7 = im.at<uchar>(i + 1, j - 1);
			uchar p8 = im.at<uchar>(i, j - 1);
			uchar p9 = im.at<uchar>(i - 1, j - 1);

			int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
				(p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
				(p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
				(p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
			int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
			int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
			int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

			if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
				marker.at<uchar>(i, j) = 1;
		}
	}

	im &= ~marker;
}

// 50
void thinning(Mat& im) {
	/**
	* Function for thinning the given binary image
	*
	* @param  im  Binary image with range = 0-255
	*/
	im /= 255;

	Mat prev = Mat::zeros(im.size(), CV_8UC1);
	Mat diff;

	do {
		thinningIteration(im, 0);
		thinningIteration(im, 1);
		cv::absdiff(im, prev, diff);
		im.copyTo(prev);
	} while (cv::countNonZero(diff) > 0);

	im *= 255;
	Size originSize = im.size();
	Mat imdst = Mat::zeros(originSize, CV_8UC1);
	Mat tmp = im(Range(1, originSize.height - 1), Range(1, originSize.width - 1));
	tmp.copyTo(imdst(Range(1, originSize.height - 1), Range(1, originSize.width - 1)));
	im = imdst;
}

// 51
Mat cut3(vector<cuttingListStru> cutPointPairs, Mat originalObjI, Mat objMask,
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

			Mat notObjMask; bitwise_not(objMask, notObjMask);
			bitwise_and(cutLine, 0, cutLine, notObjMask);

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

// 52
Mat findSkeleton(Mat obj, int thresh, vector<Point> & ep, vector<Point> & bp) {
	Mat skr = skeleton(obj);

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
			if (contours.empty()) {
				ep.clear();
				bp.clear();
				return skr;
			}

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

				vector<Point> chosenPixelList = sExtCrossingPoints.pixelList_[chosen - 1];
				for (int n = 0; n < chosenPixelList.size(); n++) {
					tmp.at<uchar>(chosenPixelList[n].x, chosenPixelList[n].y) = 0;
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

// 53
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

// 54.预分割
void preSeg(Mat img, bool bIntensityReverse, bool bCutTouching,
	int & preSingleNum, int & preSingleArea, int & preSingleEnvelope) {
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

	// 1217新增
	Mat singlesForCalcArea = Mat::zeros(singlesForExtraction.size(), CV_8UC1);
	Mat singlesForCalcAreaTmp = singlesForExtraction.clone();
	for (int i = 0; i < 3; i++) {
		calcEnvelope(singlesForCalcAreaTmp, singlesForCalcArea);
		singlesForCalcAreaTmp = singlesForCalcArea.clone();
	}


	ConnectedRegion singlesCC(singlesForExtraction, 4);

	preSingleNum = singlesCC.connNum_;
	preSingleArea = countNonZero(singlesForExtraction);
	preSingleEnvelope = countNonZero(singlesForCalcArea);

	return;
}

// 55.打分
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

