#include "stdafx.h"
#include "main.h"
#include "conn.h"

// 预处理
Mat imgUniform(const Mat imgGray, int& resizeH, int& resizeW) {

	//////////////////////////////////////////////////////////////
	////////////// resize the image //////////////////////////////
	int origH = imgGray.rows;
	int origW = imgGray.cols;

	// resize 图像
	Size dsize = Size(targetW, targetH);
	Mat scalingImgGray;
	resize(imgGray, scalingImgGray, dsize, 0.0, 0.0, INTER_CUBIC);
	resizeH = scalingImgGray.rows;
	resizeW = scalingImgGray.cols;

	/////////////////////////////////////////////////////////////
	////////////// adjust the background to 255 /////////////////
	// median filter 中值滤波
	Mat imgGrayMf;
	medianBlur(scalingImgGray, imgGrayMf, 5);

	Mat img2 = scalingImgGray.clone();
	Mat darkForeGround = imgGrayMf < 255; // 得到的 darkForeGround 所有元素非0即255

	double darkForeRatio = (double)countNonZero(darkForeGround) / resizeH / resizeW;
	if (darkForeRatio > 0.2) { // 背景亮度不够
		darkForeGround = imgGrayMf < 204;	// 降低分割阈值
		medianBlur(darkForeGround, darkForeGround, 3);
		Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
		morphologyEx(darkForeGround, darkForeGround, MORPH_OPEN, element);
	}

	Mat BW3;
	bitwise_not(darkForeGround, BW3);
	bitwise_or(img2, BW3, img2); // 即 img2(BW3)=255。BW3为真的部分，则置为255，即或运算；BW3为假的部分，保留img2的值，仍然是或运算

								 /////////////////////////////////////////////////////////////////////////////////
								 ////////////// remove the nucleus with a round shape ////////////////////////////
	double limitedSize = 0.005 * resizeH * resizeW;
	Mat darkForeGroundFilled = imFill(darkForeGround);

	ConnectedRegion s(darkForeGroundFilled, 8);
	s.calculateConvexArea();
	s.calculateArea();

	for (int index = 1; index <= s.connNum_; index++) {
		int area = s.area_[index - 1];
		if (area > limitedSize) {
			int convexArea = s.convexArea_[index - 1];
			if ((double)area / convexArea > 0.9) {
				vector<Point> pixelList = s.pixelList_[index - 1];
				for (int idx = 0; idx < pixelList.size(); idx++) {
					int row = pixelList[idx].x;
					int col = pixelList[idx].y;
					darkForeGroundFilled.at<uchar>(row, col) = 0;
				}
			}
		}
	}

	bitwise_not(darkForeGroundFilled, BW3);
	bitwise_or(img2, BW3, img2);
	return img2;
}

// imFill 'holes'
// 空洞填充
// 应该是按照 matlab 原函数原理实现的
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