#include "stdafx.h"
#include "conn.h"
#include "main.h"

// 构造函数定义
ConnectedRegion::ConnectedRegion(Mat BW, int connectivity) {
	Mat label, stats, centroids;
	int connNum = connectedComponentsWithStats(BW, label,
		stats, centroids, connectivity);

	// 类成员赋值
	connNum_ = connNum - 1;
	connectivity_ = connectivity;
	// 矩阵成员直接空间共享，不用深复制（copyTo或clone）
	BW_ = BW;
	label_ = label;
	stats_ = stats;
	centroids_ = centroids;

	pixelIdxListTrue_ = false;
	pixelListTrue_ = false;
	imageTrue_ = false;
	convexImageTrue_ = false;
}

// 析构函数定义
ConnectedRegion::~ConnectedRegion() {	
}

// 成员函数定义
void ConnectedRegion::calculatePixelIdxList() {
	if (connNum_ == 0) {
		return;
	}

	if (!pixelIdxListTrue_) {
		vector<vector<int>> pixelIdxList(connNum_, vector<int>(1, 0));
		for (int row = 0; row < label_.rows; row++) {
			int * rowPt = label_.ptr<int>(row);
			for (int col = 0; col < label_.cols; col++) {
				int labelValue = rowPt[col];

				for (int index = 1; index <= connNum_; index++) {
					if (labelValue == index) {
						pixelIdxList[index - 1].push_back(row + col * label_.cols);
					}
				}
			}
		}

		for (int index = 1; index <= connNum_; index++) {
			pixelIdxList[index - 1].erase(pixelIdxList[index - 1].begin());
		}

		pixelIdxList_ = pixelIdxList;
		pixelIdxListTrue_ = true;
	}
	
}

void ConnectedRegion::calculatePixelList() {
	if (connNum_ == 0) {
		return;
	}

	if (!pixelListTrue_) {
		vector<vector<Point>> pixelList(connNum_, vector<Point>(1, Point(0, 0)));
		for (int row = 0; row < label_.rows; row++) {
			int * rowPt = label_.ptr<int>(row);
			for (int col = 0; col < label_.cols; col++) {
				int labelValue = rowPt[col];

				for (int index = 1; index <= connNum_; index++) {
					if (labelValue == index) {
						pixelList[index - 1].push_back(Point(row, col));
					}
				}
			}
		}

		for (int index = 1; index <= connNum_; index++) {
			pixelList[index - 1].erase(pixelList[index - 1].begin());
		}

		pixelList_ = pixelList;
		pixelListTrue_ = false;
	}
}

void ConnectedRegion::calculateArea() {
	if (connNum_ == 0) {
		return;
	}

	vector<int> area(connNum_);
	if(!pixelListTrue_)
		calculatePixelList();		// attention!!!
	for (int index = 1; index <= connNum_; index++) {
		vector<Point> element = pixelList_[index - 1];
		area[index - 1] = element.size();
	}

	area_ = area;
}

void ConnectedRegion::calculateImage() {
	if (connNum_ == 0) {
		return;
	}

	if (!imageTrue_) {
		vector<Mat> image;
		if (!pixelListTrue_)
			calculatePixelList();		// attention!!!
		for (int i = 1; i <= connNum_; i++) {
			vector<Point> list = pixelList_[i - 1];
			int minC = list[0].x;
			int maxC = list[0].x;
			int minR = list[0].y;
			int maxR = list[0].y;
			for (int j = 1; j < list.size(); j++) {
				if (list[j].x < minC) minC = list[j].x;
				if (list[j].x > maxC) maxC = list[j].x;
				if (list[j].y < minR) minR = list[j].y;
				if (list[j].y > maxR) maxR = list[j].y;
			}
			Range R1;
			Range R2;
			R1.start = minR;
			R1.end = maxR + 1;
			R2.start = minC;
			R2.end = maxC + 1;

			Mat label = Mat::Mat(label_, R2, R1);
			Mat currlabel = label == i;

			image.push_back(currlabel);
		}

		image_ = image;
		imageTrue_ = true;
	}
}

void ConnectedRegion::calculateConvexImage() {
	if (connNum_ == 0) {
		return;
	}

	if (!convexImageTrue_) {
		vector<Mat> convexImage;
		if (!imageTrue_)
			calculateImage();
		for (int i = 1; i <= connNum_; i++) {
			Mat connImage = image_[i - 1];
			Mat convexImageElement = Mat::zeros(connImage.size(), CV_8UC1);

			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;
			findContours(connImage, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

			vector<vector<Point>>hull(contours.size());
			Scalar color = Scalar(255, 255, 255);
			for (int i = 0; i < contours.size(); i++)
			{
				convexHull(Mat(contours[i]), hull[i], false);
				drawContours(convexImageElement, contours, i, color, -1, 8, vector<Vec4i>(), 0, Point());
				drawContours(convexImageElement, hull, i, color, -1, 8, vector<Vec4i>(), 0, Point());
			}
			convexImage.push_back(convexImageElement);
		}

		convexImageTrue_ = true;
		convexImage_ = convexImage;
	}
}

void ConnectedRegion::calculateConvexArea() {
	if (connNum_ == 0) {
		return;
	}

	vector<int> convexArea(connNum_);
	if (!convexImageTrue_)
		calculateConvexImage();
	for (int index = 1; index <= connNum_; index++) {
		Mat convexImage = convexImage_[index - 1];
		convexArea[index - 1] = countNonZero(convexImage);
	}

	convexArea_ = convexArea;
}


void ConnectedRegion::calculateBoundingBox() {
	if (connNum_ == 0) {
		return;
	}

	vector<vector<int>> boundingBox(connNum_, vector<int>(4, 0));
	if (!pixelListTrue_)
		calculatePixelList();
	
	for (int index = 1; index <= connNum_; index++) {
		vector<Point> list = pixelList_[index - 1];
		vector<int> bb(4, 0);
		
		int width = 0, height = 0;
		int minC = list[0].x;
		int maxC = list[0].x;
		int minR = list[0].y;
		int maxR = list[0].y;
		for (int j = 1; j < list.size(); j++) {
			if (list[j].x < minC) minC = list[j].x;
			if (list[j].x > maxC) maxC = list[j].x;
			if (list[j].y < minR) minR = list[j].y;
			if (list[j].y > maxR) maxR = list[j].y;
		}

		maxC = maxC + 1;
		maxR = maxR + 1;
		width = maxC - minC;
		height = maxR - minR;

		bb[0] = minC; bb[1] = minR;
		bb[2] = width; bb[3] = height;
		boundingBox[index - 1] = bb;
	}

	boundingBox_ = boundingBox;
}

void ConnectedRegion::calculateOrientation() {
	if (connNum_ == 0) {
		return;
	}

	vector<double> orientation(connNum_, 0.0);

	if (!pixelListTrue_)
		calculatePixelList();
	for (int idx = 1; idx <= connNum_; idx++) {
		vector<Point> list = pixelList_[idx - 1];
		int length = list.size();

		double xbar = centroids_.at<double>(idx, 0);
		double ybar = centroids_.at<double>(idx, 1);

		double sumX = 0.0, sumY = 0.0, sumXY = 0.0;
		for (int j = 0; j < length; j++) {
			double xx = (list[j].y - xbar) * (list[j].y - xbar);
			double yy = (list[j].x - ybar) * (list[j].x - ybar);
			double xy = -(list[j].y - xbar) * (list[j].x - ybar);

			sumX = sumX + xx; sumY = sumY + yy; sumXY = sumXY + xy;
		}

		double uxx = sumX / length + (double)1 / 12;
		double uyy = sumY / length + (double)1 / 12;
		double uxy = sumXY / length;

		double num = 0.0, den = 0.0;
		if (uyy > uxx) {
			num = uyy - uxx + sqrt((uyy - uxx)*(uyy - uxx) + 4 * uxy * uxy);
			den = 2 * uxy;
		}
		else {
			num = 2 * uxy;
			den = uxx - uyy + sqrt((uxx - uyy)*(uxx - uyy) + 4 * uxy * uxy);
		}
		if (abs(num) < 1e-8 && abs(den) < 1e-8)
			orientation[idx - 1] = 0;
		else
			orientation[idx - 1] = (180 / PI) * atan(num / den);
	}
	orientation_ = orientation;
}