#pragma once
#pragma once
#include "main.h"

class ConnectedRegion
{
private:
	//int connNum_;			// 连通域数目（不包括背景）
	Mat BW_;				// 待提取连通域的二值图像
	Mat label_;				// 标注连通域的结果图像（背景有标注）
	Mat stats_;				// 不知道
							//Mat centroids_;		// 每个连通域中心的坐标
	int connectivity_;		// 指定几连通域

							//vector<vector<int>> pixelIdxList_;		// 各个连通域所有点的线性坐标
	bool pixelIdxListTrue_;

	//vector<vector<Point2d>> pixelList_;		// 各个连通域所有点的坐标
	bool pixelListTrue_;

	//vector<Mat> image_;						// 各个连通域的小图像
	bool imageTrue_;

	//vector<Mat> convexImage_;				// 各个小连通域的凸包
	bool convexImageTrue_;

	//vector<int> area_;						// 各个连通域面积
	//vector<int> convexArea_;				// 各个连通域凸包面积

public:
	int connNum_;							// 连通域数目（不包括背景）
	Mat centroids_;							// 每个连通域中心的坐标
	vector<vector<int>> pixelIdxList_;		// 各个连通域所有点的线性坐标
	vector<vector<Point>> pixelList_;		// 各个连通域所有点的坐标
	vector<Mat> image_;						// 各个连通域的小图像
	vector<Mat> convexImage_;				// 各个小连通域的凸包
	vector<int> area_;						// 各个连通域面积
	vector<int> convexArea_;				// 各个连通域凸包面积
	vector<vector<int>> boundingBox_;		// 各个连通域的包络矩形
											// 的左上角顶点坐标和高度和宽度
											// 存储顺序依次为：
											// leftTopRow, leftTopCol, h, w
	vector<double> orientation_;			// 各个连通域的orientation

	ConnectedRegion(Mat BW, int connectivity);
	~ConnectedRegion();
	void calculatePixelIdxList();
	void calculatePixelList();

	void calculateArea();

	void calculateImage();
	void calculateConvexImage();
	void calculateConvexArea();

	void calculateBoundingBox();

	void calculateOrientation();
};