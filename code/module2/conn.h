#pragma once
#pragma once
#include "main.h"

class ConnectedRegion
{
private:
	//int connNum_;			// ��ͨ����Ŀ��������������
	Mat BW_;				// ����ȡ��ͨ��Ķ�ֵͼ��
	Mat label_;				// ��ע��ͨ��Ľ��ͼ�񣨱����б�ע��
	Mat stats_;				// ��֪��
							//Mat centroids_;		// ÿ����ͨ�����ĵ�����
	int connectivity_;		// ָ������ͨ��

							//vector<vector<int>> pixelIdxList_;		// ������ͨ�����е����������
	bool pixelIdxListTrue_;

	//vector<vector<Point2d>> pixelList_;		// ������ͨ�����е������
	bool pixelListTrue_;

	//vector<Mat> image_;						// ������ͨ���Сͼ��
	bool imageTrue_;

	//vector<Mat> convexImage_;				// ����С��ͨ���͹��
	bool convexImageTrue_;

	//vector<int> area_;						// ������ͨ�����
	//vector<int> convexArea_;				// ������ͨ��͹�����

public:
	int connNum_;							// ��ͨ����Ŀ��������������
	Mat centroids_;							// ÿ����ͨ�����ĵ�����
	vector<vector<int>> pixelIdxList_;		// ������ͨ�����е����������
	vector<vector<Point>> pixelList_;		// ������ͨ�����е������
	vector<Mat> image_;						// ������ͨ���Сͼ��
	vector<Mat> convexImage_;				// ����С��ͨ���͹��
	vector<int> area_;						// ������ͨ�����
	vector<int> convexArea_;				// ������ͨ��͹�����
	vector<vector<int>> boundingBox_;		// ������ͨ��İ������
											// �����ϽǶ�������͸߶ȺͿ��
											// �洢˳������Ϊ��
											// leftTopRow, leftTopCol, h, w
	vector<double> orientation_;			// ������ͨ���orientation

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