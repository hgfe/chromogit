#pragma once
#include <opencv2/opencv.hpp>
#include <numeric>

class ConnectedRegion;

using namespace cv;
using namespace std;

#define targetH 1200
#define targetW 1600

#define PI 3.1415926

Mat imgUniform(const Mat imgGray, int& resizeH, int& resizeW);
Mat imFill(const Mat BW);

Mat clearBorder(const Mat BW);
vector<int> imhist(Mat &srcImage, unsigned int n = 256);
void stretchlim(Mat& src, Mat& lowHigh, double tol_low, double tol_high);
void imadjust(Mat& src, Mat& dst, Mat& lowHighIn, Mat&lowHighOut, double gamma);

// skeleton �����������
enum direction { North, South, East, West, None };
const direction dircode[16] = {
	None, West, North, West, East, None, North, West,
	South, South, None, South, East, East, North, None
};
#define SQR(x) (x)*(x)
#define MIN(x,y) (((x) < (y)) ? (x):(y))
#define MAX(x,y) (((x) > (y)) ? (x):(y))
#define ABS(x) (((x) < 0) ? (-(x)):(x))
#define MOD(x,n) (((x)%(n)<0) ? ((x)%(n)+(n)):((x)%(n)))
Mat skeleton(const Mat BW);
int jointNeighborhood(const Mat img, const int rowIdx, const int colIdx);
void quickSort(vector<int>& arr, int n);
// //////////////////////

Mat ThiningDIBSkeleton(Mat BW);			// �� MATLAB ����ȫһ��

// innerCutting ������ؽṹ�뺯������
struct cuttingListStru {
	Point point1;
	Point point2;
	double dist;
};
Mat getCutPoints(Mat objMask, double paramCurv, double paramAngle, String logic);			// ��������
cuttingListStru pDist2(const vector<Point>pointList1, const vector<Point>pointList2);		// ��������
bool compDistAscend(const cuttingListStru & a, const cuttingListStru & b);					// ��������
Mat findCutLine(cuttingListStru cuttingListEle, Mat originalObjI, Mat objMask);				// �ƺ���������
Mat drawThinLine(Point point1, Point point2, Size imgSize);									// ��������
Mat bwareaopen(const Mat BW, const int threshold, const int conn);
Mat innerCutting(Mat objMask, Mat originalObjI, Mat innerPointsMap, double globalAvg, double minArea);
// /////////////////////////////////////

void findClusters(Mat BW, Mat & singles, Mat & clusters, Mat & bwThin);	// ���Ի������󣬹Ǽ���ƫ��
Mat imreconstruct(Mat marker, Mat mask);

// anaskel ���������������������
const int connected_nbrs[256] = {
	0,1,1,1,1,1,1,1,
	1,2,2,2,1,1,1,1,
	1,2,2,2,1,1,1,1,
	1,2,2,2,1,1,1,1,
	1,2,2,2,2,2,2,2,
	2,3,3,3,2,2,2,2,
	1,2,2,2,1,1,1,1,
	1,2,2,2,1,1,1,1,
	1,1,2,1,2,1,2,1,
	2,2,3,2,2,1,2,1,
	1,1,2,1,1,1,1,1,
	1,1,2,1,1,1,1,1,
	1,1,2,1,2,1,2,1,
	2,2,3,2,2,1,2,1,
	1,1,2,1,1,1,1,1,
	1,1,2,1,1,1,1,1,
	1,1,2,1,2,1,2,1,
	2,2,3,2,2,1,2,1,
	2,2,3,2,2,1,2,1,
	2,2,3,2,2,1,2,1,
	2,2,3,2,3,2,3,2,
	3,3,4,3,3,2,3,2,
	2,2,3,2,2,1,2,1,
	2,2,3,2,2,1,2,1,
	1,1,2,1,2,1,2,1,
	2,2,3,2,2,1,2,1,
	1,1,2,1,1,1,1,1,
	1,1,2,1,1,1,1,1,
	1,1,2,1,2,1,2,1,
	2,2,3,2,2,1,2,1,
	1,1,2,1,1,1,1,1,
	1,1,2,1,1,1,1,1
};
const int nbr_branches[256] = {
	0,1,1,1,1,2,1,2,
	1,2,2,2,1,2,2,2,
	1,2,2,2,2,3,2,3,
	1,2,2,2,2,3,2,3,
	1,2,2,2,2,3,2,3,
	2,3,3,3,2,3,3,3,
	1,2,2,2,2,3,2,3,
	2,3,3,3,2,3,3,3,
	1,2,2,2,2,3,2,3,
	2,3,3,3,2,3,3,3,
	2,3,3,3,3,4,3,4,
	2,3,3,3,3,4,3,4,
	1,2,2,2,2,3,2,3,
	2,3,3,3,2,3,3,3,
	2,3,3,3,3,4,3,4,
	2,3,3,3,3,4,3,4,
	1,1,2,2,2,2,2,2,
	2,2,3,3,2,2,3,3,
	2,2,3,3,3,3,3,3,
	2,2,3,3,3,3,3,3,
	2,2,3,3,3,3,3,3,
	3,3,4,4,3,3,4,4,
	2,2,3,3,3,3,3,3,
	3,3,4,4,3,3,4,4,
	1,2,2,2,2,3,2,3,
	2,3,3,3,2,3,3,3,
	2,3,3,3,3,4,3,4,
	2,3,3,3,3,4,3,4,
	2,2,3,3,3,3,3,3,
	3,3,4,4,3,3,4,4,
	2,3,3,3,3,4,3,4,
	3,3,4,4,3,4,4,4,
};
int neighborhood(const Mat img, const int rowIdx, const int colIdx);
Mat doctrim(Mat skel);
void anaskel(Mat skel, vector<Point>& endPoints, vector<Point>& junctions);
// ///////////////////////

// ��ؾ���任�ĺ�������������
Mat bwdistgeodesic(const Mat src, double  cols, double rows);
// /////////////////////////////////////

// ��ȡ�ֲ���Сֵ
void imregionalmin(Mat src, Mat& path, int& len);											// ��������

void findPathNLength(Mat skeletonPath, Point point1, Point point2, Mat& path, int& len);	// ��������

vector<cuttingListStru> extractCutPointPairs(ConnectedRegion cutPointRegionProps);		// ���Ի�������

vector<cuttingListStru> reduceCuttingList(vector<cuttingListStru> cutPointPairs, Mat skel);		// ���Ի�������

Mat cut(vector<cuttingListStru> cutPointPairs, Mat originalObjI, Mat objMask, 
	double globalAvg, double minArea, String logic);	// ��������

void extendSkeleton(Mat objMask, Mat& skel, vector<Point>& ep, vector<Point>& bp);		// ���Ի�������

void fitExtention(Point& pointEx, Point point1, Point ref, Size imgSize);	// ���Ի�������

vector<cuttingListStru> findClosestToReference(ConnectedRegion cutPointProps, vector<Point> refBp);	// ��������

double angle3points(Point point1, Point point2, Point point3);		// ��������

Mat findAngleChanges(Mat line, Point startPoint);					// ���Ի�������
// ������
Mat cutTouching(Mat objMask, Mat originalObjI, Mat cutPointsMap, double globalAvg, double avgThicknes, double minArea);

Mat imrotate(Mat src, double angle, String model);					// ��������

// ���¾�Ϊ������
vector<vector<Mat>> separateMultipleOverlapped2new(Mat obj_mask, Mat obj_img, double globalAvg, double minArea);
vector<Mat> findEnd2EndPathOnSkeleton2(Mat skel, vector<Point> ep);
void findPointMuiltipleCluster(Mat obj_mask, Mat cutPoints_map, Mat skel, vector<Point> bp, vector<Point> ep,
	Mat& CutPoints_map, vector<cuttingListStru>& cutPoint_Pairs, vector<Mat>& commonArea);
Mat splitSkeleton(Mat skel, int k, vector<Point> bp);
void findCutPoints(Mat cutPoints_map, Mat obj_mask, Mat skel, vector<Point> ep, vector<Point>bp,
	Mat& CutPoints_map, vector<cuttingListStru>& cutPoint_pairs);
double meanpdist2(vector<Point> pointList1, vector<Point> pointList2);
vector<Point> extentdPoints(vector<Point>points1, Point ref, double dist, Size imgSize);

void dec2bin(int num, int size, Mat & str);
void findPointsOnSegments_forAngleCalc(Mat skel, double radius, vector<Point>& points_forAngleCalc, vector<Point>& ep, vector<Point>& bp);