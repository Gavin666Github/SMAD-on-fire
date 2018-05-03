
#include<opencv/cv.h>
#include<opencv/highgui.h>
#include <math.h>
#include <iostream>  
#include<opencv2\opencv.hpp>  
#include"opencv2/xfeatures2d.hpp"

#define min(x,y) (x<y?x:y)
#define R_THRESHHOLD 100
#define S_THRESHHOLD 40
using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;


void colorModel(IplImage *src, IplImage * dst) {
	int step = NULL;
	int rows = src->height;
	int cols = src->width;
	for (int i = 0; i < rows; i++) {
		//uchar* dataS = src.ptr<uchar>(i);
		//uchar* dataD = dst.ptr<uchar>(i);
		uchar *dataS = (uchar*)src->imageData;
		uchar *dataD = (uchar*)dst->imageData;
		for (int j = 0; j < cols; j++) {
			step = i*src->widthStep + j*src->nChannels;
			float S;
			float b = dataS[step] / 255.0;
			float g = dataS[step + 1] / 255.0;
			float r = dataS[step + 2] / 255.0;
			float minRGB = min(min(r, g), b);
			float den = r + g + b;
			if (den == 0)	
				S = 0;
			else
				S = (1 - 3 * minRGB / den) * 100;
			if (dataS[step + 2] <= R_THRESHHOLD) {
				dataD[step] = 0;
				dataD[step + 1] = 0;
				dataD[step + 2] = 0;
			}
			else if (dataS[step + 2] <= (dataS[step + 1] + 5) || dataS[step + 1] <= (dataS[step] + 5)) {
				dataD[step] = 0;
				dataD[step + 1] = 0;
				dataD[step + 2] = 0;
			}
			else if (S <= (float)(S_THRESHHOLD*(255 - dataS[step + 2])) / R_THRESHHOLD) {
				dataD[step] = 0;
				dataD[step + 1] = 0;
				dataD[step + 2] = 0;
			}
			else {
				dataD[step] = dataS[step];
				dataD[step + 1] = dataS[step + 1];
				dataD[step + 2] = dataS[step + 2];
			}
		}
	}
}


void fillSeg(IplImage *src, IplImage *tempdst)
{
	CvSeq * contour = NULL;
	CvMemStorage * storage = cvCreateMemStorage();
	//CV_CHAIN_APPROX_SIMPLE - 压缩水平、垂直和对角分割
	cvFindContours(src, storage, &contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	cvZero(tempdst);
	for (contour; contour != 0; contour = contour->h_next)
	{
		
		double area = cvContourArea(contour, CV_WHOLE_SEQ);
		
		if (fabs(area) < 50)
		{
			continue;
		}
		//	CvScalar color = CV_RGB( 255, 255, 255 );
		CvPoint *point = new CvPoint[contour->total];
		CvPoint *Point;

		//printf("图像分割contour->total\t%d\n",contour->total);
		for (int i = 0; i<contour->total; i++)
		{
			Point = (CvPoint*)cvGetSeqElem(contour, i);
			point[i].x = Point->x;
			point[i].y = Point->y;
		}
		int pts[1] = { contour->total };
		cvFillPoly(tempdst, &point, pts, 1, CV_RGB(255, 255, 255));//填充多边形内部 
	}
}


IplImage *firecut(IplImage *img)
{

	//初始化
	IplImage *colTemp = NULL;	
	IplImage *gray = NULL;		
	IplImage *mask = NULL;		
	IplImage *dst = NULL;		

	colTemp = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
	gray = cvCreateImage(cvGetSize(img), img->depth, 1);
	mask = cvCreateImage(cvGetSize(img), img->depth, 1);
	dst = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);		//经过填补后的火焰图片
	cvZero(dst);

	colorModel(img, colTemp);
	cvCvtColor(colTemp, gray, CV_BGR2GRAY);

	fillSeg(gray, mask);
	cvCopy(img, dst, mask);

	//cvShowImage("原始图片", img);
	//cvShowImage("颜色分割处理", colTemp);
	//cvShowImage("填充处理图片", dst);
	//cvShowImage("mask", mask);
	imwrite("lcut.png", cvarrToMat(dst));
	return dst;
}




void distance(Mat imgL, Mat imgR)
{

	Mat a = imgL;
	Mat b = imgR;

	//surf初始化
	Ptr<SURF> surf;
	surf = SURF::create(800);       //阈值  
	BFMatcher matcher;                //匹配器  
	Mat c, d;
	vector<KeyPoint> key1, key2;
	vector<DMatch> matches;

	//结果为一个Mat矩阵，它的行数与特征点向量中元素个数是一致的。每行都是一个N维描述子的向量  
	surf->detectAndCompute(a, Mat(), key1, c);      //检测关键点和匹配描述子  
	surf->detectAndCompute(b, Mat(), key2, d);

	matcher.match(c, d, matches);         // 匹配，得到匹配向量  

	sort(matches.begin(), matches.end());  // 匹配点排序  
	vector< DMatch > good_matches;            // 匹配两幅图像的描述子  
	int ptsPairs = min(40, (int)(matches.size() * 0.15));
	cout << ptsPairs << endl;

	int i = 0, j = 0;
	while (i < ptsPairs)       // 将匹配较好的特征点存入good_matches中  
	{
		Point2f Pleft = key1[matches[j].queryIdx].pt;
		Point2f Pright = key2[matches[j].trainIdx].pt;
		float dy = Pleft.y - Pright.y;
		float dx = Pleft.x - Pright.x;
		if ((fabs(dy) < 13))
		{
			good_matches.push_back(matches[j]);
			i++;
		}

		j++;
	}
	Mat outimg;
	drawMatches(                               // 绘制匹配点  
		a,                                    // 原图像1  
		key1,                                 // 原图像1的特征点  
		b,                                    // 原图像2  
		key2,                                 // 原图像2的特征点  
		good_matches,                         // 原图像1的特征点匹配原图像2的特征点[matches[i]]  
		outimg,                               // 输出图像具体由flags决定  
		Scalar::all(-1),                    // 匹配的颜色（特征点和连线),若matchColor==Scalar::all(-1)，颜色随机  
		Scalar::all(-1),                    // 单个点的颜色，即未配对的特征点，若matchColor==Scalar::all(-1)，颜色随机  
		vector<char>(),                       // Mask决定哪些点将被画出，若为空，则画出所有匹配点  
		DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);  //Fdefined by DrawMatchesFlags  

	namedWindow("匹配图", 0);
	imshow("匹配图", outimg);

	//相机标定参数
	float db = -100;
	float f_x = 390.0;
	float f_y = 390.0;
	float u0 = 305;
	float v0 = 310;

	//vector<Point2f> Pleft;
	//vector<Point2f> Pright;

	//遍历匹配点，计算距离平均值
	double sum = 0;
	for (size_t i = 0; i < good_matches.size(); i++)
	{
		//good_matches[i].queryIdx保存着第一张图片匹配点的序号，keypoints_1[good_matches[i].queryIdx].pt.x 为该序号对应的点的x坐标  
		Point2f Pleft = key1[good_matches[i].queryIdx].pt;
		Point2f Pright = key2[good_matches[i].trainIdx].pt;

		Point3f point;
		point.x = db*(Pleft.x - u0) / (Pleft.x - Pright.x);
		point.y = db*f_x*(Pleft.y - v0) / (f_y*(Pleft.x - Pright.x));
		point.z = db*f_x / (Pleft.x - Pright.x);
		//cout << "position is " << point << endl;
		sum += point.z;
	}
	sum = sum / good_matches.size();
	cout << "deption is " << sum << endl;

	waitKey();
}

void main()
{
	IplImage *imge = NULL;
	imge = cvLoadImage("l1c.jpg");
	firecut(imge);
	Mat u1 = imread("lcut.png");
	imge = cvLoadImage("r1c.jpg");
	firecut(imge);
	Mat u2 = imread("lcut.png");

	distance(u1, u2);

}
