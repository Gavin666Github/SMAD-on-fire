#include <iostream>
#include "cv.h"  
#include "opencv2/opencv.hpp" 
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include<math.h>
#include"opencv2/xfeatures2d.hpp"


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

#define min(x,y) (x<y?x:y)
#define R_THRESHHOLD 160
#define S_THRESHHOLD 115

int redThre = 49; // 115~135  
int saturationTh = 7; //55~65  
RNG g_rng(12345);
int g_nElementShape = MORPH_RECT;
Mat usedframe1;
Mat usedframe2;



Mat CheckColor(Mat &inImg);
int FT(Mat &inImg);
void DrawFire(Mat &inputImg, Mat foreImg);
void colorModel(IplImage *src, IplImage * dst);
void fillSeg(IplImage *src, IplImage *tempdst);
void distance(Mat imgL, Mat imgR);


int main()
{
	// 将视频帧转成图片输出  
	VideoCapture cap("C:\\Users\\a.mp4");

	
	long totalFrameNumber = cap.get(CV_CAP_PROP_FRAME_COUNT);
	cout << "total frames: " << totalFrameNumber << endl;

	Mat frame;
	bool flags = true;
	long currentFrame = 0;

	while (flags)
	{
		
		cap.read(frame);// 

		stringstream str;
		str << currentFrame << ".jpg";

		 
		if (currentFrame % 10 == 0) // 设置每10帧获取一次帧 
		{
			imwrite("C:\\Users\\" + str.str(), frame);// 将帧转成图片输出  
		}
		 
		if (currentFrame >= totalFrameNumber)
		{
			flags = false;
		}
		currentFrame++;
	}

	system("pause");

	for (int a = 0; a <40; a++)
	{
		string colorframename1 = format("C:\\Users\\%d.jpg", a * 10);
		string colorframename2 = format("C:\\Users\\%d.jpg", a * 10);
		usedframe1 = imread(colorframename1);
		usedframe2 = imread(colorframename2);
		string grayframename = format("gray%d.jpg", a * 10);
		imwrite(grayframename, CheckColor(usedframe1));//
		waitKey(0);
	}
	return 0;
}


Mat CheckColor(Mat &inImg)
{
	Mat fireImg;
	fireImg.create(inImg.size(), CV_8UC1);
	Mat multiRGB[3];
	int a = inImg.channels();
	split(inImg, multiRGB); 

	for (int i = 0; i < inImg.rows; i++)
	{
		for (int j = 0; j < inImg.cols; j++)
		{
			float B, G, R;
			B = multiRGB[0].at<uchar>(i, j); //动态地址计算法  
			G = multiRGB[1].at<uchar>(i, j);
			R = multiRGB[2].at<uchar>(i, j);

			float maxValue = max(max(B, G), R);
			float minValue = min(min(B, G), R);
			
			double S = (1 - 3.0*minValue / (R + G + B)); 
			if (R > redThre &&R >= G && G >= B && S >((255 - R) * saturationTh / redThre))
			{
				fireImg.at<uchar>(i, j) = 255;
			}
			else
			{
				fireImg.at<uchar>(i, j) = 0;
			}
		}
	}

	erode(fireImg, fireImg, Mat(3, 3, CV_8UC1));
	GaussianBlur(fireImg, fireImg, Size(5, 5), 0, 0);
	medianBlur(fireImg, fireImg, 5);
	dilate(fireImg, fireImg, Mat(5, 5, CV_8UC1));
	imshow("Binary", fireImg);
	int b = FT(fireImg);
	if (b == 1)
	{
		DrawFire(inImg, fireImg);
		return fireImg;
	}
	else if (b == 0)
	{
		return fireImg;
	}

}

int FT(Mat &inImg)
{
	Mat I;
	I.create(inImg.size(), CV_8UC1);

	Mat padded;                 
	int m = getOptimalDFTSize(I.rows);
	int n = getOptimalDFTSize(I.cols);

	
	copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(),CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);     

	dft(complexI, complexI);        //傅里叶变换

	split(complexI, planes);        //planes[0] = Re(DFT(I),planes[1] = Im(DFT(I))
									
	magnitude(planes[0], planes[1], planes[0]);     //planes[0] = magnitude
	Mat magI = planes[0];

	magI += Scalar::all(1);
	log(magI, magI);                //转换到对数尺度(logarithmic scale)

									//如果有奇数行或列，则对频谱进行裁剪
	magI = magI(Rect(0, 0, magI.cols&-2, magI.rows&-2));

	//重新排列傅里叶图像中的象限，使得原点位于图像中心
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	Mat q0(magI, Rect(0, 0, cx, cy));       //左上角图像划定ROI区域
	Mat q1(magI, Rect(cx, 0, cx, cy));      //右上角图像
	Mat q2(magI, Rect(0, cy, cx, cy));      //左下角图像
	Mat q3(magI, Rect(cx, cy, cx, cy));     //右下角图像

											//变换左上角和右下角象限
	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	//变换右上角和左下角象限
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	//归一化处理，用0-1之间的浮点数将矩阵变换为可视的图像格式
	normalize(magI, magI, 0, 1, CV_MINMAX);

	if (1)//输出傅里叶变换的平均灰度与灰度标准差
	{
		Mat gray, mat_mean, mat_stddev;
		meanStdDev(magI, mat_mean, mat_stddev);
		double m, s;
		m = mat_mean.at<double>(0, 0);
		s = mat_stddev.at<double>(0, 0);
		if (m>0.51 || s>0.19)
		{
			cout << "观测到疑似火焰区域" << endl;
			cout << "灰度均值是：" << m << endl;
			cout << "标准差是：" << s << endl;
			imshow("频谱图", magI);
			return 1;
		}
		else
		{
			cout << "未观测到疑似火焰区域" << endl;
			return 0;
		}

	}

	waitKey(0);
}

void DrawFire(Mat &inputImg, Mat foreImg)
{
	std::vector<std::vector<Point>> contours_set;//保存轮廓提取后的点集及拓扑关系  
	findContours(foreImg, contours_set, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	Point point1;
	Point point2;
	float a = 0.4, b = 0.75;
	float xmin1 = a*inputImg.cols, ymin1 = inputImg.rows, xmax1 = 0, ymax1 = 0;
	float xmin2 = b*inputImg.cols, ymin2 = inputImg.rows, xmax2 = a*inputImg.cols, ymax2 = 0;
	float xmin3 = inputImg.cols, ymin3 = inputImg.rows, xmax3 = b*inputImg.cols, ymax3 = 0;
	Rect finalRect1;
	Rect finalRect2;
	Rect finalRect3;

	std::vector<std::vector<Point>>  ::iterator iter = contours_set.begin();
	for (; iter != contours_set.end();)
	{
		Rect rect = boundingRect(*iter);
		float radius;
		Point2f center;
		minEnclosingCircle(*iter, center, radius);

		if (rect.area()> 0)
		{
			point1.x = rect.x;
			point1.y = rect.y;
			point2.x = point1.x + rect.width;
			point2.y = point1.y + rect.height;

			if (point2.x< a*inputImg.cols)
			{
				if (point1.x < xmin1)
					xmin1 = point1.x;
				if (point1.y < ymin1)
					ymin1 = point1.y;
				if (point2.x > xmax1 && point2.x < xmax2)
					xmax1 = point2.x;
				if (point2.y > ymax1)
					ymax1 = point2.y;
			}

			if (point2.x < b*inputImg.cols&&point2.x > a*inputImg.cols)
			{
				if (point1.x < xmin2 && point1.x>xmin1)
					xmin2 = point1.x;
				if (point1.y < ymin2)
					ymin2 = point1.y;
				if (point2.x > xmax2 && point2.x < xmax3)
					xmax2 = point2.x;
				if (point2.y > ymax2)
					ymax2 = point2.y;
			}

			if (point2.x < inputImg.cols&&point2.x > b*inputImg.cols)
			{
				if (point1.x < xmin3 && point1.x>xmin2)
					xmin3 = point1.x;
				if (point1.y < ymin3)
					ymin3 = point1.y;
				if (point2.x > xmax3)
					xmax3 = point2.x;
				if (point2.y > ymax3)
					ymax3 = point2.y;
			}

			++iter;
		}
		else
		{
			iter = contours_set.erase(iter);
		}

	}


	if (xmin1 == a*inputImg.cols&& ymin1 == inputImg.rows&&xmax1 == 0 && ymax1 == 0)
	{
		xmin1 = ymin1 = xmax1 = ymax1 = 0;
	}
	if (xmin2 == b*inputImg.cols&& ymin2 == inputImg.rows&& xmax2 == a*inputImg.cols&& ymax2 == 0)
	{
		xmin2 = ymin2 = xmax2 = ymax2 = 0;
	}
	if (xmin3 == inputImg.cols&&ymin3 == inputImg.rows&& xmax3 == b*inputImg.cols&& ymax3 == 0)
	{
		xmin3 = ymin3 = xmax3 = ymax3 = 0;
	}


	Mat imageROI1 = foreImg(Rect(xmin1, ymin1, xmax1 - xmin1, ymax1 - ymin1));
	Mat imageROI2 = foreImg(Rect(xmin2, ymin2, xmax2 - xmin2, ymax2 - ymin2));
	Mat imageROI3 = foreImg(Rect(xmin3, ymin3, xmax3 - xmin3, ymax3 - ymin3));//选出初步ROI
	finalRect1 = Rect(xmin1, ymin1, xmax1 - xmin1, ymax1 - ymin1);
	finalRect2 = Rect(xmin2, ymin2, xmax2 - xmin2, ymax2 - ymin2);
	finalRect3 = Rect(xmin3, ymin3, xmax3 - xmin3, ymax3 - ymin3);
	

	rectangle(inputImg, finalRect1, Scalar(0, 255, 0));
	rectangle(inputImg, finalRect2, Scalar(0, 255, 0));
	rectangle(inputImg, finalRect3, Scalar(0, 255, 0));
	imshow("Fire_Detection", inputImg);
	distance(usedframe1, usedframe2);

	//对初步ROI进行圆形度二次筛选
	/*Mat imageROI5 = inputImg(Rect(xmin2, ymin2, xmax2 - xmin2, ymax2 - ymin2));
	if (imageROI2.empty())
	{
		return;
	}
	else
	{

		Mat dstImage = Mat::zeros(imageROI2.rows, imageROI2.cols, CV_8UC3);
		Mat element = getStructuringElement(g_nElementShape,
			Size(5, 5), Point(-1, -1));
		morphologyEx(imageROI2, dstImage, MORPH_OPEN, element, Point(-1, -1), 2);
		vector<vector<Point>>contour;//用来储存轮廓
		vector<Vec4i>hierarchy;
		findContours(dstImage, contour, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		Mat drawing = Mat::zeros(dstImage.size(), CV_8UC3);
		for (unsigned int i = 0; i < contour.size(); i++)//计算圆形度
		{
			double factor = (contourArea(contour[i]) * 4 * CV_PI) /
				(pow(arcLength(contour[i], true), 2));
			if (factor > 0.2 && factor <0.8)
			{
				cout << "factor:" << factor << endl;
				Scalar color = Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255), g_rng.uniform(0, 255));
				drawContours(drawing, contour, i, color, 1, 8, vector<Vec4i>(), 0, Point());
				vector<Moments>mu(contour.size());//计算矩
				mu[i] = moments(contour[i], false);
				vector<Point2f>mc(contour.size());//计算矩中心
				mc[i] = Point2f(static_cast<float>(mu[i].m10 / mu[i].m00), static_cast<float>(mu[i].m01 / mu[i].m00));
				circle(drawing, mc[i], 5, Scalar(0, 0, 255), 1, 8, 0);
				rectangle(drawing, boundingRect(contour.at(i)), Scalar(0, 255, 0));
				imwrite("rectangles.jpg", drawing);
				Mat mask = imread("rectangles.jpg", 0);
				drawing.copyTo(imageROI5, mask);
				imshow("Contours", inputImg);
				
			}
		}
	}*/

	waitKey(0);
}

void colorModel(IplImage *src, IplImage * dst) 
{
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
			if (den == 0)	//分母不能为0
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

//图像分割_function2根据分割结果确定轮廓并填充
void fillSeg(IplImage *src, IplImage *tempdst)
{
	CvSeq * contour = NULL;
	CvMemStorage * storage = cvCreateMemStorage();
	//在二值图像中寻找轮廓,CV_CHAIN_APPROX_SIMPLE - 压缩水平、垂直和对角分割，即函数只保留末端的象素点
	cvFindContours(src, storage, &contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	cvZero(tempdst);
	for (contour; contour != 0; contour = contour->h_next)
	{
		//轮廓的方向影响面积的符号。因此函数也许会返回负的结果。应用函数 fabs() 得到面积的绝对值。 
		double area = cvContourArea(contour, CV_WHOLE_SEQ);
		//计算整个轮廓或部分轮廓的面积
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

//图像分割
IplImage *firecut(IplImage *img)
{

	//初始化
	IplImage *colTemp = NULL;	//颜色分割后(有内部空洞)的火焰图片
	IplImage *gray = NULL;		//灰度图
	IplImage *mask = NULL;		//二值图，用于复制图像的掩膜
	IplImage *dst = NULL;		//输出火焰疑似图像，8bit、3通道

	colTemp = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);//经过颜色分割后(有内部空洞)的火焰图片
	gray = cvCreateImage(cvGetSize(img), img->depth, 1);
	mask = cvCreateImage(cvGetSize(img), img->depth, 1);
	dst = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);		//经过填补后的火焰图片
	cvZero(dst);

	colorModel(img, colTemp);//function1（见上）
	cvCvtColor(colTemp, gray, CV_BGR2GRAY);

	fillSeg(gray, mask);//function2(见上)
	cvCopy(img, dst, mask);

	//cvShowImage("原始图片", img);
	//cvShowImage("颜色分割处理", colTemp);
	//cvShowImage("填充处理图片", dst);
	//cvShowImage("mask", mask);
	return dst;
}

void distance(Mat imgL, Mat imgR)
{

	Mat a = cvarrToMat(firecut(&IplImage(imgL)));
	Mat b = cvarrToMat(firecut(&IplImage(imgR)));

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
	int ptsPairs = min(25, (int)(matches.size() * 0.1));
	cout << ptsPairs << endl;
	for (int i = 0; i < ptsPairs; i++)       // 将匹配较好的特征点存入good_matches中  
	{
		good_matches.push_back(matches[i]);
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
	float db = -74.49;
	float f_x = 959.0;
	float f_y = 960.0;
	float u0 = 349.5;
	float v0 = 251.5;

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

