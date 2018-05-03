#include<opencv2/opencv.hpp>
#include<cv.h>

#include "highgui.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <ctime>
//#include<vector>

using namespace cv;
int redThre = 120; // 115~135  
int saturationTh = 55; //55~65  
Mat CheckColor(Mat &inImg);
void DrawFire(Mat &inputImg, Mat foreImg);

int main()
{
	
	VideoCapture capture("b.mp4");
	while (1)
	{
		Mat frame;

		capture >> frame;
		if (frame.empty())
			break;
		namedWindow("Control", CV_WINDOW_AUTOSIZE);
		//cvCreateTrackbar("redThre", "Control", &redThre, 255);
		//cvCreateTrackbar("saturationTh", "Control", &saturationTh, 255);
		CheckColor(frame);
		waitKey(1);
	}
	

	/*
	Mat inputImg = imread("4.jpg");
	CheckColor(inputImg);
	waitKey(0);
	*/

	return 0;
}

//The Color Check is According to "An Early Fire-Detection Method Based on Image Processing"   

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
			B = multiRGB[0].at<uchar>(i, j); 
			G = multiRGB[1].at<uchar>(i, j);
			R = multiRGB[2].at<uchar>(i, j);

			float maxValue = max(max(B, G), R);
			float minValue = min(min(B, G), R);
		
			double S = (1 - 3.0*minValue / (R + G + B));//

														
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
	//GaussianBlur(fireImg, fireImg, Size(5, 5), 0, 0);
	medianBlur(fireImg, fireImg, 5);
	dilate(fireImg, fireImg, Mat(5, 5, CV_8UC1));
	imshow("Binary", fireImg);
	DrawFire(inImg, fireImg);
	return fireImg;
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
	finalRect1 = Rect(xmin1, ymin1, xmax1 - xmin1, ymax1 - ymin1);
	finalRect2 = Rect(xmin2, ymin2, xmax2 - xmin2, ymax2 - ymin2);
	finalRect3 = Rect(xmin3, ymin3, xmax3 - xmin3, ymax3 - ymin3);
	rectangle(inputImg, finalRect1, Scalar(0, 255, 0));
	rectangle(inputImg, finalRect2, Scalar(0, 255, 0));
	rectangle(inputImg, finalRect3, Scalar(0, 255, 0));
	namedWindow("Fire_Detection", 0);
	imshow("Fire_Detection", inputImg);
}


