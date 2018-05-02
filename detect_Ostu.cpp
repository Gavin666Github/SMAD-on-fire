#include "cv.h"  
#include "highgui.h"  
#include <stdio.h>  
#include <iostream>  
#include<opencv2/opencv.hpp>

#include <stdlib.h>  
#include <time.h>  

#define cvQueryHistValue_1D( hist, idx0 )   ((float)cvGetReal1D((hist)->bins, (idx0)))
using namespace cv;

//OSTU算法  
int HistogramBins = 256;
float HistogramRange1[2] = { 0,255 };
float *HistogramRange[1] = { &HistogramRange1[0] };
typedef enum { back, object } entropy_state;

double caculateCurrentEntropy(CvHistogram * Histogram1, int cur_threshold, entropy_state state)
{
	int start, end;
	if (state == back)
	{
		start = 0;
		end = cur_threshold;
	}
	else
	{
		start = cur_threshold;
		end = 256;
	}
	int  total = 0;
	for (int i = start; i < end; i++)
	{
		total += (int)cvQueryHistValue_1D(Histogram1, i);
	}
	double cur_entropy = 0.0;
	for (int i = start; i < end; i++)
	{
		if ((int)cvQueryHistValue_1D(Histogram1, i) == 0)
			continue;
		double percentage = cvQueryHistValue_1D(Histogram1, i) / total;
		cur_entropy += -percentage * logf(percentage);
	}
	return cur_entropy;
}

IplImage* MaxEntropy(IplImage *src, IplImage *dst)
{
	assert(src != NULL);
	assert(src->depth == 8 && dst->depth == 8);
	assert(src->nChannels == 1);
	CvHistogram * hist = cvCreateHist(1, &HistogramBins, CV_HIST_ARRAY, HistogramRange);
	cvCalcHist(&src, hist);
	double maxentropy = -1.0;
	int max_index = -1;
	for (int i = 0; i < HistogramBins; i++)
	{
		double cur_entropy = caculateCurrentEntropy(hist, i, object) + caculateCurrentEntropy(hist, i, back);
		if (cur_entropy > maxentropy)
		{
			maxentropy = cur_entropy;
			max_index = i;
		}
	}
	printf("%f", max_index);
	cvThreshold(src, dst, (double)max_index, 255, CV_THRESH_BINARY);
	cvReleaseHist(&hist);
	return dst;
}



IplImage *img = cvLoadImage("0.jpg");

IplImage *b = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
IplImage *g = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
IplImage *temp1 = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
IplImage *temp2 = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
IplImage *r = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
IplImage *dst = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 3);


//颜色分割算法  
IplImage* process_rgb(IplImage*img) {
	cvSplit(img, b, g, r, NULL);
	cvAddWeighted(b, 1.0 / 3.0, g, 1.0 / 3.0, 0.0, g);
	cvAddWeighted(r, 1.0 / 3.0, g, 1.0, 0.0, g);
	cvInRangeS(g, cvScalar(100.0, 0.0, 0.0), cvScalar(255.0, 0.0, 0.0), temp1);
	cvSub(r, b, g);
	cvAddWeighted(g, 1.0 / 2.0, r, 0.0, 0.0, r);

	cvAnd(r, temp1, r);
	cvSmooth(r, r, CV_GAUSSIAN, 3, 0, 0, 0);
	return r;   //r=(r-b)/2  求r的最佳阈值分割  

}

int main(int argc, char* argv[])
{
	CvSeq * contour = 0;
	CvMemStorage * storage = cvCreateMemStorage();
	IplImage * tempdst1 = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);

	clock_t start, finish;
	double duration;

	cvNamedWindow("img", 0);
	//cvNamedWindow("OstuImg", 0);
	//cvNamedWindow("OstuImg1", 0);
	//cvNamedWindow("tian", 0);
	cvNamedWindow("dst", 0);

	IplImage *OtsuImg = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
	IplImage *OtsuImg1 = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
	OtsuImg = process_rgb(img);
	OtsuImg1 = MaxEntropy(OtsuImg, OtsuImg1);


	cvFindContours(OtsuImg1, storage, &contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	cvZero(tempdst1);
	//  cvZero(temp_iamge);  
	for (; contour != 0; contour = contour->h_next)
	{   //应用函数 fabs() 得到面积的绝对值。   
		double area = cvContourArea(contour, CV_WHOLE_SEQ);
		//计算整个轮廓或部分轮廓的面积  
		if (fabs(area) < 50)
		{
			continue;
		}
		CvPoint *point = new CvPoint[contour->total];
		CvPoint *Point;
		for (int i = 0; i<contour->total; i++)
		{
			Point = (CvPoint*)cvGetSeqElem(contour, i);
			point[i].x = Point->x;
			point[i].y = Point->y;
		}
		int pts[1] = { contour->total };
		cvFillPoly(tempdst1, &point, pts, 1, CV_RGB(255, 255, 255));//填充多边形内部   

	}

	cvCopy(img, dst, tempdst1);

	cvShowImage("img", img);

	cvShowImage("OstuImg", OtsuImg);
	cvShowImage("OstuImg1", OtsuImg1);
	cvShowImage("tian", tempdst1);
	cvShowImage("dst", dst);
	while (1) {
		if (cvWaitKey(100) == 27)
			break;
	}
	cvReleaseImage(&r);
	cvReleaseImage(&g);
	cvReleaseImage(&b);
	cvReleaseImage(&img);
	cvReleaseImage(&dst);
	cvReleaseImage(&temp1);
	cvReleaseImage(&temp2);
	cvReleaseImage(&OtsuImg);
	cvReleaseImage(&OtsuImg1);
	cvReleaseImage(&tempdst1);
	cvReleaseMemStorage(&storage);


	return 0;


}
