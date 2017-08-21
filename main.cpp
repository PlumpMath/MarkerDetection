#define _CRT_SECURE_NO_WARNINGS
#define MARKER_SIZE 90
#define MARKER_NUM 6
#define MARKER_UNIT MARKER_SIZE/MARKER_NUM
#define MARKING(x) x>127?1:0
#define CNTR_NUM 100
#define ADJUSTVAL(x,y,z) x<y?y:(x>z?z:x)

#include <opencv2\opencv.hpp>
#include <stdio.h>

using namespace cv;
using namespace std;

RNG rng(12345);

double distLinPnt(Point inPnt1, Point inPnt2, double pntX, double pntY);

void convertImage(Mat src, Mat dst, int valBright, int valContrast)
{
	int varBright = ADJUSTVAL(valBright, -100, 100);
	int varContrast = ADJUSTVAL(valContrast, -100, 100);

	double alpha, beta;

	if (varContrast > 0)
	{
		double delta = 127.0*varContrast / 100;
		alpha = 255.0 / (255.0 - delta * 2);
		beta = alpha * (varBright - delta);
	}
	else
	{
		double delta = -128.0*varContrast / 100;
		alpha = (256.0 - delta * 2) / 255.0;
		beta = alpha * varBright + delta;
	}

	src.convertTo(dst, CV_8U, alpha, beta);
}

void main()
{
	VideoCapture cap(0);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	VideoCapture vCap("d:/FlexibleQuad.avi");
	if (!vCap.isOpened())
	{
		cout << "웹캠을 열수 없습니다." << endl;
		return;
	}

	Mat frame, grayImg, binImg, cntrImg, boxImg, markerImg, vFrame;

	while (1)
	{
		cap >> frame;

		cvtColor(frame, grayImg, CV_BGR2GRAY);

		convertImage(grayImg, grayImg, -15, 55);
		convertImage(grayImg, grayImg, -15, 55);

		adaptiveThreshold(grayImg, binImg, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 201, 7);
		//threshold(grayImg, binImg, 128, 255, THRESH_BINARY);
		//threshold(grayImg, binImg, 125, 255, THRESH_BINARY | THRESH_OTSU);


		/*int dilation_size = 3;
		Mat kernel = getStructuringElement(MORPH_RECT, Size(2 * dilation_size + 1, 2 * dilation_size + 1),Point(dilation_size, dilation_size));
		dilate(binImg, binImg, kernel);
		erode(binImg, binImg, kernel);*/

		binImg = 255 - binImg;

		medianBlur(binImg, binImg, 7);

		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours(binImg, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

		cntrImg = frame.clone();
		boxImg = frame.clone();

		int numCntr = 0;
		int cntrsIdx[CNTR_NUM];

		for (size_t i = 0; i< contours.size(); i++)
		{
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

			int cntrIdx = 0;
			for (int j = 1; j < 4; j++)
			{
				if (hierarchy[i][j] != -1) cntrIdx = j;
			}

			switch (cntrIdx)
			{
			case 0:
				color = Scalar(0, 255, 255);
				break;
			case 1:
				color = Scalar(255, 0, 0);
				break;
			case 2:
				color = Scalar(0, 255, 0);
				break;
			case 3:
				color = Scalar(0, 0, 255);
				break;
			}
			if (true)//cntrIdx == 2)
			{
				drawContours(cntrImg, contours, (int)i, color, 1, 8, hierarchy, 0, Point());
				numCntr++;
				if (numCntr < CNTR_NUM + 1)
				{
					cntrsIdx[numCntr - 1] = i;
				}
			}
		}

		double maxArea = 0;
		int maxIndex = 0;
		vector<Point> maxPoly;

		for (int i = 0; i < numCntr; i++)
		{
			if (i < CNTR_NUM)
			{
				vector<Point> approxedPoly;
				approxPolyDP(contours[cntrsIdx[i]], approxedPoly, 10.0, true);
				if (approxedPoly.size() == 4)
				{
					double tmpArea = contourArea(contours[cntrsIdx[i]]);

					if (tmpArea > maxArea)
					{
						maxIndex = cntrsIdx[i];
						maxPoly = approxedPoly;
						maxArea = tmpArea;
					}
				}
			}
		}

		Point2f srcBox[4], dstBox[4];

		if (maxArea > 0)
		{
			//double dist1 = distLinPnt(maxPoly[0], maxPoly[2], cntX, cntY);
			//double dist2 = distLinPnt(maxPoly[1], maxPoly[3], cntX, cntY);

			if (true)
			{
				vector<vector<Point>> tmpcntr(1);
				tmpcntr[0] = maxPoly;
				drawContours(boxImg, tmpcntr, 0, Scalar(0, 0, 255));

				dstBox[0] = Point2f(0, 0);
				dstBox[1] = Point2f(MARKER_SIZE, 0);
				dstBox[2] = Point2f(MARKER_SIZE, MARKER_SIZE);
				dstBox[3] = Point2f(0, MARKER_SIZE);
				srcBox[0] = (Point2f)maxPoly[0];
				srcBox[1] = (Point2f)maxPoly[1];
				srcBox[2] = (Point2f)maxPoly[2];
				srcBox[3] = (Point2f)maxPoly[3];

				Mat psptv = getPerspectiveTransform(srcBox, dstBox);
				warpPerspective(frame, markerImg, psptv, Size(MARKER_SIZE, MARKER_SIZE));
			}
		}
		else
		{
			markerImg = Mat(MARKER_SIZE, MARKER_SIZE, CV_8UC3);
		}

		int marking[MARKER_NUM][MARKER_NUM];

		Rect rtROI;
		Mat meanMat = Mat(MARKER_UNIT, MARKER_UNIT, CV_8UC1);
		Mat checkerImg = Mat(MARKER_SIZE, MARKER_SIZE, CV_8UC1);
		Mat binChkImg = Mat(MARKER_SIZE, MARKER_SIZE, CV_8UC1);
		Mat rotBinImg = Mat(MARKER_SIZE, MARKER_SIZE, CV_8UC1);

		bool bMarker = true;
		int nParity = 0;
		int nShell = 0;
		int whiteBright = 255;
		for (int i = 0; i < MARKER_NUM; i++)
		{
			for (int j = 0; j < MARKER_NUM; j++)
			{
				if (bMarker)
				{
					rtROI = Rect(i*MARKER_UNIT, j*MARKER_UNIT, MARKER_UNIT, MARKER_UNIT);

					meanMat = mean(markerImg(rtROI));
					checkerImg(rtROI) = mean(markerImg(rtROI));

					int binVal = MARKING(meanMat.at<uchar>(0, 0));
					marking[i][j] = binVal;

					binChkImg(rtROI) = 255 * binVal * Mat::ones(MARKER_UNIT, MARKER_UNIT, CV_8UC1);

					if ((i == 0 || i == MARKER_NUM - 1) && (j == 0 || j == MARKER_NUM - 1))
					{
						bMarker = bMarker && binVal == 0;
						nShell += binVal;
					}
					else if ((i == 1 || i == MARKER_NUM - 2) && (j == 1 || j == MARKER_NUM - 2))
					{
						if (binVal == 0) nParity++;
						else whiteBright = meanMat.at<uchar>(0, 0);
						bMarker = bMarker && nParity < 2;
					}
				}
			}
		}

		//printf("%d\n", nParity);
		bMarker = bMarker && nParity == 1;
		//for (int i = 0; i < MARKER_NUM; i++)
		//{
		//	for (int j = 0; j < MARKER_NUM; j++)
		//	{
		//		printf("%d, ", marking[i][j]);
		//	}
		//	printf("\n");
		//}
		//printf("\n\n");


		if (bMarker)
		{
			//int i = 0;

			//const double subPixSize = 20.0;
			//Mat subPixImg = frame(Rect(srcBox[i].x - subPixSize*0.5, srcBox[i].y - subPixSize*0.5, subPixSize, subPixSize));
			//
			//cvtColor(subPixImg, subPixImg, CV_BGR2GRAY);
			//convertImage(subPixImg, subPixImg, -50, 75);

			//Mat corners;
			//cornerSubPix(subPixImg, corners, Size(subPixSize*0.5,subPixSize*0.5), Size(3, 3), TermCriteria(1,100,0.5));

			//imshow("subPix", subPixImg);
			///*for (int i = 0; i < 4; i++)
			//{

			//}*/
			vector<Point2f> tmpPoints(srcBox, srcBox + sizeof(srcBox) / sizeof(srcBox)[0]);

			cornerSubPix(grayImg, tmpPoints, Size(5, 5), Size(-1, -1), TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 30, 0.01));

			for (int i = 0; i < 4; i++)
			{
				srcBox[i] = tmpPoints[i];
			}

			Mat psptv = getPerspectiveTransform(srcBox, dstBox);
			warpPerspective(frame, markerImg, psptv, Size(MARKER_SIZE, MARKER_SIZE));

			double cntX = (srcBox[0].x + srcBox[1].x + srcBox[2].x + srcBox[3].x)*0.25;
			double cntY = (srcBox[0].y + srcBox[1].y + srcBox[2].y + srcBox[3].y)*0.25;

			circle(boxImg, Point(cntX, cntY), 5, Scalar(255, 0, 0), 3);
			line(boxImg, srcBox[0], srcBox[2], Scalar(0, 255, 0), 3);
			line(boxImg, srcBox[1], srcBox[3], Scalar(0, 255, 0), 3);

		}


		for (int i = 1; i < MARKER_NUM; i++)
		{
			line(markerImg, Point(MARKER_UNIT*i, 0), Point(MARKER_UNIT*i, MARKER_SIZE), Scalar(0, 0, 255));
			line(markerImg, Point(0, MARKER_UNIT*i), Point(MARKER_SIZE, MARKER_UNIT*i), Scalar(0, 0, 255));
		}

		namedWindow("rotatedBinary", WINDOW_KEEPRATIO);
		namedWindow("binMarker", WINDOW_KEEPRATIO);
		namedWindow("markerMosaic", WINDOW_KEEPRATIO);
		namedWindow("marker", WINDOW_KEEPRATIO);
		imshow("gray", grayImg);
		imshow("binary", binImg);
		imshow("contours", cntrImg);
		imshow("boxed", boxImg);

		if (bMarker)
		{
			int rotNum = 0;
			int rotMark[MARKER_NUM][MARKER_NUM];

			for (int k = 0; k < 3; k++)
			{
				if (marking[1][1])
				{
					rotNum++;
					for (int i = 0; i < MARKER_NUM; i++)
					{
						for (int j = 0; j < MARKER_NUM; j++)
						{
							rotMark[j][MARKER_NUM - i - 1] = marking[i][j];
						}
					}

					for (int i = 0; i < MARKER_NUM; i++)
					{
						for (int j = 0; j < MARKER_NUM; j++)
						{
							marking[i][j] = rotMark[i][j];

							rtROI = Rect(i*MARKER_UNIT, j*MARKER_UNIT, MARKER_UNIT, MARKER_UNIT);
							rotBinImg(rtROI) = 255 * marking[i][j] * Mat::ones(MARKER_UNIT, MARKER_UNIT, CV_8UC1);
						}
					}
				}
			}

			for (int i = 0; i < rotNum; i++)
			{
				Point2f tmpPoint = srcBox[0];
				for (int j = 0; j < 3; j++)
				{
					srcBox[j] = srcBox[j + 1];
				}
				srcBox[3] = tmpPoint;
			}

			for (int i = 0; i < MARKER_NUM; i++)
			{
				for (int j = 0; j < MARKER_NUM; j++)
				{
					marking[i][j] = rotMark[i][j];

					rtROI = Rect(i*MARKER_UNIT, j*MARKER_UNIT, MARKER_UNIT, MARKER_UNIT);
					rotBinImg(rtROI) = 255 * marking[i][j] * Mat::ones(MARKER_UNIT, MARKER_UNIT, CV_8UC1);
				}
			}

			imshow("marker", markerImg);
			imshow("markerMosaic", checkerImg);
			imshow("binMarker", binChkImg);
			imshow("rotatedBinary", rotBinImg);


			Point2f tmpBox[4];
			tmpBox[0] = Point2f(0, 0);
			tmpBox[1] = Point2f(vCap.get(CV_CAP_PROP_FRAME_WIDTH), 0);
			tmpBox[2] = Point2f(vCap.get(CV_CAP_PROP_FRAME_WIDTH), vCap.get(CV_CAP_PROP_FRAME_HEIGHT));
			tmpBox[3] = Point2f(0, vCap.get(CV_CAP_PROP_FRAME_HEIGHT));

			vCap >> vFrame;

			if (vFrame.empty())
			{
				vCap = VideoCapture("d:/FlexibleQuad.avi");
				if (!vCap.isOpened())
				{
					cout << "웹캠을 열수 없습니다." << endl;
					return;
				}
				vCap >> vFrame;
			}

			convertImage(vFrame, vFrame, (whiteBright - 255) * 2, 0);

			Mat tmpFrame;
			Mat maskFrame;
			Mat viewFrame = frame.clone();

			Mat psptv = getPerspectiveTransform(tmpBox, srcBox);
			warpPerspective(vFrame, tmpFrame, psptv, frame.size());
			warpPerspective(Mat(vFrame.size(), CV_8UC1, Scalar(255)), maskFrame, psptv, frame.size());
			tmpFrame.copyTo(viewFrame, maskFrame);

			imshow("vFrame", viewFrame);
		}

		if (waitKey(1) == 27) break;
	}
}


double distLinPnt(Point inPnt1, Point inPnt2, double pntX, double pntY)
{
	if (inPnt2.x == inPnt1.x)
		return 10000;
	else
	{
		double constA = (inPnt1.y - inPnt2.y) / (inPnt2.x - inPnt1.x);
		double constB = 1;
		double constC = -(constA*inPnt1.x + inPnt1.y);

		double nume = (constA*pntX + constB*pntY + constC);

		return sqrtf(nume*nume / (constA*constA + constB*constB));
	}
}

