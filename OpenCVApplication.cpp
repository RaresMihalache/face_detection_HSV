// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include "Functions.h"
#include <chrono>
#include <thread>
#include <fstream>
#include <direct.h>

#define MAX_HUE 256
#define FILTER_HISTOGRAM 1
#define THRESHOLD_PERCENTAGE 0.25f;
int histG_hue[MAX_HUE]; // histograma globala / cumulativa
int histG_saturation[MAX_HUE];
int total_calculated_px;
int hue_std;
int hue_mean;
int saturation_std;
int saturation_mean;
Point Pstart, Pend; // Punctele / colturile aferente selectiei ROI curente (declarate global)

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
CascadeClassifier nose_cascade;
CascadeClassifier mouth_cascade;

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = MAX_PATH-val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int w = src.step; // no dword alignment is done !!!
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
				/* sau puteti scrie:
				uchar val = lpSrc[i*width + j];
				lpDst[i*width + j] = 255 - val;
				//	w = width pt. imagini cu 8 biti / pixel
				//	w = 3*width pt. imagini cu 24 biti / pixel
				*/
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);
		
		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;
		int w = src.step; // latimea in octeti a unei linii de imagine
		
		Mat dstH = Mat(height, width, CV_8UC1);
		Mat dstS = Mat(height, width, CV_8UC1);
		Mat dstV = Mat(height, width, CV_8UC1);
		
		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* dstDataPtrH = dstH.data;
		uchar* dstDataPtrS = dstS.data;
		uchar* dstDataPtrV = dstV.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);
		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				// sau int hi = i*w + j * 3;	//w = 3*width pt. imagini 24 biti/pixel
				int gi = i*width + j;
				
				dstDataPtrH[gi] = hsvDataPtr[hi] * 510/360;		// H = 0 .. 255
				dstDataPtrS[gi] = hsvDataPtr[hi + 1];			// S = 0 .. 255
				dstDataPtrV[gi] = hsvDataPtr[hi + 2];			// V = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", dstH);
		imshow("S", dstS);
		imshow("V", dstV);
		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int k = 0.4;
		int pH = 50;
		int pL = k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey();
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey();  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

// Lab 2:
void RGB2HSV() {
	char fname[MAX_PATH];
	Mat rez;
	while (openFileDlg(fname)) {
		Mat src = imread(fname, 1); // open in RGB_mode (1)
		Mat h(src.rows, src.cols, CV_8UC1);
		Mat s(src.rows, src.cols, CV_8UC1);
		Mat v(src.rows, src.cols, CV_8UC1);
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				float r, g, b, M, m, C, V, S, H;
				r = (float)src.at<Vec3b>(i, j)[2] / 255;
				g = (float)src.at<Vec3b>(i, j)[1] / 255;
				b = (float)src.at<Vec3b>(i, j)[0] / 255;
				M = r > g ? (r > b ? r : b) : (g > b ? g : b);
				m = r < g ? (r < b ? r : b) : (g < b ? g : b);
				C = M - m;

				V = M;
				S = V != 0 ? C / V : 0;
				if (C != 0) {
					if (M == r) H = 60 * (g - b) / C;
					if (M == g) H = 120 + 60 * (b - r) / C;
					if (M == b) H = 240 + 60 * (r - g) / C;
				}
				else // grayscale
					H = 0;
				H = H < 0 ? H + 360 : H;

				h.at<uchar>(i, j) = H * 255 / 360;
				s.at<uchar>(i, j) = S * 255;
				v.at<uchar>(i, j) = V * 255;
			}
		}
		int h1[256] = { 0 };
		int h2[256] = { 0 };
		int h3[256] = { 0 };
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				h1[(int)h.at<uchar>(i, j)]++;
				h2[(int)s.at<uchar>(i, j)]++;
				h3[(int)v.at<uchar>(i, j)]++;
			}
		}

		imshow("Hue", h);
		showHistogram("H", h1, 256, 200, true);
		imshow("Saturation", s);
		showHistogram("S", h2, 256, 200, true);
		imshow("Value", v);
		showHistogram("V", h3, 256, 200, true);
		imshow("original image", src);
	}
	waitKey(0);
}

void calculateFDP(Mat srcH, Mat srcS) {
	FILE* ptr_fdpH_wr = fopen("D:\an4\proiect_ioc\OpenCVApplication-VS2019_OCV3411_basic_IOM\H_fdp.txt", "w");
	FILE* ptr_fdpS_wr = fopen("D:\an4\proiect_ioc\OpenCVApplication-VS2019_OCV3411_basic_IOM\S_fdp.txt", "r");

	int height = srcH.rows;
	int width = srcH.cols;
	int M = height * width;
	int H_FDP[255];
	int S_FDP[255];

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			H_FDP[srcH.at<uchar>(i, j)]++;
			H_FDP[srcH.at<uchar>(i, j)]++;
		}
	}
}

void binarizare() {
	char fname[MAX_PATH];
	int x; // threshold
	scanf("%d", &x);
	if (x < 0 || x > 255) {
		printf("Valoare depasita");
		return;
	}
	while (openFileDlg(fname)) {
		Mat src = imread(fname, 0); // open in GRAYSCALE_mode (0)

		Mat bw(src.rows, src.cols, CV_8UC1);
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if (src.at<uchar>(i, j) < x)
					bw.at<uchar>(i, j) = 255;
				else
					bw.at<uchar>(i, j) = 0;
			}
		}

		imshow("Original", src);
		imshow("Binarizare", bw);
	}

	waitKey(0);
}

void binarizare_automata() {
	float media1 = 0;
	float media2 = 0;
	float pixeli = 0;
	float pixeli2 = 0;
	float T = 0;
	float T2 = 0;

	int min = 256;
	int max = 0;
	int N1 = 0;
	int N2 = 0;

	int hist[256] = {};
	int histCumul[256] = {};

	char fname[MAX_PATH];

	while (openFileDlg(fname)) {
		Mat src = imread(fname, 0);
		Mat img = src.clone();

		for (int i = 1; i < img.rows; i++) {
			for (int j = 1; j < img.cols; j++) {
				hist[img.at<uchar>(i, j)]++;
				if (img.at<uchar>(i, j) < min)
					min = img.at<uchar>(i, j);
				if (img.at<uchar>(i, j) > max)
					max = img.at<uchar>(i, j);
			}
		}

		T = (min + max) / 2;
		while (abs(T - T2) >= 0.1) {
			T2 = T;
			media1 = 0;
			media2 = 0;
			N1 = 0;
			N2 = 0;
			for (int i = min; i < T; i++) {
				media1 += i * hist[i];
				N1 += hist[i];
			}
			for (int i = T + 1; i <= max; i++) {
				media2 += i * hist[i];
				N2 += hist[i];
			}

			media1 /= N1;
			media2 /= N2;
			T = (media1 + media2) / 2;
		}
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				if (img.at<uchar>(i, j) <= T)
					img.at<uchar>(i, j) = 255;
				else
					img.at<uchar>(i, j) = 0;
			}
		}

		imshow("Original", src);
imshow("Binarizare automata", img);
waitKey(0);
	}
}

void binarizare_H() {
	char fname[MAX_PATH];
	int x; // threshold
	scanf("%d", &x);
	if (x < 0 || x > 255) {
		printf("Valoare depasita");
		return;
	}
	Mat rez;
	while (openFileDlg(fname)) {
		Mat src = imread(fname, 1);
		Mat h(src.rows, src.cols, CV_8UC1);
		Mat s(src.rows, src.cols, CV_8UC1);
		Mat v(src.rows, src.cols, CV_8UC1);
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				float r, g, b, M, m, C, V, S, H;
				r = (float)src.at<Vec3b>(i, j)[2] / 255;
				g = (float)src.at<Vec3b>(i, j)[1] / 255;
				b = (float)src.at<Vec3b>(i, j)[0] / 255;
				M = r > g ? (r > b ? r : b) : (g > b ? g : b);
				m = r < g ? (r < b ? r : b) : (g < b ? g : b);
				C = M - m;

				V = M;
				S = V != 0 ? C / V : 0;
				if (C != 0) {
					if (M == r) H = 60 * (g - b) / C;
					if (M == g) H = 120 + 60 * (b - r) / C;
					if (M == b) H = 240 + 60 * (r - g) / C;
				}
				else // grayscale
					H = 0;
				H = H < 0 ? H + 360 : H;

				h.at<uchar>(i, j) = H * 255 / 360;
				s.at<uchar>(i, j) = S * 255;
				v.at<uchar>(i, j) = V * 255;
			}
		}
		int h1[256] = { 0 };
		int h2[256] = { 0 };
		int h3[256] = { 0 };
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				h1[(int)h.at<uchar>(i, j)]++;
				h2[(int)s.at<uchar>(i, j)]++;
				h3[(int)v.at<uchar>(i, j)]++;
			}
		}
		Mat bw(src.rows, src.cols, CV_8UC1);

		for (int i = 0; i < h.rows; i++) {
			for (int j = 0; j < h.cols; j++) {
				if (h.at<uchar>(i, j) < x) {
					bw.at<uchar>(i, j) = 255;
				}
				else
					bw.at<uchar>(i, j) = 0;
			}
		}
		imshow("Hue", h);
		imshow("Binarizare H", bw);
	}

	waitKey(0);
}

// Lab 3
void L3_ColorModel_Build() {
	int hue_mean = 16;
	int hue_std = 5;
	Mat src;
	Mat hsv;
	// Segmentare
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		src = imread(fname);
		Mat_<uchar> dst(src.rows, src.cols);
		GaussianBlur(src, src, Size(5, 5), 0, 0);
		namedWindow("src", 1);
		Mat_<uchar> H(src.rows, src.cols);
		uchar* lpH = H.data;
		cvtColor(src, hsv, CV_BGR2HSV);
		uchar* hsvDataPtr = hsv.data;
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				int hi = i * src.cols * 3 + j * 3;
				int gi = i * src.cols + j;
				lpH[gi] = hsvDataPtr[hi] * 510 / 360;
			}
		}
		float k = 2.5;
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if ((hue_mean - k * hue_std) <= H(i, j) && ((hue_mean + k * hue_std) >= H(i, j))){
					dst(i, j) = 255;
				}
				else {
					dst(i, j) = 0;
				}
			}
			imshow("Segmentare", dst);
		
			//Asociere functie de tratare a evenimentelor Mouse cu fereastra curenta.
			//Ultimul parametru este matricea H
			//setMouseCallback("src", CallBackFuncL3, &H);
			//Postprocesare
			imshow("src", src);
			Mat elementStruct = getStructuringElement(MORPH_RECT, Size(3, 3));
			erode(dst, dst, elementStruct, Point(-1, -1), 2);
			dilate(dst, dst, elementStruct, Point(-1, -1), 4);
			erode(dst, dst, elementStruct, Point(-1, -1), 2);
			imshow("img dupa postprocesare", dst);

			//Extragerea conturului
			Labeling("contur", dst, false);
			
			/*Mat conturImg = dst.clone();
			std::vector<std::vector<cv::Point>> contours;
			findContours(conturImg, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
			for (int i = 0; i < contours.size(); i++) {
				drawContours(conturImg, contours, i, Scalar(255, 255, 255));
			}
			imshow("contur", conturImg);*/

			// Desenarea axei de alungire
			// ...
			
		}
	}
}

void calculate_saturation_mean() {
	
	for (int i = 0; i < MAX_HUE; i++) {
		saturation_mean += (histG_saturation[i] * i);
	}
	saturation_mean /= total_calculated_px;

	printf("Saturation Mean: %d", saturation_mean);

	std::chrono::seconds dura(5);
	std::this_thread::sleep_for(dura);
}

void calculate_hue_mean() {

	for (int i = 0; i < MAX_HUE; i++) {
		hue_mean += (histG_hue[i] * i);
	}
	hue_mean /= total_calculated_px;

	printf("Hue Mean: %d", hue_mean);

	std::chrono::seconds dura(5);
	std::this_thread::sleep_for(dura);
}

void calculate_saturation_std() {

	for (int i = 0; i < MAX_HUE; i++) {
		saturation_std += pow((histG_saturation[i] - saturation_mean), 2);
	}
	saturation_std /= total_calculated_px;
	saturation_std = sqrt(saturation_std);
	printf("Saturation Std: %d", saturation_std);

	std::chrono::seconds dura(5);
	std::this_thread::sleep_for(dura);
}

void calculate_hue_std() {

	for (int i = 0; i < MAX_HUE; i++) {
		hue_std += pow((histG_hue[i] - hue_mean), 2);
	}
	hue_std /= total_calculated_px;
	hue_std = sqrt(hue_std);
	printf("Hue Std: %d", hue_std);

	std::chrono::seconds dura(5);
	std::this_thread::sleep_for(dura);
}



void CallBackFuncL3(int event, int x, int y, int flags, void* userdata) {
	std::tuple<Mat*, Mat*>* userdata_tuple = (std::tuple<Mat*, Mat*>*)userdata;

	Mat* H = std::get<0>(*userdata_tuple);
	Mat* S = std::get<1>(*userdata_tuple);
	Rect roi; // regiunea de interes curenta (ROI)

	if (event == EVENT_LBUTTONDOWN) {
		// punctul de start al ROI
		Pstart.x = x;
		Pstart.y = y;
		printf("Pstart: (%d, %d) ", Pstart.x, Pstart.y);
	}
	else if (event == EVENT_RBUTTONDOWN) {
		// punctul de final (diametral opus) al ROI
		Pend.x = x;
		Pend.y = y;
		printf("Pend: (%d, %d) ", Pend.x, Pend.y);
		// sortare puncte dupa x si y
		// (parametrii width si height ai structurii Rect > 0)
		roi.x = min(Pstart.x, Pend.x);
		roi.y = min(Pstart.y, Pend.y);
		roi.width = abs(Pstart.x - Pend.x);
		roi.height = abs(Pstart.y - Pend.y);
		printf("Local ROI: (%d, %d), (%d, %d)\n", roi.x, roi.y, roi.x + roi.width, roi.y + roi.height);

		int hist_hue[MAX_HUE];	// histograma locala a lui Hue
		int hist_saturation[MAX_HUE];
		memset(hist_hue, 0, MAX_HUE * sizeof(int));
		memset(hist_saturation, 0, MAX_HUE * sizeof(int));

		// Din toata imaginea H se selecteaza o subimagine (Hroi) aferenta ROI
		Mat Hroi = (*H)(roi);
		Mat Sroi = (*S)(roi);

		uchar hue;
		uchar saturation;
		// construieste histograma locala aferenta ROI
		for (int y = 0; y < roi.height; y++) {
			for (int x = 0; x < roi.width; x++) {
				hue = Hroi.at<uchar>(y, x);
				saturation = Sroi.at<uchar>(y, x);
				hist_hue[hue]++;
				hist_saturation[saturation]++;
			}
		}
		// acumuleaza histograma locala in cea globala
		for (int i = 0; i < MAX_HUE; i++) {
			histG_hue[i] += hist_hue[i];
			histG_saturation[i] += hist_saturation[i];
		}

		// afiseaza histograma locala
		showHistogram("H local histogram", hist_hue, MAX_HUE, 200, true);
		// afiseaza histograma globala / cumulativa
		showHistogram("H global histogram", histG_hue, MAX_HUE, 200, true);

		// afiseaza histograma locala
		showHistogram("S local histogram", hist_saturation, MAX_HUE, 200, true);
		// afiseaza histograma globala / cumulativa
		showHistogram("S global histogram", histG_saturation, MAX_HUE, 200, true);

		total_calculated_px += (Pend.x - Pstart.x) * (Pend.y - Pstart.y);
		
		for (int i = 0; i < 255; i++) {
			std::cout << histG_hue[i] << ' ';
		}
		std::cout << endl;
	}
}


void L3_ColorModel_Init() {
	memset(histG_hue, 0, sizeof(unsigned int) * MAX_HUE);
}

void L3_ColorModel_Build_2() {
	Mat src;
	Mat hsv;
	// Read image from file
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		src = imread(fname);
		int height = src.rows;
		int width = src.cols;
		Mat dst(height, width, 0);

		// Aplicare FTJ gaussian pt. eliminarea zgomote: esential sa il aplicam
		GaussianBlur(src, src, Size(5, 5), 0, 0);

		// Creare fereastra pt. afisare
		namedWindow("src", 1);

		// Componenta de culoare Hue a modelului HSV
		Mat H = Mat(height, width, CV_8UC1);

		// definire pointeri la matricea (8 biti/pixeli) folosita la stocarea
		// componentei individuale H
		uchar* lpH = H.data;

		cvtColor(src, hsv, CV_BGR2HSV); // conversie RGB -> HSV
		
		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsv.data;
		
		for (int i = 0; i < height;i++) {
			for (int j = 0; j < width; j++) {
				// index in matricea hsv (24 biti/pixel)
				int hi = i * width * 3 + j * 3;
				int gi = i * width + j; // index in matricea H (8biti/pixel)
				lpH[gi] = hsvDataPtr[hi] * 510 / 360; //lpH = 0 .. 255
			}
		}

		int hue_mean = 16;
		int hue_std = 5;
		float k = 2.5;
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if((hue_mean - k * hue_std) <= H.at<uchar>(i, j) && (hue_mean + k * hue_std >= H.at<uchar>(i, j))){
					dst.at<uchar>(i, j) = 255;
				}
				else {
					dst.at<uchar>(i, j) = 0;
				}
			}
		}

		imshow("src", src);
		imshow("segmentare", dst);

		// Postprocesare
		Mat elementStruct = getStructuringElement(MORPH_RECT, Size(3, 3));
		erode(dst, dst, elementStruct, Point(-1, -1), 2);
		dilate(dst, dst, elementStruct, Point(-1, -1), 4);
		erode(dst, dst, elementStruct, Point(-1, -1), 2);
		imshow("img dupa postprocesare", dst);

		// Asociere functie de tratare a evenimentelor MOUSE cu fereastra curenta
		// Ultimul parametru este matricea H (valorile componentei Hue)
		setMouseCallback("src", CallBackFuncL3, &H);

		// Wait until user press some key
		waitKey(0);
	}
}

void L3_ColorModel_Build_2_Personal(Mat src) {
	int height = src.rows;
	int width = src.cols;
	Mat hsv;
	Mat dst(height, width, 0);

	// Creare fereastra pt. afisare
	namedWindow("src", 1);

	// Componenta de culoare Hue a modelului HSV
	Mat H = Mat(height, width, CV_8UC1);
	Mat S = Mat(height, width, CV_8UC1);

	// definire pointeri la matricea (8biti / pixeli) folosita la stocarea componentei individuale H
	uchar* lpH = H.data;
	uchar* lpS = S.data;

	cvtColor(src, hsv, CV_BGR2HSV); // conversie RGB -> HSV

	// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
	uchar* hsvDataPtr = hsv.data;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			// index in matricea hsv (24 biti/pixel)
			int hi = i * width * 3 + j * 3;
			int gi = i * width + j; // index in matricea H (8biti/pixel)
			lpH[gi] = hsvDataPtr[hi] * 510 / 360; //lpH = 0 .. 255
			lpS[gi] = hsvDataPtr[hi + 1] * 255; //lpS = 0 .. 255
		}
	}

	imshow("src", src);

	std::tuple<Mat*, Mat*> userdata(&H, &S);
	// Asociere functie de tratare a evenimentelor MOUSE cu fereastra curenta
	// Ultimul parametru este matricea H (valorile componentei Hue)
	setMouseCallback("src", CallBackFuncL3, &userdata);
	//setMouseCallback("src", CallBackFuncL3, &S);

	// Wait until user press some key
	waitKey(0);

}

void L3_ColorModel_Save() {
	int hue, sat, i, j;
	int histF_hue[MAX_HUE]; // histograma filtrata cu FTJ
	memset(histF_hue, 0, MAX_HUE * sizeof(unsigned int));

	#if FILTER_HISTOGRAM == 1
	// filtrare histograma cu filtru gaussian 1D de dimensiune w=7
	float gauss[7];
	float sqrt2pi = sqrtf(2 * PI);
	float sigma = 1.5;
	float e = 2.718;
	float sum = 0;
	// construire Gaussian
	for (i = 0; i < 7; i++) {
		gauss[i] = 1.0 / (sqrt2pi * sigma) * powf(e, -(float)(i - 3) * (i - 3) / (2 * sigma * sigma));
		sum += gauss[i];
	}
	// filtrare cu Gaussian
	for (j = 3; j < MAX_HUE - 3; j ++ ) {
		for (i = 0; i < 7; i++) {
			histF_hue[j] += (float)histG_hue[j + i - 3] * gauss[i];
		}
	}
#elif
	for (j = 0; k < MAX_HUE; j++) {
		histF_hue[j] = histG_hue[j]
	}
#endif // End of "Filtrare Gaussiana Histograma Hue"

	showHistogram("H global histogram", histG_hue, MAX_HUE, 200, true);
	showHistogram("H global filtered histogram", histF_hue, MAX_HUE, 200, true);

	// Wait until user press some key
	waitKey(0);
}



// lab 9
void FaceDetectandDisplayEyes(const string& window_name, Mat frame,
	int minFaceSize, int minEyeSize)
{
	std::vector<Rect> faces;
	Mat frame_gray;
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
		Size(minFaceSize, minFaceSize));
	for (int i = 0; i < faces.size(); i++) {
		Point center(faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height * 0.5);
		ellipse(frame, center, Size(faces[i].width * 0.5, faces[i].height * 0.5), 0, 0,
			360, Scalar(0, 255, 255), 4, 8, 0);
		Mat faceROI = frame_gray(faces[i]);
		std::vector<Rect> eyes;
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
			Size(minEyeSize, minEyeSize));
		for (int j = 0; j < eyes.size(); j++)
		{
			Point center(faces[i].x + eyes[j].x + eyes[j].width * 0.5,
				faces[i].y + eyes[j].y + eyes[j].height * 0.5);
			int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
			circle(frame, center, radius, Scalar(0, 255, 0), 4, 8, 0);
		}
	}
	imshow(window_name, frame);
	waitKey();
}


void face_detection_eyes() {
	String face_cascade_name = "haarcascade_frontalface_alt.xml";
	String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";


	if (!face_cascade.load(face_cascade_name))
	{
		printf("Error loading face cascades !\n");
		return;
	}

	if (!eyes_cascade.load(eyes_cascade_name))
	{
		printf("Error loading eyes cascades !\n");
		return;
	}


	Mat src;
	Mat dst;
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{

		src = imread(fname, CV_LOAD_IMAGE_COLOR);
		dst = src.clone();
		int minFaceSize = 30;
		int minEyeSize = minFaceSize / 5;
		FaceDetectandDisplayEyes("Dst", dst, minFaceSize, minEyeSize);
	}
}

void FaceDetectandDisplayMouthNose(const string& window_name, Mat frame,
	int minFaceSize, int minEyeSize)
{
	std::vector<Rect> faces;
	Mat frame_gray;
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
		Size(minFaceSize, minFaceSize));
	for (int i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height * 0.5);
		ellipse(frame, center, Size(faces[i].width * 0.5, faces[i].height * 0.5), 0, 0,
			360, Scalar(0, 255, 255), 4, 8, 0);

		Mat faceROI = frame_gray(faces[i]);
		std::vector<Rect> eyes;
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
			Size(minEyeSize, minEyeSize));
		for (int j = 0; j < eyes.size(); j++)
		{
			Point center(faces[i].x + eyes[j].x + eyes[j].width * 0.5,
				faces[i].y + eyes[j].y + eyes[j].height * 0.5);
			int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
			circle(frame, center, radius, Scalar(0, 255, 0), 4, 8, 0);
		}


		Rect nose_rect;
		nose_rect.x = faces[i].x;
		nose_rect.y = faces[i].y + 0.4 * faces[i].height;
		nose_rect.width = faces[i].width;
		nose_rect.height = 0.35 * faces[i].height;

		Mat nose_ROI = frame_gray(nose_rect);
		std::vector<Rect> nose;

		nose_cascade.detectMultiScale(nose_ROI, nose, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
			Size(15, 15));

		for (int j = 0; j < nose.size(); j++) {
			Point center(nose_rect.x + nose[j].x + nose[j].width * 0.5,
				nose_rect.y + nose[j].y + nose[j].height * 0.5);

			int radius = cvRound((nose[j].width + nose[j].height) * 0.25);
			circle(frame, center, radius, Scalar(0, 0, 255), 4, 8, 0);
		}


		Rect mouth_rect;
		mouth_rect.x = faces[i].x;
		mouth_rect.y = faces[i].y + 0.7 * faces[i].height;
		mouth_rect.width = faces[i].width;
		mouth_rect.height = 0.29 * faces[i].height;
		std::vector<Rect> mouth;
		Mat mouth_ROI = frame_gray(mouth_rect);

		mouth_cascade.detectMultiScale(mouth_ROI, mouth, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
			Size(minEyeSize, minEyeSize));

		for (int j = 0; j < mouth.size(); j++) {
			Point center(mouth_rect.x + mouth[j].x + mouth[j].width * 0.5,
				mouth_rect.y + mouth[j].y + mouth[j].height * 0.5);

			int radius = cvRound((mouth[j].width + mouth[j].height) * 0.25);

			circle(frame, center, radius, Scalar(255, 51, 153), 4, 8, 0);
		}
	}
	imshow(window_name, frame);
	waitKey();

}

void face_detection_mouth_nose() {
	String face_cascade_name = "haarcascade_frontalface_alt.xml";
	String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
	String mouth_cascade_name = "haarcascade_mcs_mouth.xml";
	String nose_cascade_name = "haarcascade_mcs_nose.xml";

	if (!face_cascade.load(face_cascade_name))
	{
		printf("Error loading face cascades !\n");
		return;
	}

	if (!eyes_cascade.load(eyes_cascade_name))
	{
		printf("Error loading eyes cascades !\n");
		return;
	}

	if (!mouth_cascade.load(mouth_cascade_name))
	{
		printf("Error loading mouth cascades !\n");
		return;
	}

	if (!nose_cascade.load(nose_cascade_name))
	{
		printf("Error loading nose cascades !\n");
		return;
	}

	Mat src;
	Mat dst;
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_COLOR);
		dst = src.clone();
		int minFaceSize = 30;
		int minEyeSize = minFaceSize / 5;
		FaceDetectandDisplayMouthNose("Dst", dst, minFaceSize, minEyeSize);
	}
}


void face_detect_video() {
	String face_cascade_name = "haarcascade_frontalface_alt.xml";
	String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";

	int minFaceSize = 30;
	int minEyeSize = minFaceSize / 5;
	if (!face_cascade.load(face_cascade_name))
	{
		printf("Error loading face cascades !\n");
		return;
	}

	if (!eyes_cascade.load(eyes_cascade_name))
	{
		printf("Error loading eyes cascades !\n");
		return;
	}

	char c;
	VideoCapture cap("Videos/Megamind.avi");
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey();
		return;
	}
	Mat frame;
	for (;;) {
		double t = (double)getTickCount();
		cap >> frame;
		if (frame.empty())
		{
			printf("End of video file\n");
			break;
		}
		t = ((double)getTickCount() - t) / getTickFrequency();
		printf("Curent frame time: %.3f [ms]\n", t * 1000);
		FaceDetectandDisplayEyes("Dst", frame, minFaceSize, minEyeSize);

		c = cvWaitKey();
		if (c == 27) {
			printf("ESC pressed - capture finished\n");
			break;
		};
	}
}


void FaceDetectandDisplay(const string& window_name, Mat frame,
	int minFaceSize, int minEyeSize)
{
	std::vector<Rect> faces;
	Mat frame_gray;
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
		Size(minFaceSize, minFaceSize));
	for (int i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height * 0.5);
		ellipse(frame, center, Size(faces[i].width * 0.5, faces[i].height * 0.5), 0, 0,
			360, Scalar(0, 255, 255), 4, 8, 0);
	}
	imshow(window_name, frame);
	waitKey();
}


void face_detection_lbp() {
	String face_cascade_name = "lbpcascade_frontalface.xml";


	if (!face_cascade.load(face_cascade_name))
	{
		printf("Error loading face cascades !\n");
		return;
	}


	Mat src;
	Mat dst;
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{

		src = imread(fname, CV_LOAD_IMAGE_COLOR);
		dst = src.clone();
		int minFaceSize = 30;
		int minEyeSize = minFaceSize / 5;
		FaceDetectandDisplay("Dst", dst, minFaceSize, minEyeSize);
	}
}


//////////////////////////////////////LABORATOR 10

bool isSkinPx_RGB(Vec3b px) {
	if (px[2] > 95 && px[1] > 40 && px[0] > 20 && px[2] > px[1] && px[2] > px[0] && abs(px[2] - px[1]) > 15)
		return true;
	return false;
}

Rect reduceRectangles(const string& window_name, Mat frame, std::vector<Rect> faces) {
	
	int height = frame.rows;
	int width = frame.cols;
	Mat hsv;
	Mat gray;
	cvtColor(frame, hsv, COLOR_BGR2HSV);
	cvtColor(frame, gray, CV_BGR2GRAY);
	imshow("hsv", hsv);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			//if (frame.at<Vec3b>(i, j)[2] > 95)
			if (isSkinPx_RGB(frame.at<Vec3b>(i, j)))
				gray.at<uchar>(i, j) = 255;
			else
				gray.at<uchar>(i, j) = 0;
		}
	}

	Mat elementStruct = getStructuringElement(MORPH_RECT, Size(3, 3));
	erode(gray, gray, elementStruct, Point(-1, -1), 3);
	dilate(gray, gray, elementStruct, Point(-1, -1), 3);
	erode(gray, gray, elementStruct, Point(-1, -1), 3);


	for (int i = 0; i < faces.size(); i++) {
		int xmin = 5000;
		int ymin = 5000;
		int xmax = 0;
		int ymax = 0;

		for (int row = faces[i].y; row < faces[i].y + faces[i].height; row++) {
			for (int col = faces[i].x; col < faces[i].x + faces[i].width; col++) {
				if (gray.at<uchar>(row, col) == 255) {
					xmin = min(xmin, col);
					xmax = max(xmax, col);
					ymin = min(ymin, row);
					ymax = max(ymax, row);
				}
			}
		}

		faces[i].x = xmin;
		faces[i].y = ymin;
		faces[i].height = ymax - ymin;
		faces[i].width = xmax - xmin;
	}

	imshow("reduced_gray", gray);
	for (int i = 0; i < faces.size(); i++) {
		rectangle(frame, faces[i], Scalar(0, 0, 255));
	}
	imshow(window_name, frame);
	return faces[0];
	waitKey();
}

Rect FaceDetect(const string& window_name, Mat frame, int minFaceSize, int minEyeSize, bool hasFace, bool hasNose, bool hasEyes, bool hasMouth) {

	std::vector<Rect> faces;
	Mat grayFrame;
	cvtColor(frame, grayFrame, CV_BGR2GRAY);
	equalizeHist(grayFrame, grayFrame);

	face_cascade.detectMultiScale(grayFrame, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
		Size(minFaceSize, minFaceSize));
	Rect faceROI = faces[0];

	//reduceRectangles(window_name + "_reduced", frame, faces);
	for (int i = 0; i < faces.size(); i++)
	{
		/*Point center(faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height * 0.5);
		Point first(faces[i].x, faces[i].y);
		Point second(faces[i].x + faces[i].width, faces[i].y + faces[i].height);*/

		/*faces[i].x += 15;
		faces[i].y += 15;
		faces[i].width -= 15;
		faces[i].height -= 15;*/
		rectangle(frame, faces[i], Scalar(0, 0, 255));

		/*std::vector<Rect> eyes;
		Mat faceROI = grayFrame(faces[i]);
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
			Size(minEyeSize, minEyeSize));

		for (int j = 0; hasEyes && j < eyes.size(); j++)
		{
			Point center(faces[i].x + eyes[j].x + eyes[j].width * 0.5,
				faces[i].y + eyes[j].y + eyes[j].height * 0.5);
			int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
			circle(frame, center, radius, Scalar(255, 0, 0), 4, 8, 0);
		}

		Rect rect_mouth;
		rect_mouth.x = faces[i].x + faces[i].width / 3;
		rect_mouth.y = faces[i].y + 0.65 * faces[i].height;
		rect_mouth.width = faces[i].width / 2;
		rect_mouth.height = 0.35 * faces[i].height;

		Mat mouth_ROI = grayFrame(rect_mouth);
		std::vector<Rect> mouth;
		int minMouthSize = 0.2 * minFaceSize;
		mouth_cascade.detectMultiScale(mouth_ROI, mouth, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(minMouthSize, minMouthSize));

		for (int j = 0; hasMouth && j < mouth.size(); j++)
		{
			Point center(rect_mouth.x + mouth[j].x + mouth[j].width * 0.5,
				rect_mouth.y + mouth[j].y + mouth[j].height * 0.5);
			int radius = cvRound((mouth[j].width + mouth[j].height) * 0.25);
			circle(frame, center, radius, Scalar(0, 255, 0), 4, 8, 0);
		}
		Rect rect_nose;
		rect_nose.x = faces[i].x + faces[i].width / 3;
		rect_nose.y = faces[i].y + 0.3 * faces[i].height;
		rect_nose.width = faces[i].width / 2;
		rect_nose.height = 0.5 * faces[i].height;

		Mat nose_ROI = grayFrame(rect_nose);
		std::vector<Rect> nose;
		int minNoseSize = 0.10 * minFaceSize;
		nose_cascade.detectMultiScale(nose_ROI, nose, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(minNoseSize, minNoseSize));

		for (int j = 0; hasNose && j < nose.size(); j++)
		{
			Point center(rect_nose.x + nose[j].x + nose[j].width * 0.5,
				rect_nose.y + nose[j].y + nose[j].height * 0.5);
			int radius = cvRound((nose[j].width + nose[j].height) * 0.25);
			circle(frame, center, radius, Scalar(0, 0, 255), 4, 8, 0);
		}*/
	}
	L3_ColorModel_Build_2_Personal(frame);
	//imshow(window_name, frame);
	return faces[0];
	waitKey();
}

//void ViolaJonesDetection() {
//	String face_cascade_name = "haarcascade_frontalface_alt.xml";
//	String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
//	String mouth_cascade_name = "haarcascade_mcs_mouth.xml";
//	String nose_cascade_name = "haarcascade_mcs_nose.xml";
//
//	if (!face_cascade.load(face_cascade_name))
//	{
//		printf("Error loading face cascades !\n");
//		return;
//	}
//	if (!eyes_cascade.load(eyes_cascade_name))
//	{
//		printf("Error loading eyes cascades !\n");
//		return;
//	}
//	if (!mouth_cascade.load(mouth_cascade_name))
//	{
//		printf("Error loading mouth cascades !\n");
//		return;
//	}
//	if (!nose_cascade.load(nose_cascade_name))
//	{
//		printf("Error loading nose cascades !\n");
//		return;
//	}
//	VideoCapture cap("Videos/test_msv1_short.avi");
//	if (!cap.isOpened()) {
//		printf("Cannot open video capture device.\n");
//		waitKey();
//		return;
//	}
//	Mat frame, gray;
//	Mat backgnd;
//	Mat diff;
//	Mat dst;
//	char c;
//	int frameNum = -1;
//
//	cap.read(frame);
//	const unsigned char Th = 25;
//	const double alpha = 0.05;
//
//	for (;;) {
//		double t = (double)getTickCount();
//
//		cap >> frame;
//		if (frame.empty())
//		{
//			printf("End of video file\n");
//			break;
//		}
//		++frameNum;
//		if (frameNum == 0)
//			imshow("src", frame);
//		cvtColor(frame, gray, CV_BGR2GRAY);
//		//GaussianBlur(gray, gray, Size(5, 5), 0.8, 0.8);
//		dst = Mat::zeros(gray.size(), gray.type());
//		const int channels_gray = gray.channels();
//		if (channels_gray > 1)
//			return;
//		if (frameNum > 0)
//		{
//			int minFaceSize = 50;
//			int minEyeSize = minFaceSize / 5;
//			Rect faceROI = FaceDetect("face", frame, minFaceSize, minEyeSize, true, false, false, false);
//			absdiff(gray, backgnd, diff);
//			backgnd = gray.clone();
//			imshow("diff", diff);
//			for (int i = 0; i < dst.rows; i++)
//			{
//				for (int j = 0; j < dst.cols; j++) {
//					if (diff.at<uchar>(i, j) > Th)
//					{
//						dst.at<uchar>(i, j) = 255;
//					}
//				}
//			}
//			Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));
//			Mat temp = dst(faceROI);
//			imshow("src", frame);
//			erode(temp, temp, element, Point(-1, -1), 1);
//			dilate(temp, temp, element, Point(-1, -1), 1);
//			//dilate(temp, temp, element, Point(-1, -1), 1);
//			//erode(temp, temp, element, Point(-1, -1), 1);
//			imshow("tempFinal", temp);
//
//			typedef struct {
//				double arie;
//				double xc;
//				double yc;
//			} mylist;
//			vector<mylist> candidates;
//			candidates.clear();
//			vector<vector<Point> > contours;
//			vector<Vec4i> hierarchy;
//			Mat roi = Mat::zeros(temp.rows, temp.cols, CV_8UC3);
//			findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
//			Moments m;
//			if (contours.size() > 0)
//			{
//				int idx = 0;
//				for (; idx >= 0; idx = hierarchy[idx][0])
//				{
//					const vector<Point>& c = contours[idx];
//					m = moments(c);
//					double arie = m.m00;
//					double xc = m.m10 / m.m00;
//					double yc = m.m01 / m.m00;
//					Scalar color(rand() & 255, rand() & 255, rand() & 255);
//					drawContours(roi, contours, idx, color, CV_FILLED, 8, hierarchy);
//					mylist elem;
//					elem.arie = arie;
//					elem.xc = xc;
//					elem.yc = yc;
//					candidates.push_back(elem);
//				}
//			}
//			if (candidates.size() >= 2)
//			{
//				mylist leftEye = candidates[0], rightEye = candidates[0];
//				double arie1 = 0, arie2 = 0;
//				for (mylist e : candidates)
//				{
//					if (e.arie > arie1)
//					{
//						arie2 = arie1;
//						leftEye = rightEye;
//						arie1 = e.arie;
//						rightEye = e;
//					}
//					else
//					{
//						if (e.arie > arie2)
//						{
//							arie2 = e.arie;
//							leftEye = e;
//						}
//					}
//				}
//				if ((abs(rightEye.yc - leftEye.yc) < 0.1 * faceROI.height && abs(rightEye.yc - leftEye.yc) < (faceROI.height) / 2))
//
//					if (abs(rightEye.xc - leftEye.xc) > 0.3 * faceROI.width && abs(rightEye.xc - leftEye.xc) < 0.5 * faceROI.width)
//						if (rightEye.xc - leftEye.xc > 0) {
//							if (leftEye.xc <= (faceROI.width) / 2 && rightEye.xc >= (faceROI.width) / 2)
//							{
//								DrawCross(roi, Point(rightEye.xc, rightEye.yc), 15, Scalar(255, 0, 0), 2);
//								DrawCross(roi, Point(leftEye.xc, leftEye.yc), 15, Scalar(0, 0, 255), 2);
//								rectangle(frame, faceROI, Scalar(0, 255, 0));
//								imshow("sursa", frame);
//							}
//						}
//						else if (leftEye.xc >= (faceROI.width) / 2 && rightEye.xc <= (faceROI.width) / 2) {
//							{
//								DrawCross(roi, Point(leftEye.xc, leftEye.yc), 15, Scalar(0, 255, 255), 2);
//								DrawCross(roi, Point(rightEye.xc, rightEye.yc), 15, Scalar(0, 255, 0), 2);
//								rectangle(frame, faceROI, Scalar(0, 255, 0));
//								imshow("sursa", frame);
//							}
//						}
//			}
//			imshow("colored", roi);
//		}
//		else
//			backgnd = gray.clone();
//		c = cvWaitKey(0);
//		if (c == 27) {
//			printf("ESC pressed - playback finished\n");
//			break;
//		}
//		t = ((double)getTickCount() - t) / getTickFrequency();
//		printf("%d - %.3f [ms]\n", frameNum, t * 1000);
//	}
//}

void ViolaJonesPersonal() {
	String face_cascade_name = "haarcascade_frontalface_alt.xml";
	String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
	String mouth_cascade_name = "haarcascade_mcs_mouth.xml";
	String nose_cascade_name = "haarcascade_mcs_nose.xml";

	if (!face_cascade.load(face_cascade_name)) {
		printf("Error loading face cascades! \n");
		return;
	}

	if (!eyes_cascade.load(eyes_cascade_name)) {
		printf("Error loading eyes cascades! \n");
		return;
	}

	if (!mouth_cascade.load(mouth_cascade_name)) {
		printf("Error loading mouth cascades! \n");
		return;
	}

	if (!nose_cascade.load(nose_cascade_name)) {
		printf("Error loading nose cascades! \n");
		return;
	}

	int option;

	printf("Select:\n1.Image\n2.Video\n");
	scanf("%d", &option);
	switch (option) {
	case 1: {
		Mat src;
		Mat hsv;


		char fname[MAX_PATH];
		while (openFileDlg(fname)) {
			src = imread(fname);
			Mat_<uchar> dst(src.rows, src.cols);
			int minFaceSize = 50;
			int minEyeSize = minFaceSize / 5;
			Rect faceROI = FaceDetect("face", src, minFaceSize, minEyeSize, true, false, false, false);
			//imshow("abc", dst);
			/*GaussianBlur(src, src, Size(5, 5), 0, 0);
			namedWindow("src", 1);
			Mat_<uchar> H(src.rows, src.cols);
			uchar* lpH = H.data;
			cvtColor(src, hsv, CV_BGR2HSV);
			uchar* hsvDataPtr = hsv.data;
			for (int i = 0; i < src.rows; i++) {
				for (int j = 0; j < src.cols; j++) {
					int hi = i * src.cols * 3 + j * 3;
					int gi = i * src.cols + j;
					lpH[gi] = hsvDataPtr[hi] * 510 / 360;
				}
			}
			float k = 2.5;
			for (int i = 0; i < src.rows; i++) {
				for (int j = 0; j < src.cols; j++) {
					if ((hue_mean - k * hue_std) <= H(i, j) && ((hue_mean + k * hue_std) >= H(i, j))) {
						dst(i, j) = 255;
					}
					else {
						dst(i, j) = 0;
					}
				}
			}
			imshow("segmentare", dst);*/
		}
		break;
	}
	case 2:
		VideoCapture cap = ("Videos/Megamind.avi");
		if (!cap.isOpened()) {
			printf("Cannot open video capture device.\n");
			waitKey();
			return;
		}

		Mat frame;
		Mat gray;
		Mat backgnd;
		Mat diff;
		Mat dst;
		char c;
		int frameNum = -1;

		cap.read(frame);
		const unsigned char Th = 25;
		const double alpha = 0.05;

		for (;;) {
			double t = (double)getTickCount();

			cap >> frame;
			if (frame.empty()) {
				printf("End of video file\n");
				break;
			}
			++frameNum;
			if (frameNum == 0)
				imshow("src", frame);
			cvtColor(frame, gray, CV_BGR2GRAY);
			dst = Mat::zeros(gray.size(), gray.type());
			const int channels_gray = gray.channels();
			if (channels_gray > 1)
				return;
			if (frameNum > 0) {
				int minFaceSize = 50;
				int minEyeSize = minFaceSize / 5;
				Rect faceROI = FaceDetect("face", frame, minFaceSize, minEyeSize, true, false, false, false);
				/*absdiff(gray, backgnd, diff);
				backgnd = gray.clone();
				imshow("diff", diff);
				for (int i = 0; i < dst.rows; i++) {
					for (int j = 0; j < dst.cols; j++) {
						if (diff.at<uchar>(i, j) > Th) {
							dst.at<uchar>(i, j) = 255;
						}
					}
				}

				Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));
				Mat temp = dst(faceROI);
				imshow("src", frame);*/
			}
			else {
				backgnd = gray.clone();
			}
			c = cvWaitKey(0);
			if (c == 27) {
				printf("ESC pressed - playback finished\n");
				break;
			}
			t = ((double)getTickCount() - t) / getTickFrequency();
			printf("%d - %.3f [ms]\n", frameNum, t * 1000);
		}


		break;
	}
}

void write_to_file() {
	std::ofstream outFile("..\\..\\values.txt");
	if (!outFile.is_open()) {
		std::cout << "Error opening file for writing." << std::endl;
		std::chrono::seconds dura(5);
		std::this_thread::sleep_for(dura);
		return;
	}

	// write the values to the file
	outFile << hue_mean << std::endl;
	outFile << saturation_mean << std::endl;
	outFile << hue_std << std::endl;
	outFile << saturation_std << std::endl;

	outFile.close();
	std::cout << "Values written to file successfully!" << std::endl;
	std::chrono::seconds dura(5);
	std::this_thread::sleep_for(dura);
}

std::tuple<int, int, int, int> read_from_file() {
	char cwd[1024];
	if (_getcwd(cwd, sizeof(cwd)) != NULL) {
		std::cout << "Current working directory: " << cwd << std::endl;
	}
	else {
		std::cout << "Error getting current working directory" << std::endl;
	}

	std::ifstream inFile("values.txt");
	if (!inFile.is_open()) {
		std::cout << "Error opening file for reading." << std::endl;
		return std::make_tuple(-1, -1, -1, -1);
	}

	// read the values from the file
	int hue_mean, saturation_mean, hue_std, saturation_std;
	inFile >> hue_mean >> saturation_mean >> hue_std >> saturation_std;

	// close the file
	inFile.close();

	// print the values to the console
	std::cout << "Read values:\n" <<"Hue Mean: "<< hue_mean  <<"\nSaturation Mean: "<< saturation_mean << "\nHue Std : "<< hue_std << "\nSaturation Std: " << saturation_std << '\n';
	std::cout << endl;
	std::cout << "Thresholds:\nHue Mean: (" << hue_mean - 0.25 * hue_mean <<", "<<hue_mean + 0.25 * hue_mean<<")\n";
	std::cout << "Saturation Mean: (" << saturation_mean - 0.25 * saturation_mean << ", " << saturation_mean + 0.25 * saturation_mean << ")\n";
	std::cout << "Hue Std: (" << hue_std - 0.25 * hue_std << ", " << hue_std + 0.25 * hue_std << ")\n";
	std::cout << "Saturation Std: (" << saturation_std - 0.25 * saturation_std << ", " << saturation_std + 0.25 * saturation_std << ")\n";

	return std::make_tuple(hue_mean, saturation_mean, hue_std, saturation_std);
}

void ViolaJonesPersonalTest() {
	String face_cascade_name = "haarcascade_frontalface_alt.xml";
	String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
	String mouth_cascade_name = "haarcascade_mcs_mouth.xml";
	String nose_cascade_name = "haarcascade_mcs_nose.xml";

	if (!face_cascade.load(face_cascade_name)) {
		printf("Error loading face cascades! \n");
		return;
	}

	if (!eyes_cascade.load(eyes_cascade_name)) {
		printf("Error loading eyes cascades! \n");
		return;
	}

	if (!mouth_cascade.load(mouth_cascade_name)) {
		printf("Error loading mouth cascades! \n");
		return;
	}

	if (!nose_cascade.load(nose_cascade_name)) {
		printf("Error loading nose cascades! \n");
		return;
	}

	int option;
	printf("Select:\n1.Image\n2.Video\n");
	scanf("%d", &option);
	switch (option) {
	case 1: {
		Mat src;
		Mat hsv;

		int hue_mean, saturation_mean, hue_std, saturation_std;
		std::tie(hue_mean, saturation_mean, hue_std, saturation_std) = read_from_file();
		std::chrono::seconds dura(5);
		std::this_thread::sleep_for(dura);
		break;


		char fname[MAX_PATH];
		while (openFileDlg(fname)) {
			src = imread(fname);
			Mat_<uchar> dst(src.rows, src.cols);
			int minFaceSize = 50;
			int minEyeSize = minFaceSize / 5;
			//Rect faceROI = FaceDetect("face", src, minFaceSize, minEyeSize, true, false, false, false);
		}
		break;
	}
	case 2: {
		break;
	}
	
	}
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - RGB2HSV\n");
		printf(" 11 - Binarizare cu threshold\n");
		printf(" 12 - Binarizare automata\n");
		printf(" 13 - binarizare H\n");
		printf(" 14 - L3_ColorModel_Build\n");
		printf(" 15 - L3_ColorModel_Build2\n");
		printf(" 16 - Face Detection Eyes\n");
		printf(" 17 - Face Detection Mouth & Nose\n");
		printf(" 18 - Face Detection Video\n");
		printf(" 19 - Face Detection lbp\n");
		printf(" 20 - Face Detection Viola-Jones\n");
		printf(" 21 - Face detection Viola-Jones Personal\n");
		printf(" 22 - Calculate Hue mean\n");
		printf(" 23 - Calculate Saturation mean\n");
		printf(" 24 - Calculate Hue std\n");
		printf(" 25 - Calculate Saturation std\n");
		printf(" 26 - Write values (means, stds) to file\n");
		printf(" 27 - Test HSV face detection\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				RGB2HSV();
				break;
			case 11:
				binarizare();
				break;
			case 12:
				binarizare_automata();
				break;
			case 13:
				binarizare_H();
				break;
			case 14:
				L3_ColorModel_Build();
				break;
			case 15:
				L3_ColorModel_Build_2();
				//L3_ColorModel_Save();
				break;
			case 16:
				face_detection_eyes();
				//L3_ColorModel_Save();
				break;
			case 17:
				face_detection_mouth_nose();
				//L3_ColorModel_Save();
				break;
			case 18:
				face_detect_video();
				//L3_ColorModel_Save();
				break;
			case 19:
				face_detection_lbp();
				//L3_ColorModel_Save();
				break;
			/*case 20:
				ViolaJonesDetection();
				break;*/
			case 21:
				ViolaJonesPersonal();
				break;
			case 22:
				calculate_hue_mean();
				break;
			case 23:
				calculate_saturation_mean();
				break;
			case 24:
				calculate_hue_std();
				break;
			case 25:
				calculate_saturation_std();
				break;
			case 26:
				write_to_file();
				break;
			case 27:
				ViolaJonesPersonalTest();
				break;

		}
	}
	while (op!=0);
	return 0;
}