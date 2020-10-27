#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Environment.h"
#include "Raytracing.h"
#include "CudaOps.cuh"
//#include "Camera.h"
//#include "Tools.h"
//#include <opencv2/imgproc/imgproc.hpp>
//#include <cv.h>
using namespace std;
using namespace cv;


Mat HU_norm_image(Mat img, float max, float min) {
	float norm_key = 255 / (max - min);
	Mat img_ = cv::Mat::zeros(Size(512, 512), CV_8UC1);
	for (int y = 0; y < img.cols; y++) {
		for (int x = 0; x < img.rows; x++) {			
			double hu = img.at<uint16_t>(y, x) - 32768;
			if (hu > max)
				img_.at<uint8_t>(y, x) = 255;
			if (hu > min)
				img_.at<uint8_t>(y, x) = int((hu - min) * norm_key);

		}
	}
	return img_;
} 
int main() {
	CudaOperator CudaO();
	int* a, * b, * c;
	a = (int*)malloc(1024 * sizeof(int));
	b = (int*)malloc(1024 * sizeof(int));
	c = (int*)malloc(1024 * sizeof(int));
	return 0;
}
int proxyMain() {
	Environment Env;
	//Env.Run();
	//Ray R(camera, 3.14 * 0.5, 3.14*0., 1.);

	return 0;

	string image_path = "E:\\NIH_images\\000001_01_01\\110.png";
	Mat img = imread(image_path, cv::IMREAD_UNCHANGED);
	cout << img.type() << endl;
	//img = HU_norm_image(img, -100, -750);
	imshow("Image", img);
	waitKey();
}
