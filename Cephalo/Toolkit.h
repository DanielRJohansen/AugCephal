#pragma once

#include <iostream>
#include <string>
#include <windows.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "Constants.h"

using namespace std;
using namespace cv;

typedef vector<string> stringvec;

void hello(int a);
void read_directory(const string& name, stringvec& v);
void saveNormIm(Mat im, int number, string foldername);
