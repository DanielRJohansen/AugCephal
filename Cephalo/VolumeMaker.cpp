#include "VolumeMaker.h"


void read_directory(const string& name, stringvec& v)
{
    string pattern(name);
    pattern.append("\\*");
    WIN32_FIND_DATA data;
    HANDLE hFind;
    if ((hFind = FindFirstFile(pattern.c_str(), &data)) != INVALID_HANDLE_VALUE) {
        do {
            v.push_back(data.cFileName);
        } while (FindNextFile(hFind, &data) != 0);
        FindClose(hFind);
    }
}


VolumeMaker::VolumeMaker() {
    volume = new Block[512 * 512 * VOL_Z];
    loadScans();
}
void VolumeMaker::loadScans() {
    stringvec v;
    read_directory(folder_path, v);
    for (int i = 2; i < VOL_Z+2; i++) {
        string im_path = folder_path;
        im_path.append(v[i]);
        cout << im_path << endl;

        Mat img = imread(im_path, cv::IMREAD_UNCHANGED);
        int z = i - 2;
        if (img.empty()) {
            cout << "imload failed" << endl;
            return;
        }
        insertImInVolume(img, z);
    }
}

void VolumeMaker::insertImInVolume(Mat img, int z) {
    float min = -400;
    float max = 400;
    float norm_key = 1. / (max - min);
    Mat img_ = cv::Mat::zeros(Size(512, 512), CV_8UC1);
    for (int y = 0; y < img.cols; y++) {
        for (int x = 0; x < img.rows; x++) {
            double hu = img.at<uint16_t>(y, x) - 32768;
            if (hu < min)
                volume[xyzToIndex(x, y, z)].air = true;

            if (hu > max)
                volume[xyzToIndex(x, y, z)].value = 1;
            else if (hu > min)
                volume[xyzToIndex(x, y, z)].value = (hu - min) * norm_key;
            else
                volume[xyzToIndex(x, y, z)].value = 0;
        }
    }

}