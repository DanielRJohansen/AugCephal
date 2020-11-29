#include "Toolkit.h"


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
void saveNormIm(Mat im, int number, string foldername) {
    float norm_key = 1. / (700 + 200);
    int a;
    for (int y = 0; y < im.cols; y++) {
        for (int x = 0; x < im.rows; x++) {
            double hu = im.at<uint16_t>(y, x) - 32768;
            if (hu < HU_MIN) {
                im.at<uint16_t>(y, x) = 0;
                a = 0;
            }
            else if (hu > HU_MAX) {
                im.at<uint16_t>(y, x) = 1 * 65500;
                a = 1 * 65500;
            }
            else {
                im.at<uint16_t>(y, x) = (hu - HU_MIN) * norm_key * 65500;
                a = (hu - HU_MIN) * norm_key * 65500;
            }
            //outfile << to_string(a) + ' ';
        }
    }
    imwrite("E:\\NormImages\\" + foldername + "\\" + to_string(number) + ".png", im);
}