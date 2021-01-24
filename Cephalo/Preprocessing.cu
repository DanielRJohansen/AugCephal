#include "Preprocessing.cuh"



void Preprocessor::insertImInVolume(cv::Mat img, int z) {
    for (int y = 0; y < input_size.y; y++) {
        for (int x = 0; x < input_size.x; x++) {
            int hu = img.at<uint16_t>(y, x) - 32768;
            //if (hu < HU_MIN || hu > HU_MAX) 
              //  volume[xyzToIndex(x, y, z)].ignore = true;
            raw_scan[xyzToIndex(Int3(x,y,z), input_size)] = (float) hu;
        }
    }
}

void Preprocessor::loadScans(string folder_path) {
    stringvec v;
    read_directory(folder_path, v);
    for (int i = 2; i < input_size.z + 2; i++) {
        string im_path = folder_path;
        im_path.append(v[i]);
        cout << '\r' << im_path;

        cv::Mat img = imread(im_path, cv::IMREAD_UNCHANGED);
        int z = input_size.z - 1 - i + 2;
        if (img.empty()) {
            cout << "imload failed" << endl;
            return;
        }
        //saveNormIm(img, i - 2, "163slices");
        insertImInVolume(img, z);
    }
}