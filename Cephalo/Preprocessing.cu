#include "Preprocessing.cuh"



void Preprocessor::insertImInVolume(cv::Mat img, int z) {
    for (int y = 0; y < input_size.y; y++) {
        for (int x = 0; x < input_size.x; x++) {
            float hu = img.at<uint16_t>(y, x) - 32768.; // Load as float here, as we do all further calcs on floats in GPU
            raw_scan[xyzToIndex(Int3(x,y,z), input_size)] = hu;
        }
    }
}

void Preprocessor::loadScans(string folder_path) {
    int successes = 0;
    stringvec v;
    printf("Reading directory %s\n", folder_path.c_str());
    read_directory(folder_path, v);
    for (int i = 2; i < input_size.z + 2; i++) {
        string im_path = folder_path;
        im_path.append(v[i]);
        printf("Loading slice: %s               \r", im_path.c_str());

        cv::Mat img = imread(im_path, cv::IMREAD_UNCHANGED);
        int z = input_size.z - 1 - i + 2;
        if (img.empty()) {
            cout << "\n        Failed!\n" << endl;
            return;
        }
        else successes++;
        insertImInVolume(img, z);
    }

    printf("\n%d Slices loaded succesfully\n", successes);
}