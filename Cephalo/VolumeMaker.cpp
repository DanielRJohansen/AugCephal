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
    volume = new Block[VOL_X * VOL_Y * VOL_Z];
    loadScans();
    CudaOperator CudaOps;
    CudaOps.medianFilter(copyVolume(volume), volume);
 
    categorizeBlocks();
    //medianFilter();
    //copyVolume(volume, volume_original);
    //medianFilter();
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
    
    float norm_key = 1. / (HU_MAX - HU_MIN);
    Mat img_ = cv::Mat::zeros(Size(512, 512), CV_8UC1);
    for (int y = 0; y < img.cols; y++) {
        for (int x = 0; x < img.rows; x++) {
            double hu = img.at<uint16_t>(y, x) - 32768;
            if (hu < HU_MIN) {
                volume[xyzToIndex(x, y, z)].air = true;
                volume[xyzToIndex(x, y, z)].value = 0;
            }
            else if (hu > HU_MAX) {
                volume[xyzToIndex(x, y, z)].value = 1;
                volume[xyzToIndex(x, y, z)].bone = true;

            }
            else if (hu > 100 && hu < 300) {
                volume[xyzToIndex(x, y, z)].soft_tissue = true;
                volume[xyzToIndex(x, y, z)].value = (hu - HU_MIN) * norm_key;

            }
            else if (hu > -120 && hu < -90) {
                volume[xyzToIndex(x, y, z)].fat = true;
                volume[xyzToIndex(x, y, z)].value = (hu - HU_MIN) * norm_key;
            }
            else 
                volume[xyzToIndex(x, y, z)].value = (hu - HU_MIN) * norm_key;
            
            //volume[xyzToIndex(x, y, z)].cluster->mean = volume[xyzToIndex(x, y, z)].value;
        }
    }
}

Block* VolumeMaker::copyVolume(Block* from) {
    Block* to = new Block[VOL_X * VOL_Y * VOL_Z];
    for (int i = 0; i < VOL_X * VOL_Y * VOL_Z; i++) {
        to[i] = from[i];
    }
    return to;
}

void VolumeMaker::cluster() {
    Block* vol_copy = copyVolume(volume);
    for (int z = 0; z < VOL_Z; z++) {
        printf("Clustering Z %d  \n", z);
        for (int y = 0; y < VOL_Y; y++) {
            for (int x = 0; x < VOL_X; x++) {
                
                for (int zoff = 0; zoff < 2; zoff++) {
                    for (int yoff = 0; yoff < 2; yoff++) {
                        for (int xoff = 0; xoff < 2; xoff++) {
                            if (xoff + yoff + zoff == 0) // Cannot cluster with itself
                                continue;
                            float clusterdif = vol_copy[xyzToIndex(x + xoff, y + yoff, z + zoff)].cluster->mean -
                                vol_copy[xyzToIndex(x, y, z)].cluster->mean;
                            if (abs(clusterdif) > CLUSTER_MAX_SEP) {

                            }
                        }
                    }
                }
            }
        }
    }
}

void VolumeMaker::categorizeBlocks() {
    ColorScheme colorscheme;
    for (int z = 0; z < VOL_Z; z++) {
        printf("Categorizing z level %d  \n", z);
        for (int y = 0; y < VOL_Y; y++) {
            for (int x = 0; x < VOL_X; x++) {
                int block_index = xyzToIndex(x, y, z);
                int hu_index =(int) (volume[block_index].value * (colorscheme.upper_limit- colorscheme.lower_limit -1));
                //if (z == 2)
                  //  cout << block_index << " " << hu_index  << "  " << volume[block_index].value << endl;
                
                volume[block_index].color = colorscheme.colors[hu_index];
                //volume[block_index].name = colorscheme.names[hu_index];
            }
        }
    }
}












void VolumeMaker::medianFilter() {
    Block* vol_copy = copyVolume(volume);
    for (int z = 0; z < VOL_Z; z++) {
        printf("Filtering layer %d  \n", z);
        for (int y = 0; y < VOL_Y; y++) {
            for (int x = 0; x < VOL_X; x++) {

                int block_index = xyzToIndex(x, y, z);
                if (x * y * z == 0 || x == VOL_X - 1 || y == VOL_Y - 1 || z == VOL_Z - 1) {    // Set all edges to air to no (out of mem problems)
                    volume[block_index].air = true;
                }     
                else {
                    //float window_values[27];
                    vector <float>window(27);
                    int i = 0;
                    for (int z_off = -1; z_off < 2; z_off++) {
                        for (int y_off = -1; y_off < 2; y_off++) {
                            for (int x_off = -1; x_off < 2; x_off++) {
                                window[i] = vol_copy[xyzToIndex(x+x_off, y+y_off, z+z_off)].value;
                                //cout << window[i] << endl;
                                i++;
                            }
                        }
                    }
                    sort(window.begin(), window.end());
                    volume[block_index].value = window[14];
                }
            }
        }
    }
}